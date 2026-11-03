#!/usr/bin/env python3
"""Pilot annual, JJA, and same-year DJF air-temperature exports from GEE.

The pilot uses the existing GECC 2020 Census Tract and ZCTA assets and two
daily Earth Engine collections:

* GRIDMET: IDAHO_EPSCOR/GRIDMET (tmmx/tmmn, Kelvin, 1979-present)
* PRISM 800 m: projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_DAILY
  (tmax/tmin, degrees Celsius, 1981-present)

No source rasters are downloaded. The script builds six temporal-mean bands
per year (annual/JJA/DJF x tmax/tmin), computes polygon statistics for a
deterministic stratified sample, and optionally exports four pilot CSVs to
Google Drive (one per dataset/geography combination). In native mode, the
zonal reductions use the source band's exact CRS and affine transform rather
than only its nominal pixel scale.

Dry run is the default. Examples:

    python pilot_prism_gridmet_airtemp.py
    python pilot_prism_gridmet_airtemp.py --preview-rows 3
    python pilot_prism_gridmet_airtemp.py --submit

The default winter convention matches the existing GECC workflow: January,
February, and December from the same labeled calendar year. Pass
--djf-convention ending-year to use December of year-1 plus January-February
of the labeled year.
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from typing import Iterable

import ee


EE_PROJECT = "your-ee-project"
DEFAULT_YEARS = "1980,1981,2000,2020"
DEFAULT_DRIVE_FOLDERS = {
    "native": "GECC_PRISM_GRIDMET_airtemp_native_aligned_pilot",
    "gecc-1km": "GECC_PRISM_GRIDMET_airtemp_pilot",
}

CONUS_BOUNDS = (-125.0, 24.0, -66.0, 50.0)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    collection_id: str
    tmax_band: str
    tmin_band: str
    units: str
    first_year: int
    native_scale_m: float


@dataclass(frozen=True)
class GeographyConfig:
    name: str
    asset_id: str
    id_field: str
    area_field: str


DATASETS = {
    "gridmet": DatasetConfig(
        name="GRIDMET",
        collection_id="IDAHO_EPSCOR/GRIDMET",
        tmax_band="tmmx",
        tmin_band="tmmn",
        units="K",
        first_year=1979,
        native_scale_m=4638.312116386398,
    ),
    "prism": DatasetConfig(
        name="PRISM",
        collection_id=(
            "projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_DAILY"
        ),
        tmax_band="tmax",
        tmin_band="tmin",
        units="C",
        first_year=1981,
        native_scale_m=927.6624232401731,
    ),
}

GEOGRAPHIES = {
    "tracts": GeographyConfig(
        name="CensusTracts",
        asset_id="projects/your-ee-project/assets/GECC/censusTracts2020US",
        id_field="GEOID",
        area_field="Area_Land",
    ),
    "zctas": GeographyConfig(
        name="ZCTAs",
        asset_id="projects/your-ee-project/assets/GECC/tl_2020_us_zcta520",
        id_field="GEOID20",
        area_field="ALAND20",
    ),
}

PERIODS = ("annual", "JJA", "DJF")
COUNT_VALIDATION_YEARS_PER_REQUEST = 8
TEMPERATURE_BANDS = tuple(
    f"{period}_{variable}"
    for period in PERIODS
    for variable in ("tmax", "tmin")
)


def parse_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def validate_names(values: Iterable[str], choices: dict[str, object], label: str) -> None:
    invalid = sorted(set(values) - set(choices))
    if invalid:
        raise ValueError(
            f"Unknown {label}: {', '.join(invalid)}. "
            f"Choose from: {', '.join(sorted(choices))}"
        )


def ee_init(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"Earth Engine initialized with project: {project}")


def standardized_daily_collection(config: DatasetConfig) -> ee.ImageCollection:
    """Return a two-band daily collection named tmax/tmin in degrees Celsius."""
    source = ee.ImageCollection(config.collection_id)

    def standardize(image: ee.Image) -> ee.Image:
        result = image.select(
            [config.tmax_band, config.tmin_band],
            ["tmax", "tmin"],
        )
        if config.units == "K":
            result = result.subtract(273.15)
        return result.toFloat().copyProperties(image, image.propertyNames())

    return source.map(standardize)


def source_projection(config: DatasetConfig) -> ee.Projection:
    """Return the exact projection and grid transform of the source tmax band."""
    first = ee.Image(ee.ImageCollection(config.collection_id).first())
    return first.select(config.tmax_band).projection()


def period_collection(
    daily: ee.ImageCollection,
    year: int,
    period: str,
    djf_convention: str,
) -> ee.ImageCollection:
    if period == "annual":
        return daily.filterDate(f"{year}-01-01", f"{year + 1}-01-01")

    if period == "JJA":
        return daily.filterDate(f"{year}-06-01", f"{year}-09-01")

    if period != "DJF":
        raise ValueError(f"Unknown period: {period}")

    if djf_convention == "ending-year":
        return daily.filterDate(f"{year - 1}-12-01", f"{year}-03-01")

    same_year = daily.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    winter_months = ee.Filter.Or(
        ee.Filter.calendarRange(1, 2, "month"),
        ee.Filter.calendarRange(12, 12, "month"),
    )
    return same_year.filter(winter_months)


def expected_period_count(year: int, period: str, djf_convention: str) -> int:
    if period == "annual":
        return 366 if calendar.isleap(year) else 365
    if period == "JJA":
        return 92
    if period == "DJF":
        february_year = year
        if djf_convention == "same-year":
            february_year = year
        return 31 + 31 + (29 if calendar.isleap(february_year) else 28)
    raise ValueError(f"Unknown period: {period}")


def validate_requested_years(
    daily: ee.ImageCollection,
    config: DatasetConfig,
    years: list[int],
    djf_convention: str,
) -> tuple[list[int], dict[int, dict[str, int]]]:
    """Validate all requested daily counts with one metadata request."""
    count_expressions: dict[str, ee.Number] = {}
    eligible_years: list[int] = []

    for year in years:
        if year < config.first_year:
            print(
                f"  skip {config.name} {year}: collection begins in "
                f"{config.first_year}"
            )
            continue
        eligible_years.append(year)
        for period in PERIODS:
            count_expressions[f"{year}:{period}"] = period_collection(
                daily,
                year,
                period,
                djf_convention,
            ).size()

    if not count_expressions:
        return [], {}

    resolved: dict[str, int] = {}
    count_items = list(count_expressions.items())
    chunk_size = COUNT_VALIDATION_YEARS_PER_REQUEST * len(PERIODS)
    for start in range(0, len(count_items), chunk_size):
        chunk = dict(count_items[start : start + chunk_size])
        resolved.update(ee.Dictionary(chunk).getInfo())
    valid_years: list[int] = []
    counts_by_year: dict[int, dict[str, int]] = {}

    for year in eligible_years:
        counts = {
            period: int(resolved[f"{year}:{period}"])
            for period in PERIODS
        }
        incomplete = next(
            (
                period
                for period in PERIODS
                if counts[period]
                != expected_period_count(year, period, djf_convention)
            ),
            None,
        )
        if incomplete is not None:
            expected = expected_period_count(year, incomplete, djf_convention)
            print(
                f"  skip {config.name} {year}: {incomplete} has "
                f"{counts[incomplete]} daily images; expected {expected}"
            )
            continue

        print(
            f"  valid {config.name} {year}: "
            + ", ".join(f"{period}={counts[period]}" for period in PERIODS)
        )
        valid_years.append(year)
        counts_by_year[year] = counts

    return valid_years, counts_by_year


def build_temperature_image(
    daily: ee.ImageCollection,
    year: int,
    djf_convention: str,
) -> ee.Image:
    images: list[ee.Image] = []
    for period in PERIODS:
        period_mean = period_collection(daily, year, period, djf_convention).mean()
        images.extend(
            [
                period_mean.select("tmax").rename(f"{period}_tmax"),
                period_mean.select("tmin").rename(f"{period}_tmin"),
            ]
        )
    return ee.Image.cat(images).toFloat()


def tag_stratum(
    features: ee.FeatureCollection,
    stratum: str,
) -> ee.FeatureCollection:
    return features.map(lambda feature: feature.set("pilot_stratum", stratum))


def prepare_pilot_sample(
    config: GeographyConfig,
    sample_size: int,
    seed: int,
) -> ee.FeatureCollection:
    """Sample small, large, and random CONUS polygons deterministically."""
    bounds = ee.Geometry.Rectangle(list(CONUS_BOUNDS), geodesic=False)
    candidates = (
        ee.FeatureCollection(config.asset_id)
        .filterBounds(bounds)
        .filter(ee.Filter.notNull([config.id_field, config.area_field]))
        .filter(ee.Filter.gt(config.area_field, 0))
    )

    quarter = max(1, sample_size // 4)
    random_count = max(1, sample_size - (2 * quarter))

    small = tag_stratum(
        candidates.sort(config.area_field).limit(quarter),
        "small_area",
    )
    large = tag_stratum(
        candidates.sort(config.area_field, False).limit(quarter),
        "large_area",
    )
    random = tag_stratum(
        candidates.randomColumn("pilot_random", seed).sort("pilot_random").limit(
            random_count
        ),
        "random",
    )

    sample = small.merge(large).merge(random).distinct([config.id_field])

    def retain_fields(feature: ee.Feature) -> ee.Feature:
        return ee.Feature(
            feature.geometry(),
            {
                config.id_field: feature.get(config.id_field),
                "source_area_m2": feature.get(config.area_field),
                "pilot_stratum": feature.get("pilot_stratum"),
            },
        )

    return sample.map(retain_fields)


def combined_reducer() -> ee.Reducer:
    return (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.count(), sharedInputs=True)
    )


def output_metric_fields() -> list[str]:
    return [
        f"{band}_{statistic}"
        for band in TEMPERATURE_BANDS
        for statistic in ("mean", "median", "stdDev", "count")
    ]


def reduce_one_year(
    image: ee.Image,
    polygons: ee.FeatureCollection,
    dataset: DatasetConfig,
    geography: GeographyConfig,
    year: int,
    period_counts: dict[str, int],
    djf_convention: str,
    reduction_scale_m: float,
    tile_scale: int,
    reduction_projection: ee.Projection | None = None,
    reduction_transform: list[float] | None = None,
) -> ee.FeatureCollection:
    reduction_args: dict[str, object] = {
        "collection": polygons,
        "reducer": combined_reducer(),
        "tileScale": tile_scale,
    }
    if reduction_projection is None:
        # Retain the established GECC scale-only behavior for 1 km products.
        reduction_args["scale"] = reduction_scale_m
        grid_alignment = "scale_only"
    else:
        # A temporal ImageCollection mean loses the source image's default
        # projection. Supply both pieces of the native grid explicitly so GEE
        # cannot choose a shifted grid from only the nominal scale.
        if reduction_transform is None:
            raise ValueError(
                "reduction_transform is required with reduction_projection"
            )
        reduction_args["crs"] = reduction_projection
        reduction_args["crsTransform"] = reduction_transform
        grid_alignment = "source_crs_transform"

    raw = image.reduceRegions(**reduction_args)

    mean_fields = [f"{band}_mean" for band in TEMPERATURE_BANDS]

    def add_metadata(feature: ee.Feature) -> ee.Feature:
        missing_count = ee.Number(
            ee.List(mean_fields)
            .map(
                lambda field: ee.Number(
                    ee.Algorithms.If(
                        ee.Algorithms.IsEqual(feature.get(field), None),
                        1,
                        0,
                    )
                )
            )
            .reduce(ee.Reducer.sum())
        )
        return feature.set(
            {
                "dataset": dataset.name,
                "geography_type": geography.name,
                "year": year,
                "djf_convention": djf_convention,
                "annual_day_count": period_counts["annual"],
                "JJA_day_count": period_counts["JJA"],
                "DJF_day_count": period_counts["DJF"],
                "source_scale_m": dataset.native_scale_m,
                "reduction_scale_m": reduction_scale_m,
                "reduction_grid_alignment": grid_alignment,
                "missing_mean_count": missing_count,
                "all_six_means_valid": missing_count.eq(0),
                "fallback_applied": False,
            }
        )

    return raw.map(add_metadata)


def output_selectors(geography: GeographyConfig) -> list[str]:
    return [
        geography.id_field,
        "source_area_m2",
        "pilot_stratum",
        "dataset",
        "geography_type",
        "year",
        "djf_convention",
        "annual_day_count",
        "JJA_day_count",
        "DJF_day_count",
        "source_scale_m",
        "reduction_scale_m",
        "reduction_grid_alignment",
        "missing_mean_count",
        "all_six_means_valid",
        "fallback_applied",
        *output_metric_fields(),
    ]


def requested_scale(config: DatasetConfig, scale_profile: str) -> float:
    if scale_profile == "gecc-1km":
        return 1000.0
    return config.native_scale_m


def requested_projection(
    config: DatasetConfig,
    scale_profile: str,
) -> ee.Projection | None:
    if scale_profile == "native":
        return source_projection(config)
    return None


def export_description(
    dataset: DatasetConfig,
    geography: GeographyConfig,
    years: list[int],
    scale_profile: str,
    djf_convention: str,
) -> str:
    year_label = str(years[0]) if len(years) == 1 else f"{years[0]}_{years[-1]}"
    djf_label = djf_convention.replace("-", "")
    scale_label = (
        "nativealigned"
        if scale_profile == "native"
        else scale_profile.replace("-", "")
    )
    return (
        f"Pilot_{dataset.name}_{geography.name}_AirTemp_"
        f"{year_label}_{djf_label}_{scale_label}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=EE_PROJECT)
    parser.add_argument("--years", type=parse_years, default=parse_years(DEFAULT_YEARS))
    parser.add_argument(
        "--datasets",
        type=parse_csv,
        default=parse_csv("gridmet,prism"),
        help="Comma-separated choices: gridmet,prism",
    )
    parser.add_argument(
        "--geographies",
        type=parse_csv,
        default=parse_csv("tracts,zctas"),
        help="Comma-separated choices: tracts,zctas",
    )
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--djf-convention",
        choices=("same-year", "ending-year"),
        default="same-year",
    )
    parser.add_argument(
        "--scale-profile",
        choices=("native", "gecc-1km"),
        default="native",
        help=(
            "Use each source's exact native CRS/transform or the existing "
            "GECC 1 km scale-only reduction."
        ),
    )
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument(
        "--drive-folder",
        default=None,
        help="Google Drive folder; defaults according to --scale-profile.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=0,
        help="Evaluate and print this many rows per planned export during dry run.",
    )
    parser.add_argument(
        "--qc-summary",
        action="store_true",
        help=(
            "Evaluate the pilot and print IDs with missing seasonal means. "
            "This is slower than the default coverage-only dry run."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Start Google Drive export tasks. Default is a read-only dry run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_names(args.datasets, DATASETS, "datasets")
    validate_names(args.geographies, GEOGRAPHIES, "geographies")
    if args.sample_size < 4:
        raise ValueError("--sample-size must be at least 4")
    if args.preview_rows < 0:
        raise ValueError("--preview-rows cannot be negative")
    drive_folder = (
        args.drive_folder or DEFAULT_DRIVE_FOLDERS[args.scale_profile]
    )

    ee_init(args.project)

    samples: dict[str, ee.FeatureCollection] = {}
    print("\nPreparing deterministic CONUS pilot samples:")
    for geography_key in args.geographies:
        geography = GEOGRAPHIES[geography_key]
        sample = prepare_pilot_sample(geography, args.sample_size, args.seed)
        sample_count = sample.size().getInfo()
        samples[geography_key] = sample
        print(
            f"  {geography.name}: {sample_count} polygons "
            f"(requested {args.sample_size})"
        )

    planned_exports = 0
    started_tasks = []

    for dataset_key in args.datasets:
        dataset = DATASETS[dataset_key]
        daily = standardized_daily_collection(dataset)
        print(f"\nValidating daily coverage for {dataset.name}:")
        valid_years, counts_by_year = validate_requested_years(
            daily,
            dataset,
            args.years,
            args.djf_convention,
        )

        if not valid_years:
            print(f"  no complete requested years for {dataset.name}; skipping")
            continue

        scale_m = requested_scale(dataset, args.scale_profile)
        projection = requested_projection(dataset, args.scale_profile)
        transform = None
        if projection is not None:
            projection_info = projection.getInfo()
            transform = projection_info["transform"]
            crs_label = projection_info.get("crs") or "custom source WKT"
            print(
                "  native reduction grid: "
                f"{crs_label} transform={transform}"
            )

        for geography_key in args.geographies:
            geography = GEOGRAPHIES[geography_key]
            combined = ee.FeatureCollection([])

            for year in valid_years:
                image = build_temperature_image(daily, year, args.djf_convention)
                yearly = reduce_one_year(
                    image=image,
                    polygons=samples[geography_key],
                    dataset=dataset,
                    geography=geography,
                    year=year,
                    period_counts=counts_by_year[year],
                    djf_convention=args.djf_convention,
                    reduction_scale_m=scale_m,
                    tile_scale=args.tile_scale,
                    reduction_projection=projection,
                    reduction_transform=transform,
                )
                combined = combined.merge(yearly)

            description = export_description(
                dataset,
                geography,
                valid_years,
                args.scale_profile,
                args.djf_convention,
            )
            planned_exports += 1
            print(
                f"\nPlanned export: {description}\n"
                f"  years: {valid_years}\n"
                f"  reduction scale: {scale_m:.3f} m\n"
                f"  grid alignment: "
                f"{'source CRS + transform' if projection is not None else 'scale only'}\n"
                f"  Drive folder: {drive_folder}"
            )

            if args.preview_rows:
                preview = combined.limit(args.preview_rows).getInfo()
                properties = [
                    feature.get("properties", {})
                    for feature in preview.get("features", [])
                ]
                print(f"  preview rows: {properties}")

            if args.qc_summary:
                missing = combined.filter(ee.Filter.gt("missing_mean_count", 0))
                missing_ids = missing.aggregate_array(geography.id_field).distinct().getInfo()
                print(
                    f"  QC missing polygons: {len(missing_ids)} "
                    f"({missing_ids})"
                )

            if args.submit:
                task = ee.batch.Export.table.toDrive(
                    collection=combined,
                    description=description,
                    folder=drive_folder,
                    fileNamePrefix=description,
                    fileFormat="CSV",
                    selectors=output_selectors(geography),
                )
                task.start()
                started_tasks.append((description, task.id))
                print(f"  started task: {task.id}")

    print(f"\nPlanned exports: {planned_exports}")
    if args.submit:
        print(f"Started tasks: {len(started_tasks)}")
        for description, task_id in started_tasks:
            print(f"  {description}: {task_id}")
        print("Monitor at https://code.earthengine.google.com/tasks")
    else:
        print("Dry run only. Add --submit to start Drive exports.")


if __name__ == "__main__":
    main()
