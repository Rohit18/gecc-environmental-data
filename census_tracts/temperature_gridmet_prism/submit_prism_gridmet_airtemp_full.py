#!/usr/bin/env python3
"""Submit resumable full GECC PRISM/GRIDMET air-temperature exports.

The production workflow exports one dataset x geography x year CSV from
Earth Engine. It processes all CONUS 2020 Census Tracts and ZCTAs at either
the established 1 km reduction scale or each source dataset's exact native
CRS and affine transform.

Dry run is the default. Submission is deliberately capped with
--max-new-tasks (default: 8). Re-running the same command skips tasks whose
descriptions are already READY, RUNNING, or COMPLETED, so batches are
resumable.

GRIDMET years: 1980-2025
PRISM years:   1981-2025 (the daily collection has no 1980 data)

Polygons without a native pixel are retained with null temperature metrics
and identified by missing_mean_count; no fallback is applied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import ee

import pilot_prism_gridmet_airtemp as pilot


DEFAULT_PROJECT = "your-ee-project"
DEFAULT_YEARS = "1980-2025"
DEFAULT_DRIVE_FOLDERS = {
    "1km": "GECC_PRISM_GRIDMET_airtemp_full",
    "native": "GECC_PRISM_GRIDMET_airtemp_native_aligned",
}
DEFAULT_MAX_NEW_TASKS = 8
TASK_PREFIX = "GECC_"

ACTIVE_OR_DONE_STATES = {"READY", "RUNNING", "COMPLETED"}


@dataclass(frozen=True)
class ExportJob:
    dataset_key: str
    geography_key: str
    year: int
    description: str


def full_conus_features(config: pilot.GeographyConfig) -> ee.FeatureCollection:
    bounds = ee.Geometry.Rectangle(list(pilot.CONUS_BOUNDS), geodesic=False)
    source = (
        ee.FeatureCollection(config.asset_id)
        .filterBounds(bounds)
        .filter(ee.Filter.notNull([config.id_field]))
    )

    return source.map(
        lambda feature: ee.Feature(
            feature.geometry(),
            {
                config.id_field: feature.get(config.id_field),
                "source_area_m2": feature.get(config.area_field),
            },
        )
    )


def output_selectors(geography: pilot.GeographyConfig) -> list[str]:
    return [
        geography.id_field,
        "source_area_m2",
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
        *pilot.output_metric_fields(),
    ]


def task_description(
    dataset: pilot.DatasetConfig,
    geography: pilot.GeographyConfig,
    year: int,
    scale_mode: str,
) -> str:
    scale_label = "native_aligned" if scale_mode == "native" else scale_mode
    return (
        f"{TASK_PREFIX}{dataset.name}_{geography.name}_AirTemp_"
        f"{year}_sameyear_{scale_label}"
    )


def reduction_scale_m(dataset: pilot.DatasetConfig, scale_mode: str) -> float:
    if scale_mode == "1km":
        return 1000.0
    if scale_mode == "native":
        return dataset.native_scale_m
    raise ValueError(f"Unsupported scale mode: {scale_mode}")


def build_jobs(
    years_by_dataset: dict[str, list[int]],
    dataset_keys: list[str],
    geography_keys: list[str],
    scale_mode: str,
) -> list[ExportJob]:
    """Interleave sources/geographies so early batches exercise each path."""
    jobs: list[ExportJob] = []
    all_years = sorted(
        {year for years in years_by_dataset.values() for year in years}
    )
    for year in all_years:
        for dataset_key in dataset_keys:
            if year not in years_by_dataset[dataset_key]:
                continue
            dataset = pilot.DATASETS[dataset_key]
            for geography_key in geography_keys:
                geography = pilot.GEOGRAPHIES[geography_key]
                jobs.append(
                    ExportJob(
                        dataset_key=dataset_key,
                        geography_key=geography_key,
                        year=year,
                        description=task_description(
                            dataset,
                            geography,
                            year,
                            scale_mode,
                        ),
                    )
                )
    return jobs


def existing_task_states() -> dict[str, set[str]]:
    states: dict[str, set[str]] = {}
    for task in ee.batch.Task.list():
        description = task.config.get("description") or ""
        if not description.startswith(TASK_PREFIX):
            continue
        state = str(task.status().get("state", "UNKNOWN"))
        states.setdefault(description, set()).add(state)
    return states


def should_skip(job: ExportJob, states: dict[str, set[str]]) -> bool:
    return bool(states.get(job.description, set()) & ACTIVE_OR_DONE_STATES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--years",
        type=pilot.parse_years,
        default=pilot.parse_years(DEFAULT_YEARS),
    )
    parser.add_argument(
        "--datasets",
        type=pilot.parse_csv,
        default=pilot.parse_csv("gridmet,prism"),
        help="Comma-separated choices: gridmet,prism",
    )
    parser.add_argument(
        "--geographies",
        type=pilot.parse_csv,
        default=pilot.parse_csv("tracts,zctas"),
        help="Comma-separated choices: tracts,zctas",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("1km", "native"),
        default="1km",
        help=(
            "Reduction scale: the established 1 km analysis grid or each "
            "dataset's exact native CRS and affine transform."
        ),
    )
    parser.add_argument(
        "--drive-folder",
        default=None,
        help="Google Drive folder; defaults according to --scale-mode.",
    )
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument(
        "--max-new-tasks",
        type=int,
        default=DEFAULT_MAX_NEW_TASKS,
        help="Maximum tasks to start in this invocation.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Start exports. Default is a read-only dry run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pilot.validate_names(args.datasets, pilot.DATASETS, "datasets")
    pilot.validate_names(args.geographies, pilot.GEOGRAPHIES, "geographies")
    if args.max_new_tasks < 1:
        raise ValueError("--max-new-tasks must be at least 1")
    drive_folder = args.drive_folder or DEFAULT_DRIVE_FOLDERS[args.scale_mode]

    pilot.ee_init(args.project)

    polygons: dict[str, ee.FeatureCollection] = {}
    print("\nPreparing full CONUS geographies:")
    for geography_key in args.geographies:
        geography = pilot.GEOGRAPHIES[geography_key]
        collection = full_conus_features(geography)
        polygons[geography_key] = collection
        print(f"  {geography.name}: {collection.size().getInfo()} features")

    daily_by_dataset: dict[str, ee.ImageCollection] = {}
    native_grid_by_dataset: dict[
        str, tuple[ee.Projection, list[float]]
    ] = {}
    years_by_dataset: dict[str, list[int]] = {}
    counts_by_dataset_year: dict[str, dict[int, dict[str, int]]] = {}

    for dataset_key in args.datasets:
        dataset = pilot.DATASETS[dataset_key]
        daily = pilot.standardized_daily_collection(dataset)
        daily_by_dataset[dataset_key] = daily
        if args.scale_mode == "native":
            projection = pilot.source_projection(dataset)
            projection_info = projection.getInfo()
            native_grid_by_dataset[dataset_key] = (
                projection,
                projection_info["transform"],
            )
        print(f"\nValidating complete daily coverage for {dataset.name}:")
        valid_years, counts_by_year = pilot.validate_requested_years(
            ee.ImageCollection(dataset.collection_id),
            dataset,
            args.years,
            "same-year",
        )
        years_by_dataset[dataset_key] = valid_years
        counts_by_dataset_year[dataset_key] = counts_by_year

    jobs = build_jobs(
        years_by_dataset,
        args.datasets,
        args.geographies,
        args.scale_mode,
    )
    states = existing_task_states()
    pending_jobs = [job for job in jobs if not should_skip(job, states)]
    skipped_jobs = [job for job in jobs if should_skip(job, states)]

    print("\nProduction plan:")
    print(f"  total expected tasks: {len(jobs)}")
    print(f"  already active/completed: {len(skipped_jobs)}")
    print(f"  available to submit: {len(pending_jobs)}")
    print(f"  invocation cap: {args.max_new_tasks}")
    print(f"  scale mode: {args.scale_mode}")
    print(f"  Drive folder: {drive_folder}")
    for dataset_key in args.datasets:
        dataset = pilot.DATASETS[dataset_key]
        print(
            f"  {dataset.name} reduction scale: "
            f"{reduction_scale_m(dataset, args.scale_mode):.6f} m"
        )
    for job in pending_jobs[: args.max_new_tasks]:
        print(f"  next: {job.description}")

    if not args.submit:
        print("\nDry run only. Add --submit to start the listed batch.")
        return

    started: list[tuple[str, str]] = []
    for job in pending_jobs[: args.max_new_tasks]:
        dataset = pilot.DATASETS[job.dataset_key]
        geography = pilot.GEOGRAPHIES[job.geography_key]
        image = pilot.build_temperature_image(
            daily_by_dataset[job.dataset_key],
            job.year,
            "same-year",
        )
        native_projection = None
        native_transform = None
        if args.scale_mode == "native":
            native_projection, native_transform = native_grid_by_dataset[
                job.dataset_key
            ]
        table = pilot.reduce_one_year(
            image=image,
            polygons=polygons[job.geography_key],
            dataset=dataset,
            geography=geography,
            year=job.year,
            period_counts=counts_by_dataset_year[job.dataset_key][job.year],
            djf_convention="same-year",
            reduction_scale_m=reduction_scale_m(dataset, args.scale_mode),
            tile_scale=args.tile_scale,
            reduction_projection=native_projection,
            reduction_transform=native_transform,
        )
        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=job.description,
            folder=drive_folder,
            fileNamePrefix=job.description,
            fileFormat="CSV",
            selectors=output_selectors(geography),
        )
        task.start()
        started.append((job.description, task.id))
        print(f"  started {job.description}: {task.id}")

    print(f"\nStarted {len(started)} production tasks.")
    print("Monitor at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
