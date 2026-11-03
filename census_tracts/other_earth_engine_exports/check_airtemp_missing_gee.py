#!/usr/bin/env python3
"""Recheck native-pilot missing polygons at alternate Earth Engine scales."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ee

import pilot_prism_gridmet_airtemp as pilot


FILE_CONFIGS = (
    (
        "Pilot_GRIDMET_CensusTracts_AirTemp_1980_2020_sameyear_native.csv",
        "gridmet",
        "tracts",
    ),
    (
        "Pilot_GRIDMET_ZCTAs_AirTemp_1980_2020_sameyear_native.csv",
        "gridmet",
        "zctas",
    ),
    (
        "Pilot_PRISM_CensusTracts_AirTemp_1981_2020_sameyear_native.csv",
        "prism",
        "tracts",
    ),
    (
        "Pilot_PRISM_ZCTAs_AirTemp_1981_2020_sameyear_native.csv",
        "prism",
        "zctas",
    ),
)


def read_missing_ids(path: Path, id_field: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return sorted(
            {
                row[id_field]
                for row in rows
                if float(row["missing_mean_count"] or 0) > 0
            }
        )


def minimal_features(
    geography: pilot.GeographyConfig,
    identifiers: list[str],
) -> ee.FeatureCollection:
    source = ee.FeatureCollection(geography.asset_id).filter(
        ee.Filter.inList(geography.id_field, identifiers)
    )
    return source.map(
        lambda feature: ee.Feature(
            feature.geometry(),
            {
                geography.id_field: feature.get(geography.id_field),
                "source_area_m2": feature.get(geography.area_field),
                "pilot_stratum": "native_missing",
            },
        )
    )


def missing_after_reduction(
    image: ee.Image,
    polygons: ee.FeatureCollection,
    dataset: pilot.DatasetConfig,
    geography: pilot.GeographyConfig,
    scale_m: float,
) -> ee.List:
    stats = pilot.reduce_one_year(
        image=image,
        polygons=polygons,
        dataset=dataset,
        geography=geography,
        year=2020,
        period_counts={"annual": 366, "JJA": 92, "DJF": 91},
        djf_convention="same-year",
        reduction_scale_m=scale_m,
        tile_scale=4,
    )
    return (
        stats.filter(ee.Filter.gt("missing_mean_count", 0))
        .aggregate_array(geography.id_field)
        .distinct()
        .sort()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--project", default="your-ee-project")
    parser.add_argument("--scales", default="1000,500")
    args = parser.parse_args()

    scales = [float(value) for value in args.scales.split(",") if value.strip()]
    pilot.ee_init(args.project)
    results: dict[str, ee.List | list[str]] = {}

    for filename, dataset_key, geography_key in FILE_CONFIGS:
        dataset = pilot.DATASETS[dataset_key]
        geography = pilot.GEOGRAPHIES[geography_key]
        identifiers = read_missing_ids(
            args.folder.resolve() / filename,
            geography.id_field,
        )
        label = f"{dataset.name}_{geography.name}"
        results[f"{label}_native_missing"] = identifiers
        if not identifiers:
            continue

        polygons = minimal_features(geography, identifiers)
        daily = pilot.standardized_daily_collection(dataset)
        image = pilot.build_temperature_image(daily, 2020, "same-year")
        for scale_m in scales:
            results[f"{label}_{int(scale_m)}m_missing"] = missing_after_reduction(
                image,
                polygons,
                dataset,
                geography,
                scale_m,
            )

    print(ee.Dictionary(results).getInfo())


if __name__ == "__main__":
    main()
