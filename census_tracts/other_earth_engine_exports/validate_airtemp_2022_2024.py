#!/usr/bin/env python3
"""Validate the 2022-2024 SVCM-SP air-temperature release in Earth Engine.

This script is read-only. It checks every regional collection for exact JJA
and same-year-DJF dates, validates source band/projection metadata, verifies the
GECC Census Tract, ZCTA, and FUA assets, and compares raw 2020 seasonal
composites with the existing GECC mosaics at one inland point per region.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta

import ee

import airtemp_2022_2024_common as common


BASELINE_POINTS = {
    "africa": ("Nairobi", 36.8219, -1.2921),
    "australia": ("Sydney", 151.2093, -33.8688),
    "europe_asia": ("London", -0.1276, 51.5072),
    "latin_america": ("Sao Paulo", -46.6333, -23.5505),
    "north_america": ("Chicago", -87.6298, 41.8781),
}
BASELINE_TOLERANCE_C = 1e-5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=common.DEFAULT_PROJECT)
    parser.add_argument(
        "--years",
        type=common.parse_years,
        default=list(common.RELEASE_YEARS),
    )
    parser.add_argument(
        "--seasons",
        type=common.parse_csv,
        default=["jja", "djf"],
        help="Comma-separated choices: jja,djf",
    )
    return parser


def full_year_date_anomalies(
    collection: ee.ImageCollection,
    year: int,
    prop_type: str,
) -> tuple[list[str], list[str]]:
    millis = (
        collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.eq("prop_type", prop_type))
        .sort("system:time_start")
        .aggregate_array("system:time_start")
        .getInfo()
    )
    actual = common.millis_to_utc_dates(millis)
    expected: list[str] = []
    current = date(year, 1, 1)
    while current < date(year + 1, 1, 1):
        expected.append(current.isoformat())
        current += timedelta(days=1)
    missing = sorted(set(expected) - set(actual))
    duplicates = sorted(value for value, count in Counter(actual).items() if count > 1)
    return missing, duplicates


def legacy_asset(season_key: str) -> ee.Image:
    suffix = "_1km" if season_key == "jja" else ""
    label = common.SEASONS[season_key].band_label
    return ee.Image(
        f"projects/your-ee-project/assets/GlobalAirTemp_{label}_2020{suffix}"
    )


def main() -> None:
    args = build_parser().parse_args()
    common.validate_release_years(args.years)
    common.validate_choices(args.seasons, common.SEASONS, "seasons")
    common.initialize_earth_engine(args.project)

    collections = common.regional_collections()
    failures: list[str] = []
    warnings: list[str] = []

    print("\nRelease metadata:")
    print(f"  catalog: {common.CATALOG_URL}")
    print(f"  catalog GEE update: {common.CATALOG_UPDATED}")
    print(f"  Earth Engine Python API: {ee.__version__}")
    print(f"  scale factor: {common.SCALE_FACTOR} (stored 0.1 deg C -> deg C)")

    print("\nSource image metadata:")
    for region, collection in collections.items():
        first = ee.Image(collection.filterDate("2022-01-01", "2023-01-01").first())
        info = ee.Dictionary(
            {
                "bands": first.bandNames(),
                "types": first.bandTypes(),
                "projection": first.select(0).projection(),
            }
        ).getInfo()
        print(f"  {region}: bands={info['bands']}, types={info['types']}")
        projection = info["projection"]
        print(f"    projection={projection}")
        if info["bands"] != ["b1"]:
            failures.append(f"{region}: expected one source band named b1")

    print("\nExact full-year date audit:")
    for region, collection in collections.items():
        for year in args.years:
            for prop_type in ("tmax", "tmin"):
                missing, duplicates = full_year_date_anomalies(
                    collection, year, prop_type
                )
                status = "PASS" if not missing and not duplicates else "WARN"
                print(
                    f"  {status} {region} {year} {prop_type}: "
                    f"missing={missing or '-'}, duplicates={duplicates or '-'}"
                )
                if missing or duplicates:
                    warnings.append(
                        f"{region} {year} {prop_type}: "
                        f"missing={missing}, duplicates={duplicates}"
                    )

    print("\nTarget-period counts:")
    counts = common.resolve_period_counts(collections, args.years, args.seasons)
    for year in args.years:
        for season_key in args.seasons:
            expected = common.expected_day_count(year, season_key)
            for region in collections:
                actual = {
                    prop_type: counts[(region, year, season_key, prop_type)]
                    for prop_type in ("tmax", "tmin")
                }
                status = "PASS" if set(actual.values()) == {expected} else "WARN"
                print(
                    f"  {status} {region} {year} {season_key.upper()}: "
                    f"expected={expected}, tmax={actual['tmax']}, "
                    f"tmin={actual['tmin']}"
                )
                if status == "WARN":
                    warnings.append(
                        f"incomplete target period: {region} {year} "
                        f"{season_key} {actual}, expected={expected}"
                    )

    print("\nGECC geography assets:")
    expected_properties = {
        "tracts": {"GEOID", "StFIPS", "CoFIPS", "TrFIPS"},
        "zctas": {"GEOID20", "ZCTA5CE20"},
        "fuas": {"eFUA_ID", "eFUA_name"},
    }
    for key, config in common.GEOGRAPHIES.items():
        collection = ee.FeatureCollection(config.asset_id)
        count = collection.size().getInfo()
        properties = set(ee.Feature(collection.first()).propertyNames().getInfo())
        asset = ee.data.getAsset(config.asset_id)
        print(
            f"  {config.name}: count={count}, updated={asset.get('updateTime')}, "
            f"asset={config.asset_id}"
        )
        missing = expected_properties[key] - properties
        if missing:
            failures.append(f"{config.name}: missing properties {sorted(missing)}")

    print("\n2020 raw-to-legacy mosaic comparison:")
    for region, (place, lon, lat) in BASELINE_POINTS.items():
        point = ee.Geometry.Point([lon, lat])
        source = collections[region]
        for season_key in args.seasons:
            label = common.SEASONS[season_key].band_label
            raw = common.build_season_image(source, 2020, season_key)
            legacy = legacy_asset(season_key)
            values = raw.addBands(
                legacy.select(
                    [f"{label}_tmax", f"{label}_tmin"],
                    [f"legacy_{label}_tmax", f"legacy_{label}_tmin"],
                )
            ).reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=common.REDUCE_SCALE_M,
                maxPixels=1_000_000,
            ).getInfo()
            for prop_type in ("tmax", "tmin"):
                raw_value = values.get(f"{label}_{prop_type}")
                legacy_value = values.get(f"legacy_{label}_{prop_type}")
                if raw_value is None or legacy_value is None:
                    failures.append(
                        f"{place} {label} {prop_type}: masked baseline sample"
                    )
                    continue
                difference = float(raw_value) - float(legacy_value)
                status = (
                    "PASS"
                    if abs(difference) <= BASELINE_TOLERANCE_C
                    else "FAIL"
                )
                print(
                    f"  {status} {place} {label} {prop_type}: "
                    f"raw={raw_value:.8f}, legacy={legacy_value:.8f}, "
                    f"difference={difference:.3g} C"
                )
                if status == "FAIL":
                    failures.append(
                        f"{place} {label} {prop_type}: "
                        f"baseline difference {difference} C"
                    )

    print("\nValidation summary:")
    print(f"  failures: {len(failures)}")
    print(f"  warnings: {len(warnings)}")
    for item in failures:
        print(f"  FAIL: {item}")
    for item in warnings:
        print(f"  WARN: {item}")

    if failures:
        raise SystemExit(1)
    print("  RESULT: PASS WITH WARNINGS" if warnings else "  RESULT: PASS")


if __name__ == "__main__":
    main()
