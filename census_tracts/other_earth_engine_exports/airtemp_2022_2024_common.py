#!/usr/bin/env python3
"""Shared Earth Engine helpers for the GECC 2022-2024 air-temperature extension."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import ee


DEFAULT_PROJECT = "your-ee-project"
RELEASE_YEARS = (2022, 2023, 2024)
SCALE_FACTOR = 0.1
REDUCE_SCALE_M = 1000
CATALOG_UPDATED = "2026-08-02"
CATALOG_URL = (
    "https://gee-community-catalog.org/projects/airtemp/"
    "?h=air+temperature"
)

REGIONAL_COLLECTIONS = {
    "africa": (
        "projects/sat-io/open-datasets/global-daily-air-temp/africa"
    ),
    "australia": (
        "projects/sat-io/open-datasets/global-daily-air-temp/australia"
    ),
    "europe_asia": (
        "projects/sat-io/open-datasets/global-daily-air-temp/europe_asia"
    ),
    "latin_america": (
        "projects/sat-io/open-datasets/global-daily-air-temp/latin_america"
    ),
    "north_america": (
        "projects/sat-io/open-datasets/global-daily-air-temp/north_america"
    ),
}


@dataclass(frozen=True)
class GeographyConfig:
    key: str
    name: str
    asset_id: str
    task_stem: str
    drive_folder: str


GEOGRAPHIES = {
    "tracts": GeographyConfig(
        key="tracts",
        name="Census Tracts",
        asset_id=(
            "projects/your-ee-project/assets/GECC/censusTracts2020US"
        ),
        task_stem="CensusTract_AirTemp",
        drive_folder="GECC_CensusTract_AirTemp_2022_2024",
    ),
    "zctas": GeographyConfig(
        key="zctas",
        name="ZCTAs",
        asset_id=(
            "projects/your-ee-project/assets/GECC/tl_2020_us_zcta520"
        ),
        task_stem="ZCTA_AirTemp",
        drive_folder="GECC_ZCTA_AirTemp_2022_2024",
    ),
    "fuas": GeographyConfig(
        key="fuas",
        name="FUAs",
        asset_id="projects/your-ee-project/assets/EC_FUA",
        task_stem="FUA_AirTemp",
        drive_folder="GECC_FUA_AirTemp_2022_2024",
    ),
}


@dataclass(frozen=True)
class SeasonConfig:
    key: str
    band_label: str


SEASONS = {
    "jja": SeasonConfig(key="jja", band_label="JJA"),
    "djf": SeasonConfig(key="djf", band_label="DJF"),
}


def parse_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def validate_choices(
    values: Iterable[str], choices: dict[str, object], label: str
) -> None:
    invalid = sorted(set(values) - set(choices))
    if invalid:
        raise ValueError(
            f"Unknown {label}: {', '.join(invalid)}. "
            f"Choose from: {', '.join(sorted(choices))}"
        )


def validate_release_years(years: Iterable[int]) -> None:
    invalid = sorted(set(years) - set(RELEASE_YEARS))
    if invalid:
        raise ValueError(
            "This workflow is intentionally limited to the newly released "
            f"years {RELEASE_YEARS[0]}-{RELEASE_YEARS[-1]}; received: "
            + ", ".join(map(str, invalid))
        )


def initialize_earth_engine(project: str) -> None:
    """Initialize without starting an interactive auth flow in batch runs."""
    ee.Initialize(project=project)
    print(f"Earth Engine initialized with project: {project}")


def regional_collections() -> dict[str, ee.ImageCollection]:
    return {
        region: ee.ImageCollection(collection_id)
        for region, collection_id in REGIONAL_COLLECTIONS.items()
    }


def global_collection(
    collections: dict[str, ee.ImageCollection] | None = None,
) -> ee.ImageCollection:
    collections = collections or regional_collections()
    merged: ee.ImageCollection | None = None
    for region in REGIONAL_COLLECTIONS:
        collection = collections[region]
        merged = collection if merged is None else merged.merge(collection)
    if merged is None:  # pragma: no cover - constant mapping is never empty.
        raise RuntimeError("No regional air-temperature collections configured")
    return merged


def period_collection(
    collection: ee.ImageCollection,
    year: int,
    season_key: str,
) -> ee.ImageCollection:
    if season_key == "jja":
        return collection.filterDate(f"{year}-06-01", f"{year}-09-01")
    if season_key != "djf":
        raise ValueError(f"Unknown season: {season_key}")
    jan_feb = collection.filterDate(f"{year}-01-01", f"{year}-03-01")
    december = collection.filterDate(
        f"{year}-12-01", f"{year + 1}-01-01"
    )
    return jan_feb.merge(december)


def seasonal_composite(
    collection: ee.ImageCollection,
    year: int,
    season_key: str,
    prop_type: str,
) -> ee.Image:
    return (
        period_collection(collection, year, season_key)
        .filter(ee.Filter.eq("prop_type", prop_type))
        .mean()
        .multiply(SCALE_FACTOR)
        .toFloat()
    )


def build_season_image(
    collection: ee.ImageCollection,
    year: int,
    season_key: str,
) -> ee.Image:
    label = SEASONS[season_key].band_label
    tmax = seasonal_composite(
        collection, year, season_key, "tmax"
    ).rename(f"{label}_tmax")
    tmin = seasonal_composite(
        collection, year, season_key, "tmin"
    ).rename(f"{label}_tmin")
    return ee.Image.cat([tmax, tmin])


def expected_dates(year: int, season_key: str) -> list[str]:
    if season_key == "jja":
        start = date(year, 6, 1)
        end = date(year, 9, 1)
        return _date_range(start, end)
    if season_key == "djf":
        return _date_range(date(year, 1, 1), date(year, 3, 1)) + _date_range(
            date(year, 12, 1), date(year + 1, 1, 1)
        )
    raise ValueError(f"Unknown season: {season_key}")


def _date_range(start: date, end: date) -> list[str]:
    values: list[str] = []
    current = start
    while current < end:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def expected_day_count(year: int, season_key: str) -> int:
    if season_key == "jja":
        return 92
    if season_key == "djf":
        return 31 + (29 if calendar.isleap(year) else 28) + 31
    raise ValueError(f"Unknown season: {season_key}")


def millis_to_utc_dates(values: Iterable[int]) -> list[str]:
    return [
        datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        .date()
        .isoformat()
        for value in values
    ]


def resolve_period_counts(
    collections: dict[str, ee.ImageCollection],
    years: Iterable[int],
    season_keys: Iterable[str],
) -> dict[tuple[str, int, str, str], int]:
    expressions: dict[str, ee.Number] = {}
    for region, collection in collections.items():
        for year in years:
            for season_key in season_keys:
                period = period_collection(collection, year, season_key)
                for prop_type in ("tmax", "tmin"):
                    key = f"{region}|{year}|{season_key}|{prop_type}"
                    expressions[key] = period.filter(
                        ee.Filter.eq("prop_type", prop_type)
                    ).size()

    resolved = ee.Dictionary(expressions).getInfo()
    return {
        tuple_key(key): int(value) for key, value in resolved.items()
    }


def tuple_key(value: str) -> tuple[str, int, str, str]:
    region, year, season_key, prop_type = value.split("|")
    return region, int(year), season_key, prop_type


def task_description(
    geography_key: str,
    season_key: str,
    year: int,
) -> str:
    geography = GEOGRAPHIES[geography_key]
    if geography_key == "fuas" and season_key == "djf":
        return f"{geography.task_stem}_DJFsameYear_{year}"
    return (
        f"{geography.task_stem}_{SEASONS[season_key].band_label}_{year}"
    )


def months_label(geography_key: str, season_key: str) -> str:
    if geography_key == "fuas" and season_key == "djf":
        return "DJF_same_year"
    return SEASONS[season_key].band_label
