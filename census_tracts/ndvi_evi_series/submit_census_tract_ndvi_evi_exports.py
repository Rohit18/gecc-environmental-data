#!/usr/bin/env python3
"""Plan, preview, or submit Census Tract NDVI and EVI exports.

This workflow runs against the ``your-ee-project`` Earth Engine project and
covers all 85,528 2020 Census Tracts, matching the all-US release: the project
asset (84,414 tracts, 50 states + DC) plus the 1,114 Puerto Rico, U.S. Virgin
Islands, Guam, Northern Mariana Islands, and American Samoa tracts from Earth
Engine's TIGER/2020/TRACT (the source of the release's territory geography). It applies the established GECC Landsat
Collection 2 Level-2 masking/scaling logic, calculates NDVI and EVI together,
builds JJA and same-calendar-year DJF median composites, and exports tract
mean, median, standard deviation, and valid-pixel count for both indices, plus
clear_obs_mean: the average number of clear Landsat observations behind each
valid pixel's seasonal median (a composite-quality indicator; 1 means the
pixel's value comes from a single scene).

Completed-season defaults cover 1985-2025. The current 2026 JJA and DJF
seasons are intentionally excluded because they are incomplete on 2026-08-20.

Values are released as produced (no Landsat 7/8 cross-sensor harmonization);
see harmonization_test/README.md for the known offset and the version-2 fix.

Water pixels are excluded: each seasonal composite is masked where GLC-FCS30D
is class 210 (water body) or 187 (tidal flat) in the GLC map for that year, or the latest earlier
map (1985, 1990, 1995 five-year maps; 2000-2022 annual; 2023-2025 use 2022).
GLC-FCS30D has no data over Hawaii, American Samoa, Guam, and the Northern
Mariana Islands; there (and anywhere else GLC is unclassified) ESA WorldCover
2020 is used instead, excluding its water class (80) and its unclassified open
ocean. Each row records glc_water_mask_year. The FUA tables were not water-masked.

Counts and grid: like the FUA notebook, reductions run at scale=30 on the
composite's default EPSG:4326 grid, so a counted cell is 30 m north-south but
30*cos(latitude) m east-west. Counts are therefore only meaningful as ratios
(e.g. the observed fraction below); do not convert them to area with 900 m2.
Counts include partially covered edge pixels (count is combined with weighted
reducers). Medians use Earth Engine's histogram-based median and can differ
slightly (~1e-3) if a task is re-run at a different tileScale; means, standard
deviations, and counts do not. regional_scene_count is the number of Landsat
scenes over the whole batch area, not over the individual tract.

Each task first restricts every Landsat collection to the season's dates, then to
the batch area, before masking. Tasks start at --tile-scale and anything that
previously FAILED is resubmitted at --retry-tile-scale; a task that has FAILED
twice is reported but not resubmitted automatically. --pixel-totals submits one
export per batch of each tract's total pixel count, its GLC-gap pixel count
(pixels with no GLC-FCS30D class; the WorldCover fallback decides these), and its
non-water pixel count for every GLC map year, using the same grid, scale, and
edge-pixel counting as NDVI_count, so that
NDVI_count / pixel_nonwater_count_<glc_water_mask_year> is the observed fraction
of the tract's non-water area.

Examples:
    python submit_census_tract_ndvi_evi_exports.py \
        --years 2022 --seasons jja --chunks 0 --preview
    python submit_census_tract_ndvi_evi_exports.py \
        --years 2022 --seasons jja --chunks 0 --submit
    python submit_census_tract_ndvi_evi_exports.py \
        --years 2022 --seasons jja,djf --submit --max-new-tasks 2
    python submit_census_tract_ndvi_evi_exports.py \
        --years 1985-2025 --completed-dir ~/Downloads/drive_zips/ --submit
    python submit_census_tract_ndvi_evi_exports.py --pixel-totals --submit
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ee


DEFAULT_PROJECT = "your-ee-project"
TRACT_ASSET = "projects/your-ee-project/assets/GECC/censusTracts2020US"
TERRITORY_SOURCE = "TIGER/2020/TRACT"
TERRITORY_STATEFPS = ("60", "66", "69", "72", "78")
EXPECTED_TRACT_COUNT = 85528
# Version-1 production names. They keep these outputs apart from the unmasked
# August 2022 pilot (CensusTract_NDVI_EVI_*) and the two October 2022
# New England test tasks (CensusTract_NDVI_EVI_nowater_*, no clear_obs_mean).
DRIVE_FOLDER = "GECC_CensusTract_NDVI_EVI_v1"
ID_FIELD = "GEOID"
STATE_FIELD = "StFIPS"
SCALE_M = 30
DEFAULT_TILE_SCALE = 4
DEFAULT_RETRY_TILE_SCALE = 16
# Guard only. Alaska North Slope tracts reach ~3.4e10 cells on the default
# EPSG:4326 grid at 30 m (cells narrow with latitude), above the 2e10 default.
DEFAULT_MAX_PIXELS_PER_REGION = int(1e12)
DEFAULT_MAX_NEW_TASKS = 50
DEFAULT_PREVIEW_GEOID = "06001420302"
TASK_PREFIX = "CensusTract_NDVI_EVI_v1_"
MAX_FAILED_ATTEMPTS = 2
# Batches that need a higher tileScale from the first attempt. Alaska (14):
# huge high-latitude tracts ran out of memory at tileScale 4 (pixel totals and
# JJA 1985) and succeeded at 16.
MIN_TILE_SCALE_BY_CHUNK = {14: 16}
# Batches left out unless named with --chunks. Alaska (14) NDVI/EVI tasks ran
# out of memory even at tileScale 16 (2026-10-10) and need their own approach.
DEFERRED_CHUNKS = {14}
# Batches reduced on an explicit projection instead of the composite's default
# EPSG:4326 grid. Alaska (14): at 60-70 N a 30 m EPSG:4326 cell is only 10-15 m
# wide, which inflated pixel counts 2-3x and ran out of memory; on Alaska Albers
# (EPSG:3338) the largest tract (234,600 km2) succeeded (2026-10-10 test).
# Alaska counts are therefore true 30 m pixels; its pixel totals use the same grid.
CRS_BY_CHUNK = {14: "EPSG:3338"}
# Batches split into fixed tract groups, each its own task. Alaska (14): the
# whole batch ran out of memory even on EPSG:3338, while a single 234,600 km2
# tract and 25 Anchorage tracts each succeeded. Groups follow GEOID order and
# hold at most GROUP_MAX_TRACTS tracts and GROUP_MAX_KM2 of land + water.
GROUPED_CHUNKS = {14}
GROUP_MAX_KM2 = 250_000
GROUP_MAX_TRACTS = 25
# Per-tier group limits (km2, tracts). Smaller groups proved cheaper and faster:
# JJA 1985 Alaska cost 96.7 EECU-h in 4 large groups (up to 137 tracts /
# 726,000 km2, tasks up to 62 min) but ~34 EECU-h at 100 m in 11 small groups
# (2-15 min). Values do not depend on grouping (identical to ~1e-15 at 30 m,
# ~1e-6 at 100 m). Keys use "p" (parts) so they never collide with the
# superseded 4-group layout's "g" keys.
GROUP_LIMITS_BY_SCALE = {30: (250_000, 25), 100: (250_000, 25)}
GROUP_KEY_LETTER = "p"
# Resolution tiers by tract size (land + water) for grouped batches. Alaska:
# tracts up to 5,000 km2 (137 tracts, 84% of Alaskans, 2.9% of the area) at
# 30 m; the 40 larger tracts at 100 m. Test (JJA 1985, 2026-10-12): 100 m cost
# 10.6x less and moved tract NDVI means by at most 0.006 for tracts > 5,000 km2
# (median 0.001; r = 0.997 over all Alaska tracts).
TIER_MAX_KM2_AT_30M = {14: 5_000}
COARSE_SCALE_M = 100


def group_scale(group_key: str | None) -> int:
    """Reduction scale encoded in a group key such as '100m_p02' (30 m if none)."""
    return int(group_key.split("m_", 1)[0]) if group_key else SCALE_M


def tiered_groups(chunk: int, rows: list[tuple[str, float]]) -> dict[str, list[str]]:
    """Group keys -> GEOIDs: size tiers (30 m / 100 m), then GEOID-order groups."""
    limit = TIER_MAX_KM2_AT_30M.get(chunk)
    tiers = {SCALE_M: [r for r in rows if limit is None or r[1] <= limit]}
    if limit is not None:
        tiers[COARSE_SCALE_M] = [r for r in rows if r[1] > limit]
    keyed: dict[str, list[str]] = {}
    for scale, tier_rows in tiers.items():
        max_km2, max_tracts = GROUP_LIMITS_BY_SCALE.get(
            scale, (GROUP_MAX_KM2, GROUP_MAX_TRACTS)
        )
        for n, geoids in enumerate(group_tracts(tier_rows, max_km2, max_tracts), 1):
            keyed[f"{scale}m_{GROUP_KEY_LETTER}{n:02d}"] = geoids
    return keyed


def group_tracts(
    rows: list[tuple[str, float]],
    max_km2: float = GROUP_MAX_KM2,
    max_tracts: int = GROUP_MAX_TRACTS,
) -> list[list[str]]:
    """Greedy GEOID-order grouping of (GEOID, km2) rows; deterministic."""
    groups: list[list[str]] = []
    current: list[str] = []
    area = 0.0
    for geoid, km2 in sorted(rows):
        if current and (len(current) >= max_tracts or area + km2 > max_km2):
            groups.append(current)
            current, area = [], 0.0
        current.append(geoid)
        area += km2
    if current:
        groups.append(current)
    return groups


def reduce_grid_kwargs(chunk: int) -> dict:
    """Projection argument for reduceRegions (empty = default grid, as the FUA notebook)."""
    return {"crs": CRS_BY_CHUNK[chunk]} if chunk in CRS_BY_CHUNK else {}
GLC_FIVE_YEAR_COLLECTION = "projects/sat-io/open-datasets/GLC-FCS30D/five-years-map"
GLC_ANNUAL_COLLECTION = "projects/sat-io/open-datasets/GLC-FCS30D/annual"
# Water body and tidal flat (tidal flats behave as water: mean JJA NDVI -0.32
# in a San Francisco Bay test tract). Other wetland classes are kept.
GLC_EXCLUDED_CLASSES = (210, 187)
GLC_MAP_YEARS = (1985, 1990, 1995, *range(2000, 2023))
# Fallback where GLC-FCS30D has no class (Hawaii and the Pacific territories).
WORLDCOVER_IMAGE = "ESA/WorldCover/v100"
WORLDCOVER_WATER_CLASS = 80
ACTIVE_OR_DONE_STATES = {"READY", "RUNNING", "COMPLETED"}
EXPECTED_YEARS = tuple(range(1985, 2026))
SEASONS = ("jja", "djf")

# Spatially coherent groups limit each task's Landsat scene search. Groups are
# intentionally balanced more finely in the populous South and Pacific. Alaska,
# Hawaii, and Oregon/Washington are separate: a combined bounding box spans most
# of the North Pacific (and the Aleutians cross the antimeridian).
REGION_STATEFPS: dict[int, tuple[str, ...]] = {
    0: ("09", "23", "25", "33", "44", "50"),
    1: ("34", "36"),
    2: ("10", "11", "24", "42"),
    3: ("17", "18", "39"),
    4: ("26", "55"),
    5: ("19", "20", "27", "29", "31", "38", "46"),
    6: ("12", "13"),
    7: ("37", "45", "51", "54"),
    8: ("01", "21", "28", "47"),
    9: ("48",),
    10: ("05", "22", "40"),
    11: ("04", "08", "16", "30", "32", "35", "49", "56"),
    12: ("06",),
    13: ("41", "53"),
    14: ("02",),
    15: ("15",),
    16: ("72", "78"),
    17: ("66", "69"),
    18: ("60",),
}
# Batches whose bounding box is unusable for the scene search (antimeridian).
GEOMETRY_SCENE_FILTER_CHUNKS = {14}
REGION_LABELS = {
    0: "new_england",
    1: "ny_nj",
    2: "mid_atlantic",
    3: "il_in_oh",
    4: "mi_wi",
    5: "west_north_central",
    6: "fl_ga",
    7: "carolinas_va_wv",
    8: "east_south_central",
    9: "texas",
    10: "ar_la_ok",
    11: "mountain",
    12: "california",
    13: "or_wa",
    14: "alaska",
    15: "hawaii",
    16: "pr_vi",
    17: "guam_cnmi",
    18: "american_samoa",
}

NDVI_EXPR = 'clamp((b("nir") - b("red")) / (b("nir") + b("red")), -1, 1)'
EVI_EXPR = (
    'clamp(2.5 * ((b("nir") - b("red")) / '
    'clamp(b("nir") + 6 * b("red") - 7.5 * b("blue") + 1, 0, 10)), '
    "-1.0, 1.0)"
)


@dataclass(frozen=True)
class ExportJob:
    year: int
    season: str
    chunk: int
    description: str
    group: str | None = None


def parse_int_ranges(value: str) -> list[int]:
    values: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def validate_subset(
    values: Iterable[object], allowed: Iterable[object], label: str
) -> None:
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}; allowed: {list(allowed)}")


def initialize_earth_engine(project: str) -> None:
    ee.Initialize(project=project)
    print(f"Earth Engine initialized with project: {project}")


def prep_c2sr_l4l5l7(image: ee.Image) -> ee.Image:
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_band = image.select("ST_B6").multiply(0.00341802).add(149.0)
    scaled = optical_bands.addBands(thermal_band).select(
        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6"],
        ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"],
    )
    qa = image.select("QA_PIXEL")
    # One bit test equal to the two original checks: bits 0-5 (fill, dilated
    # cloud, cirrus, cloud, shadow, snow) all 0 and bits 8-13 (cloud, shadow,
    # snow confidence) == 010101 (low).
    qa_clear = qa.bitwiseAnd(0x3F3F).eq(0x1500)
    mask3 = image.select("QA_RADSAT").eq(0)
    # One min/max pass over the bands (same masks as separate min and max reduces).
    band_range = optical_bands.reduce(ee.Reducer.minMax())
    mask4 = band_range.select("min").gt(0)
    mask5 = band_range.select("max").lt(1)
    mask6 = image.select("SR_ATMOS_OPACITY").unmask(-1).lt(300)
    return (
        image.select()
        .addBands(scaled)
        .updateMask(qa_clear.And(mask3).And(mask4).And(mask5).And(mask6))
    )


def prep_c2sr_l8l9(image: ee.Image) -> ee.Image:
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_band = image.select("ST_B10").multiply(0.00341802).add(149.0)
    scaled = optical_bands.addBands(thermal_band).select(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"],
        ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"],
    )
    qa = image.select("QA_PIXEL")
    # One bit test equal to the two original checks: bits 0-5 all 0 and
    # bits 8-15 (cloud, shadow, snow, cirrus confidence) == 01010101 (low).
    qa_clear = qa.bitwiseAnd(0xFF3F).eq(0x5500)
    mask3 = image.select("QA_RADSAT").eq(0)
    # One min/max pass over the bands (same masks as separate min and max reduces).
    band_range = optical_bands.reduce(ee.Reducer.minMax())
    mask4 = band_range.select("min").gt(0)
    mask5 = band_range.select("max").lt(1)
    mask6 = image.select("SR_QA_AEROSOL").rightShift(6).neq(int("11", 2))
    return (
        image.select()
        .addBands(scaled)
        .updateMask(qa_clear.And(mask3).And(mask4).And(mask5).And(mask6))
    )


def season_date_ranges(year: int, season: str) -> list[tuple[str, str]]:
    """Half-open date ranges; DJF is Jan-Feb plus Dec of the same calendar year."""
    if season == "jja":
        return [(f"{year}-06-01", f"{year}-09-01")]
    if season == "djf":
        return [(f"{year}-01-01", f"{year}-03-01"), (f"{year}-12-01", f"{year + 1}-01-01")]
    raise ValueError(f"Unknown season: {season}")


def build_landsat_sr_collection(
    region: ee.Geometry, date_ranges: list[tuple[str, str]]
) -> ee.ImageCollection:
    """Merged, masked L4/5/7/8/9 C2 SR, filtered by season dates and region first."""
    season_dates = ee.Filter.Or(*[ee.Filter.date(start, end) for start, end in date_ranges])

    def sensor(collection_id: str) -> ee.ImageCollection:
        return (
            ee.ImageCollection(collection_id)
            .filter(season_dates)
            .filterBounds(region)
            .filter("WRS_ROW < 122")
        )

    l4 = sensor("LANDSAT/LT04/C02/T1_L2").map(prep_c2sr_l4l5l7)
    l5 = sensor("LANDSAT/LT05/C02/T1_L2").map(prep_c2sr_l4l5l7)
    l7 = (
        sensor("LANDSAT/LE07/C02/T1_L2")
        .filterDate("1984-01-01", "2017-01-01")
        .map(prep_c2sr_l4l5l7)
    )
    l8 = (
        sensor("LANDSAT/LC08/C02/T1_L2")
        .filterDate("2013-05-01", "2099-01-01")
        .filter(ee.Filter.neq("NADIR_OFFNADIR", "OFFNADIR"))
        .map(prep_c2sr_l8l9)
    )
    l9 = (
        sensor("LANDSAT/LC09/C02/T1_L2")
        .filter(ee.Filter.neq("NADIR_OFFNADIR", "OFFNADIR"))
        .map(prep_c2sr_l8l9)
    )
    return l4.merge(l5).merge(l7).merge(l8).merge(l9)


def add_indices(image: ee.Image) -> ee.Image:
    ndvi = image.expression(NDVI_EXPR).rename("NDVI")
    evi = image.expression(EVI_EXPR).rename("EVI")
    return ndvi.addBands(evi).copyProperties(image, ["system:time_start"])


def seasonal_composite(
    region: ee.Geometry, year: int, season: str
) -> tuple[ee.Image, ee.Number]:
    seasonal_sr = build_landsat_sr_collection(region, season_date_ranges(year, season))
    scene_count = seasonal_sr.size()
    seasonal_vi = seasonal_sr.map(add_indices)
    # One pass over the collection: per-pixel median of each index plus the
    # number of clear observations behind it (NDVI and EVI share one mask).
    reduced = seasonal_vi.reduce(
        ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True)
    )
    median_and_depth = reduced.select(
        ["NDVI_median", "EVI_median", "NDVI_count"], ["NDVI", "EVI", "clear_obs"]
    )
    empty = (
        ee.Image.constant([0, 0, 0])
        .rename(["NDVI", "EVI", "clear_obs"])
        .updateMask(0)
    )
    composite = ee.Image(
        ee.Algorithms.If(scene_count.gt(0), median_and_depth, empty)
    )
    # Depth is reported only where a composite value exists.
    composite = composite.addBands(
        composite.select("clear_obs").updateMask(composite.select("NDVI").mask()),
        overwrite=True,
    )
    return composite, scene_count


def scene_search_region(chunk_tracts: ee.FeatureCollection, chunk: int) -> ee.Geometry:
    if chunk in GEOMETRY_SCENE_FILTER_CHUNKS:
        return chunk_tracts.geometry(maxError=1000)
    return chunk_tracts.geometry(maxError=100).bounds(maxError=100)


def state_to_chunk_mapping() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for chunk, state_codes in REGION_STATEFPS.items():
        for state_code in state_codes:
            if state_code in mapping:
                raise ValueError(f"Duplicate state mapping: {state_code}")
            mapping[state_code] = chunk
    return mapping


def territory_tracts(state_codes: Iterable[str] = TERRITORY_STATEFPS) -> ee.FeatureCollection:
    """2020 territory tracts from TIGER/2020/TRACT, renamed to the asset's fields."""
    return (
        ee.FeatureCollection(TERRITORY_SOURCE)
        .filter(ee.Filter.inList("STATEFP", list(state_codes)))
        .map(
            lambda f: ee.Feature(
                f.geometry(),
                {
                    ID_FIELD: f.get("GEOID"),
                    STATE_FIELD: f.get("STATEFP"),
                    "Area_Land": f.get("ALAND"),
                    "Area_Water": f.get("AWATER"),
                },
            )
        )
    )


def all_tracts() -> ee.FeatureCollection:
    """All 85,528 2020 tracts: project asset (states + DC) plus territories."""
    return (
        ee.FeatureCollection(TRACT_ASSET)
        .filter(ee.Filter.notNull([ID_FIELD, STATE_FIELD]))
        .merge(territory_tracts())
    )


def prepared_tracts() -> ee.FeatureCollection:
    mapping = ee.Dictionary(state_to_chunk_mapping())
    source = all_tracts()

    def prepare(feature: ee.Feature) -> ee.Feature:
        state_code = ee.String(feature.get(STATE_FIELD))
        return ee.Feature(
            feature.geometry(),
            {
                ID_FIELD: ee.String(feature.get(ID_FIELD)),
                STATE_FIELD: state_code,
                "_vi_chunk": mapping.get(state_code),
            },
        )

    return source.map(prepare)


def chunk_source(chunk: int) -> ee.FeatureCollection:
    """The batch's tracts straight from the asset, filtered by state code.

    Filtering the stored table (rather than a mapped national collection) keeps
    each task from re-evaluating all 84k tracts; profiling showed that overhead
    at ~1.2-1.4 EECU-hours per task regardless of batch size. Territory
    batches come from TIGER/2020/TRACT (small, so mapping them is cheap).
    """
    codes = REGION_STATEFPS[chunk]
    if set(codes) <= set(TERRITORY_STATEFPS):
        return territory_tracts(codes)
    return ee.FeatureCollection(TRACT_ASSET).filter(
        ee.Filter.And(
            ee.Filter.inList(STATE_FIELD, list(REGION_STATEFPS[chunk])),
            ee.Filter.notNull([ID_FIELD, STATE_FIELD]),
        )
    )


def glc_map_year(year: int) -> int:
    """Latest GLC-FCS30D map year at or before ``year`` (2022 for later years)."""
    earlier = [map_year for map_year in GLC_MAP_YEARS if map_year <= year]
    if not earlier:
        raise ValueError(f"No GLC-FCS30D map at or before {year}")
    return earlier[-1]


def glc_class_image(map_year: int) -> ee.Image:
    """GLC-FCS30D class codes for one map year (band layout as the GLC export)."""
    if map_year in (1985, 1990, 1995):
        band = f"b{(map_year - 1985) // 5 + 1}"
        collection = GLC_FIVE_YEAR_COLLECTION
    else:
        band = f"b{map_year - 1999}"
        collection = GLC_ANNUAL_COLLECTION
    return ee.ImageCollection(collection).mosaic().select(band).rename("glc_class")


def glc_gap_image() -> ee.Image:
    """1 where GLC-FCS30D (2022) has no class; WorldCover decides these pixels
    (keeping its land classes and dropping its water and open ocean)."""
    return glc_class_image(2022).mask().Not()


def not_water_mask(map_year: int) -> ee.Image:
    """1 = keep, 0 = water/tidal flat/no land-cover class.

    GLC-FCS30D decides where it has a class; elsewhere ESA WorldCover 2020
    decides (keep if classified and not water). Pixels neither product
    classifies (open ocean) are dropped.
    """
    glc = glc_class_image(map_year)
    glc_excluded = ee.Image.constant(0)
    for code in GLC_EXCLUDED_CLASSES:
        glc_excluded = glc_excluded.Or(glc.eq(code))
    glc_keep = glc_excluded.Not()
    worldcover = ee.ImageCollection(WORLDCOVER_IMAGE).first().select("Map")
    worldcover_keep = worldcover.neq(WORLDCOVER_WATER_CLASS).unmask(0, False)
    return glc_keep.unmask(worldcover_keep, False).rename("keep")


def chunk_groups(chunk: int) -> dict[str, list[str]]:
    """Fixed, keyed tract groups for a grouped batch (empty for other batches)."""
    if chunk not in GROUPED_CHUNKS:
        return {}
    fc = chunk_source(chunk)
    geoids = fc.aggregate_array(ID_FIELD).getInfo()
    land = fc.aggregate_array("Area_Land").getInfo()
    water = fc.aggregate_array("Area_Water").getInfo()
    return tiered_groups(chunk, [(g, (l + w) / 1e6) for g, l, w in zip(geoids, land, water)])


def tract_source(
    chunk: int, groups: dict[str, list[str]], group: str | None
) -> ee.FeatureCollection:
    """The batch's tracts, or one group's tracts for a grouped batch."""
    if group is None:
        return chunk_source(chunk)
    return chunk_source(chunk).filter(ee.Filter.inList(ID_FIELD, groups[group]))


def stats_reducer() -> ee.Reducer:
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )


def seasonal_stats_table(
    chunk_tracts: ee.FeatureCollection,
    year: int,
    season: str,
    chunk: int,
    tile_scale: int,
    max_pixels_per_region: int,
    scale_m: int = SCALE_M,
) -> ee.FeatureCollection:
    """Per-tract stats; chunk_tracts must already be limited to this batch."""
    region = scene_search_region(chunk_tracts, chunk)
    composite, scene_count = seasonal_composite(region, year, season)
    water_year = glc_map_year(year)
    composite = composite.updateMask(not_water_mask(water_year))
    stats = composite.select(["NDVI", "EVI", "clear_obs"]).reduceRegions(
        collection=chunk_tracts,
        reducer=stats_reducer(),
        scale=scale_m,
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels_per_region,
        **reduce_grid_kwargs(chunk),
    )

    def flatten(feature: ee.Feature) -> ee.Feature:
        return ee.Feature(
            None,
            {
                ID_FIELD: feature.get(ID_FIELD),
                STATE_FIELD: feature.get(STATE_FIELD),
                "year": year,
                "season": "JJA" if season == "jja" else "DJF_same_year",
                "batch": chunk,
                "batch_label": REGION_LABELS[chunk],
                "regional_scene_count": scene_count,
                "glc_water_mask_year": water_year,
                "NDVI_mean": feature.get("NDVI_mean"),
                "NDVI_median": feature.get("NDVI_median"),
                "NDVI_stdDev": feature.get("NDVI_stdDev"),
                "NDVI_count": feature.get("NDVI_count"),
                "EVI_mean": feature.get("EVI_mean"),
                "EVI_median": feature.get("EVI_median"),
                "EVI_stdDev": feature.get("EVI_stdDev"),
                "EVI_count": feature.get("EVI_count"),
                "clear_obs_mean": feature.get("clear_obs_mean"),
            },
        )

    return stats.map(flatten)


def pixel_totals_selectors() -> list[str]:
    return [ID_FIELD, STATE_FIELD, "pixel_total_count", "pixel_glc_gap_count"] + [
        f"pixel_nonwater_count_{map_year}" for map_year in GLC_MAP_YEARS
    ]


def pixel_count_reducer() -> ee.Reducer:
    """Count combined with a weighted reducer, as in stats_reducer().

    On its own Reducer.count() counts only pixels whose centre is inside the
    tract; combined with weighted reducers it also counts partially covered
    edge pixels. The denominators must use the latter to match NDVI_count
    (an independent review measured stand-alone counts 8-25% lower).
    """
    return ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)


def pixel_totals_table(
    tracts: ee.FeatureCollection,
    tile_scale: int,
    max_pixels_per_region: int,
    chunk: int | None = None,
    scale_m: int = SCALE_M,
) -> ee.FeatureCollection:
    """Total, GLC-gap, and per-GLC-year non-water pixel counts per tract."""
    bands = [
        ee.Image.constant(1).rename("total"),
        ee.Image.constant(1).updateMask(glc_gap_image()).rename("glc_gap"),
    ] + [
        ee.Image.constant(1)
        .updateMask(not_water_mask(map_year))
        .rename(f"nonwater_{map_year}")
        for map_year in GLC_MAP_YEARS
    ]
    stats = ee.Image.cat(bands).reduceRegions(
        collection=tracts,
        reducer=pixel_count_reducer(),
        scale=scale_m,
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels_per_region,
        **(reduce_grid_kwargs(chunk) if chunk is not None else {}),
    )

    def flatten(feature: ee.Feature) -> ee.Feature:
        props = {
            ID_FIELD: feature.get(ID_FIELD),
            STATE_FIELD: feature.get(STATE_FIELD),
            "pixel_total_count": feature.get("total_count"),
            "pixel_glc_gap_count": feature.get("glc_gap_count"),
        }
        for map_year in GLC_MAP_YEARS:
            props[f"pixel_nonwater_count_{map_year}"] = feature.get(
                f"nonwater_{map_year}_count"
            )
        return ee.Feature(None, props)

    return stats.map(flatten)


def pixel_totals_description(chunk: int, group: str | None = None) -> str:
    """Batches on an explicit grid get a grid suffix, so they are not confused
    with (or skipped because of) an earlier default-grid export."""
    suffix = f"_{group}" if group else ""
    if chunk in CRS_BY_CHUNK:
        suffix += "_" + CRS_BY_CHUNK[chunk].replace(":", "").lower()
    if group:  # grouped batches name the group's own resolution (in the key)
        return f"{TASK_PREFIX}pixel_totals_b{chunk:03d}{suffix}"
    return f"{TASK_PREFIX}pixel_totals_30m_b{chunk:03d}{suffix}"


def completed_descriptions(paths: Iterable[str] | None) -> set[str]:
    """Export names already downloaded: CSVs and Drive zips (also inside folders)."""
    names: list[str] = []
    for path in paths or []:
        source = Path(path).expanduser()
        zips = [source] if source.suffix == ".zip" else sorted(source.rglob("*.zip"))
        if source.is_dir():
            names += [str(p) for p in source.rglob("*.csv")]
        for archive in zips:
            names += zipfile.ZipFile(archive).namelist()
    return {Path(name).stem for name in names if name.endswith(".csv")}


def output_selectors() -> list[str]:
    return [
        ID_FIELD,
        STATE_FIELD,
        "year",
        "season",
        "batch",
        "batch_label",
        "regional_scene_count",
        "glc_water_mask_year",
        "NDVI_mean",
        "NDVI_median",
        "NDVI_stdDev",
        "NDVI_count",
        "EVI_mean",
        "EVI_median",
        "EVI_stdDev",
        "EVI_count",
        "clear_obs_mean",
    ]


def task_description(year: int, season: str, chunk: int, group: str | None = None) -> str:
    season_label = "JJA" if season == "jja" else "DJFsameYear"
    suffix = f"_{group}" if group else ""
    return f"{TASK_PREFIX}{season_label}_{year}_b{chunk:03d}{suffix}"


def build_jobs(
    years: Iterable[int],
    seasons: Iterable[str],
    chunks: Iterable[int],
    groups_by_chunk: dict[int, list[str]] | None = None,
) -> list[ExportJob]:
    """One job per year, season and batch, or per group for grouped batches."""
    groups_by_chunk = groups_by_chunk or {}
    return [
        ExportJob(
            year=year,
            season=season,
            chunk=chunk,
            description=task_description(year, season, chunk, group),
            group=group,
        )
        for year in years
        for season in seasons
        for chunk in chunks
        for group in (groups_by_chunk[chunk] if chunk in groups_by_chunk else [None])
    ]


# Operation states (listOperations) mapped to the Task.list() vocabulary.
OPERATION_STATE = {"PENDING": "READY", "SUCCEEDED": "COMPLETED", "CANCELLING": "CANCEL_REQUESTED"}


def existing_task_states(other_projects: Iterable[str] = ()) -> dict[str, list[str]]:
    """State of every attempt per task description: this project's task list,
    plus the task history of other_projects (so work finished in another
    Earth Engine project is not redone)."""
    states: dict[str, list[str]] = {}
    for task in ee.batch.Task.list():
        description = task.config.get("description") or ""
        if not description.startswith(TASK_PREFIX):
            continue
        states.setdefault(description, []).append(str(task.state or "UNKNOWN"))
    for project in other_projects:
        for operation in ee.data.listOperations(f"projects/{project}"):
            meta = operation.get("metadata", {})
            description = meta.get("description", "")
            if description.startswith(TASK_PREFIX):
                state = meta.get("state", "UNKNOWN")
                states.setdefault(description, []).append(OPERATION_STATE.get(state, state))
    return states


def plan_tasks(
    descriptions: Iterable[str],
    states: dict[str, list[str]],
    downloaded: set[str],
    retry_given_up: bool = False,
) -> tuple[list[str], list[str], set[str], list[str]]:
    """Split into (already done/active, available, retries, given up)."""
    done, available, retries, given_up = [], [], set(), []
    for description in descriptions:
        attempts = states.get(description, [])
        if set(attempts) & ACTIVE_OR_DONE_STATES or description in downloaded:
            done.append(description)
        elif attempts.count("FAILED") >= MAX_FAILED_ATTEMPTS and not retry_given_up:
            given_up.append(description)
        else:
            available.append(description)
            if "FAILED" in attempts:
                retries.add(description)
    return done, available, retries, given_up


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--also-check-projects",
        default="",
        help="Comma-separated other Earth Engine projects whose finished or active "
        "tasks count as done (e.g. your-ee-project when running elsewhere).",
    )
    parser.add_argument(
        "--years",
        type=parse_int_ranges,
        default=list(EXPECTED_YEARS),
        help="Comma-separated years/ranges; defaults to completed 1985-2025 seasons.",
    )
    parser.add_argument(
        "--seasons",
        type=parse_csv,
        default=list(SEASONS),
        help="Comma-separated choices: jja,djf",
    )
    parser.add_argument(
        "--chunks",
        type=parse_int_ranges,
        default=[c for c in REGION_STATEFPS if c not in DEFERRED_CHUNKS],
        help="Comma-separated spatial batches; defaults to all except "
        f"{sorted(DEFERRED_CHUNKS)} (deferred; name them to include).",
    )
    parser.add_argument("--tile-scale", type=int, default=DEFAULT_TILE_SCALE)
    parser.add_argument(
        "--retry-tile-scale",
        type=int,
        default=DEFAULT_RETRY_TILE_SCALE,
        help="tileScale for tasks whose earlier attempt FAILED.",
    )
    parser.add_argument(
        "--completed-dir",
        nargs="+",
        help="Downloaded outputs (CSVs, Drive zips, or folders of either); "
        "matching tasks are skipped.",
    )
    parser.add_argument(
        "--retry-given-up",
        action="store_true",
        help="Also resubmit tasks that already FAILED twice (after fixing the cause).",
    )
    parser.add_argument(
        "--pixel-totals",
        action="store_true",
        help="Submit only the one-time per-tract total pixel count export.",
    )
    parser.add_argument(
        "--max-pixels-per-region",
        type=int,
        default=DEFAULT_MAX_PIXELS_PER_REGION,
    )
    parser.add_argument("--drive-folder", default=DRIVE_FOLDER)
    parser.add_argument(
        "--max-new-tasks", type=int, default=DEFAULT_MAX_NEW_TASKS
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-geoid", default=DEFAULT_PREVIEW_GEOID)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_subset(args.years, EXPECTED_YEARS, "years")
    validate_subset(args.seasons, SEASONS, "seasons")
    validate_subset(args.chunks, REGION_STATEFPS, "chunks")
    if args.tile_scale < 1:
        raise ValueError("--tile-scale must be at least 1")
    if args.max_pixels_per_region < 1:
        raise ValueError("--max-pixels-per-region must be at least 1")
    if args.max_new_tasks < 1:
        raise ValueError("--max-new-tasks must be at least 1")

    initialize_earth_engine(args.project)
    tracts = prepared_tracts()
    tract_count = int(tracts.size().getInfo())
    chunk_counts = tracts.aggregate_histogram("_vi_chunk").getInfo()
    mapped_chunk_count = int(
        tracts.filter(ee.Filter.notNull(["_vi_chunk"])).size().getInfo()
    )
    missing_chunk_count = tract_count - mapped_chunk_count
    if missing_chunk_count:
        raise RuntimeError(
            f"{missing_chunk_count} tracts have unmapped state codes"
        )
    unique_geoids = int(tracts.aggregate_count_distinct(ID_FIELD).getInfo())
    if tract_count != EXPECTED_TRACT_COUNT or unique_geoids != tract_count:
        raise RuntimeError(
            f"Expected {EXPECTED_TRACT_COUNT} unique 2020 tract GEOIDs; "
            f"found {tract_count} rows and {unique_geoids} unique GEOIDs"
        )

    print("\nValidated inputs:")
    print(f"  Earth Engine project: {args.project}")
    print(f"  Census Tracts: {tract_count} features; ID={ID_FIELD}")
    print(f"  completed years: {min(EXPECTED_YEARS)}-{max(EXPECTED_YEARS)}")
    print(f"  selected seasons: {args.seasons}")
    print(
        "  selected spatial batches: "
        + ", ".join(
            f"b{chunk:03d}={chunk_counts.get(str(chunk), 0)}"
            for chunk in args.chunks
        )
    )

    states = existing_task_states(
        [p.strip() for p in args.also_check_projects.split(",") if p.strip()]
    )
    downloaded = completed_descriptions(args.completed_dir)
    groups = {chunk: chunk_groups(chunk) for chunk in args.chunks if chunk in GROUPED_CHUNKS}
    for chunk, keyed in groups.items():
        print(f"  b{chunk:03d} groups: {len(keyed)} ("
              + ", ".join(f"{k}={len(g)}" for k, g in keyed.items()) + " tracts)")
    if args.pixel_totals:
        exports = {
            pixel_totals_description(chunk, group): (
                lambda ts, chunk=chunk, group=group: pixel_totals_table(
                    tract_source(chunk, groups.get(chunk, {}), group),
                    ts, args.max_pixels_per_region, chunk, group_scale(group),
                ),
                pixel_totals_selectors(),
                chunk,
            )
            for chunk in args.chunks
            for group in (list(groups[chunk]) if chunk in groups else [None])
        }
    else:
        jobs = build_jobs(args.years, args.seasons, args.chunks,
                          {c: list(g) for c, g in groups.items()})
        exports = {
            job.description: (
                lambda ts, job=job: seasonal_stats_table(
                    tract_source(job.chunk, groups.get(job.chunk, {}), job.group),
                    job.year,
                    job.season,
                    job.chunk,
                    ts,
                    args.max_pixels_per_region,
                    group_scale(job.group),
                ),
                output_selectors(),
                job.chunk,
            )
            for job in jobs
        }
    already, available, retries, given_up = plan_tasks(
        exports, states, downloaded, args.retry_given_up
    )

    print("\nRun plan:" + (" (pixel totals)" if args.pixel_totals else ""))
    print(f"  expected tasks: {len(exports)}")
    print(f"  already active/completed/downloaded: {len(already)}")
    print(f"  previously failed (retry at tileScale {args.retry_tile_scale}): {len(retries)}")
    print(f"  failed {MAX_FAILED_ATTEMPTS}+ times, not resubmitted: {len(given_up)}")
    for description in given_up:
        print(f"  GIVEN UP: {description}")
    print(f"  available to submit: {len(available)}")
    print(f"  invocation cap: {min(args.max_new_tasks, len(available))}")
    for description in available[: max(args.max_new_tasks, 10)]:
        print(f"  AVAILABLE: {description} -> Drive/{args.drive_folder}")

    if args.preview and not args.pixel_totals:
        if not jobs:
            raise RuntimeError("No jobs selected for preview")
        preview_tracts = all_tracts().filter(
            ee.Filter.eq(ID_FIELD, args.preview_geoid)
        )
        preview_count = int(preview_tracts.size().getInfo())
        if preview_count != 1:
            raise ValueError(
                f"Expected one preview tract for {args.preview_geoid}; "
                f"found {preview_count}"
            )
        preview_chunk = state_to_chunk_mapping()[
            ee.Feature(preview_tracts.first()).get(STATE_FIELD).getInfo()
        ]
        job = jobs[0]
        preview_table = seasonal_stats_table(
            preview_tracts,
            job.year,
            job.season,
            preview_chunk,
            args.tile_scale,
            args.max_pixels_per_region,
        )
        preview = (
            ee.Feature(preview_table.first())
            .toDictionary(output_selectors())
            .getInfo()
        )
        print("\nPreview row:")
        for key in output_selectors():
            print(f"  {key}: {preview.get(key)}")

    if not args.submit:
        print("\nDry run only. Add --submit to start available tasks.")
        return

    started: list[str] = []
    for description in available[: args.max_new_tasks]:
        build_table, selectors, chunk = exports[description]
        tile_scale = max(
            args.retry_tile_scale if description in retries else args.tile_scale,
            MIN_TILE_SCALE_BY_CHUNK.get(chunk, 0),
        )
        task = ee.batch.Export.table.toDrive(
            collection=build_table(tile_scale),
            description=description,
            folder=args.drive_folder,
            fileNamePrefix=description,
            fileFormat="CSV",
            selectors=selectors,
        )
        task.start()
        started.append(description)
        print(f"STARTED: {description} (tileScale {tile_scale})")

    print(f"\nStarted {len(started)} task(s).")


if __name__ == "__main__":
    main()
