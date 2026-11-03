#!/usr/bin/env python3
"""Plan, preview, or submit 2020 Census Tract urban/rural NDVI and EVI exports.

NDVI/EVI counterpart of the WorldCover 2020 urban-rural release: each tract is
split into an urban zone (inside any 2020 Census Urban Area) and a rural zone
(the rest), and NDVI and EVI statistics are exported for urban, rural, and the
whole tract. Urban - rural differences are derived when packaging.

Everything except the zone split is the NDVI/EVI v1 workflow, imported from
submit_census_tract_ndvi_evi_exports: Landsat C2 L2 masking, JJA and
same-calendar-year DJF median composites, the GLC-FCS30D water mask with the
WorldCover fallback, the 85,528-tract universe, spatial batches, and Alaska's
EPSG:3338 30 m / 100 m tract groups. The whole-tract columns therefore
reproduce the released v1 2020 values.

Zones are rasterized by pixel centre, as in the WorldCover run: a pixel is
urban when its centre is inside both this tract and a 2020 Census Urban Area,
and rural when its centre is inside the tract but outside every UA. Each tract
is reduced once with v1's weighted reducer, and the three zones are separate
masked bands. The whole-tract bands are not centre-masked: like v1, they
include edge pixels that straddle the tract boundary. Urban + rural therefore
partition the pixels centred in the tract (tract_centre_pixel_count), not
tract_pixel_count.

Version 2 adds the centre-in-tract condition. Version 1 (two JJA test tasks,
2026-10-12) assigned straddling edge pixels by their centre outside the tract.
That gave about 20 spurious "rural" pixels to wholly urban tracts whose edge
is a UA boundary, e.g. 646 such tracts in California.

Plan: census_tract_ndvi_evi_2020_urban_rural_plan.md

Examples:
    python submit_census_tract_ndvi_evi_urban_rural_2020.py --preview
    python submit_census_tract_ndvi_evi_urban_rural_2020.py \
        --seasons jja --chunks 0 --submit --max-new-tasks 1
    python submit_census_tract_ndvi_evi_urban_rural_2020.py --submit
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import ee

import submit_census_tract_ndvi_evi_exports as vi


# Runs on your own Earth Engine project; the input assets
# stay in your-ee-project. Task-state lookup is per project, so keep this
# default when resuming or the planner will resubmit tasks that already ran.
DEFAULT_PROJECT = "your-ee-project"
UA_ASSET = "projects/your-ee-project/assets/GECC/tl_2025_us_uac20"
EXPECTED_UA_COUNT = 2644
YEAR = 2020
DRIVE_FOLDER = "GECC_CensusTract_NDVI_EVI_UrbanRural_2020"
TASK_PREFIX = "CensusTract_NDVI_EVI_UR2020_v2_"
OUT = Path(__file__).resolve().parent / "outputs/census_tract_ndvi_evi_urban_rural_2020"
MANIFEST = OUT / "submitted_tasks.jsonl"
ZONES = ("urban", "rural", "tract")
INDICES = ("NDVI", "EVI")
INDEX_STATS = ("mean", "median", "stdDev", "count")
VALUE_BANDS = ("NDVI", "EVI", "clear_obs")
# Edge cases from the WorldCover zones: ~50/50 split, wholly urban, wholly
# rural, a 1,251 m2 urban sliver, two UAs, Puerto Rico, and Alaska (EPSG:3338).
DEFAULT_PREVIEW_GEOIDS = (
    "06001401700",
    "36001000100",
    "38001965600",
    "01043964400",
    "01003011418",
    "72001956600",
    "02020000206",
)
# Preview tolerances against v1's seasonal_stats_table for the same tract.
# Medians are histogram-based and can move ~1e-3 with tileScale; means,
# standard deviations, and counts come from the same pixels and weights.
MEAN_TOLERANCE = 1e-9
MEDIAN_TOLERANCE = 2e-3


def urban_mask() -> ee.Image:
    """1 where the reduction grid's pixel lies in a 2020 Census Urban Area, else 0."""
    return (
        ee.Image.constant(0)
        .byte()
        .paint(ee.FeatureCollection(UA_ASSET), 1)
        .rename("urban")
    )


def zone_image(composite: ee.Image, keep: ee.Image, inside: ee.Image) -> ee.Image:
    """Value and pixel-count bands for each zone, named '<zone>_<band>'.

    ``inside`` is 1 where a pixel's centre is in the tract being reduced.
    """
    urban = urban_mask()
    masks = {
        "urban": urban.And(inside),
        "rural": urban.Not().And(inside),
        "tract": ee.Image.constant(1),
    }
    one = ee.Image.constant(1)
    bands = [one.updateMask(inside).rename("tract_centre_pixel")]
    for zone, mask in masks.items():
        bands += [
            composite.select(list(VALUE_BANDS))
            .updateMask(keep)
            .updateMask(mask)
            .rename([f"{zone}_{band}" for band in VALUE_BANDS]),
            one.updateMask(mask).rename(f"{zone}_pixel"),
            one.updateMask(keep).updateMask(mask).rename(f"{zone}_nonwater"),
        ]
    return ee.Image.cat(bands)


def reduction_crs(chunk: int) -> str:
    return vi.CRS_BY_CHUNK.get(chunk, "EPSG:4326")


def zone_stats_table(
    tracts: ee.FeatureCollection,
    season: str,
    chunk: int,
    tile_scale: int,
    max_pixels_per_region: int,
    scale_m: int = vi.SCALE_M,
    group: str | None = None,
) -> ee.FeatureCollection:
    """Per-tract urban, rural, and whole-tract stats for one 2020 season."""
    region = vi.scene_search_region(tracts, chunk)
    composite, scene_count = vi.seasonal_composite(region, YEAR, season)
    water_year = vi.glc_map_year(YEAR)
    keep = vi.not_water_mask(water_year)

    def reduce(feature: ee.Feature) -> ee.Feature:
        # The centre-in-tract mask differs per tract, so reduce each tract on
        # its own (reduceRegions applies one image to every feature).
        inside = ee.Image.constant(0).byte().paint(feature.geometry(), 1)
        return feature.set(zone_image(composite, keep, inside).reduceRegion(
            reducer=vi.stats_reducer(),
            geometry=feature.geometry(),
            scale=scale_m,
            tileScale=tile_scale,
            maxPixels=max_pixels_per_region,
            **vi.reduce_grid_kwargs(chunk),
        ))

    stats = tracts.map(reduce)

    def flatten(feature: ee.Feature) -> ee.Feature:
        props = {
            vi.ID_FIELD: feature.get(vi.ID_FIELD),
            vi.STATE_FIELD: feature.get(vi.STATE_FIELD),
            "year": YEAR,
            "season": "JJA" if season == "jja" else "DJF_same_year",
            "batch": chunk,
            "batch_label": vi.REGION_LABELS[chunk],
            "tract_group": group or "",
            "reduction_scale_m": scale_m,
            "reduction_crs": reduction_crs(chunk),
            "regional_scene_count": scene_count,
            "glc_water_mask_year": water_year,
        }
        for zone in ZONES:
            for index in INDICES:
                for stat in INDEX_STATS:
                    name = f"{zone}_{index}_{stat}"
                    props[name] = feature.get(name)
            props[f"{zone}_clear_obs_mean"] = feature.get(f"{zone}_clear_obs_mean")
            props[f"{zone}_pixel_count"] = feature.get(f"{zone}_pixel_count")
            props[f"{zone}_pixel_nonwater_count"] = feature.get(f"{zone}_nonwater_count")
        props["tract_centre_pixel_count"] = feature.get("tract_centre_pixel_count")
        return ee.Feature(None, props)

    return stats.map(flatten)


def output_selectors() -> list[str]:
    fields = [
        vi.ID_FIELD, vi.STATE_FIELD, "year", "season", "batch", "batch_label",
        "tract_group", "reduction_scale_m", "reduction_crs",
        "regional_scene_count", "glc_water_mask_year",
    ]
    for zone in ZONES:
        fields += [f"{zone}_{index}_{stat}" for index in INDICES for stat in INDEX_STATS]
        fields += [
            f"{zone}_clear_obs_mean",
            f"{zone}_pixel_count",
            f"{zone}_pixel_nonwater_count",
        ]
    return fields + ["tract_centre_pixel_count"]


def task_description(season: str, chunk: int, group: str | None = None) -> str:
    season_label = "JJA" if season == "jja" else "DJFsameYear"
    suffix = f"_{group}" if group else ""
    return f"{TASK_PREFIX}{season_label}_{YEAR}_b{chunk:03d}{suffix}"


def build_jobs(
    seasons: list[str], chunks: list[int], groups_by_chunk: dict[int, list[str]]
) -> list[tuple[str, int, str | None]]:
    """(season, chunk, group) per task; grouped batches get one task per group."""
    return [
        (season, chunk, group)
        for season in seasons
        for chunk in chunks
        for group in groups_by_chunk.get(chunk, [None])
    ]


def existing_task_states() -> dict[str, list[str]]:
    states: dict[str, list[str]] = {}
    for task in ee.batch.Task.list():
        description = task.config.get("description") or ""
        if description.startswith(TASK_PREFIX):
            states.setdefault(description, []).append(str(task.state or "UNKNOWN"))
    return states


def preview_scale(chunk: int, area_km2: float) -> int:
    """The scale this tract gets in production (Alaska's size tiers)."""
    limit = vi.TIER_MAX_KM2_AT_30M.get(chunk)
    return vi.COARSE_SCALE_M if limit is not None and area_km2 > limit else vi.SCALE_M


def close(a: float | None, b: float | None, tolerance: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def check_preview_row(row: dict, v1_row: dict) -> list[str]:
    """Zone partition and v1 agreement for one previewed tract; [] when it passes."""
    problems = []
    centre = row.get("tract_centre_pixel_count") or 0
    urban, rural = row.get("urban_pixel_count") or 0, row.get("rural_pixel_count") or 0
    if urban + rural != centre:
        problems.append(f"pixel_count: urban {urban} + rural {rural} != centre-in-tract {centre}")
    if centre > (row.get("tract_pixel_count") or 0):
        problems.append("centre-in-tract pixels exceed tract pixels")
    for band in ("NDVI_count", "EVI_count", "pixel_nonwater_count"):
        urban = row.get(f"urban_{band}") or 0
        rural = row.get(f"rural_{band}") or 0
        if urban + rural > (row.get(f"tract_{band}") or 0):
            problems.append(f"{band}: urban {urban} + rural {rural} > tract {row.get(f'tract_{band}')}")
    for index in INDICES:
        for stat, tolerance in (("mean", MEAN_TOLERANCE), ("stdDev", MEAN_TOLERANCE),
                                ("median", MEDIAN_TOLERANCE), ("count", 0)):
            ours, theirs = row.get(f"tract_{index}_{stat}"), v1_row.get(f"{index}_{stat}")
            if not close(ours, theirs, tolerance):
                problems.append(f"tract_{index}_{stat} {ours} != v1 {theirs}")
    if not close(row.get("tract_clear_obs_mean"), v1_row.get("clear_obs_mean"), MEAN_TOLERANCE):
        problems.append("tract_clear_obs_mean differs from v1")
    for zone in ("urban", "rural"):
        for index in INDICES:
            value = row.get(f"{zone}_{index}_mean")
            if (value is None) != (not row.get(f"{zone}_{index}_count")):
                problems.append(f"{zone}_{index}_mean null-ness disagrees with its count")
    return problems


def run_previews(geoids: list[str], seasons: list[str], args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = vi.state_to_chunk_mapping()
    failures = 0
    for geoid in geoids:
        tracts = vi.all_tracts().filter(ee.Filter.eq(vi.ID_FIELD, geoid))
        info = ee.Dictionary({
            "n": tracts.size(),
            "state": tracts.first().get(vi.STATE_FIELD),
            "land": tracts.first().get("Area_Land"),
            "water": tracts.first().get("Area_Water"),
        }).getInfo()
        if info["n"] != 1:
            raise ValueError(f"Expected one tract for {geoid}; found {info['n']}")
        chunk = mapping[info["state"]]
        scale = preview_scale(chunk, (info["land"] + info["water"]) / 1e6)
        tile_scale = max(args.tile_scale, vi.MIN_TILE_SCALE_BY_CHUNK.get(chunk, 0))
        for season in seasons:
            ours = zone_stats_table(tracts, season, chunk, tile_scale,
                                    args.max_pixels_per_region, scale)
            v1 = vi.seasonal_stats_table(tracts, YEAR, season, chunk, tile_scale,
                                         args.max_pixels_per_region, scale)
            both = ee.Dictionary({
                "row": ee.Feature(ours.first()).toDictionary(),
                "v1": ee.Feature(v1.first()).toDictionary(),
            }).getInfo()
            row, v1_row = both["row"], both["v1"]
            problems = check_preview_row(row, v1_row)
            (OUT / f"preview_{geoid}_{season}.json").write_text(
                json.dumps({"row": row, "v1": v1_row, "problems": problems}, indent=2)
            )
            diffs = {
                index: (None if row.get(f"urban_{index}_mean") is None
                        or row.get(f"rural_{index}_mean") is None
                        else round(row[f"urban_{index}_mean"] - row[f"rural_{index}_mean"], 4))
                for index in INDICES
            }
            counts = "/".join(str(row.get(f"{z}_NDVI_count", 0)) for z in ZONES)
            pixels = (f"{row.get('urban_pixel_count', 0)}+{row.get('rural_pixel_count', 0)}"
                      f"={row.get('tract_centre_pixel_count')} of {row.get('tract_pixel_count')}")
            status = "PASS" if not problems else "FAIL"
            failures += bool(problems)
            print(f"PREVIEW {status} {geoid} {season} b{chunk:03d} {scale} m: "
                  f"NDVI counts u/r/t {counts}; pixels u+r=centre of tract {pixels}; "
                  f"urban-rural mean diff "
                  f"NDVI {diffs['NDVI']}, EVI {diffs['EVI']}", flush=True)
            for problem in problems:
                print(f"  {problem}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} preview(s) failed; see {OUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--seasons", type=vi.parse_csv, default=list(vi.SEASONS),
                        help="Comma-separated choices: jja,djf")
    parser.add_argument("--chunks", type=vi.parse_int_ranges,
                        default=list(vi.REGION_STATEFPS),
                        help="Comma-separated spatial batches; defaults to all 19.")
    parser.add_argument("--tile-scale", type=int, default=vi.DEFAULT_TILE_SCALE)
    parser.add_argument("--retry-tile-scale", type=int, default=vi.DEFAULT_RETRY_TILE_SCALE,
                        help="tileScale for tasks whose earlier attempt FAILED.")
    parser.add_argument("--completed-dir", nargs="+",
                        help="Downloaded outputs (CSVs, Drive zips, or folders); "
                        "matching tasks are skipped.")
    parser.add_argument("--retry-given-up", action="store_true")
    parser.add_argument("--max-pixels-per-region", type=int,
                        default=vi.DEFAULT_MAX_PIXELS_PER_REGION)
    parser.add_argument("--drive-folder", default=DRIVE_FOLDER)
    parser.add_argument("--max-new-tasks", type=int, default=vi.DEFAULT_MAX_NEW_TASKS)
    parser.add_argument("--preview", nargs="*", metavar="GEOID",
                        help="Preview tracts (default: the edge-case set) and compare "
                        "whole-tract values with v1; runs no exports.")
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vi.validate_subset(args.seasons, vi.SEASONS, "seasons")
    vi.validate_subset(args.chunks, vi.REGION_STATEFPS, "chunks")
    if args.tile_scale < 1 or args.max_new_tasks < 1 or args.max_pixels_per_region < 1:
        raise ValueError("--tile-scale, --max-new-tasks and --max-pixels-per-region must be positive")

    vi.initialize_earth_engine(args.project)
    ua_count = ee.FeatureCollection(UA_ASSET).size().getInfo()
    if ua_count != EXPECTED_UA_COUNT:
        raise RuntimeError(f"Urban Area asset has {ua_count} features, expected "
                           f"{EXPECTED_UA_COUNT}; inspect before exporting")

    if args.preview is not None:
        run_previews(args.preview or list(DEFAULT_PREVIEW_GEOIDS), args.seasons, args)
        return

    tracts = vi.prepared_tracts()
    tract_count = int(tracts.size().getInfo())
    unique_geoids = int(tracts.aggregate_count_distinct(vi.ID_FIELD).getInfo())
    if tract_count != vi.EXPECTED_TRACT_COUNT or unique_geoids != tract_count:
        raise RuntimeError(f"Expected {vi.EXPECTED_TRACT_COUNT} unique 2020 tract GEOIDs; "
                           f"found {tract_count} rows and {unique_geoids} unique")

    groups = {chunk: vi.chunk_groups(chunk) for chunk in args.chunks
              if chunk in vi.GROUPED_CHUNKS}
    jobs = build_jobs(args.seasons, args.chunks, {c: list(g) for c, g in groups.items()})
    exports = {task_description(season, chunk, group): (season, chunk, group)
               for season, chunk, group in jobs}
    already, available, retries, given_up = vi.plan_tasks(
        exports, existing_task_states(),
        vi.completed_descriptions(args.completed_dir), args.retry_given_up,
    )

    print(f"\nValidated inputs: {tract_count} tracts, {ua_count} Urban Areas, year {YEAR}")
    for chunk, keyed in groups.items():
        print(f"  b{chunk:03d} groups: " + ", ".join(f"{k}={len(g)}" for k, g in keyed.items()))
    print("\nRun plan:")
    print(f"  expected tasks: {len(exports)}")
    print(f"  already active/completed/downloaded: {len(already)}")
    print(f"  previously failed (retry at tileScale {args.retry_tile_scale}): {len(retries)}")
    print(f"  failed {vi.MAX_FAILED_ATTEMPTS}+ times, not resubmitted: {len(given_up)}")
    for description in given_up:
        print(f"  GIVEN UP: {description}")
    print(f"  available to submit: {len(available)}; invocation cap: "
          f"{min(args.max_new_tasks, len(available))}")
    for description in available[: max(args.max_new_tasks, 10)]:
        print(f"  AVAILABLE: {description} -> Drive/{args.drive_folder}")

    if not args.submit:
        print("\nDry run only. Add --submit to start available tasks.")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    started = 0
    for description in available[: args.max_new_tasks]:
        season, chunk, group = exports[description]
        tile_scale = max(
            args.retry_tile_scale if description in retries else args.tile_scale,
            vi.MIN_TILE_SCALE_BY_CHUNK.get(chunk, 0),
        )
        table = zone_stats_table(
            vi.tract_source(chunk, groups.get(chunk, {}), group), season, chunk,
            tile_scale, args.max_pixels_per_region, vi.group_scale(group), group,
        )
        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=description,
            folder=args.drive_folder,
            fileNamePrefix=description,
            fileFormat="CSV",
            selectors=output_selectors(),
        )
        task.start()
        started += 1
        with MANIFEST.open("a") as log:
            log.write(json.dumps({
                "description": description, "task_id": task.id, "season": season,
                "chunk": chunk, "group": group, "tile_scale": tile_scale,
                "drive_folder": args.drive_folder, "project": args.project,
                "submitted_utc": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
        print(f"STARTED: {description} (tileScale {tile_scale}) {task.id}", flush=True)
    print(f"\nStarted {started} task(s).")


if __name__ == "__main__":
    main()
