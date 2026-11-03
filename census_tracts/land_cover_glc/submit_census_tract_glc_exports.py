#!/usr/bin/env python3
"""Plan, preview, or submit Census Tract GLC-FCS30D exports.

The workflow aggregates exact pixel area by land-cover class for 2020 US
Census Tracts. It covers the same 26 GLC map years as the existing FUA
product: 1985, 1990, 1995, and annually from 2000 through 2022.

Exports use deterministic, GEOID-seeded balanced batches. Dry run is the
default, a single-feature preview is available with --preview, and submission
is capped with --max-new-tasks. Existing READY, RUNNING, or COMPLETED tasks
are skipped.

Examples:
    python submit_census_tract_glc_exports.py --years 2022 --chunks 0 --preview
    python submit_census_tract_glc_exports.py --years 2022 --chunks 0 --submit
    python submit_census_tract_glc_exports.py --submit --max-new-tasks 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import ee


DEFAULT_PROJECT = "your-ee-project"
TRACT_ASSET = "projects/your-ee-project/assets/GECC/censusTracts2020US"
DRIVE_FOLDER = "GECC_CensusTract_GLC_FCS30D"
FIVE_YEAR_COLLECTION = (
    "projects/sat-io/open-datasets/GLC-FCS30D/five-years-map"
)
ANNUAL_COLLECTION = "projects/sat-io/open-datasets/GLC-FCS30D/annual"

ID_FIELD = "GEOID"
SCALE_M = 30
DEFAULT_TILE_SCALE = 8
DEFAULT_N_CHUNKS = 10
DEFAULT_MAX_NEW_TASKS = 4
DEFAULT_MAX_PIXELS_PER_REGION = int(2e10)
DEFAULT_PREVIEW_GEOID = "06001420302"
CHUNK_SEED = 20260820
TASK_PREFIX = "GLC_CensusTract_class_areas_"
ACTIVE_OR_DONE_STATES = {"READY", "RUNNING", "COMPLETED"}

EXPECTED_YEARS = (1985, 1990, 1995, *range(2000, 2023))

GLC_CLASSES = (
    0,
    10,
    11,
    12,
    20,
    51,
    52,
    61,
    62,
    71,
    72,
    81,
    82,
    91,
    92,
    120,
    121,
    122,
    130,
    140,
    150,
    152,
    153,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    190,
    200,
    201,
    202,
    210,
    220,
    250,
)

# These definitions match the completed FUA vegetation products.
TREE_CLASSES = (51, 52, 61, 62, 71, 72, 81, 82, 91, 92)
SHRUB_CLASSES = (120, 121, 122)
GRASS_CLASSES = (130,)
CROP_CLASSES = (10, 11, 20)
VEGETATION_CLASSES = (
    10,
    11,
    12,
    20,
    51,
    52,
    61,
    62,
    71,
    72,
    81,
    82,
    91,
    92,
    120,
    121,
    122,
    130,
    140,
    150,
    152,
    153,
    181,
    182,
    185,
    186,
)


@dataclass(frozen=True)
class ExportJob:
    year: int
    chunk: int
    description: str


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


def validate_subset(
    values: Iterable[int], allowed: Iterable[int], label: str
) -> None:
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(
            f"Invalid {label}: {invalid}; allowed values are "
            f"{min(allowed)} through {max(allowed)} with the documented gaps"
        )


def initialize_earth_engine(project: str) -> None:
    ee.Initialize(project=project)
    print(f"Earth Engine initialized with project: {project}")


def year_images() -> dict[int, ee.Image]:
    five_year = ee.ImageCollection(FIVE_YEAR_COLLECTION).mosaic()
    annual = ee.ImageCollection(ANNUAL_COLLECTION).mosaic()
    images = {
        1985: five_year.select("b1").rename("class_code"),
        1990: five_year.select("b2").rename("class_code"),
        1995: five_year.select("b3").rename("class_code"),
    }
    for band_number in range(1, 24):
        year = 1999 + band_number
        images[year] = annual.select(f"b{band_number}").rename("class_code")
    return images


def prepared_tracts(n_chunks: int) -> ee.FeatureCollection:
    source = ee.FeatureCollection(TRACT_ASSET).filter(
        ee.Filter.notNull([ID_FIELD])
    )

    def prepare(feature: ee.Feature) -> ee.Feature:
        geoid = ee.String(feature.get(ID_FIELD))
        return ee.Feature(
            feature.geometry(),
            {ID_FIELD: geoid},
        )

    prepared = source.map(prepare).randomColumn(
        columnName="_glc_random",
        seed=CHUNK_SEED,
        distribution="uniform",
        rowKeys=[ID_FIELD],
    )
    return prepared.map(
        lambda feature: feature.set(
            "_glc_chunk",
            ee.Number(feature.get("_glc_random"))
            .multiply(n_chunks)
            .floor()
            .toInt(),
        )
    )


def sum_classes(
    areas: dict[int, ee.Number], class_codes: Iterable[int]
) -> ee.Number:
    total = ee.Number(0)
    for code in class_codes:
        total = total.add(areas[code])
    return total


def flatten_grouped_areas(
    feature: ee.Feature, year: int, chunk: int
) -> ee.Feature:
    groups = ee.List(
        ee.Algorithms.If(feature.get("groups"), feature.get("groups"), [])
    )
    keys = groups.map(
        lambda item: ee.Number(
            ee.Dictionary(item).get("class_code")
        ).format("%.0f")
    )
    values = groups.map(lambda item: ee.Dictionary(item).get("sum"))
    by_class = ee.Dictionary.fromLists(keys, values)

    areas: dict[int, ee.Number] = {}
    properties: dict[str, object] = {
        ID_FIELD: feature.get(ID_FIELD),
        "year": year,
        "batch": chunk,
    }
    for code in GLC_CLASSES:
        key = str(code)
        area = ee.Number(
            ee.Algorithms.If(by_class.contains(key), by_class.get(key), 0)
        )
        areas[code] = area
        properties[f"area_class_{code}"] = area

    group_sums = groups.map(lambda item: ee.Dictionary(item).get("sum"))
    total_area = ee.Number(
        ee.Algorithms.If(
            group_sums.size().gt(0),
            group_sums.reduce(ee.Reducer.sum()),
            0,
        )
    )
    source_area = feature.geometry().area(maxError=1)

    subtype_definitions = {
        "vegetation": VEGETATION_CLASSES,
        "tree": TREE_CLASSES,
        "shrub": SHRUB_CLASSES,
        "grass": GRASS_CLASSES,
        "crop": CROP_CLASSES,
    }
    properties["total_area_m2"] = total_area
    properties["source_area_m2"] = source_area
    properties["glc_coverage_fraction"] = ee.Number(
        ee.Algorithms.If(
            source_area.gt(0), total_area.divide(source_area), 0
        )
    )
    for label, codes in subtype_definitions.items():
        area = sum_classes(areas, codes)
        properties[f"{label}_area_m2"] = area
        properties[f"{label}_fraction"] = ee.Number(
            ee.Algorithms.If(total_area.gt(0), area.divide(total_area), 0)
        )

    return ee.Feature(None, properties)


def class_area_table(
    lc_image: ee.Image,
    tracts: ee.FeatureCollection,
    year: int,
    chunk: int,
    tile_scale: int,
    max_pixels_per_region: int,
) -> ee.FeatureCollection:
    chunk_tracts = tracts.filter(ee.Filter.eq("_glc_chunk", chunk))
    area_and_class = ee.Image.pixelArea().rename("area_m2").addBands(
        lc_image.rename("class_code")
    )
    grouped = area_and_class.reduceRegions(
        collection=chunk_tracts,
        reducer=ee.Reducer.sum().group(
            groupField=1, groupName="class_code"
        ),
        scale=SCALE_M,
        crs=lc_image.projection(),
        tileScale=tile_scale,
        maxPixelsPerRegion=max_pixels_per_region,
    )
    return grouped.map(
        lambda feature: flatten_grouped_areas(feature, year, chunk)
    )


def output_selectors() -> list[str]:
    selectors = [
        ID_FIELD,
        "year",
        "batch",
        "source_area_m2",
        "total_area_m2",
        "glc_coverage_fraction",
        "vegetation_area_m2",
        "vegetation_fraction",
        "tree_area_m2",
        "tree_fraction",
        "shrub_area_m2",
        "shrub_fraction",
        "grass_area_m2",
        "grass_fraction",
        "crop_area_m2",
        "crop_fraction",
    ]
    selectors.extend(f"area_class_{code}" for code in GLC_CLASSES)
    return selectors


def task_description(year: int, chunk: int, n_chunks: int) -> str:
    return f"{TASK_PREFIX}{year}_b{chunk:03d}of{n_chunks:03d}"


def build_jobs(
    years: Iterable[int], chunks: Iterable[int], n_chunks: int
) -> list[ExportJob]:
    return [
        ExportJob(
            year=year,
            chunk=chunk,
            description=task_description(year, chunk, n_chunks),
        )
        for year in years
        for chunk in chunks
    ]


def existing_task_states() -> dict[str, set[str]]:
    states: dict[str, set[str]] = {}
    for task in ee.batch.Task.list():
        description = task.config.get("description") or ""
        if not description.startswith(TASK_PREFIX):
            continue
        state = str(task.status().get("state", "UNKNOWN"))
        states.setdefault(description, set()).add(state)
    return states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--years",
        type=parse_int_ranges,
        default=list(EXPECTED_YEARS),
        help="Comma-separated years/ranges; defaults to all 26 map years.",
    )
    parser.add_argument("--n-chunks", type=int, default=DEFAULT_N_CHUNKS)
    parser.add_argument(
        "--chunks",
        type=parse_int_ranges,
        default=None,
        help="Comma-separated batch indexes/ranges; defaults to every batch.",
    )
    parser.add_argument("--tile-scale", type=int, default=DEFAULT_TILE_SCALE)
    parser.add_argument(
        "--max-pixels-per-region",
        type=int,
        default=DEFAULT_MAX_PIXELS_PER_REGION,
        help="Per-tract safety ceiling; large Alaska tracts exceed 500M pixels.",
    )
    parser.add_argument(
        "--drive-folder",
        default=DRIVE_FOLDER,
        help="Google Drive destination folder.",
    )
    parser.add_argument(
        "--max-new-tasks", type=int, default=DEFAULT_MAX_NEW_TASKS
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Evaluate and print one output row from the first selected job.",
    )
    parser.add_argument(
        "--preview-geoid",
        default=DEFAULT_PREVIEW_GEOID,
        help="Census tract GEOID evaluated by --preview.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Start export tasks. Default is a read-only dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_chunks < 1:
        raise ValueError("--n-chunks must be at least 1")
    if args.tile_scale < 1:
        raise ValueError("--tile-scale must be at least 1")
    if args.max_new_tasks < 1:
        raise ValueError("--max-new-tasks must be at least 1")
    if args.max_pixels_per_region < 1:
        raise ValueError("--max-pixels-per-region must be at least 1")
    validate_subset(args.years, EXPECTED_YEARS, "years")

    chunks = args.chunks or list(range(args.n_chunks))
    validate_subset(chunks, range(args.n_chunks), "chunks")

    initialize_earth_engine(args.project)
    images = year_images()
    tracts = prepared_tracts(args.n_chunks)
    tract_count = int(tracts.size().getInfo())
    chunk_counts = tracts.aggregate_histogram("_glc_chunk").getInfo()

    print("\nValidated inputs:")
    print(f"  Census Tracts: {tract_count} features; ID={ID_FIELD}")
    print(f"  GLC map years: {len(images)} ({min(images)}-{max(images)})")
    print(f"  Batches: {args.n_chunks}; selected={chunks}")
    print(
        "  Selected batch sizes: "
        + ", ".join(
            f"b{chunk:03d}={chunk_counts.get(str(chunk), 0)}"
            for chunk in chunks
        )
    )

    jobs = build_jobs(args.years, chunks, args.n_chunks)
    states = existing_task_states()
    already = [
        job
        for job in jobs
        if states.get(job.description, set()) & ACTIVE_OR_DONE_STATES
    ]
    available = [job for job in jobs if job not in already]

    print("\nRun plan:")
    print(f"  expected tasks: {len(jobs)}")
    print(f"  already active/completed: {len(already)}")
    print(f"  available to submit: {len(available)}")
    print(f"  invocation cap: {min(args.max_new_tasks, len(available))}")
    for job in available[: max(args.max_new_tasks, 10)]:
        print(
            f"  AVAILABLE: {job.description} -> "
            f"Drive/{args.drive_folder}"
        )

    if args.preview:
        if not jobs:
            raise RuntimeError("No jobs selected for preview")
        job = jobs[0]
        preview_tracts = tracts.filter(
            ee.Filter.eq(ID_FIELD, args.preview_geoid)
        )
        preview_count = int(preview_tracts.size().getInfo())
        if preview_count != 1:
            raise ValueError(
                f"Expected one tract for preview GEOID {args.preview_geoid}, "
                f"found {preview_count}"
            )
        preview_chunk = int(
            ee.Feature(preview_tracts.first()).get("_glc_chunk").getInfo()
        )
        preview_table = class_area_table(
            images[job.year],
            preview_tracts,
            job.year,
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
    for job in available[: args.max_new_tasks]:
        table = class_area_table(
            images[job.year],
            tracts,
            job.year,
            job.chunk,
            args.tile_scale,
            args.max_pixels_per_region,
        )
        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=job.description,
            folder=args.drive_folder,
            fileNamePrefix=job.description,
            fileFormat="CSV",
            selectors=output_selectors(),
        )
        task.start()
        started.append(job.description)
        print(f"STARTED: {job.description}")

    print(f"\nStarted {len(started)} task(s).")


if __name__ == "__main__":
    main()
