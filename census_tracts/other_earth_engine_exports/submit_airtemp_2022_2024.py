#!/usr/bin/env python3
"""Plan or submit GECC 2022-2024 SVCM-SP air-temperature exports.

The default is a read-only ZCTA dry run. Census Tracts and FUAs are explicit
additional stages.
Each output is one geography x season x year CSV in Google Drive and follows
the established GECC naming and same-year DJF convention.

Examples:

    python submit_airtemp_2022_2024.py
    python submit_airtemp_2022_2024.py --geographies fuas
    python submit_airtemp_2022_2024.py --geographies tracts
    python submit_airtemp_2022_2024.py --geographies zctas --submit

Requested batches containing an incomplete source period are blocked before
any task starts. The August 2026 release currently lacks 2024-12-31 in every
regional collection, so 2024 same-year DJF requires the explicit
--allow-incomplete-periods override.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import ee

import airtemp_2022_2024_common as common


ACTIVE_OR_DONE_STATES = {
    "PENDING",
    "READY",
    "RUNNING",
    "SUCCEEDED",
    "COMPLETED",
}


@dataclass(frozen=True)
class ExportJob:
    geography_key: str
    season_key: str
    year: int
    description: str
    drive_folder: str


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
    parser.add_argument(
        "--geographies",
        type=common.parse_csv,
        default=["zctas"],
        help="Comma-separated choices: tracts,zctas,fuas. Default: zctas",
    )
    parser.add_argument(
        "--drive-folder",
        default=None,
        help=(
            "Override the geography-specific Google Drive folder. Only valid "
            "when one geography is requested."
        ),
    )
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument(
        "--max-new-tasks",
        type=int,
        default=6,
        help="Maximum number of new tasks to start in this invocation.",
    )
    parser.add_argument(
        "--allow-incomplete-periods",
        action="store_true",
        help=(
            "Allow exports whose daily source count is below the calendar "
            "expectation. Output rows retain completeness metadata."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Start exports. Default is a read-only dry run.",
    )
    return parser


def make_jobs(
    geography_keys: list[str],
    years: list[int],
    season_keys: list[str],
    drive_folder_override: str | None,
) -> list[ExportJob]:
    jobs: list[ExportJob] = []
    for geography_key in geography_keys:
        geography = common.GEOGRAPHIES[geography_key]
        folder = drive_folder_override or geography.drive_folder
        for year in years:
            for season_key in season_keys:
                jobs.append(
                    ExportJob(
                        geography_key=geography_key,
                        season_key=season_key,
                        year=year,
                        description=common.task_description(
                            geography_key, season_key, year
                        ),
                        drive_folder=folder,
                    )
                )
    return jobs


def existing_operation_states(project: str) -> dict[str, set[str]]:
    project_path = project if project.startswith("projects/") else f"projects/{project}"
    states: dict[str, set[str]] = {}
    for operation in ee.data.listOperations(project_path):
        metadata = operation.get("metadata", {})
        description = str(metadata.get("description", ""))
        if not description:
            continue
        state = str(metadata.get("state", "UNKNOWN"))
        states.setdefault(description, set()).add(state)
    return states


def period_summary(
    counts: dict[tuple[str, int, str, str], int],
    year: int,
    season_key: str,
) -> dict[str, object]:
    expected = common.expected_day_count(year, season_key)
    tmax_counts = [
        counts[(region, year, season_key, "tmax")]
        for region in common.REGIONAL_COLLECTIONS
    ]
    tmin_counts = [
        counts[(region, year, season_key, "tmin")]
        for region in common.REGIONAL_COLLECTIONS
    ]
    return {
        "expected": expected,
        "tmax_min": min(tmax_counts),
        "tmax_max": max(tmax_counts),
        "tmin_min": min(tmin_counts),
        "tmin_max": max(tmin_counts),
        "complete": set(tmax_counts + tmin_counts) == {expected},
    }


def output_collection(
    image: ee.Image,
    polygons: ee.FeatureCollection,
    job: ExportJob,
    summary: dict[str, object],
    tile_scale: int,
) -> ee.FeatureCollection:
    reducer = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.count(), sharedInputs=True)
    )
    metadata = {
        "year": job.year,
        "months": common.months_label(job.geography_key, job.season_key),
        "scale_factor_applied": common.SCALE_FACTOR,
        "source_catalog_updated": common.CATALOG_UPDATED,
        "source_expected_day_count": summary["expected"],
        "source_tmax_day_count_min": summary["tmax_min"],
        "source_tmax_day_count_max": summary["tmax_max"],
        "source_tmin_day_count_min": summary["tmin_min"],
        "source_tmin_day_count_max": summary["tmin_max"],
        "source_period_complete": summary["complete"],
        "djf_convention": "same-year" if job.season_key == "djf" else "not_applicable",
    }
    return (
        image.reduceRegions(
            collection=polygons,
            reducer=reducer,
            scale=common.REDUCE_SCALE_M,
            tileScale=tile_scale,
        )
        .map(
            lambda feature: ee.Feature(None, feature.toDictionary()).set(
                metadata
            )
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    common.validate_release_years(args.years)
    common.validate_choices(args.seasons, common.SEASONS, "seasons")
    common.validate_choices(
        args.geographies, common.GEOGRAPHIES, "geographies"
    )
    if args.max_new_tasks < 1:
        raise ValueError("--max-new-tasks must be at least 1")
    if args.tile_scale < 1:
        raise ValueError("--tile-scale must be at least 1")
    if args.drive_folder and len(args.geographies) != 1:
        raise ValueError(
            "--drive-folder can only override a single requested geography"
        )

    common.initialize_earth_engine(args.project)
    collections = common.regional_collections()
    merged_source = common.global_collection(collections)
    counts = common.resolve_period_counts(
        collections, args.years, args.seasons
    )
    summaries = {
        (year, season_key): period_summary(counts, year, season_key)
        for year in args.years
        for season_key in args.seasons
    }

    print("\nValidated source periods:")
    incomplete: list[tuple[int, str]] = []
    for (year, season_key), summary in summaries.items():
        status = "PASS" if summary["complete"] else "INCOMPLETE"
        print(
            f"  {status} {year} {season_key.upper()}: "
            f"expected={summary['expected']}; "
            f"tmax={summary['tmax_min']}..{summary['tmax_max']}; "
            f"tmin={summary['tmin_min']}..{summary['tmin_max']}"
        )
        if not summary["complete"]:
            incomplete.append((year, season_key))

    polygons: dict[str, ee.FeatureCollection] = {}
    print("\nGeographies:")
    for geography_key in args.geographies:
        config = common.GEOGRAPHIES[geography_key]
        collection = ee.FeatureCollection(config.asset_id)
        polygons[geography_key] = collection
        print(
            f"  {config.name}: {collection.size().getInfo()} features; "
            f"asset={config.asset_id}"
        )

    jobs = make_jobs(
        args.geographies,
        args.years,
        args.seasons,
        args.drive_folder,
    )
    operation_states = existing_operation_states(args.project)
    pending = [
        job
        for job in jobs
        if not (operation_states.get(job.description, set()) & ACTIVE_OR_DONE_STATES)
    ]
    skipped = [job for job in jobs if job not in pending]

    print("\nRun plan:")
    print(f"  expected tasks: {len(jobs)}")
    print(f"  already active/completed: {len(skipped)}")
    print(f"  available to submit: {len(pending)}")
    print(f"  invocation cap: {args.max_new_tasks}")
    for job in pending:
        summary = summaries[(job.year, job.season_key)]
        marker = "READY" if summary["complete"] else "BLOCKED-INCOMPLETE"
        print(
            f"  {marker}: {job.description} -> Drive/{job.drive_folder}"
        )

    if not args.submit:
        print("\nDry run only. Add --submit to start this stage.")
        if incomplete:
            print(
                "Incomplete requested periods remain blocked unless "
                "--allow-incomplete-periods is supplied."
            )
        return

    if incomplete and not args.allow_incomplete_periods:
        labels = ", ".join(
            f"{year} {season_key.upper()}" for year, season_key in incomplete
        )
        raise SystemExit(
            "No tasks started. Requested source periods are incomplete: "
            f"{labels}. Review validation and either exclude them or pass "
            "--allow-incomplete-periods."
        )

    started: list[tuple[str, str]] = []
    for job in pending[: args.max_new_tasks]:
        summary = summaries[(job.year, job.season_key)]
        image = common.build_season_image(
            merged_source, job.year, job.season_key
        )
        table = output_collection(
            image,
            polygons[job.geography_key],
            job,
            summary,
            args.tile_scale,
        )
        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=job.description,
            folder=job.drive_folder,
            fileNamePrefix=job.description,
            fileFormat="CSV",
        )
        task.start()
        started.append((job.description, task.id))
        print(f"  started {job.description}: {task.id}")

    print(f"\nStarted {len(started)} task(s).")
    print("Monitor at https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
