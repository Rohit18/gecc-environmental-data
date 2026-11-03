#!/usr/bin/env python3
"""Incremental GLC-FCS30D exports for Puerto Rico and the Virgin Islands.

Uses the existing census-tract GLC method and public 2020 TIGER boundaries.
The current GEE GLC asset has no sampled pixels in Hawaii or the Pacific
Island Areas; Alaska was already included in the national GLC release.
"""

import argparse
import json
from pathlib import Path

import ee

import submit_census_tract_glc_exports as prior


PROJECT = "your-ee-project"
TRACTS = "TIGER/2020/TRACT"
STATES = {"72": 981, "78": 32}
FOLDER = "GECC_nonCONUS_CensusTract_GLC_FCS30D"
PREFIX = "GLC_NONCONUS_CensusTract_v1"
OUT = Path(__file__).resolve().parent / "outputs/glc_nonconus"


def parse_years(value):
    years = prior.parse_int_ranges(value)
    prior.validate_subset(years, prior.EXPECTED_YEARS, "years")
    return years


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--states", nargs="+", default=["72", "78"])
    parser.add_argument("--years", type=parse_years, default=list(prior.EXPECTED_YEARS))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-new-tasks", type=int, default=2)
    parser.add_argument("--eecu-cap-hours", type=float, default=950)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    states = tuple(dict.fromkeys(args.states))
    if not states or set(states) - STATES.keys():
        parser.error("Current GLC source support is restricted to states 72 and 78")
    if args.batch_size < 1 or args.max_new_tasks < 1 or args.eecu_cap_hours <= 0:
        parser.error("Batch size, task limit, and EECU cap must be positive")

    ee.Initialize(project=args.project)
    source = ee.FeatureCollection(TRACTS).filter(ee.Filter.inList("STATEFP", list(states)))
    info = ee.Dictionary({"count": source.size(), "states": source.aggregate_histogram("STATEFP"),
                          "unique_geoids": source.aggregate_count_distinct("GEOID")}).getInfo()
    if info["count"] != info["unique_geoids"]:
        raise RuntimeError("GEOIDs are not unique")
    for state in states:
        if int(info["states"].get(state, 0)) != STATES[state]:
            raise RuntimeError(f"Unexpected tract count in {state}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_validation.json").write_text(json.dumps(info, indent=2))
    print(info)

    operations = ee.data.listOperations(f"projects/{args.project}")
    existing = {}
    total_seconds = 0.0
    active_glc = []
    for operation in operations:
        meta = operation.get("metadata", {})
        name = meta.get("description", "")
        if name.startswith(("WC2020_NONCONUS_tract_", PREFIX)):
            total_seconds += float(meta.get("batchEecuUsageSeconds", 0))
        if name.startswith(PREFIX):
            existing[name] = meta.get("state")
            if meta.get("state") in {"PENDING", "READY", "RUNNING"}:
                active_glc.append(name)
    hours = total_seconds / 3600
    print(f"Measured WC+GLC usage: {hours:.4f} EECU-hours; cap {args.eecu_cap_hours:.1f}")
    if args.submit and (active_glc or hours >= args.eecu_cap_hours):
        raise RuntimeError("Active GLC tasks or EECU cap; no new task submitted")

    manifest = OUT / "submitted_tasks.jsonl"
    recorded = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            item = json.loads(line)
            recorded[item["description"]] = item["task_id"]
    images = prior.year_images()
    started = 0
    for state in states:
        state_tracts = source.filter(ee.Filter.eq("STATEFP", state)).sort("GEOID")
        for offset in range(0, STATES[state], args.batch_size):
            count = min(args.batch_size, STATES[state] - offset)
            chunk = ee.FeatureCollection(state_tracts.toList(count, offset)).map(
                lambda feature: ee.Feature(feature.geometry(),
                                           {"GEOID": feature.get("GEOID"),
                                            "_glc_chunk": offset // args.batch_size}))
            for year in args.years:
                name = (f"{PREFIX}_{year}_s{state}_n{args.batch_size}_"
                        f"b{offset // args.batch_size:03d}")
                if existing.get(name) in {"PENDING", "READY", "RUNNING", "SUCCEEDED", "COMPLETED"}:
                    continue
                if name in recorded:
                    print(f"SKIP recorded {name} {recorded[name]}")
                    continue
                if not args.submit:
                    print(f"PLAN {name} ({count} tracts)")
                    continue
                if started >= args.max_new_tasks:
                    print(f"Task limit reached: {started}")
                    return
                table = prior.class_area_table(images[year], chunk, year,
                                               offset // args.batch_size, 8, int(2e10))
                task = ee.batch.Export.table.toDrive(
                    collection=table, description=name, folder=FOLDER,
                    fileNamePrefix=name, fileFormat="CSV",
                    selectors=prior.output_selectors())
                task.start()
                with manifest.open("a") as log:
                    log.write(json.dumps({"description": name, "task_id": task.id,
                                          "state": state, "year": year,
                                          "tract_count": count}) + "\n")
                started += 1
                print(f"STARTED {name} {task.id}")
    print(f"New tasks started: {started}")


if __name__ == "__main__":
    main()
