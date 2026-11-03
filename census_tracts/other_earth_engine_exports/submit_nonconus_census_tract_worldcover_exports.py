#!/usr/bin/env python3
"""Incremental 2020 WorldCover exports for tracts absent from the CONUS release.

Dry run by default. Uses the public TIGER/2020/TRACT collection, so no boundary
upload is needed. Alaska requires an explicit benchmark or opt-in.
"""

import argparse
import json
from pathlib import Path

import ee

import submit_census_tract_worldcover_exports as prior


TRACT_ASSET = "TIGER/2020/TRACT"
STATES = {"02": 177, "15": 461, "60": 18, "66": 57, "69": 26, "72": 981, "78": 32}
ROUTINE_STATES = ("15", "60", "66", "69", "72", "78")
PREFIX = "WC2020_NONCONUS_tract_v2"
FOLDER = "GECC_nonCONUS_CensusTract_WorldCover2020"
OUT = Path(__file__).resolve().parent / "outputs/worldcover_2020_nonconus"


def normalized_tracts(states):
    source = ee.FeatureCollection(TRACT_ASSET).filter(
        ee.Filter.inList("STATEFP", list(states))
    )

    def normalize(feature):
        feature = ee.Feature(feature)
        return feature.set({
            "StFIPS": feature.get("STATEFP"),
            "Area_Land": feature.get("ALAND"),
            "Area_Water": feature.get("AWATER"),
        })

    return source.map(normalize)


def batch_name(kind, state, batch_size, batch_index):
    return f"{PREFIX}_{kind}_s{state}_n{batch_size}_b{batch_index:03d}"


def alaska_batches(tracts, batch_size, max_area_km2):
    """Deterministic GEOID-order chunks, each with a bounded total area."""
    rows = tracts.filter(ee.Filter.eq("StFIPS", "02")).sort("GEOID")
    geoids = rows.aggregate_array("GEOID").getInfo()
    land = rows.aggregate_array("Area_Land").getInfo()
    water = rows.aggregate_array("Area_Water").getInfo()
    batches, current, area = [], [], 0.0
    for geoid, land_m2, water_m2 in zip(geoids, land, water):
        size = (land_m2 + water_m2) / 1e6
        if current and (len(current) >= batch_size or area + size > max_area_km2):
            batches.append(current)
            current, area = [], 0.0
        current.append(geoid)
        area += size
    if current:
        batches.append(current)
    if len(geoids) != STATES["02"] or sum(map(len, batches)) != STATES["02"]:
        raise RuntimeError("Alaska batch partition is incomplete")
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="your-ee-project")
    parser.add_argument("--states", nargs="+", default=list(ROUTINE_STATES))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--alaska-max-area-km2", type=int, default=20000,
                        help="Maximum combined tract area per Alaska batch, except single large tracts")
    parser.add_argument("--kinds", nargs="+", choices=("areas", "crosswalk"),
                        default=["areas", "crosswalk"])
    parser.add_argument("--preview", nargs="*", default=[])
    parser.add_argument("--benchmark-geoid", help="Export one selected tract as a cost benchmark")
    parser.add_argument("--allow-alaska", action="store_true",
                        help="Permit routine Alaska batches after checking benchmark costs")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--allow-active", action="store_true",
                        help="Allow a monitored queue to submit alongside existing add-on tasks")
    parser.add_argument("--max-new-tasks", type=int, default=2)
    parser.add_argument("--eecu-cap-hours", type=float, default=1.0,
                        help="Stop new submissions once this measured task total is reached")
    args = parser.parse_args()
    states = tuple(dict.fromkeys(args.states))
    if not states or set(states) - STATES.keys():
        parser.error("States must be drawn from 02, 15, 60, 66, 69, 72, 78")
    if "02" in states and args.submit and not (args.benchmark_geoid or args.allow_alaska):
        parser.error("Alaska needs --benchmark-geoid or --allow-alaska")
    if args.benchmark_geoid and states != (args.benchmark_geoid[:2],):
        parser.error("Benchmark requires --states with only its own state")
    if args.batch_size < 1 or args.max_new_tasks < 1 or args.alaska_max_area_km2 < 1:
        parser.error("Batch size and task limit must be positive")
    if args.eecu_cap_hours <= 0:
        parser.error("EECU cap must be positive")

    ee.Initialize(project=args.project)
    prior.TRACT_ASSET = TRACT_ASSET  # Output provenance in the existing row schema.
    all_tracts = normalized_tracts(states)
    uas = ee.FeatureCollection(prior.UA_ASSET)
    wc = ee.ImageCollection(prior.WC_ASSET).first().select("Map")
    info = ee.Dictionary({
        "tract_count": all_tracts.size(),
        "state_counts": all_tracts.aggregate_histogram("StFIPS"),
        "unique_geoids": all_tracts.aggregate_count_distinct("GEOID"),
        "ua_count": uas.size(),
        "projection": wc.projection(),
    }).getInfo()
    if info["tract_count"] != info["unique_geoids"]:
        raise RuntimeError("Nonunique tract GEOID in TIGER/2020/TRACT")
    for state in states:
        if int(info["state_counts"].get(state, 0)) != STATES[state]:
            raise RuntimeError(f"Unexpected 2020 tract count for {state}")
    if info["ua_count"] != 2644:
        raise RuntimeError("Urban Area asset changed; inspect before exporting")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_validation.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({key: value for key, value in info.items() if key != "projection"}))
    tracts = all_tracts
    if args.benchmark_geoid:
        tracts = all_tracts.filter(ee.Filter.eq("GEOID", args.benchmark_geoid))
        if tracts.size().getInfo() != 1:
            raise ValueError(f"Benchmark GEOID not found exactly once: {args.benchmark_geoid}")

    image = prior.area_image(wc)
    projection = info["projection"]
    for geoid in args.preview:
        selected = tracts.filter(ee.Filter.eq("GEOID", geoid))
        if selected.size().getInfo() != 1:
            raise ValueError(f"Preview GEOID not found exactly once: {geoid}")
        feature = ee.Feature(selected.first())
        row = prior.tract_row(feature, uas, image, projection,
                              single_pass=True).toDictionary().getInfo()
        pairs = prior.pair_rows(feature, uas).getInfo()["features"]
        prior.check_row(row, pairs)
        (OUT / f"preview_{PREFIX}_{geoid}.json").write_text(
            json.dumps({"row": row, "pairs": pairs}, indent=2)
        )
        print(f"PREVIEW PASS {geoid}: {row['ua_count']} UA matches")

    operations = ee.data.listOperations(f"projects/{args.project}")
    relevant = [operation.get("metadata", {}) for operation in operations
                if operation.get("metadata", {}).get("description", "").startswith(
                    "WC2020_NONCONUS_tract_")]
    existing = {meta["description"]: meta.get("state") for meta in relevant}
    eecu_hours = sum(float(meta.get("batchEecuUsageSeconds", 0))
                     for meta in relevant) / 3600
    print(f"Measured add-on task usage: {eecu_hours:.4f} EECU-hours; "
          f"submission cap: {args.eecu_cap_hours:.4f}")
    if args.submit and not args.allow_active and any(
            meta.get("state") in {"PENDING", "READY", "RUNNING"} for meta in relevant):
        raise RuntimeError("Existing add-on tasks are still active; inspect them first")
    if args.submit and eecu_hours >= args.eecu_cap_hours:
        raise RuntimeError("EECU cap reached; no tasks submitted")
    started = 0
    manifest = OUT / "submitted_tasks.jsonl"
    recorded = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            item = json.loads(line)
            recorded[item["description"]] = item["task_id"]
    for state in states:
        state_tracts = tracts.filter(ee.Filter.eq("StFIPS", state)).sort("GEOID")
        if args.benchmark_geoid:
            partitions = [(state_tracts, 1)]
        elif state == "02":
            partitions = [
                (state_tracts.filter(ee.Filter.inList("GEOID", geoids)), len(geoids))
                for geoids in alaska_batches(tracts, args.batch_size,
                                             args.alaska_max_area_km2)
            ]
        else:
            count = STATES[state]
            partitions = [
                (ee.FeatureCollection(state_tracts.toList(
                    min(args.batch_size, count - offset), offset)),
                 min(args.batch_size, count - offset))
                for offset in range(0, count, args.batch_size)
            ]
        for batch_index, (chunk_fc, chunk_count) in enumerate(partitions):
            chunk = chunk_fc.toList(chunk_count)
            for kind in args.kinds:
                name = (f"{PREFIX}_{kind}_benchmark_{args.benchmark_geoid}"
                        if args.benchmark_geoid else
                        (f"WC2020_NONCONUS_tract_v{'3' if kind == 'areas' else '2'}_"
                         f"{kind}_s02_n{args.batch_size}_a"
                         f"{args.alaska_max_area_km2}_b{batch_index:03d}"
                         if state == "02" else
                         batch_name(kind, state, args.batch_size, batch_index)))
                # Five Alaska area exports were cancelled on 2026-09-25.
                # A new description preserves both attempts' measured EECU use.
                if (state == "02" and kind == "areas" and
                        batch_index in {31, 33, 37, 41, 42} and
                        existing.get(name) in {"CANCELLING", "CANCELLED"}):
                    name = name.replace("_tract_v3_areas_", "_tract_v5_areas_")
                if existing.get(name) in {"PENDING", "READY", "RUNNING", "SUCCEEDED", "COMPLETED"}:
                    print(f"SKIP {name} {existing[name]}")
                    continue
                if name in recorded:
                    print(f"SKIP {name} recorded task {recorded[name]}; inspect before retrying")
                    continue
                # The one-pass change affects only areas. Reuse an identical
                # completed v1 crosswalk rather than spending EECU twice.
                legacy_name = name.replace("_tract_v2_", "_tract_v1_")
                if kind == "crosswalk" and existing.get(legacy_name) in {"SUCCEEDED", "COMPLETED"}:
                    print(f"REUSE {legacy_name} for {name}")
                    continue
                if not args.submit:
                    print(f"PLAN {name} ({chunk_count} tracts)")
                    continue
                if started >= args.max_new_tasks:
                    print(f"Task limit reached: {started}")
                    return
                if kind == "areas":
                    table = ee.FeatureCollection(chunk).map(
                        lambda feature: prior.tract_row(feature, uas, image,
                                                        projection, single_pass=True)
                    )
                    fields = prior.selectors()
                else:
                    table = ee.FeatureCollection(chunk.map(
                        lambda feature: prior.pair_rows(feature, uas)
                    )).flatten()
                    fields = prior.PAIR_FIELDS
                task = ee.batch.Export.table.toDrive(
                    collection=table, description=name, folder=FOLDER,
                    fileNamePrefix=name, fileFormat="CSV", selectors=fields,
                )
                task.start()
                with manifest.open("a") as log:
                    log.write(json.dumps({"description": name, "task_id": task.id,
                                          "state": state, "tract_count": chunk_count,
                                          "kind": kind, "project": args.project}) + "\n")
                started += 1
                print(f"STARTED {name} {task.id}")
    print(f"New tasks started: {started}")


if __name__ == "__main__":
    main()
