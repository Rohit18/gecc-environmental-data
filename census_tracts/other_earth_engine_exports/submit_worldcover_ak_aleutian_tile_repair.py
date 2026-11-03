#!/usr/bin/env python3
"""Repair the timed-out Aleutian tract as disjoint native-grid 10 m tiles."""

import argparse
import json
import math
from pathlib import Path

import ee

import submit_census_tract_worldcover_exports as prior


PROJECT = "your-ee-project"
GEOID = "02016000100"
TRACTS = "TIGER/2020/TRACT"
FOLDER = "GECC_nonCONUS_CensusTract_WorldCover2020"
PREFIX = "WC2020_NONCONUS_tract_v4_areas_ak_aleutian_02016000100"
OUT = Path(__file__).resolve().parent / "outputs/worldcover_2020_nonconus"
PIXEL_DEGREES = 1 / 12000


def tiles_for_bounds(bounds):
    west, south = bounds[0]
    east, north = bounds[2]
    x0 = math.floor((west + 180) / PIXEL_DEGREES)
    x1 = math.ceil((east + 180) / PIXEL_DEGREES)
    y0 = math.floor((84 - north) / PIXEL_DEGREES)
    y1 = math.ceil((84 - south) / PIXEL_DEGREES)
    # Tile edges coincide with the actual WorldCover 2020 pixel grid.
    xs = [-180 + (x0 + (x1 - x0) * i // 5) * PIXEL_DEGREES
          for i in range(6)]
    ys = [84 - (y1 - (y1 - y0) * j // 4) * PIXEL_DEGREES
          for j in range(5)]
    return [(i * 4 + j, [xs[i], ys[j], xs[i + 1], ys[j + 1]])
            for i in range(5) for j in range(4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--max-new-tasks", type=int, default=2)
    parser.add_argument("--eecu-cap-hours", type=float, default=950)
    args = parser.parse_args()
    if args.max_new_tasks < 1 or args.eecu_cap_hours <= 0:
        parser.error("Limits must be positive")
    ee.Initialize(project=PROJECT)
    prior.TRACT_ASSET = TRACTS
    source = ee.FeatureCollection(TRACTS).filter(ee.Filter.eq("GEOID", GEOID))
    if source.size().getInfo() != 1:
        raise RuntimeError("Aleutian tract not found exactly once")
    raw = ee.Feature(source.first())
    tract = raw.set({"StFIPS": "02", "Area_Land": raw.get("ALAND"),
                     "Area_Water": raw.get("AWATER")})
    wc = ee.ImageCollection(prior.WC_ASSET).first().select("Map")
    projection = wc.projection().getInfo()
    if projection["crs"] != "EPSG:4326" or projection["transform"] != [
            PIXEL_DEGREES, 0, -180, 0, -PIXEL_DEGREES, 84]:
        raise RuntimeError("WorldCover pixel grid changed")
    all_tiles = tiles_for_bounds(tract.geometry().bounds(1).coordinates().getInfo()[0])
    areas = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Rectangle(bounds, None, False),
                   {"index": index, "km2": tract.geometry().intersection(
                       ee.Geometry.Rectangle(bounds, None, False), 1).area(1).divide(1e6)})
        for index, bounds in all_tiles
    ]).aggregate_array("km2").getInfo()
    tiles = [(index, bounds, areas[index]) for index, bounds in all_tiles
             if areas[index] > 0]
    if not math.isclose(sum(area for _, _, area in tiles),
                        tract.geometry().area(1).getInfo() / 1e6,
                        abs_tol=0.001):
        raise RuntimeError("Tile partition does not cover entire tract")
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"geoid": GEOID, "projection": projection,
            "tiles": [{"index": index, "bounds": bounds, "geometry_km2": area}
                      for index, bounds, area in tiles]}
    (OUT / "ak_aleutian_tile_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Aleutian repair: {len(tiles)} nonempty tiles; "
          f"largest {max(area for _, _, area in tiles):.1f} km2")
    operations = ee.data.listOperations(f"projects/{PROJECT}")
    states = {op.get("metadata", {}).get("description"): op.get("metadata", {}).get("state")
              for op in operations}
    active = sum(state in {"PENDING", "READY", "RUNNING"}
                 for name, state in states.items() if name.startswith(PREFIX))
    if args.submit and active >= 2:
        print(f"WAIT: {active} tile tasks active")
        return
    ledger = OUT / "eecu_ledger.json"
    hours = sum(json.loads(ledger.read_text()).values()) / 3600 if ledger.exists() else 0
    if args.submit and hours >= args.eecu_cap_hours:
        raise RuntimeError("Combined EECU cap reached")
    manifest = OUT / "ak_aleutian_tile_tasks.jsonl"
    recorded = ({json.loads(line)["description"] for line in manifest.read_text().splitlines()}
                if manifest.exists() else set())
    image = prior.area_image(wc)
    uas = ee.FeatureCollection(prior.UA_ASSET)
    started = 0
    for index, bounds, area in tiles:
        name = f"{PREFIX}_tile_t{index:02d}"
        if states.get(name) in {"PENDING", "READY", "RUNNING", "SUCCEEDED"} or name in recorded:
            continue
        if not args.submit:
            print(f"PLAN {name} {area:.1f} km2")
            continue
        if started >= min(args.max_new_tasks, 2 - active):
            break
        row = prior.tract_row(tract, uas, image, projection, single_pass=True,
                              raster_tile_bounds=bounds).set("repair_tile_index", index)
        task = ee.batch.Export.table.toDrive(
            collection=ee.FeatureCollection([row]), description=name,
            folder=FOLDER, fileNamePrefix=name, fileFormat="CSV",
            selectors=prior.selectors() + ["repair_tile_index"])
        task.start()
        with manifest.open("a") as stream:
            stream.write(json.dumps({"description": name, "task_id": task.id,
                                     "index": index, "bounds": bounds,
                                     "geometry_km2": area}) + "\n")
        print(f"STARTED {name} {task.id}")
        started += 1
    print(f"New tasks started: {started}")


if __name__ == "__main__":
    main()
