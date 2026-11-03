#!/usr/bin/env python3
"""Export the 1,752 missing TIGER/2020 tract polygons for NERSC jobs."""

import argparse
import json
from pathlib import Path

import ee


PROJECT = "your-ee-project"
STATES = {"02": 177, "15": 461, "60": 18, "66": 57, "69": 26, "72": 981, "78": 32}
DESCRIPTION = "GECC_nonCONUS_2020_tract_boundaries_v2"
FOLDER = "GECC_nonCONUS_CensusTract_boundaries"
OUT = Path(__file__).resolve().parent / "outputs/nonconus_tract_boundaries"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    ee.Initialize(project=PROJECT)
    tracts = ee.FeatureCollection("TIGER/2020/TRACT").filter(
        ee.Filter.inList("STATEFP", list(STATES)))
    counts = tracts.aggregate_histogram("STATEFP").getInfo()
    if {k: int(v) for k, v in counts.items()} != STATES:
        raise RuntimeError(f"Unexpected 2020 tract counts: {counts}")
    if tracts.aggregate_count_distinct("GEOID").getInfo() != sum(STATES.values()):
        raise RuntimeError("Nonunique GEOIDs in source boundaries")
    OUT.mkdir(parents=True, exist_ok=True)
    ops = ee.data.listOperations(f"projects/{PROJECT}")
    matches = [op for op in ops if op.get("metadata", {}).get("description") == DESCRIPTION]
    if matches:
        op = matches[-1]
        print(json.dumps({"description": DESCRIPTION, "existing_operation": op["name"],
                          "state": op.get("metadata", {}).get("state")}, indent=2))
        return
    if not args.submit:
        print(json.dumps({"description": DESCRIPTION, "state_counts": counts,
                          "total": sum(STATES.values())}, indent=2))
        return
    fields = ["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "ALAND", "AWATER"]
    selected = tracts.map(lambda f: ee.Feature(f.geometry(), f.toDictionary(fields)))
    task = ee.batch.Export.table.toDrive(
        collection=selected, description=DESCRIPTION, folder=FOLDER,
        fileNamePrefix=DESCRIPTION, fileFormat="GeoJSON")
    task.start()
    rec = {"description": DESCRIPTION, "task_id": task.id,
           "state": task.status().get("state"), "state_counts": counts,
           "total": sum(STATES.values()), "source": "TIGER/2020/TRACT"}
    (OUT / "task.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
