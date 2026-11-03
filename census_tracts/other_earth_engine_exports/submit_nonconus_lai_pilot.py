#!/usr/bin/env python3
"""Submit one small 2020 LAI diagnostic tract per non-CONUS jurisdiction.

This export measures Earth Engine behavior and EECU before any large queue.
The published CONUS LAI release was computed at NERSC with coverage-weighted
native-grid statistics; these Earth Engine reducer values are not identical.
"""

import argparse
import json
from pathlib import Path

import ee


PROJECT = "your-ee-project"
TRACTS = "TIGER/2020/TRACT"
LAI = "projects/your-ee-project/assets/LAI_2020_Global_Mosaic_uint8"
STATES = ("02", "15", "60", "66", "69", "72", "78")
DESCRIPTION = "GECC_nonCONUS_LAI2020_7tract_diagnostic_v2"
FOLDER = "GECC_nonCONUS_CensusTract_LAI_2020_pilot"
OUT = Path(__file__).resolve().parent / "outputs/lai_nonconus"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    ee.Initialize(project=PROJECT)
    source = ee.FeatureCollection(TRACTS)
    image = ee.Image(LAI).select("LAI_uint8").divide(10).rename("lai_native")
    selected = []
    for state in STATES:
        tract = ee.Feature(source.filter(ee.Filter.eq("STATEFP", state))
                           .filter(ee.Filter.gt("ALAND", 1_000_000))
                           .sort("ALAND").first())
        selected.append(tract.set("StFIPS", state))
    reducer = (ee.Reducer.mean()
               .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
               .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
               .combine(reducer2=ee.Reducer.count(), sharedInputs=True)
               .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
               .combine(reducer2=ee.Reducer.percentile([25, 75], ["p25", "p75"]),
                        sharedInputs=True))
    reduced = image.reduceRegions(collection=ee.FeatureCollection(selected),
                                  reducer=reducer, scale=10, crs="EPSG:4326",
                                  tileScale=8)
    fields = ["GEOID", "StFIPS", "ALAND", "AWATER", "lai_native_mean",
              "lai_native_median", "lai_native_stdDev", "lai_native_count",
              "lai_native_min", "lai_native_max", "lai_native_p25",
              "lai_native_p75"]
    stat_names = ["mean", "median", "stdDev", "count", "min", "max", "p25", "p75"]
    def normalize(feature):
        feature = ee.Feature(feature)
        props = feature.toDictionary(["GEOID", "StFIPS", "ALAND", "AWATER"])
        props = ee.Dictionary(props)
        for source_name in stat_names:
            props = props.set("lai_native_" + source_name, feature.get(source_name))
        return ee.Feature(None, props)
    rows = reduced.map(normalize)
    ops = ee.data.listOperations(f"projects/{PROJECT}")
    matches = [op for op in ops if op.get("metadata", {}).get("description") == DESCRIPTION]
    if matches:
        meta = matches[-1].get("metadata", {})
        print(json.dumps({"description": DESCRIPTION, "existing_operation": matches[-1]["name"],
                          "state": meta.get("state"), "eecu_seconds": meta.get("batchEecuUsageSeconds")},
                         indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.submit:
        print(json.dumps({"description": DESCRIPTION, "states": STATES,
                          "tract_count": len(STATES), "source": LAI,
                          "note": "Pass --submit to start the diagnostic export."}, indent=2))
        return
    task = ee.batch.Export.table.toDrive(collection=rows, description=DESCRIPTION,
                                         folder=FOLDER, fileNamePrefix=DESCRIPTION,
                                         fileFormat="CSV", selectors=fields)
    task.start()
    record = {"description": DESCRIPTION, "task_id": task.id,
              "state": task.status().get("state"), "source": LAI,
              "states": STATES, "tract_count": len(STATES),
              "purpose": "cost and output diagnostic; not a NERSC-equivalent release"}
    (OUT / "pilot_task.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
