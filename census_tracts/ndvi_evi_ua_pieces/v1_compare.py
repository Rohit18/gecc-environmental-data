"""Stage 1: information-only comparison with GECC v1 tract NDVI_mean (JJA 2020) for Columbus tracts.

No v1 JJA 2020 tract table exists on NERSC or in Drive (the only v1 export archive, GECC_CensusTract_NDVI_EVI-
20260914T153945Z-1-001.zip, holds 2022 only), so v1 values are recomputed here with the v1 export script itself
($HOME/GECC/GECC_NERSC_NDVI_EVI_code_validation_no_boundaries/submit_census_tract_ndvi_evi_exports.py):
build_landsat_sr_collection(region) -> seasonal_composite(sr, 2020, "jja") -> reduceRegions(mean + count,
scale=30, tileScale=8), as in its seasonal_stats_table. That script has no GLC water mask. Tract geometry: the v1
tract asset (projects/your-ee-project/assets/GECC/censusTracts2020US, field GEOID) if readable, else TIGER 2020.

Selection: Columbus UA (19234) pieces with overlap_share > 0.99 and piece water share (WC 80 area / pixel area)
< 1%, from the Stage 1 draft tables.
  urbveg env:   python v1_compare.py --prepare
  s1aef-train:  python v1_compare.py --run
Output: qa/stage1_v1_compare.json, qa/stage1_v1_compare.csv.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndvi_common import load_config, utc_now, write_json_atomic  # noqa: E402

cfg = load_config()
QA = cfg["paths"]["qa"]
TEST = os.path.join(cfg["paths"]["output"], "stage1_test")
SEL = os.path.join(QA, "stage1_v1_compare_tracts.geojson")
V1 = "/path/to/home/GECC/GECC_NERSC_NDVI_EVI_code_validation_no_boundaries"


def prepare():
    import geopandas as gpd
    import pandas as pd
    tp = pd.read_parquet(os.path.join(TEST, "tract_piece_ndvi_evi_2020.parquet"))
    dg = pd.read_parquet(os.path.join(TEST, "piece_diagnostics_ndvi.parquet"))
    t = tp[(tp.GEOID20 == "19234") & (tp.season == "JJA")].merge(dg[["piece_id", "water_m2", "pixel_area_m2"]], on="piece_id")
    t["water_share"] = t.water_m2 / t.pixel_area_m2
    t = t[(t.overlap_share > 0.99) & (t.water_share < 0.01)]
    pcs = gpd.read_parquet(cfg["domains"]["pieces_all"])
    tr = pcs[pcs.GEOID.isin(t.GEOID)]
    # TIGER tract geometry = the union of the tract's pieces only if the tract is wholly in the UA; use the tract file
    files = [f"/path/to/scratch/urbveg/data/raw/boundaries/tracts_2020/tl_2020_39_tract/tl_2020_39_tract.shp"]
    tg = gpd.read_file(files[0]).to_crs("EPSG:4326")
    tg = tg[tg.GEOID.isin(t.GEOID)][["GEOID", "geometry"]].merge(
        t[["GEOID", "piece_id", "urban_ndvi", "ndvi_urban_valid_coverage", "overlap_share", "water_share"]], on="GEOID")
    tg.to_file(SEL, driver="GeoJSON")
    print(f"{len(tg)} tracts -> {SEL} (of {len(tr)} pieces)")


def run():
    import ee
    import numpy as np
    import ee_composite as ec
    sys.path.insert(0, V1)
    import submit_census_tract_ndvi_evi_exports as v1
    ec.init_ee(cfg["earth_engine"]["project"], cfg["earth_engine"]["url"])
    sel = json.load(open(SEL))
    geoids = [f["properties"]["GEOID"] for f in sel["features"]]
    ours = {f["properties"]["GEOID"]: f["properties"] for f in sel["features"]}
    source = "v1 asset " + v1.TRACT_ASSET
    try:
        fc = ee.FeatureCollection(v1.TRACT_ASSET).filter(ee.Filter.inList("GEOID", geoids)).select(["GEOID"])
        n = fc.size().getInfo()
        if n != len(geoids):
            raise RuntimeError(f"asset returned {n} of {len(geoids)} tracts")
    except Exception as e:  # noqa: BLE001
        source = f"TIGER 2020 (v1 asset not usable: {type(e).__name__}: {str(e)[:200]})"
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry(f["geometry"]), {"GEOID": f["properties"]["GEOID"]})
                                   for f in sel["features"]])
    region = fc.geometry(maxError=100).bounds(maxError=100)
    sr = v1.build_landsat_sr_collection(region)
    comp, nsc = v1.seasonal_composite(sr, 2020, "jja")
    red = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    t0 = time.time()
    res = comp.select(["NDVI", "EVI"]).reduceRegions(collection=fc, reducer=red, scale=v1.SCALE_M,
                                                     tileScale=v1.DEFAULT_TILE_SCALE).getInfo()
    secs = time.time() - t0
    rows = []
    for f in res["features"]:
        p = f["properties"]
        o = ours[p["GEOID"]]
        if p.get("NDVI_mean") is None or o.get("urban_ndvi") is None:
            continue
        rows.append({"GEOID": p["GEOID"], "v1_NDVI_mean": p["NDVI_mean"], "v1_NDVI_count": p.get("NDVI_count"),
                     "urban_ndvi": o["urban_ndvi"], "diff": o["urban_ndvi"] - p["NDVI_mean"],
                     "ndvi_urban_valid_coverage": o["ndvi_urban_valid_coverage"], "water_share": o["water_share"]})
    import csv
    with open(os.path.join(QA, "stage1_v1_compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    d = np.array([r["diff"] for r in rows])
    out = {"utc": utc_now(), "tract_source": source, "n_selected": len(geoids), "n_compared": len(rows),
           "seconds": round(secs, 1), "v1_regional_scene_count": nsc.getInfo(),
           "diff_definition": "ours urban_ndvi (JJA 2020) - v1-script tract NDVI_mean (JJA 2020)",
           "median": float(np.median(d)), "p5": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
           "mean": float(d.mean()), "min": float(d.min()), "max": float(d.max()),
           "median_abs": float(np.median(np.abs(d)))}
    write_json_atomic(os.path.join(QA, "stage1_v1_compare.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.run:
        run()
