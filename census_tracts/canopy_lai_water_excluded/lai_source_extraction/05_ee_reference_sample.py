#!/usr/bin/env python3
"""Earth Engine reference statistics for a small tract sample (optional QA step).

The supplied Earth Engine CSV exports carry no LAI values (every statistic is
null), so they cannot be compared numerically. This step recomputes a small,
seeded sample in Earth Engine itself, on the uint8 asset at its own
crsTransform (no scale / generic 10 m grid), to show how Earth Engine counts and
weights pixels and how its percentiles relate to the first-value rule.

Reducers (all at the native crsTransform, geometries planar in EPSG:4326):
  weighted:   mean, stdDev, count, min, max, percentile([25, 50, 75]), fixedHistogram(0, 256, 256)
  unweighted: count, fixedHistogram(0, 256, 256)

EECU use is profiled and appended to the per-project ledger kept with the
downloader; the run refuses to start if the ledger is at its cap.

Run from a login node (needs network and Earth Engine credentials):
    python 05_ee_reference_sample.py --manifest-dir $WORK/manifest --project your-ee-project \
        --ledger $LAI_LEDGER --out $WORK/qa/ee_reference_sample.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lai_common import EE_SOURCE, ORIGIN_X, ORIGIN_Y, RES, write_json

SEED = 20260921
LEDGER_CAP_HOURS = 100.0
MARGIN = 1.25


def ledger_total_s(path: Path, project: str) -> float:
    if not path.exists():
        return 0.0
    return sum(json.loads(l)["eecu_s"] for l in path.read_text().splitlines() if l.strip()
               and json.loads(l).get("project") == project)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--ledger", required=True, help="EECU ledger JSONL shared with the downloader")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-stratum", type=int, default=20)
    ap.add_argument("--max-area-km2", type=float, default=20.0)
    a = ap.parse_args()
    import ee
    import geopandas as gpd
    import shapely

    ledger = Path(a.ledger)
    prior = ledger_total_s(ledger, a.project)
    if prior * MARGIN >= LEDGER_CAP_HOURS * 3600:
        raise SystemExit(f"ledger for {a.project} at cap ({prior / 3600:.2f} EECU-h); not running")

    tr = gpd.read_parquet(Path(a.manifest_dir) / "tracts_conus_4326.parquet")
    tr = tr[tr.geometry_area_m2 < a.max_area_km2 * 1e6]
    rng = np.random.default_rng(SEED)
    tot = tr.Area_Land + tr.Area_Water
    water = (tr.Area_Water / tot.where(tot > 0)).fillna(0)
    strata = {
        "tiny": tr.nsmallest(a.n_per_stratum, "geometry_area_m2").index.to_numpy(),
        "coastal": rng.choice(tr[water >= 0.5].index.to_numpy(), a.n_per_stratum, replace=False),
        "random": rng.choice(tr.index.to_numpy(), a.n_per_stratum, replace=False),
    }
    parts = [pd.DataFrame({"idx": v, "stratum": k}) for k, v in strata.items()]
    pick = pd.concat(parts).drop_duplicates("idx")
    s = tr.loc[pick.idx].assign(stratum=pick.stratum.values)

    warnings.filterwarnings("ignore", message=".*restricted mode.*")
    ee.Initialize(project=a.project)
    img = ee.Image(EE_SOURCE).select("LAI_uint8")
    feats = [ee.Feature(ee.Geometry(shapely.geometry.mapping(g), "EPSG:4326", False), {"GEOID": gid})
             for gid, g in zip(s.GEOID, s.geometry)]
    fc = ee.FeatureCollection(feats)
    weighted = (ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), "", True)
                .combine(ee.Reducer.count(), "", True)
                .combine(ee.Reducer.min(), "", True)
                .combine(ee.Reducer.max(), "", True)
                .combine(ee.Reducer.percentile([25, 50, 75]), "", True)
                .combine(ee.Reducer.fixedHistogram(0, 256, 256), "", True))
    unweighted = (ee.Reducer.count().unweighted()
                  .combine(ee.Reducer.fixedHistogram(0, 256, 256).unweighted(), "", True))
    reducer = weighted.combine(unweighted.setOutputs(["u_count", "u_histogram"]), "", True)
    ids: list[str] = []
    t0 = time.time()
    with ee.data.profiling(ids.append):
        res = img.reduceRegions(collection=fc, reducer=reducer, crs="EPSG:4326",
                                crsTransform=[RES, 0, ORIGIN_X, 0, -RES, ORIGIN_Y]).getInfo()
    wall = time.time() - t0

    get_profiles = ee.ApiFunction.lookup("Profile.getProfiles").call
    eecu, calls = None, 0
    for k in range(10):
        calls += 1
        try:
            txt = get_profiles(ids=ids).getInfo()
            eecu = sum(float(l.split()[0]) for l in txt.splitlines()
                       if l.split() and re.fullmatch(r"\d+(\.\d+)?", l.split()[0]))
            break
        except ee.EEException:
            time.sleep(3 * (k + 1))
    rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "project": a.project, "asset": EE_SOURCE,
           "tiles": 0, "requests": len(ids), "eecu_s": round((eecu if eecu is not None else 5.0) + 0.05 * calls, 4),
           "note": f"05_ee_reference_sample.py: reduceRegions on {len(s)} tracts"}
    if eecu is None:
        rec["estimated"] = True
    with ledger.open("a") as f:
        f.write(json.dumps(rec) + "\n")

    rows = []
    for f in res["features"]:
        p = f["properties"]

        def hist(key):
            h = p.get(key)
            return [] if h is None else [[int(round(b[0])), float(b[1])] for b in h if b[1]]

        rows.append({
            "GEOID": p["GEOID"],
            "ee_mean": None if p.get("mean") is None else p["mean"] / 10,
            "ee_stdDev": None if p.get("stdDev") is None else p["stdDev"] / 10,
            "ee_count": p.get("count"), "ee_u_count": p.get("u_count"),
            "ee_min": None if p.get("min") is None else p["min"] / 10,
            "ee_max": None if p.get("max") is None else p["max"] / 10,
            "ee_p25": None if p.get("p25") is None else p["p25"] / 10,
            "ee_p50": None if p.get("p50") is None else p["p50"] / 10,
            "ee_p75": None if p.get("p75") is None else p["p75"] / 10,
            "ee_hist_weighted": json.dumps(hist("histogram")),
            "ee_hist_unweighted": json.dumps(hist("u_histogram")),
        })
    df = pd.DataFrame(rows).merge(s[["GEOID", "stratum", "geometry_area_m2", "Area_Land", "Area_Water"]], on="GEOID")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_json(out.with_suffix(".json"), {"n": len(df), "project": a.project, "asset": EE_SOURCE,
                                           "eecu_s": rec["eecu_s"], "eecu_estimated": eecu is None,
                                           "wall_s": round(wall, 1), "seed": SEED,
                                           "crsTransform": [RES, 0, ORIGIN_X, 0, -RES, ORIGIN_Y],
                                           "strata": df.stratum.value_counts().to_dict()})
    print(f"EE reference: {len(df)} tracts, {rec['eecu_s']} EECU-s, {wall:.0f}s wall", flush=True)


if __name__ == "__main__":
    main()
