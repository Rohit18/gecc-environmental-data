"""Stage 1 grid fidelity check (conda env s1aef-train).

N random 10 m WorldCover pixel centres inside one UA (default Columbus, 19234; fixed seed). For each centre and
season, the value read from the fetched chunk store (cell i = floor(lon/s), j = floor(-lat/s)) must equal
composite.reduceRegion(ee.Reducer.first(), point, scale=30) evaluated by Earth Engine with no crs given (the v1
default grid), where composite = seasonal_composite(<chunk rectangle>, 2020, season) for the point's chunk.
Points are evaluated per chunk with FeatureCollection.map(reduceRegion), one request per chunk and season.
Mismatch: finite in one and missing in the other, |difference| > 1e-6 for NDVI/EVI, or a clear_obs difference.

Input: points file written by the worker env (--make-points, urbveg env) because the UA polygon test needs
shapely. Output: qa/stage1_grid_fidelity.json (+ per-point CSV).
Usage:
  urbveg env:   python grid_fidelity.py --make-points [--ua 19234] [--n 1000]
  s1aef-train:  python grid_fidelity.py --check
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndvi_common import SEASON_SHORT, CompositeGrid, chunk_path, load_config, utc_now, write_json_atomic  # noqa: E402

cfg = load_config()
PTS = os.path.join(cfg["paths"]["qa"], "stage1_grid_fidelity_points.csv")


def make_points(ua_id, n, seed=20261013):
    import geopandas as gpd
    import shapely
    ua = gpd.read_parquet(cfg["domains"]["uas_all"])
    g = ua.loc[ua.GEOID20 == ua_id, "geometry"].iloc[0]
    shapely.prepare(g)
    w, s, e, nn = g.bounds
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        k = rng.integers(int(np.floor(w * 12000)), int(np.ceil(e * 12000)), 4 * n)
        m = rng.integers(int(np.floor(s * 12000)), int(np.ceil(nn * 12000)), 4 * n)
        lon, lat = (k + 0.5) / 12000, (m + 0.5) / 12000      # WorldCover pixel centres (1/12000 deg grid, origin on integer degrees)
        ok = shapely.contains_xy(g, lon, lat)
        pts += list(zip(lon[ok], lat[ok]))
    pts = pts[:n]
    os.makedirs(os.path.dirname(PTS), exist_ok=True)
    with open(PTS, "w") as f:
        f.write("pid,lon,lat\n")
        for i, (x, y) in enumerate(pts):
            f.write(f"{i},{float(x)!r},{float(y)!r}\n")
    print(f"{len(pts)} points in UA {ua_id} -> {PTS}")


def check():
    import ee
    import ee_composite as ec
    ec.init_ee(cfg["earth_engine"]["project"], cfg["earth_engine"]["url"])
    grid = CompositeGrid.from_cfg(cfg)
    pts = list(csv.DictReader(open(PTS)))
    lon = np.array([float(p["lon"]) for p in pts]); lat = np.array([float(p["lat"]) for p in pts])
    ci, rj = grid.col(lon), grid.row(lat)
    CI, RJ = grid.chunk_of(ci), grid.chunk_of(rj)
    edge = np.minimum(grid.edge_distance_deg(lon, "lon"), grid.edge_distance_deg(lat, "lat"))
    rows, t0 = [], time.time()
    for season in cfg["seasons"]:
        sh = SEASON_SHORT[season]
        for I, J in sorted(set(zip(CI.tolist(), RJ.tolist()))):
            sel = np.nonzero((CI == I) & (RJ == J))[0]
            with np.load(chunk_path(cfg["paths"]["composites"], season, I, J)) as z:
                loc = {k: z[k][rj[sel] - J * grid.C, ci[sel] - I * grid.C] for k in ("NDVI", "EVI", "clear_obs")}
            comp = ec.seasonal_composite(ec.chunk_region(grid.chunk_bounds(I, J)), cfg["year"], season)
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lon[k]), float(lat[k])]), {"pid": int(k)})
                                       for k in sel])
            first = ee.Reducer.first()
            res = fc.map(lambda f: f.set(comp.reduceRegion(first, f.geometry(), scale=30))).getInfo()
            got = {f["properties"]["pid"]: f["properties"] for f in res["features"]}
            for n_, k in enumerate(sel):
                p = got[int(k)]
                r = {"pid": int(k), "season": season, "lon": lon[k], "lat": lat[k], "i": int(ci[k]), "j": int(rj[k]),
                     "edge_dist_deg": float(edge[k])}
                mism = False
                for b in ("NDVI", "EVI", "clear_obs"):
                    a = float(loc[b][n_])
                    e = p.get(b)
                    e = np.nan if e is None else float(e)
                    if b == "clear_obs":
                        e = 0.0 if np.isnan(e) else e      # fetched clear_obs is unmasked with 0
                        d = abs(a - e); bad = d != 0
                    else:
                        both_nan = np.isnan(a) and np.isnan(e)
                        d = 0.0 if both_nan else abs(a - e)
                        bad = (not both_nan) and (np.isnan(a) or np.isnan(e) or d > 1e-6)
                    r[f"{b}_fetched"], r[f"{b}_ee"], r[f"{b}_absdiff"] = a, e, d
                    mism |= bool(bad)
                r["mismatch"] = mism
                rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(PTS.replace("_points.csv", "_per_point.csv"), index=False)
    out = {"utc": utc_now(), "ua": "19234", "n_points": len(pts), "seed": 20261013, "seconds": round(time.time() - t0, 1),
           "chunks": sorted({f"{a},{b}" for a, b in zip(CI.tolist(), RJ.tolist())}), "chunk_cells": grid.C,
           "min_edge_distance_deg": float(edge.min()), "by_season": {}}
    for season, g in df.groupby("season"):
        dd = {"n": len(g), "mismatches": int(g.mismatch.sum()),
              "both_missing_ndvi": int((g.NDVI_fetched.isna() & g.NDVI_ee.isna()).sum())}
        for b in ("NDVI", "EVI", "clear_obs"):
            dd[f"max_abs_diff_{b}"] = float(np.nanmax(g[f"{b}_absdiff"]))
        out["by_season"][season] = dd
    out["pass"] = all(v["mismatches"] == 0 for v in out["by_season"].values())
    write_json_atomic(os.path.join(cfg["paths"]["qa"], "stage1_grid_fidelity.json"), out)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["pass"] else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-points", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ua", default="19234")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    if a.make_points:
        make_points(a.ua, a.n)
    if a.check:
        check()
