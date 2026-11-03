#!/usr/bin/env python3
"""How Earth Engine's reducers relate to this workflow's definitions (QA, needs 05's output).

Uses the Earth Engine reference sample (05_ee_reference_sample.py) and the
supplied Earth Engine exports to test, reproducibly:
  * count:    does Earth Engine's `count` equal the number of valid cells with exact
              coverage above some threshold, or with >= 1 of k x k sub-cell sample
              points inside the polygon (k = 4, 8, 16, 32)?
  * stdDev:   is it weighted, or the unweighted population SD of included cells?
              (recomputed from Earth Engine's own weighted / unweighted histograms)
  * mean:     does it equal the mean of Earth Engine's weighted histogram?
  * quantiles: where Earth Engine's percentile differs from lai_native_*, does the
              first-value rule applied to Earth Engine's own weighted histogram give
              the NERSC value? Is Earth Engine's value off the 0.1 grid (interpolated)?
  * area:     is the exported geometry_area_m2 a spherical area (R = 6378137 m)?

Writes $QA_DIR/ee_semantics.json, which 06_merge_validate.py folds into validation.json.

Example:
    python 07_ee_semantics.py --manifest-dir $MANIFEST_DIR --shard-dir $SHARD_DIR --qa-dir $QA_DIR \
        --vrt $LAI_VRT --ee-exports $EE_EXPORTS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lai_common import gdal_env, write_json

SEED = 20260921
THRESHOLDS = {">0": 0.0, ">=0.5/255": 0.5 / 255, ">=1/255": 1 / 255, ">=0.5/256": 0.5 / 256, ">=1/256": 1 / 256}
SUBSAMPLE_K = (4, 8, 16, 32)


def first_value(values: np.ndarray, weights: np.ndarray, p: float) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, p * cum[-1] * (1 - 1e-12))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--vrt", required=True)
    ap.add_argument("--ee-exports", required=True)
    ap.add_argument("--area-sample", type=int, default=5000)
    a = ap.parse_args()
    import os
    for k, v in gdal_env().items():
        os.environ.setdefault(k, v)
    import geopandas as gpd
    import pyproj
    import rasterio
    from exactextract import exact_extract
    from rasterio import features
    from rasterio.transform import Affine
    from rasterio.windows import Window

    qdir = Path(a.qa_dir)
    ref = pd.read_parquet(qdir / "ee_reference_sample.parquet").set_index("GEOID")
    tr = gpd.read_parquet(Path(a.manifest_dir) / "tracts_conus_4326.parquet")
    samp = tr.set_index("GEOID").loc[ref.index]
    stats = pd.concat([pd.read_parquet(p) for p in sorted(Path(a.shard_dir).glob("shard_*.parquet"))]).set_index("GEOID")

    # --- count rules: exact coverage thresholds and k x k point sampling
    ex = exact_extract(a.vrt, samp.reset_index()[["GEOID", "geometry"]], ["values", "coverage"],
                       include_cols=["GEOID"], output="pandas").set_index("GEOID")
    ds = rasterio.open(a.vrt)
    rows = []
    for g, geom in samp.geometry.items():
        e = ref.loc[g]
        c = np.asarray(ex.loc[g, "coverage"], float)
        r = {"GEOID": g, "stratum": e.stratum, "ee_count": int(e.ee_count), "touched": int((c > 0).sum())}
        for name, t in THRESHOLDS.items():
            r[f"thr {name}"] = int(((c > 0) & (c >= t)).sum())
        inv = ~ds.transform
        minx, miny, maxx, maxy = geom.bounds
        c0, r0 = [int(np.floor(x)) - 1 for x in inv * (minx, maxy)]
        c1, r1 = [int(np.floor(x)) + 2 for x in inv * (maxx, miny)]
        win = Window(c0, r0, c1 - c0, r1 - r0)
        arr = ds.read(1, window=win, boundless=True, fill_value=ds.nodata)
        wt = ds.window_transform(win)
        valid = arr != ds.nodata
        for k in SUBSAMPLE_K:
            sub = features.rasterize([(geom, 1)], out_shape=(arr.shape[0] * k, arr.shape[1] * k),
                                     transform=wt * Affine.scale(1 / k), all_touched=False, dtype="uint8")
            n_in = sub.reshape(arr.shape[0], k, arr.shape[1], k).sum(axis=(1, 3))
            r[f"sample {k}x{k}"] = int(((n_in > 0) & valid).sum())
        rows.append(r)
    cnt = pd.DataFrame(rows)
    rules = [c for c in cnt.columns if c.startswith(("thr", "sample"))]
    count_rep = {
        "n_tracts": len(cnt),
        "ee_count_minus_touched": {"median": float((cnt.ee_count - cnt.touched).median()),
                                   "min": int((cnt.ee_count - cnt.touched).min()),
                                   "max": int((cnt.ee_count - cnt.touched).max()),
                                   "median_relative": float(((cnt.ee_count - cnt.touched) / cnt.touched).median())},
        "rules_tested": {c: {"equal_to_ee_count": int((cnt[c] == cnt.ee_count).sum()),
                             "rule_minus_ee_min": int((cnt[c] - cnt.ee_count).min()),
                             "rule_minus_ee_max": int((cnt[c] - cnt.ee_count).max())} for c in rules},
    }
    count_rep["any_rule_reproduces_ee_count"] = any(v["equal_to_ee_count"] == len(cnt)
                                                   for v in count_rep["rules_tested"].values())

    # --- stdDev / mean / quantile semantics from Earth Engine's own histograms
    sd_rows, q_rows = [], []
    for g, e in ref.iterrows():
        uh = np.array(json.loads(e.ee_hist_unweighted), float)
        wh = np.array(json.loads(e.ee_hist_weighted), float)
        umu = np.average(uh[:, 0], weights=uh[:, 1])
        u_sd = np.sqrt(np.average((uh[:, 0] - umu) ** 2, weights=uh[:, 1])) / 10
        wmu = np.average(wh[:, 0], weights=wh[:, 1])
        w_sd = np.sqrt(np.average((wh[:, 0] - wmu) ** 2, weights=wh[:, 1])) / 10
        n = uh[:, 1].sum()
        sd_rows.append({"GEOID": g, "ee_sd": e.ee_stdDev, "u_pop": u_sd, "u_sample": u_sd * np.sqrt(n / (n - 1)),
                        "w_pop": w_sd, "ee_mean": e.ee_mean, "w_hist_mean": wmu / 10,
                        "nersc_sd": stats.loc[g, "lai_native_stdDev"]})
        for p, ee_col, col in ((0.25, "ee_p25", "lai_native_p25"), (0.5, "ee_p50", "lai_native_median"),
                               (0.75, "ee_p75", "lai_native_p75")):
            nersc = stats.loc[g, col]
            if abs(e[ee_col] - nersc) >= 1e-9:
                q_rows.append({"GEOID": g, "stratum": e.stratum, "quantile": p, "nersc": nersc, "ee": e[ee_col],
                               "ee_on_0p1_grid": bool(abs(e[ee_col] * 10 - round(e[ee_col] * 10)) < 1e-9),
                               "first_value_on_ee_weighted_hist": first_value(wh[:, 0], wh[:, 1], p) / 10})
    sd = pd.DataFrame(sd_rows)
    q = pd.DataFrame(q_rows)
    sd_rep = {
        "ee_stdDev_vs_unweighted_population_sd_of_ee_histogram_max_abs": float((sd.ee_sd - sd.u_pop).abs().max()),
        "ee_stdDev_vs_unweighted_sample_sd_max_abs": float((sd.ee_sd - sd.u_sample).abs().max()),
        "ee_stdDev_vs_weighted_population_sd_max_abs": float((sd.ee_sd - sd.w_pop).abs().max()),
        "ee_mean_vs_weighted_histogram_mean_max_abs": float((sd.ee_mean - sd.w_hist_mean).abs().max()),
        "nersc_weighted_sd_minus_ee_sd": {"median": float((sd.nersc_sd - sd.ee_sd).median()),
                                          "min": float((sd.nersc_sd - sd.ee_sd).min()),
                                          "max": float((sd.nersc_sd - sd.ee_sd).max())},
    }
    sd_rep["conclusion"] = ("Earth Engine stdDev = unweighted population SD of the cells it includes"
                            if sd_rep["ee_stdDev_vs_unweighted_population_sd_of_ee_histogram_max_abs"] < 1e-9
                            else "Earth Engine stdDev semantics not identified")
    q_rep = {"differing_quantile_values": len(q),
             "ee_value_off_0p1_grid": int((~q.ee_on_0p1_grid).sum()) if len(q) else 0,
             "first_value_rule_on_ee_histogram_equals_nersc": int((np.abs(q.first_value_on_ee_weighted_hist - q.nersc) < 1e-9).sum()) if len(q) else 0,
             "cases": q.to_dict("records")}

    # --- area: Earth Engine export geometry_area_m2 vs sphere and WGS84 ellipsoid
    files = sorted(Path(a.ee_exports).glob("*.csv"))
    ee = pd.concat([pd.read_csv(f, usecols=["GEOID", "geometry_area_m2"], dtype={"GEOID": str}) for f in files])
    m = tr[["GEOID", "geometry", "geometry_area_m2"]].merge(ee, on="GEOID", suffixes=("", "_ee"))
    m = m.sample(min(a.area_sample, len(m)), random_state=SEED)
    sphere = pyproj.Geod(a=6378137.0, b=6378137.0)
    sph = np.array([abs(sphere.geometry_area_perimeter(g)[0]) for g in m.geometry])
    rel_s = (sph - m.geometry_area_m2_ee) / m.geometry_area_m2_ee
    rel_e = (m.geometry_area_m2 - m.geometry_area_m2_ee) / m.geometry_area_m2_ee
    area_rep = {"n_tracts": len(m),
                "sphere_R6378137_vs_ee_abs_relative": {"median": float(rel_s.abs().median()), "max": float(rel_s.abs().max())},
                "wgs84_ellipsoid_vs_ee_abs_relative": {"median": float(rel_e.abs().median()), "max": float(rel_e.abs().max())}}

    out = {"count": count_rep, "stdDev_and_mean": sd_rep, "quantiles": q_rep, "area": area_rep}
    write_json(qdir / "ee_semantics.json", out)
    cnt.to_csv(qdir / "ee_count_rules.csv", index=False)
    print(json.dumps({k: (v if k != "quantiles" else {kk: vv for kk, vv in v.items() if kk != "cases"})
                      for k, v in out.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
