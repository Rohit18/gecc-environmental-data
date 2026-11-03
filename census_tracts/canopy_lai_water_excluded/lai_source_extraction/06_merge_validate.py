#!/usr/bin/env python3
"""Merge the shard tables, validate them, compare with Earth Engine, write the deliverables.

Refuses to merge unless raster and tract validation passed and every shard is
complete (content hashes match). Then:
  * completeness: every selected CONUS GEOID exactly once, nothing else;
  * internal consistency of the statistics (null policy, ordering, 0.1 grid, ranges,
    agreement with exactextract's own mean/stdev/min/max);
  * count definitions: canonical touched-valid-pixel count vs coverage-weighted count;
  * independent recomputation sample (04_independent_check.py);
  * the supplied Earth Engine CSV exports;
  * the Earth Engine reference sample, if 05_ee_reference_sample.py was run.

Writes to --out-dir:
  CONUS_CensusTract_LAI_2020.parquet / .csv, validation.json, tracts_without_data.csv,
  README.md, SHA256SUMS, and qa/ (per-tract QA table and comparison tables).

Example:
    python 06_merge_validate.py --manifest-dir $WORK/manifest --shard-dir $WORK/shards --qa-dir $WORK/qa \
        --ee-exports $EE_EXPORTS --tile-list $LAI_TILE_LIST --tiles-dir $LAI_TILES_DIR --out-dir $OUT_DIR
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from lai_common import (CONSTANT_COLUMNS, COUNT_DEFINITION, EE_SOURCE, ORIGIN_X, ORIGIN_Y, OUTPUT_COLUMNS, RES,
                        SCRIPT_VERSION, STAT_COLUMNS, sha256_file, shard_input_hash, shard_is_complete, shard_paths,
                        write_json)

OUT_STEM = "CONUS_CensusTract_LAI_2020"
EXPORT_RE = re.compile(r"_b(\d+)of(\d+)\.csv$")
VALUE_COLS = ["lai_native_mean", "lai_native_median", "lai_native_stdDev", "lai_native_min", "lai_native_max",
              "lai_native_p25", "lai_native_p75"]
EXACT_COLS = ["lai_native_min", "lai_native_max", "lai_native_median", "lai_native_p25", "lai_native_p75"]
TOL = 1e-9
# Mean/SD tolerance for the independent recomputation. Its coverage fractions come from GEOS
# polygon-cell intersection areas at coordinates near -100 deg with 9e-5 deg cells, which carry
# relative round-off of ~1e-10 per cell (observed: weighted counts differ by <= 6.8e-10 relative);
# that propagates to ~1e-9 LAI in mean/SD. 1e-6 LAI is 5 orders below the 0.1 storage step.
# Counts, min, max and quantiles are still required to match exactly.
IND_MEAN_SD_TOL = 1e-6


def software_versions() -> dict:
    from osgeo import gdal
    import pyproj
    v = {m: version(m) for m in ["exactextract", "geopandas", "rasterio", "shapely", "pyproj", "pandas",
                                 "numpy", "pyarrow", "pyogrio"]}
    v.update(python=platform.python_version(), gdal=gdal.__version__, proj=pyproj.proj_version_str,
             env_prefix=os.environ.get("CONDA_PREFIX"))
    return v


def size_class(area_m2: pd.Series) -> pd.Series:
    bins = [0, 1e5, 1e6, 1e7, 1e8, np.inf]
    labels = ["<0.1 km2", "0.1-1 km2", "1-10 km2", "10-100 km2", ">=100 km2"]
    return pd.cut(area_m2, bins=bins, labels=labels, right=False).astype(str)


def water_share(df: pd.DataFrame) -> pd.Series:
    tot = df.Area_Land + df.Area_Water
    return (df.Area_Water / tot.where(tot > 0)).fillna(0.0)


def describe(x: pd.Series) -> dict:
    x = pd.Series(x, dtype="float64").dropna()
    if x.empty:
        return {"n": 0}
    q = x.quantile([0.5, 0.9, 0.99])
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(q[0.5]), "p90": float(q[0.9]),
            "p99": float(q[0.99]), "max": float(x.max()), "min": float(x.min())}


# ---------------------------------------------------------------- load + build

def load_shards(mdir: Path, sdir: Path, rv: dict) -> tuple[pd.DataFrame, dict]:
    shards = pd.read_csv(mdir / "shard_manifest.csv", dtype={"first_geoid": str, "last_geoid": str})
    incomplete, runs = [], []
    for s in shards.to_dict("records"):
        h = shard_input_hash(s, rv["tile_manifest_sha256"], rv["vrt"]["sha256"])
        if not shard_is_complete(sdir, s, h):
            incomplete.append(int(s["shard_id"]))
            continue
        runs.append(json.loads(shard_paths(sdir, int(s["shard_id"]))[1].read_text()))
    info = {"n_shards": len(shards), "incomplete_shards": incomplete}
    if incomplete:
        return pd.DataFrame(), info
    df = pd.concat([pd.read_parquet(shard_paths(sdir, int(i))[0]) for i in shards.shard_id], ignore_index=True)
    secs = [r["seconds"] for r in runs]
    info.update({
        "slurm_job_ids": sorted({str(r["slurm_job_id"]) for r in runs if r.get("slurm_job_id")}),
        "shards_outside_slurm": sorted(int(r["shard_id"]) for r in runs if not r.get("slurm_job_id")),
        "hosts": sorted({r.get("host") for r in runs}),
        "shard_seconds_min_median_max": [min(secs), float(np.median(secs)), max(secs)],
        "shard_seconds_total": round(sum(secs), 1),
        "finished_last": max(r["finished"] for r in runs),
    })
    return df, info


def build_output(tracts: pd.DataFrame, stats: pd.DataFrame, tile_sha: str, ee_version: str) -> pd.DataFrame:
    df = tracts.merge(stats[["GEOID"] + STAT_COLUMNS], on="GEOID", how="left", validate="one_to_one")
    n = len(df)
    for k, v in CONSTANT_COLUMNS.items():
        df[k] = v
    for k in ("batch", "n_batches", "batch_seed"):   # Earth Engine batching does not exist here
        df[k] = pd.array([pd.NA] * n, dtype="Int64")
    df["exactextract_version"] = ee_version
    df["tile_manifest_sha256"] = tile_sha
    df["lai_native_count"] = df["lai_native_count"].astype("int64")
    for c in VALUE_COLS:
        df[c] = df[c].astype("float64")
    df["Pop"] = df["Pop"].astype("Int64")
    for c in ("boundary_year", "lai_year", "lai_input_divisor"):
        df[c] = df[c].astype("int64")
    for c in ("GEOID", "StFIPS", "CoFIPS", "TrFIPS"):
        df[c] = df[c].astype(str)
    return df[OUTPUT_COLUMNS].sort_values("GEOID").reset_index(drop=True)


# ---------------------------------------------------------------- checks

def check_stats(out: pd.DataFrame, qa: pd.DataFrame, rv: dict) -> dict:
    has = out.lai_native_count > 0
    d = out[has]
    raw_ok = lambda s: bool((np.abs(s * 10 - np.round(s * 10)) < 1e-9).all())  # noqa: E731
    lo, hi = rv["raw_value_min"] / 10, rv["raw_value_max"] / 10
    q = qa.set_index("GEOID").loc[out.GEOID[has]]
    c = {
        "tracts_with_data": int(has.sum()),
        "tracts_without_data": int((~has).sum()),
        "null_policy_violations_no_data": int(out.loc[~has, VALUE_COLS].notna().any(axis=1).sum()),
        "null_policy_violations_with_data": int(d[VALUE_COLS].isna().any(axis=1).sum()),
        "negative_counts": int((out.lai_native_count < 0).sum()),
        "order_violations_min_p25_median_p75_max": int((
            (d.lai_native_min > d.lai_native_p25 + TOL) | (d.lai_native_p25 > d.lai_native_median + TOL)
            | (d.lai_native_median > d.lai_native_p75 + TOL) | (d.lai_native_p75 > d.lai_native_max + TOL)).sum()),
        "mean_outside_min_max": int(((d.lai_native_mean < d.lai_native_min - TOL)
                                     | (d.lai_native_mean > d.lai_native_max + TOL)).sum()),
        "negative_stdDev": int((d.lai_native_stdDev < -TOL).sum()),
        "stdDev_above_half_range": int((d.lai_native_stdDev > (d.lai_native_max - d.lai_native_min) / 2 + TOL).sum()),
        "exact_fields_on_0p1_grid": all(raw_ok(d[col]) for col in EXACT_COLS),
        "values_outside_raster_range": int(((d.lai_native_min < lo - TOL) | (d.lai_native_max > hi + TOL)).sum()),
        "lai_native_range": [float(d.lai_native_min.min()), float(d.lai_native_max.max())],
        "lai_native_mean_summary": describe(d.lai_native_mean),
        "weighted_count_exceeds_touched": int((q.qa_weighted_count.values > d.lai_native_count.values + 1e-6).sum()),
        "exactextract_mean_max_abs_diff": float(np.nanmax(np.abs(q.qa_exactextract_mean.values - d.lai_native_mean.values))),
        "exactextract_stdev_max_abs_diff": float(np.nanmax(np.abs(q.qa_exactextract_stdev.values - d.lai_native_stdDev.values))),
        "exactextract_min_mismatch": int((np.abs(q.qa_exactextract_min.values - d.lai_native_min.values) > TOL).sum()),
        "exactextract_max_mismatch": int((np.abs(q.qa_exactextract_max.values - d.lai_native_max.values) > TOL).sum()),
        "geometry_area_nonpositive": int((out.geometry_area_m2 <= 0).sum()),
    }
    c["passed"] = (c["null_policy_violations_no_data"] == 0 and c["null_policy_violations_with_data"] == 0
                   and c["negative_counts"] == 0 and c["order_violations_min_p25_median_p75_max"] == 0
                   and c["mean_outside_min_max"] == 0 and c["negative_stdDev"] == 0
                   and c["stdDev_above_half_range"] == 0 and c["exact_fields_on_0p1_grid"]
                   and c["values_outside_raster_range"] == 0 and c["weighted_count_exceeds_touched"] == 0
                   and c["exactextract_mean_max_abs_diff"] < 1e-9 and c["exactextract_stdev_max_abs_diff"] < 1e-9
                   and c["exactextract_min_mismatch"] == 0 and c["exactextract_max_mismatch"] == 0
                   and c["geometry_area_nonpositive"] == 0)
    return c


def count_definitions(out: pd.DataFrame, qa: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    t = out[["GEOID", "StFIPS", "Area_Land", "Area_Water", "geometry_area_m2", "lai_native_count"]].merge(
        qa[["GEOID", "qa_weighted_count"]], on="GEOID")
    t = t.rename(columns={"lai_native_count": "count_touched", "qa_weighted_count": "count_coverage_weighted"})
    t["difference"] = t.count_touched - t.count_coverage_weighted
    t["relative_difference"] = t.difference / t.count_touched.where(t.count_touched > 0)
    t["size_class"] = size_class(t.geometry_area_m2)
    t["coastal_water_ge_50pct"] = water_share(t) >= 0.5
    rep = {
        "canonical": {"column": "lai_native_count", "definition": COUNT_DEFINITION},
        "qa_only": {"definition": "sum of exactextract coverage fractions over valid native pixels "
                                  "(coverage-weighted count); kept in qa/tract_qa.parquet, not in the table"},
        "difference_touched_minus_weighted": describe(t.difference),
        "relative_difference": describe(t.relative_difference),
        "by_size_class": {k: {"n": int(len(g)), "median_relative_difference": float(g.relative_difference.median()),
                              "max_relative_difference": float(g.relative_difference.max())}
                          for k, g in t[t.count_touched > 0].groupby("size_class")},
        "coastal_water_ge_50pct": describe(t[t.coastal_water_ge_50pct].relative_difference),
        "largest_relative_differences": t.nlargest(10, "relative_difference")[
            ["GEOID", "geometry_area_m2", "count_touched", "count_coverage_weighted", "relative_difference"]
        ].to_dict("records"),
    }
    return rep, t


def compare_independent(out: pd.DataFrame, qa: pd.DataFrame, ind: pd.DataFrame | None) -> dict:
    if ind is None:
        return {"available": False}
    m = ind.merge(out[["GEOID"] + STAT_COLUMNS], on="GEOID").merge(qa[["GEOID", "qa_weighted_count"]], on="GEOID")
    has = m.lai_native_count > 0
    rep = {"available": True, "n_tracts": int(len(m)), "strata": m.stratum.value_counts().to_dict(),
           "count_touched_equal": int((m.ind_touched == m.lai_native_count).sum()),
           "count_touched_max_abs_diff": int((m.ind_touched - m.lai_native_count).abs().max()),
           "weighted_count_max_rel_diff": float(((m.ind_weighted_count - m.qa_weighted_count).abs()
                                                 / m.qa_weighted_count.where(m.qa_weighted_count > 0)).max()),
           "no_data_agreement": int(((m.ind_touched == 0) == (m.lai_native_count == 0)).sum())}
    for col in EXACT_COLS:
        k = col.replace("lai_native_", "ind_")
        rep[f"{col}_equal"] = int((np.abs(m.loc[has, k] - m.loc[has, col]) < TOL).sum())
    for col in ("lai_native_mean", "lai_native_stdDev"):
        k = col.replace("lai_native_", "ind_")
        rep[f"{col}_max_abs_diff"] = float(np.abs(m.loc[has, k] - m.loc[has, col]).max())
    rep["tracts_with_data"] = int(has.sum())
    mism = m[has & ((m.ind_touched != m.lai_native_count)
                    | np.any([np.abs(m[c.replace("lai_native_", "ind_")] - m[c]) >= TOL for c in EXACT_COLS], axis=0))]
    rep["mismatching_tracts"] = mism[["GEOID", "stratum", "geometry_area_m2", "ind_touched", "lai_native_count"]
                                     + EXACT_COLS].head(20).to_dict("records")
    # Alternative count definitions on the same sample.
    rep["alternative_counts_vs_touched"] = {
        "gdal_all_touched_equal": int((m.ind_all_touched == m.lai_native_count).sum()),
        "gdal_all_touched_minus_touched": describe(m.ind_all_touched - m.lai_native_count),
        "pixel_centre_equal": int((m.ind_centre == m.lai_native_count).sum()),
        "pixel_centre_minus_touched": describe(m.ind_centre - m.lai_native_count),
        "coverage_weighted_minus_touched": describe(m.ind_weighted_count - m.lai_native_count),
    }
    rep["nodata_cells_touched_in_sample"] = int(m.ind_nodata_cells_touched.sum())
    n = int(has.sum())
    rep["passed"] = (rep["count_touched_equal"] == len(m) and rep["no_data_agreement"] == len(m)
                     and all(rep[f"{c}_equal"] == n for c in EXACT_COLS)
                     and rep["lai_native_mean_max_abs_diff"] < IND_MEAN_SD_TOL
                     and rep["lai_native_stdDev_max_abs_diff"] < IND_MEAN_SD_TOL)
    rep["tolerances"] = {"count_min_max_quantiles": "exact", "mean_stdDev_lai": IND_MEAN_SD_TOL,
                         "reason": "GEOS intersection-area round-off in the independent coverage fractions"}
    return rep


def compare_ee_exports(out: pd.DataFrame, ee_dir: Path) -> tuple[dict, pd.DataFrame | None]:
    files = sorted(ee_dir.glob("*.csv"))
    if not files:
        return {"available": False, "reason": f"no CSV files in {ee_dir}"}, None
    nums = {int(m.group(1)): int(m.group(2)) for f in files if (m := EXPORT_RE.search(f.name))}
    n_batches = sorted(set(nums.values()))
    expected = set(range(n_batches[0])) if len(n_batches) == 1 else set()
    str_cols = {"GEOID": str, "StFIPS": str, "CoFIPS": str, "TrFIPS": str}
    ee = pd.concat([pd.read_csv(f, dtype=str_cols) for f in files], ignore_index=True)
    rep = {"available": True, "directory": str(ee_dir), "files": len(files), "n_batches_declared": n_batches,
           "batches_missing": sorted(expected - set(nums)), "rows": int(len(ee)),
           "duplicate_geoids": int(ee.GEOID.duplicated().sum()),
           "lai_source_values": sorted(ee.lai_source.dropna().unique().tolist()),
           "script_version_values": sorted(ee.script_version.dropna().astype(str).unique().tolist()),
           "analysis_scale_m_values": sorted(ee.analysis_scale_m.dropna().unique().tolist()),
           "analysis_crs_values": sorted(ee.analysis_crs.dropna().unique().tolist())}
    conus = ee[ee.GEOID.isin(out.GEOID)]
    rep["conus_rows_in_exports"] = int(len(conus))
    rep["conus_tracts_absent_from_exports"] = int((~out.GEOID.isin(ee.GEOID)).sum())
    rep["non_conus_rows_in_exports"] = int((~ee.GEOID.isin(out.GEOID)).sum())
    rep["non_null_values_in_exports"] = {c: int(conus[c].notna().sum()) for c in STAT_COLUMNS}
    m = out.merge(conus, on="GEOID", suffixes=("", "_ee"))
    rep["attributes"] = {
        "StFIPS_CoFIPS_TrFIPS_equal": int(((m.StFIPS == m.StFIPS_ee) & (m.CoFIPS == m.CoFIPS_ee)
                                           & (m.TrFIPS == m.TrFIPS_ee)).sum()),
        "Area_Land_equal": int(np.isclose(m.Area_Land, m.Area_Land_ee, rtol=1e-12, atol=0).sum()),
        "Area_Water_equal": int(np.isclose(m.Area_Water, m.Area_Water_ee, rtol=1e-12, atol=0).sum()),
        "Pop_equal": int((m.Pop.astype("float64") == m.Pop_ee.astype("float64")).sum()),
        "matched_rows": int(len(m)),
    }
    rel = (m.geometry_area_m2 - m.geometry_area_m2_ee) / m.geometry_area_m2_ee
    rep["geometry_area_m2_relative_difference_nersc_minus_ee"] = describe(rel)
    rep["geometry_area_m2_abs_relative_difference"] = describe(rel.abs())
    both = m[(m.lai_native_count > 0) & m.lai_native_mean_ee.notna()]
    rep["numeric_comparison_feasible"] = bool(len(both))
    if both.empty:
        rep["numeric_comparison"] = ("not feasible: every LAI statistic (mean, median, stdDev, count, min, max, "
                                     "p25, p75) is null in all supplied Earth Engine export rows")
    else:
        rep["numeric_comparison"] = {f"{c}_equal": int((np.abs(both[c] - both[c + "_ee"]) < TOL).sum())
                                     for c in EXACT_COLS}
        rep["numeric_comparison"]["mean_abs_diff"] = describe((both.lai_native_mean - both.lai_native_mean_ee).abs())
        rep["numeric_comparison"]["count_diff"] = describe(both.lai_native_count - both.lai_native_count_ee)
    table = m[["GEOID", "geometry_area_m2", "geometry_area_m2_ee", "Area_Land", "Area_Land_ee", "Pop", "Pop_ee",
               "lai_native_mean_ee", "lai_native_count_ee", "batch_ee"]].copy()
    table["geometry_area_rel_diff"] = rel.values
    return rep, table


def compare_ee_reference(out: pd.DataFrame, qa: pd.DataFrame, ind: pd.DataFrame | None,
                         ref: pd.DataFrame | None) -> tuple[dict, pd.DataFrame | None]:
    if ref is None:
        return {"available": False, "reason": "05_ee_reference_sample.py not run"}, None
    m = ref.merge(out[["GEOID"] + STAT_COLUMNS], on="GEOID").merge(
        qa[["GEOID", "qa_weighted_count", "qa_hist_values", "qa_hist_frac"]], on="GEOID")
    if ind is not None:
        m = m.merge(ind[["GEOID", "ind_all_touched", "ind_centre", "ind_touched"]], on="GEOID", how="left")
    has = m.lai_native_count > 0

    def hist_diff(r):
        ee_h = dict(json.loads(r.ee_hist_weighted))
        tot = sum(ee_h.values())
        ours = dict(zip(np.asarray(r.qa_hist_values, int).tolist(), np.asarray(r.qa_hist_frac, float).tolist()))
        keys = set(ee_h) | set(ours)
        if not tot:
            return np.nan, 0.0
        return max(abs(ee_h.get(k, 0) / tot - ours.get(k, 0.0)) for k in keys), tot

    hd = m.apply(hist_diff, axis=1, result_type="expand")
    m["hist_max_abs_share_diff"], m["ee_hist_total_weight"] = hd[0], hd[1]
    rep = {"available": True, "n_tracts": int(len(m)), "strata": m.stratum.value_counts().to_dict(),
           "ee_asset": EE_SOURCE, "ee_grid": "crsTransform = native grid (no scale / 10 m target grid)",
           "tracts_with_data_nersc": int(has.sum()), "tracts_with_data_ee": int(m.ee_mean.notna().sum())}
    for c_nersc, c_ee in (("lai_native_min", "ee_min"), ("lai_native_max", "ee_max"),
                          ("lai_native_median", "ee_p50"), ("lai_native_p25", "ee_p25"),
                          ("lai_native_p75", "ee_p75")):
        rep[f"{c_nersc}_equal"] = int((np.abs(m.loc[has, c_nersc] - m.loc[has, c_ee]) < TOL).sum())
    rep["mean_abs_diff"] = describe((m.loc[has, "lai_native_mean"] - m.loc[has, "ee_mean"]).abs())
    rep["stdDev_abs_diff"] = describe((m.loc[has, "lai_native_stdDev"] - m.loc[has, "ee_stdDev"]).abs())
    rep["weighted_histogram_max_abs_share_diff"] = describe(m.loc[has, "hist_max_abs_share_diff"])
    rep["ee_weighted_count_vs_nersc_coverage_weighted"] = describe(m.ee_count - m.qa_weighted_count)
    rep["ee_histogram_total_weight_vs_nersc_coverage_weighted"] = describe(m.ee_hist_total_weight - m.qa_weighted_count)
    rep["ee_unweighted_count_equal_to"] = {"touched (canonical)": int((m.ee_u_count == m.lai_native_count).sum())}
    rep["ee_weighted_count_equal_to"] = {"touched (canonical)": int((m.ee_count == m.lai_native_count).sum())}
    if ind is not None:
        for name, col in (("gdal_all_touched", "ind_all_touched"), ("pixel_centre", "ind_centre")):
            rep["ee_unweighted_count_equal_to"][name] = int((m.ee_u_count == m[col]).sum())
            rep["ee_weighted_count_equal_to"][name] = int((m.ee_count == m[col]).sum())
    rep["ee_unweighted_count_minus_touched"] = describe(m.ee_u_count - m.lai_native_count)
    rep["ee_weighted_count_minus_touched"] = describe(m.ee_count - m.lai_native_count)
    by = {}
    for k, g in m[has].groupby("stratum"):
        by[k] = {"n": int(len(g)),
                 "mean_abs_diff_max": float((g.lai_native_mean - g.ee_mean).abs().max()),
                 "stdDev_abs_diff_max": float((g.lai_native_stdDev - g.ee_stdDev).abs().max()),
                 "ee_unweighted_count_minus_touched_median": float((g.ee_u_count - g.lai_native_count).median()),
                 "median_equal": int((np.abs(g.lai_native_median - g.ee_p50) < TOL).sum())}
    rep["by_stratum"] = by
    keep = ["GEOID", "stratum", "geometry_area_m2", "lai_native_count", "qa_weighted_count", "ee_count", "ee_u_count",
            "lai_native_mean", "ee_mean", "lai_native_stdDev", "ee_stdDev", "lai_native_min", "ee_min",
            "lai_native_max", "ee_max", "lai_native_p25", "ee_p25", "lai_native_median", "ee_p50",
            "lai_native_p75", "ee_p75", "hist_max_abs_share_diff", "ee_hist_total_weight"]
    keep += [c for c in ("ind_all_touched", "ind_centre") if c in m.columns]
    return rep, m[keep]


# ---------------------------------------------------------------- README

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if x != 0 and abs(x) < 1e-3:
        return f"{x:.2g}"
    return f"{x:,.{nd}f}"


def write_readme(path: Path, v: dict, out: pd.DataFrame) -> None:
    rv, tv, st, cd = v["raster"], v["tracts"], v["statistics"], v["count_definitions"]
    ind, eex, eer, sw = v["independent_check"], v["ee_exports"], v["ee_reference_sample"], v["software"]
    src = v["sources"]
    lines = [
        "# CONUS 2020 Census Tract Leaf Area Index (LAI)",
        "",
        f"Generated {v['generated']} by `06_merge_validate.py` ({SCRIPT_VERSION}) on NERSC Perlmutter. "
        f"Validation: **{'passed' if v['passed'] else 'FAILED'}**" + (f" — failures: {v['failures']}" if v["failures"] else "") + ".",
        "",
        f"One row per 2020 Census Tract in the 48 contiguous states plus DC: **{len(out):,} tracts**, "
        f"{st['tracts_with_data']:,} with LAI data and {st['tracts_without_data']:,} without "
        "(listed in `tracts_without_data.csv`). No geometry column.",
        "",
        "## Files",
        "",
        "| File | Content |",
        "|---|---|",
        f"| `{OUT_STEM}.parquet` | the table (typed; GEOID/StFIPS/CoFIPS/TrFIPS as text) |",
        f"| `{OUT_STEM}.csv` | the same table as CSV; empty cells are nulls |",
        "| `validation.json` | every check and comparison behind this README |",
        "| `tracts_without_data.csv` | tracts with `lai_native_count = 0` (all statistics null) |",
        "| `qa/` | per-tract QA values, count-definition table, comparison tables |",
        "| `SHA256SUMS` | checksums of the files above |",
        "",
        "## Source raster",
        "",
        f"- `lai_source` = `{EE_SOURCE}` (band `LAI_uint8`, stored value = LAI × 10). Its asset note reads "
        "\"Stored as uint8; cast from original LAI×10 without rescaling.\"",
        "- The local raster was downloaded from Earth Engine as native-grid tiles of "
        "`projects/your-ee-project/assets/LAI_Global_Mosaic_2020`, the int32 original that asset was cast from, "
        "on the same grid (4,007,502 × 2,003,752 cells, identical affine transform). Masked cells were written as -1, "
        "the declared nodata of every tile. "
        "Three 2048 × 2048 tiles fetched from the uint8 asset matched the downloaded tiles cell for cell "
        "(identical mask, identical values where valid).",
        f"- Local raster: `{rv['vrt']['path']}` (sha256 `{rv['vrt']['sha256']}`), a VRT over "
        f"{rv['tiles_listed']:,} GeoTIFF tiles in `{src['tiles_dir']}` ({rv['tile_bytes'] / 1e9:.1f} GB).",
        f"- Tile manifest: `{rv['tile_manifest']}`, sha256 `{rv['tile_manifest_sha256']}` "
        "(per-tile path, size, sha256, grid position); this hash is the `tile_manifest_sha256` column.",
        f"- Storage type of the tiles: {', '.join(rv['tile_dtypes'])}; declared nodata {rv['tile_declared_nodata']}. "
        f"Every valid value lies in {rv['raw_value_min']}–{rv['raw_value_max']} (LAI {rv['raw_value_min'] / 10}–"
        f"{rv['raw_value_max'] / 10}); values below 0 or above 254: {rv['valid_values_below_0'] + rv['valid_values_above_254']}; "
        f"255 as a valid value: {rv['valid_value_255_count']}. The raw integers are therefore exactly the uint8 values.",
        f"- Valid cells: {rv['valid_pixels']:,}; masked cells: {rv['nodata_pixels']:,}; raw 0 cells "
        f"(valid, kept): {rv['raw_zero_pixels']:,}.",
        "",
        "## Grid and CRS",
        "",
        f"- The raster is used only on its own grid: EPSG:4326, cell size {RES!r}°, origin "
        f"({ORIGIN_X!r}, {ORIGIN_Y!r}). It is never resampled, aggregated, interpolated or reprojected, and no "
        "generic 10 m target grid is used anywhere. The VRT's cell size and origin were checked against this grid, "
        "and every tile's origin against its grid position.",
        f"- Tract polygons were reprojected to EPSG:4326 for extraction ({tv['reprojection']['transformer']}; "
        f"largest coordinate change {tv['reprojection']['max_abs_coordinate_shift_deg']}°).",
        "- `geometry_area_m2` is the geodesic polygon area on the WGS84 ellipsoid (pyproj `Geod.geometry_area_perimeter`).",
        "- `analysis_scale_m` = 10 records the nominal scale of the native grid (the cell size is 10.000 m of arc at "
        "the equator) for Earth Engine compatibility; it was not used to build any grid.",
        "",
        "## Tracts",
        "",
        f"- Source `{tv['source']}` ({tv['features_in_source']:,} features). CONUS = StFIPS not in 02, 15, 60, 66, 69, "
        f"72, 78 → {tv['conus_tracts']:,} tracts in {tv['conus_states']} state codes (48 states + DC). No clipping to any other boundary.",
        f"- GEOID is 11-character text with leading zeros, unique, and equals StFIPS + CoFIPS + TrFIPS for every tract. "
        f"Invalid geometries: {tv['invalid_geometries']}; empty: {tv['empty_geometries']}.",
        f"- `Pop` is present in the source boundary file and copied unchanged (nulls: {tv['pop_nulls']}).",
        f"- Every tile a CONUS tract intersects is present ({tv['tiles_needed_by_tracts']:,} needed, "
        f"{tv['tiles_needed_but_absent']} absent).",
        "",
        "## Mask and missing data",
        "",
        "- Only the raster's own validity mask (its declared nodata, -1 = masked in Earth Engine) is applied. "
        "No water, built-up, urban/rural, vegetation, positive-LAI, cloud, canopy-height or other mask.",
        "- Raw value 0 is valid and is included in every statistic and count.",
        f"- A tract with no valid cell gets `lai_native_count = 0` and null for every other statistic "
        f"({st['tracts_without_data']:,} tracts in this run" + ("; `tracts_without_data.csv` therefore has only its header" if st["tracts_without_data"] == 0 else "")
        + "). Nulls are never replaced by zero.",
        "",
        "## Statistics",
        "",
        "For each tract, exactextract returns the distinct valid raw values of the cells the polygon covers and "
        "each value's coverage-weighted share (`unique` + `frac`), i.e. a compact histogram over at most 256 values. "
        "With w = coverage fraction of a cell and x = its raw value:",
        "",
        "- `lai_native_mean` = Σ w·x / Σ w / 10",
        "- `lai_native_stdDev` = √(E(x²) − E(x)²) / 10, population form, weights w (computed as Σ w(x − x̄)² / Σ w)",
        "- `lai_native_min` / `lai_native_max` = lowest / highest raw value with positive weight, / 10",
        "- `lai_native_p25` / `lai_native_median` / `lai_native_p75` = the first raw value whose cumulative weight "
        "≥ p × Σ w, / 10; no interpolation (a relative tolerance of 1e-12 absorbs floating-point round-off at exact ties)",
        "- `lai_native_count`: see below",
        "",
        "## Count definitions",
        "",
        f"- **`lai_native_count` (canonical)**: {COUNT_DEFINITION}. It is an integer.",
        "- **Coverage-weighted count (QA only)**: Σ w over valid cells — exactextract's default `count`. Kept in "
        "`qa/tract_qa.parquet` and `qa/count_definitions.parquet`; it is not substituted for `lai_native_count`.",
        f"- Touched minus weighted, all tracts: median {fmt(cd['difference_touched_minus_weighted'].get('median'), 1)} cells, "
        f"relative median {fmt(cd['relative_difference'].get('median'), 4)}, max {fmt(cd['relative_difference'].get('max'), 4)}. "
        "The gap is the partial boundary cells, so it is largest for the smallest tracts:",
        "",
        "| Size class | Tracts | Median (touched − weighted) / touched | Max |",
        "|---|---:|---:|---:|",
    ]
    for k in ["<0.1 km2", "0.1-1 km2", "1-10 km2", "10-100 km2", ">=100 km2"]:
        if k in cd["by_size_class"]:
            b = cd["by_size_class"][k]
            lines.append(f"| {k} | {b['n']:,} | {fmt(b['median_relative_difference'], 4)} | {fmt(b['max_relative_difference'], 4)} |")
    lines += [
        f"| coastal (water ≥ 50%) | {cd['coastal_water_ge_50pct'].get('n', 0):,} | "
        f"{fmt(cd['coastal_water_ge_50pct'].get('median'), 4)} | {fmt(cd['coastal_water_ge_50pct'].get('max'), 4)} |",
        "",
    ]
    if eer.get("available"):
        sem = eer.get("semantics") or {}
        cr = sem.get("count", {})
        d = cr.get("ee_count_minus_touched", {})
        best = max(cr.get("rules_tested", {}).items(), key=lambda kv: kv[1]["equal_to_ee_count"], default=(None, None))
        lines += [
            f"Earth Engine's own `count` on the {eer['n_tracts']} reference tracts (native crsTransform) is the same "
            "with and without weights, and is always a little below the touched count: Earth Engine minus touched = "
            f"median {d.get('median')} cells (relative {fmt(d.get('median_relative'), 4)}), range {d.get('min')} to {d.get('max')}. "
            "Earth Engine drops some thin boundary slivers. No rule tested reproduces its count exactly "
            f"(exact-coverage thresholds 0, 0.5/255, 1/255, 0.5/256, 1/256; k×k sub-cell point sampling, k = 4, 8, 16, 32; "
            f"best: {best[0]}, {best[1]['equal_to_ee_count'] if best[1] else 'n/a'}/{cr.get('n_tracts')} tracts), "
            "so an Earth Engine-identical count cannot be produced outside Earth Engine. `lai_native_count` follows the "
            f"specified definition (valid cells touched, coverage > 0); on the sample it exceeds Earth Engine's count by a "
            f"median {fmt(-d.get('median_relative', float('nan')) * 100, 2)}% ({-d.get('median', 0)} cells). Details: `validation.json` → "
            "`ee_reference_sample.semantics.count`, per tract in `qa/ee_count_rules.csv`.",
            "",
        ]
    lines += [
        "## Output columns",
        "",
        "The 31 columns of the Earth Engine export, in its order, then four NERSC provenance columns:",
        "",
        "`" + "`, `".join(OUTPUT_COLUMNS) + "`",
        "",
        "- `boundary_year` = `lai_year` = 2020; `analysis_crs` = EPSG:4326; `lai_units` = m2_leaf_per_m2_ground; "
        "`lai_input_band` = LAI_uint8; `lai_input_divisor` = 10; `lai_storage_units` = LAI_times_10_uint8.",
        "- `batch`, `n_batches`, `batch_seed` are Earth Engine batching fields and are null here; no batch "
        "assignment was invented.",
        f"- `script_version` = {SCRIPT_VERSION}; `processing_platform` = {CONSTANT_COLUMNS['processing_platform']}; "
        f"`exactextract_version` = {sw['exactextract']}.",
        "",
        "## Quality checks",
        "",
        f"- Raster validation: {'passed' if rv['passed'] else 'FAILED'} (CRS, cell size, grid alignment of every tile, "
        "integer type, declared nodata, value range, 255 absent, VRT sources = manifest).",
        f"- Tract validation: {'passed' if tv['passed'] else 'FAILED'}; completeness: every selected GEOID exactly once "
        f"({v['completeness']['missing_geoids']} missing, {v['completeness']['unexpected_geoids']} unexpected, "
        f"{v['completeness']['duplicate_geoids']} duplicated).",
        f"- Statistic consistency: {'passed' if st['passed'] else 'FAILED'} (null policy, min ≤ p25 ≤ median ≤ p75 ≤ max, "
        "mean within [min, max], SD ≤ half-range, quantiles on the 0.1 grid, exactextract's own mean/stdev/min/max "
        f"agree to {fmt(max(st['exactextract_mean_max_abs_diff'], st['exactextract_stdev_max_abs_diff']), 12)}).",
    ]
    if ind.get("available"):
        n = ind["tracts_with_data"]
        lines += [
            f"- Independent recomputation on {ind['n_tracts']} tracts ({', '.join(f'{k} {c}' for k, c in ind['strata'].items())}), "
            "using GEOS cell-intersection areas and a sort-based weighted percentile instead of exactextract and the "
            f"histogram: touched count equal in {ind['count_touched_equal']}/{ind['n_tracts']}; min/max/p25/median/p75 "
            f"equal in {min(ind[c + '_equal'] for c in EXACT_COLS)}/{n}; mean and SD within "
            f"{fmt(max(ind['lai_native_mean_max_abs_diff'], ind['lai_native_stdDev_max_abs_diff']), 12)} LAI "
            f"(required: exact for counts and quantiles, {ind['tolerances']['mean_stdDev_lai']:g} LAI for mean/SD, "
            "which covers round-off in the GEOS intersection areas). "
            f"{'Passed' if ind['passed'] else 'NOT passed — see validation.json'}.",
        ]
    lines += ["", "## Comparison with the Earth Engine exports", ""]
    if eex.get("available"):
        a = eex["attributes"]
        lines += [
            f"- Supplied: {eex['files']} CSV files in `{eex['directory']}`, declared batches {eex['n_batches_declared']}, "
            f"missing batch(es) {eex['batches_missing']}; {eex['conus_rows_in_exports']:,} CONUS rows; "
            f"{eex['conus_tracts_absent_from_exports']:,} CONUS tracts are not in the supplied files.",
            f"- **Numerical comparison: {eex['numeric_comparison'] if isinstance(eex['numeric_comparison'], str) else 'see validation.json'}.** "
            "Exact agreement of min, max, median, P25 and P75 therefore could not be tested against these files; "
            "the Earth Engine reference sample below is the substitute.",
            f"- Attributes on the {a['matched_rows']:,} matched tracts: StFIPS/CoFIPS/TrFIPS equal {a['StFIPS_CoFIPS_TrFIPS_equal']:,}, "
            f"Area_Land equal {a['Area_Land_equal']:,}, Area_Water equal {a['Area_Water_equal']:,}, Pop equal {a['Pop_equal']:,}.",
            f"- `geometry_area_m2` (WGS84 geodesic here vs Earth Engine): median relative difference (NERSC − EE)/EE "
            f"{fmt(eex['geometry_area_m2_relative_difference_nersc_minus_ee'].get('median'), 6)}, "
            f"99th percentile |difference| {fmt(eex['geometry_area_m2_abs_relative_difference'].get('p99'), 6)}, "
            f"max {fmt(eex['geometry_area_m2_abs_relative_difference'].get('max'), 6)}.",
        ]
    else:
        lines.append(f"- Not available: {eex.get('reason')}")
    if eer.get("available"):
        n = eer["tracts_with_data_nersc"]
        sem = eer.get("semantics") or {}
        sd, qs = sem.get("stdDev_and_mean", {}), sem.get("quantiles", {})
        lines += [
            "",
            f"Because the exports carry no values, {eer['n_tracts']} tracts ({', '.join(f'{k} {c}' for k, c in eer['strata'].items())}; "
            f"all < 20 km²) were recomputed in Earth Engine with `reduceRegions` on `{EE_SOURCE}` at the native "
            f"crsTransform ({fmt(eer['run']['eecu_s'], 1)} EECU-seconds on project `{eer['run']['project']}`):",
            "",
            f"- **min / max**: equal in {eer['lai_native_min_equal']}/{n} and {eer['lai_native_max_equal']}/{n}.",
            f"- **median / P25 / P75**: equal in {eer['lai_native_median_equal']}/{n}, {eer['lai_native_p25_equal']}/{n}, "
            f"{eer['lai_native_p75_equal']}/{n}. The {qs.get('differing_quantile_values')} differing values are all off the "
            f"0.1 grid ({qs.get('ee_value_off_0p1_grid')}/{qs.get('differing_quantile_values')}): Earth Engine's percentile "
            "reducer interpolates within histogram buckets. Applying this workflow's first-value rule to Earth Engine's own "
            f"weighted histogram gives the NERSC value in {qs.get('first_value_rule_on_ee_histogram_equals_nersc')}/"
            f"{qs.get('differing_quantile_values')} of them, so exact agreement is reached wherever Earth Engine does not interpolate.",
            f"- **mean**: |difference| median {fmt(eer['mean_abs_diff'].get('median'), 7)}, max {fmt(eer['mean_abs_diff'].get('max'), 7)} LAI. "
            "Earth Engine's mean equals the mean of its weighted histogram "
            f"(to {fmt(sd.get('ee_mean_vs_weighted_histogram_mean_max_abs'), 15)}), whose coverage weights are stored in steps of 1/255; "
            "the shares of each value differ from exactextract's exact coverage by at most "
            f"{fmt(eer['weighted_histogram_max_abs_share_diff'].get('max'), 6)}.",
            f"- **stdDev**: Earth Engine's `stdDev` is the *unweighted* population SD of the cells it includes (equal to the SD "
            f"of its unweighted histogram to {fmt(sd.get('ee_stdDev_vs_unweighted_population_sd_of_ee_histogram_max_abs'), 15)}), "
            "i.e. boundary cells count fully. `lai_native_stdDev` is the coverage-weighted population SD as specified. "
            f"NERSC minus Earth Engine: median {fmt(sd.get('nersc_weighted_sd_minus_ee_sd', {}).get('median'), 4)}, range "
            f"{fmt(sd.get('nersc_weighted_sd_minus_ee_sd', {}).get('min'), 4)} to {fmt(sd.get('nersc_weighted_sd_minus_ee_sd', {}).get('max'), 4)} LAI; "
            "the largest differences are in tiny tracts, where boundary cells are a large share.",
            "- **count**: see *Count definitions* above.",
            f"- Per-tract values: `qa/ee_reference_comparison.csv`; per-stratum summaries: `validation.json`.",
        ]
        ar = sem.get("area")
        if ar:
            lines.append(
                f"- `geometry_area_m2` in the Earth Engine export is neither a spherical area (R = 6,378,137 m: median |relative "
                f"difference| {fmt(ar['sphere_R6378137_vs_ee_abs_relative']['median'], 4)}) nor the WGS84 ellipsoidal area "
                f"used here (median {fmt(ar['wgs84_ellipsoid_vs_ee_abs_relative']['median'], 4)}, max "
                f"{fmt(ar['wgs84_ellipsoid_vs_ee_abs_relative']['max'], 4)}; {ar['n_tracts']:,}-tract sample). This workflow "
                "reports the WGS84 geodesic area as specified.")
    lines += [
        "",
        "## Software",
        "",
        f"Python {sw['python']}, exactextract {sw['exactextract']}, GDAL {sw['gdal']}, PROJ {sw['proj']}, "
        f"geopandas {sw['geopandas']}, rasterio {sw['rasterio']}, shapely {sw['shapely']}, pyproj {sw['pyproj']}, "
        f"pandas {sw['pandas']}, numpy {sw['numpy']}, pyarrow {sw['pyarrow']}, pyogrio {sw['pyogrio']}. "
        f"Environment `{sw['env_prefix']}`, specified by `environment.yml` and pinned in `environment.lock.txt` "
        "(code directory).",
        "",
        "## Processing",
        "",
        f"- Code: `{src['code_dir']}` (scripts 01–07, Slurm scripts, `config.sh`, tests).",
        f"- {v['shards']['n_shards']} shards (balanced tract groups in GEOID order); Slurm job(s) "
        f"{', '.join(v['shards'].get('slurm_job_ids', []))} on {', '.join(v['shards'].get('hosts', []))}; "
        f"shard time total {v['shards'].get('shard_seconds_total')} s, "
        f"per shard min/median/max {v['shards'].get('shard_seconds_min_median_max')} s"
        + (f"; shards computed outside Slurm: {v['shards']['shards_outside_slurm']}" if v['shards'].get('shards_outside_slurm') else "")
        + ".",
        "- Rerun: `bash submit_all.sh` (complete shards are skipped by content hash; delete a shard's "
        "`.done.json` to force it).",
        "",
    ]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--qa-dir", required=True)
    ap.add_argument("--ee-exports", required=True)
    ap.add_argument("--tile-list", required=True)
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import geopandas as gpd

    mdir, sdir, qdir, out_dir = map(Path, (a.manifest_dir, a.shard_dir, a.qa_dir, a.out_dir))
    (out_dir / "qa").mkdir(parents=True, exist_ok=True)
    rv = json.loads((mdir / "raster_validation.json").read_text())
    tv = json.loads((mdir / "tract_validation.json").read_text())
    failures = []
    if not rv.get("passed"):
        failures.append("raster validation did not pass")
    if not tv.get("passed"):
        failures.append("tract validation did not pass")

    stats, shard_info = load_shards(mdir, sdir, rv)
    if shard_info["incomplete_shards"]:
        sys.exit(f"cannot merge: incomplete shards {shard_info['incomplete_shards']}")

    tracts = gpd.read_parquet(mdir / "tracts_conus_4326.parquet").drop(columns="geometry")
    attrs = tracts[["GEOID", "StFIPS", "CoFIPS", "TrFIPS", "Area_Land", "Area_Water", "Pop", "geometry_area_m2"]]
    exp, got = set(tracts.GEOID), set(stats.GEOID)
    completeness = {"expected_tracts": len(exp), "shard_rows": int(len(stats)),
                    "duplicate_geoids": int(stats.GEOID.duplicated().sum()),
                    "missing_geoids": len(exp - got), "unexpected_geoids": len(got - exp)}
    completeness["passed"] = (completeness["duplicate_geoids"] == 0 and completeness["missing_geoids"] == 0
                              and completeness["unexpected_geoids"] == 0 and len(stats) == len(exp))
    if not completeness["passed"]:
        failures.append("tract completeness")

    sw = software_versions()
    out = build_output(attrs, stats, rv["tile_manifest_sha256"], sw["exactextract"])
    qa = stats[["GEOID", "shard", "qa_weighted_count", "qa_n_distinct_values", "qa_exactextract_mean",
                "qa_exactextract_stdev", "qa_exactextract_min", "qa_exactextract_max", "qa_hist_values", "qa_hist_frac"]]

    st = check_stats(out, qa, rv)
    if not st["passed"]:
        failures.append("statistic consistency")
    cd, cd_table = count_definitions(out, qa)

    ind_path = qdir / "independent_check.parquet"
    ind = pd.read_parquet(ind_path) if ind_path.exists() else None
    ind_rep = compare_independent(out, qa, ind)
    if ind_rep.get("available") and not ind_rep["passed"]:
        failures.append("independent recomputation disagrees")
    if not ind_rep.get("available"):
        failures.append("independent recomputation missing")

    eex, eex_table = compare_ee_exports(out, Path(a.ee_exports))
    ref_path = qdir / "ee_reference_sample.parquet"
    ref = pd.read_parquet(ref_path) if ref_path.exists() else None
    eer, eer_table = compare_ee_reference(out, qa, ind, ref)
    if ref is not None:
        eer["run"] = json.loads(ref_path.with_suffix(".json").read_text())
        sem_path = qdir / "ee_semantics.json"
        eer["semantics"] = json.loads(sem_path.read_text()) if sem_path.exists() else None

    v = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "script_version": SCRIPT_VERSION,
        "sources": {"tracts": tv["source"], "tracts_sha256": tv["source_sha256"], "vrt": rv["vrt"]["path"],
                    "vrt_sha256": rv["vrt"]["sha256"], "tiles_dir": a.tiles_dir, "tile_list": a.tile_list,
                    "tile_list_sha256": sha256_file(a.tile_list), "code_dir": str(Path(__file__).resolve().parent),
                    "ee_asset": EE_SOURCE},
        "raster": rv, "tracts": tv, "completeness": completeness, "shards": shard_info, "statistics": st,
        "count_definitions": cd, "independent_check": ind_rep, "ee_exports": eex, "ee_reference_sample": eer,
        "software": sw,
    }
    v["failures"] = failures
    v["passed"] = not failures

    # Deliverables (atomic writes).
    pq, csv = out_dir / f"{OUT_STEM}.parquet", out_dir / f"{OUT_STEM}.csv"
    out.to_parquet(pq.with_suffix(".parquet.tmp"), index=False)
    os.replace(pq.with_suffix(".parquet.tmp"), pq)
    out.to_csv(csv.with_suffix(".csv.tmp"), index=False)
    os.replace(csv.with_suffix(".csv.tmp"), csv)
    nod = out[out.lai_native_count == 0][["GEOID", "StFIPS", "CoFIPS", "TrFIPS", "Area_Land", "Area_Water", "Pop",
                                          "geometry_area_m2", "lai_native_count"]].merge(
        qa[["GEOID", "qa_weighted_count"]], on="GEOID")
    nod["reason"] = "no valid (unmasked) native LAI pixel touched by the tract"
    nod.to_csv(out_dir / "tracts_without_data.csv", index=False)
    qa.to_parquet(out_dir / "qa" / "tract_qa.parquet", index=False)
    cd_table.to_parquet(out_dir / "qa" / "count_definitions.parquet", index=False)
    if ind is not None:
        ind.to_parquet(out_dir / "qa" / "independent_check.parquet", index=False)
    if eex_table is not None:
        eex_table.to_csv(out_dir / "qa" / "ee_export_comparison.csv", index=False)
    if eer_table is not None:
        eer_table.to_csv(out_dir / "qa" / "ee_reference_comparison.csv", index=False)
    write_json(out_dir / "validation.json", v)
    write_readme(out_dir / "README.md", v, out)
    names = [f"{OUT_STEM}.parquet", f"{OUT_STEM}.csv", "validation.json", "tracts_without_data.csv", "README.md"]
    names += sorted(str(p.relative_to(out_dir)) for p in (out_dir / "qa").glob("*"))
    (out_dir / "SHA256SUMS").write_text("".join(f"{sha256_file(out_dir / n)}  {n}\n" for n in names))

    summary = {"passed": v["passed"], "failures": failures, "rows": len(out),
               "tracts_without_data": st["tracts_without_data"], "independent_passed": ind_rep.get("passed"),
               "ee_numeric_comparison_feasible": eex.get("numeric_comparison_feasible")}
    print(json.dumps(summary, indent=2))
    if failures:
        sys.exit(f"validation FAILED: {failures}")


if __name__ == "__main__":
    main()
