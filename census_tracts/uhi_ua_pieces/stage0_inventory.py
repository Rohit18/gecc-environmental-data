"""Stage 0, items 1-2: inventory of the 40 seasonal composites (2015-2024) and the temperature grid.

For every year x season x measure: both rasters exist; grid (CRS WKT, transform, size), nodata; from the
sidecar: expected/observed days, missing dates, period_complete, and whether exclude_implausible_daily was on.
Then the grid parameters, the sphere radius read from the CRS, and whether the grid is a whole-pixel window of
the global MODIS sinusoidal grid (43,200 x 21,600 cells of 2*pi*R/43200 m). Also checks the September
WorldCover fraction raster is on the same grid, and locates the extreme_value_cells_<year>.parquet records.
Read-only; writes qa/stage0_inventory.json and qa/stage0_composites.csv.
"""
import glob
import json
import math
import os

import pandas as pd
import rasterio
from pyproj import CRS

from uhi_common import UHI_ROOT, composite_keys, composite_paths, load_config, setup_logging, utc_now, write_json_atomic

log = setup_logging("stage0_inventory")
cfg = load_config()


def first(globs, year):
    for g in globs:
        hits = sorted(glob.glob(g.format(year=year)))
        if hits:
            return hits[0]
    return None


rows, grids = [], {}
for y, s, m in composite_keys(cfg):
    mp, dp = composite_paths(cfg, y, s, m)
    rec = {"year": y, "season": s, "measure": m, "mean_path": mp, "days_path": dp,
           "mean_exists": os.path.exists(mp), "days_exists": os.path.exists(dp),
           "mean_sidecar_exists": os.path.exists(mp + ".meta.json"),
           "days_sidecar_exists": os.path.exists(dp + ".meta.json")}
    for kind, p in (("mean", mp), ("days", dp)):
        if not os.path.exists(p):
            continue
        with rasterio.open(p) as ds:
            g = (ds.crs.to_wkt(), tuple(ds.transform)[:6], ds.width, ds.height)
            grids.setdefault(g, []).append(f"{y}_{s}_{m}_{kind}")
            rec[f"{kind}_nodata"] = ds.nodata
            rec[f"{kind}_dtype"] = ds.dtypes[0]
            rec[f"{kind}_width"], rec[f"{kind}_height"] = ds.width, ds.height
            rec[f"{kind}_transform"] = list(tuple(ds.transform)[:6])
    if rec["mean_sidecar_exists"]:
        sc = json.load(open(mp + ".meta.json"))
        p = sc.get("params", {})
        meta = p.get("meta", {})
        rec.update({
            "expected_days": meta.get("expected_days"), "observed_days": meta.get("observed_days"),
            "missing_dates": meta.get("missing_dates"), "period_complete": meta.get("period_complete"),
            "exclude_implausible_daily_key_present": "exclude_implausible_daily" in p,
            "exclude_implausible_daily": p.get("exclude_implausible_daily"),
            "plausible_range_c": p.get("plausible_range_c"),
            "implausible_obs_excluded": sc.get("stats", {}).get("implausible_obs_excluded"),
            "cells_with_excluded_obs": sc.get("stats", {}).get("cells_with_excluded_obs"),
            "composite_min_c": sc.get("stats", {}).get("min_c"), "composite_max_c": sc.get("stats", {}).get("max_c"),
            "valid_cells": sc.get("stats", {}).get("valid_cells"), "written_utc": sc.get("written_utc"),
            "n_source_files": sc.get("n_source_files"), "source_archive": meta.get("source_archive"),
            "source_archive_md5": meta.get("source_archive_md5"), "sidecar_output_path": sc.get("output"),
        })
    if rec["days_sidecar_exists"]:
        sd = json.load(open(dp + ".meta.json"))
        rec["days_sidecar_params_hash_equals_mean"] = sd.get("params_hash") == sc.get("params_hash")
    rows.append(rec)

df = pd.DataFrame(rows)
os.makedirs(f"{UHI_ROOT}/qa", exist_ok=True)
df.to_csv(f"{UHI_ROOT}/qa/stage0_composites.csv", index=False)

# ---- grid
(g_wkt, g_tr, g_w, g_h), members = max(grids.items(), key=lambda kv: len(kv[1]))
crs = CRS.from_wkt(g_wkt)
R = crs.ellipsoid.semi_major_metre
inv_flat = crs.ellipsoid.inverse_flattening
a, b, c, d, e, f = g_tr
cell_global = 2 * math.pi * R / 43200
X0, Y0 = -math.pi * R, math.pi / 2 * R
grid = {"crs_wkt": g_wkt, "transform": list(g_tr), "x0": c, "y0": f, "dx": a, "dy": abs(e),
        "rotation_terms": [b, d], "width": g_w, "height": g_h, "sphere_radius_m": R,
        "inverse_flattening": inv_flat, "projection": crs.to_dict().get("proj"),
        "lon_0": crs.to_dict().get("lon_0", 0)}
whole = {"global_cell_m": cell_global, "global_x_origin": X0, "global_y_origin": Y0,
         "col_offset": (c - X0) / cell_global, "row_offset": (Y0 - f) / cell_global,
         "dx_minus_global": a - cell_global, "dy_minus_global": abs(e) - cell_global,
         "dx_rel": (a - cell_global) / cell_global, "dy_rel": (abs(e) - cell_global) / cell_global}
whole["col_offset_frac"] = abs(whole["col_offset"] - round(whole["col_offset"]))
whole["row_offset_frac"] = abs(whole["row_offset"] - round(whole["row_offset"]))
whole["is_whole_pixel_window"] = (whole["col_offset_frac"] < 1e-6 and whole["row_offset_frac"] < 1e-6
                                  and abs(whole["dx_rel"]) < 1e-9 and abs(whole["dy_rel"]) < 1e-9)
whole["window_in_global_grid"] = {"col0": round(whole["col_offset"]), "row0": round(whole["row_offset"]),
                                  "col1": round(whole["col_offset"]) + g_w, "row1": round(whole["row_offset"]) + g_h}

# ---- WorldCover fraction raster
fr = cfg["paths"]["worldcover_fractions"]
frac = {"path": fr, "exists": os.path.exists(fr)}
if frac["exists"]:
    with rasterio.open(fr) as ds:
        frac.update({"same_grid_as_composites": (ds.crs.to_wkt() == g_wkt and tuple(ds.transform)[:6] == g_tr
                                                 and ds.width == g_w and ds.height == g_h),
                     "band_descriptions": list(ds.descriptions), "nodata": ds.nodata, "count": ds.count,
                     "tags": ds.tags()})
    sc = fr + ".meta.json"
    if os.path.exists(sc):
        j = json.load(open(sc))
        frac["sidecar_params"] = j.get("params")
        frac["sidecar_stats"] = {k: v for k, v in j.items() if k in ("stats", "n_tiles", "written_utc")} or j.get("extra")
    scratch_copy = "/path/to/scratch/gecc_census_tract_uhi/worldcover/worldcover2020_v100_class_fractions.tif"
    frac["scratch_copy"] = scratch_copy if os.path.exists(scratch_copy) else None

# ---- out-of-range records
extreme = {}
for y in cfg["years"]:
    p = first(cfg["paths"]["extreme_cells_globs"], y)
    rp = first(cfg["paths"]["inventory_report_globs"], y)
    rep = json.load(open(rp)).get("extreme_value_report", {}) if rp else {}
    extreme[y] = {"records_path": p, "inventory_report": rp,
                  "report_total_extreme_obs": rep.get("total_extreme_pixel_observations"),
                  "report_n_rasters_with_extreme": rep.get("n_rasters_with_extreme_pixels"),
                  "report_cells_file": rep.get("cells_file")}
    if p:
        ex = pd.read_parquet(p)
        extreme[y].update({"n_cells": int(len(ex)), "columns": list(ex.columns),
                           "n_obs_sum_n_files": int(ex["n_files"].sum()) if "n_files" in ex else None})
    extreme[y]["status"] = ("found" if p else
                            "absent_report_says_zero" if rep.get("total_extreme_pixel_observations") == 0 else
                            "MISSING")

# ---- equivalence of the distinct grid definitions to the reference (most common) one
equiv = []
for (wkt_i, tr_i, w_i, h_i), mem in grids.items():
    ci = CRS.from_wkt(wkt_i)
    ai, _, xi, _, ei, yi = tr_i
    # largest displacement of any cell corner between the two definitions (m)
    disp = max(abs((xi + ai * k) - (c + a * k)) for k in (0, w_i)) if w_i == g_w else None
    disp_y = max(abs((yi + ei * k) - (f + e * k)) for k in (0, h_i)) if h_i == g_h else None
    equiv.append({"n_members": len(mem), "first": mem[0], "last": mem[-1],
                  "years": sorted({int(x.split("_")[0]) for x in mem}),
                  "crs_wkt_identical": wkt_i == g_wkt,
                  "crs_same_parameters": ci.ellipsoid.semi_major_metre == R and ci.to_dict() == crs.to_dict(),
                  "transform_identical": tr_i == g_tr, "size_identical": (w_i, h_i) == (g_w, g_h),
                  "origin_dx_m": xi - c, "origin_dy_m": yi - f, "dx_diff_m": ai - a, "dy_diff_m": abs(ei) - abs(e),
                  "max_corner_displacement_x_m": disp, "max_corner_displacement_y_m": disp_y})

out = {"generated_utc": utc_now(),
       "grid_equivalence": equiv,
       "n_composites_expected": len(df), "n_mean_exist": int(df.mean_exists.sum()),
       "n_days_exist": int(df.days_exists.sum()),
       "n_distinct_grids": len(grids), "grid_members": {str(i): v for i, v in enumerate(grids.values())},
       "grid": grid, "whole_pixel_check": whole, "worldcover_fractions": frac, "extreme_value_records": extreme,
       "nodata_values": {"mean": sorted(df.mean_nodata.dropna().unique().tolist()),
                         "days": sorted(df.days_nodata.dropna().unique().tolist()) if df.days_nodata.notna().any() else [None]},
       "rule_on": df.loc[df.exclude_implausible_daily == True, ["year", "season", "measure"]].astype(str).agg("_".join, axis=1).tolist(),  # noqa: E712
       "incomplete": df.loc[df.period_complete == False, ["year", "season", "measure", "missing_dates"]].to_dict("records")}  # noqa: E712
write_json_atomic(f"{UHI_ROOT}/qa/stage0_inventory.json", out)
log.info("composites: mean %d/40, days %d/40, distinct grids %d", out["n_mean_exist"], out["n_days_exist"], len(grids))
log.info("grid: %s", json.dumps({k: v for k, v in grid.items() if k != "crs_wkt"}))
log.info("whole-pixel: %s", json.dumps(whole))
log.info("rule on: %s", out["rule_on"])
log.info("incomplete: %s", out["incomplete"])
log.info("fractions: %s", json.dumps({k: v for k, v in frac.items() if k not in ("tags", "sidecar_params")}, default=str)[:1500])
log.info("extreme: %s", json.dumps(extreme, default=str))
