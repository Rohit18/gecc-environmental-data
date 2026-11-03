"""Stage 5 statistics: piece and UA tables for the 40 composites from the reduced weight tables.

build_tables(cfg, weights_dir, ua_subset=None) -> (piece_long, ua_long_full, piece_static, grid_deviations)
Used by the Columbus test (Stage 2, ua_subset = UAs fully inside the test tile) and by the full run.

Per composite: T = mean[cell_id], valid = T != nodata; per zone sum(w*T) and sum(w) over valid cells,
sum(w) over all cells, min/max valid days over cells with w > 0 and valid T, distinct cells with w > 0.
Effective cells = valid weight / (dx*dy). Every composite's grid is checked numerically against the reference
grid before use (Grid.deviation; origin <= 0.01 m, cell size <= 1e-6 m, same projection parameters).
Nulls always carry a quality_status (precedence top to bottom):
  piece_no_pixels             piece owns no 10 m pixel (degenerate sliver)          [pieces only]
  outside_temperature_grid    zone's kept pixels all fall off the temperature grid
  ua_no_valid_temperature     no valid temperature in any of the UA's kept (urban-rule) pixels
  rural_no_valid_temperature  UA has valid temperature but its rural reference has none
  urban_no_valid_temperature  piece urban value (or, in built_status, the UA built-only value) has none
  ok
with "+incomplete_period" appended when the composite's period_complete is false. quality_status describes
uhi_c (pieces) / uhi_all_c (UAs); built_status describes uhi_built_c.
"""
import json

import numpy as np
import pandas as pd
import rasterio

from uhi_common import Grid, composite_keys, composite_paths, setup_logging
from uhi_stats_core import zone_stats

log = setup_logging("uhi_tables")
KINDS = ("urban", "rural", "ua_all", "ua_built", "ua_nowater_snowkept")
SPHERE_CEA = "+proj=cea +lon_0=0 +lat_ts=0 +R=6371007.181 +units=m +no_defs"


def read_composite(cfg, grid, y, s, m):
    mp, dp = composite_paths(cfg, y, s, m)
    devs = []
    for p in (mp, dp):
        rec = grid.deviation(p)
        rec.update({"year": y, "season": s, "measure": m, "raster": "mean_C" if p == mp else "valid_days"})
        devs.append(rec)
        if not rec["ok"]:
            raise SystemExit(f"{p}: grid differs from the reference beyond tolerance: {rec}")
    with rasterio.open(mp) as ds:
        T = ds.read(1).reshape(-1)
        nod = ds.nodata
    with rasterio.open(dp) as ds:
        D = ds.read(1).reshape(-1)
    meta = json.load(open(mp + ".meta.json"))["params"]["meta"]
    return T, D, nod, meta, devs


def with_period(base, period_complete):
    return [b + ("" if period_complete else "+incomplete_period") for b in base]


def build_tables(cfg, weights_dir, ua_subset=None, keys=None):
    import geopandas as gpd
    grid = Grid.from_cfg(cfg)
    cell_area = grid.dx * grid.dy
    W = {k: pd.read_parquet(f"{weights_dir}/uhi_weights_{k}.parquet") for k in KINDS}
    dpc = pd.read_parquet(f"{weights_dir}/uhi_diagnostics_piece.parquet").set_index("piece_id")
    dua = pd.read_parquet(f"{weights_dir}/uhi_diagnostics_ua.parquet").set_index("ua_idx")
    uas = gpd.read_parquet(cfg["urban_areas"]["kept_path"])[["ua_idx", "GEOID20", "NAME20", "geometry"]]
    pcs = gpd.read_parquet(cfg["pieces"]["path"])[["piece_id", "GEOID", "GEOID20", "ua_idx", "piece_area_m2",
                                                   "tract_area_m2", "overlap_share", "geometry"]]
    if ua_subset is not None:
        uas = uas[uas.ua_idx.isin(ua_subset)]
        pcs = pcs[pcs.ua_idx.isin(ua_subset)]
        for k in KINDS:
            ids = pcs.piece_id if k == "urban" else uas.ua_idx
            W[k] = W[k][W[k].zone_id.isin(ids)]
    pcs["piece_area_sphere_m2"] = pcs.to_crs(SPHERE_CEA).area
    uas = pd.DataFrame(uas.drop(columns="geometry")).set_index("ua_idx")
    pcs = pd.DataFrame(pcs.drop(columns="geometry")).set_index("piece_id")
    pcs["NAME20"] = pcs.ua_idx.map(uas.NAME20)

    # ---- static per-piece diagnostics (no diagnostic row = the piece owns no 10 m pixel)
    d = dpc.reindex(pcs.index)
    st = pcs[["GEOID", "GEOID20", "NAME20", "ua_idx", "overlap_share", "piece_area_m2", "piece_area_sphere_m2",
              "tract_area_m2"]].copy()
    st["owns_pixels"] = d["total_m2"].notna() & (d["total_m2"] > 0)
    st["pixel_area_m2"] = d["total_m2"].fillna(0.0)
    for c, n in (("wc0_m2", "nodata_m2"), ("wc50_m2", "built_m2"), ("wc70_m2", "snow_ice_m2"), ("wc80_m2", "water_m2")):
        st[n] = d[c].fillna(0.0)
    st["valid_m2"] = st.pixel_area_m2 - st.nodata_m2
    st["urban_weight_m2"] = d["weight_urban_m2"].fillna(0.0)
    st["urban_distinct_cells"] = d["distinct_cells_urban"].fillna(0).astype(int)
    st["outside_grid_urban_m2"] = d["outside_grid_urban_m2"].fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        st["pixel_area_ratio"] = np.where(st.piece_area_m2 > 0, st.pixel_area_m2 / st.piece_area_m2, np.nan)
        st["pixel_area_ratio_sphere"] = np.where(st.piece_area_sphere_m2 > 0,
                                                 st.pixel_area_m2 / st.piece_area_sphere_m2, np.nan)
    no_px = ~st.owns_pixels.to_numpy()

    keys = keys or composite_keys(cfg)
    prow, urow, devs = [], [], []
    for (y, s, m) in keys:
        T, D, nod, meta, dv = read_composite(cfg, grid, y, s, m)
        devs += dv
        zs = {k: zone_stats(W[k], T, D, nod) for k in KINDS}
        pc_complete = bool(meta["period_complete"])
        dayf = {"expected_days": int(meta["expected_days"]), "observed_days": int(meta["observed_days"]),
                "missing_dates": meta["missing_dates"] or "", "period_complete": pc_complete}

        def get(k, idx, col):
            return zs[k][col].reindex(idx).to_numpy()

        # ---- UA rows
        ui = uas.index
        u = pd.DataFrame({"ua_idx": ui, "GEOID20": uas.GEOID20.values, "NAME20": uas.NAME20.values,
                          "year": y, "season": s, "temperature_measure": m})
        for k, pre in (("rural", "rural"), ("ua_all", "urban_all"), ("ua_built", "urban_built"),
                       ("ua_nowater_snowkept", "regr_nowater_snowkept")):
            vw = np.nan_to_num(get(k, ui, "valid_w"))
            tw = np.nan_to_num(get(k, ui, "total_w"))
            u[f"{pre}_temperature_c"] = np.where(vw > 0, get(k, ui, "temperature"), np.nan)
            u[f"{pre}_sum_wT"] = np.nan_to_num(get(k, ui, "sum_wT"))
            u[f"{pre}_valid_weight_m2"] = vw
            u[f"{pre}_total_weight_m2"] = tw
            u[f"{pre}_effective_cells"] = vw / cell_area
            u[f"{pre}_distinct_cells"] = np.nan_to_num(get(k, ui, "distinct_cells")).astype(int)
            with np.errstate(divide="ignore", invalid="ignore"):
                u[f"{pre}_valid_coverage"] = np.where(tw > 0, vw / tw, np.nan)
            u[f"{pre}_min_valid_days"] = get(k, ui, "min_days")
            u[f"{pre}_max_valid_days"] = get(k, ui, "max_days")
        out_all = dua["outside_grid_ua_all_m2"].reindex(ui).fillna(0).to_numpy()
        no_ua = u["urban_all_valid_weight_m2"].to_numpy() <= 0
        no_rur = u["rural_valid_weight_m2"].to_numpy() <= 0
        no_blt = u["urban_built_valid_weight_m2"].to_numpy() <= 0
        off = (u["urban_all_total_weight_m2"].to_numpy() <= 0) & (out_all > 0)
        base = np.where(off, "outside_temperature_grid", np.where(no_ua, "ua_no_valid_temperature",
                        np.where(no_rur, "rural_no_valid_temperature", "ok")))
        bstat = np.where(base != "ok", base, np.where(no_blt, "urban_no_valid_temperature", "ok"))
        u["uhi_all_c"] = np.where(base == "ok", u.urban_all_temperature_c - u.rural_temperature_c, np.nan)
        u["uhi_built_c"] = np.where(bstat == "ok", u.urban_built_temperature_c - u.rural_temperature_c, np.nan)
        u["quality_status"] = with_period(base, pc_complete)
        u["built_status"] = with_period(bstat, pc_complete)
        for c, n in (("total_m2", "ua_pixel_area_m2"), ("wc0_m2", "ua_nodata_m2"), ("wc50_m2", "ua_built_m2"),
                     ("wc70_m2", "ua_snow_ice_m2"), ("wc80_m2", "ua_water_m2")):
            u[n] = dua[c].reindex(ui).to_numpy()
        for k, v in dayf.items():
            u[k] = v
        urow.append(u)

        # ---- piece rows
        pi = pcs.index
        uidx = pcs.ua_idx.to_numpy()
        p = pd.DataFrame({"piece_id": pi, "GEOID": pcs.GEOID.values, "GEOID20": pcs.GEOID20.values,
                          "NAME20": pcs.NAME20.values, "year": y, "season": s, "temperature_measure": m})
        vw = np.nan_to_num(get("urban", pi, "valid_w"))
        tw = np.nan_to_num(get("urban", pi, "total_w"))
        p["urban_temperature_c"] = np.where(vw > 0, get("urban", pi, "temperature"), np.nan)
        p["rural_temperature_c"] = u.set_index("ua_idx").rural_temperature_c.reindex(uidx).to_numpy()
        p["uhi_c"] = np.nan
        p["urban_sum_wT"] = np.nan_to_num(get("urban", pi, "sum_wT"))
        p["urban_valid_weight_m2"] = vw
        p["urban_total_weight_m2"] = tw
        p["urban_effective_cells"] = vw / cell_area
        p["urban_distinct_cells"] = np.nan_to_num(get("urban", pi, "distinct_cells")).astype(int)
        with np.errstate(invalid="ignore", divide="ignore"):
            p["urban_valid_coverage"] = np.where(tw > 0, vw / tw, np.nan)
        for c in ("overlap_share", "piece_area_m2", "tract_area_m2"):
            p[c] = pcs[c].values
        for c, n in (("water_m2", "piece_water_m2"), ("snow_ice_m2", "piece_snow_ice_m2"),
                     ("built_m2", "piece_built_m2"), ("nodata_m2", "piece_nodata_m2"),
                     ("pixel_area_m2", "piece_pixel_area_m2")):
            p[n] = st[c].values
        p["min_valid_days"] = get("urban", pi, "min_days")
        p["max_valid_days"] = get("urban", pi, "max_days")
        for k, v in dayf.items():
            p[k] = v
        ua_base = pd.Series(base, index=u.ua_idx).reindex(uidx).to_numpy()
        out_p = st["outside_grid_urban_m2"].to_numpy()
        pbase = np.where(no_px, "piece_no_pixels",
                         np.where((tw <= 0) & (out_p > 0), "outside_temperature_grid",
                                  np.where(ua_base != "ok", ua_base,
                                           np.where(vw <= 0, "urban_no_valid_temperature", "ok"))))
        p["uhi_c"] = np.where(pbase == "ok", p.urban_temperature_c - p.rural_temperature_c, np.nan)
        p["quality_status"] = with_period(pbase, pc_complete)
        prow.append(p)
        log.info("%d %s %s: pieces ok %d/%d, UAs ok %d/%d", y, s, m, int((pbase == "ok").sum()), len(p),
                 int((base == "ok").sum()), len(u))
    return (pd.concat(prow, ignore_index=True), pd.concat(urow, ignore_index=True), st.reset_index(),
            pd.DataFrame(devs))
