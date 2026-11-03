"""Stage 2: Columbus, OH test on WorldCover tile N39W084 (run after uhi_weights_worker.py N39W084 --cell-diag
and uhi_reduce.py --tiles N39W084 --name columbus). Writes qa/stage2/*.json|csv|tif and
data/output/test_columbus/*. Read-only with respect to all existing data.

0  pixel identity with the vegetation product (same tile partial): per-piece valid/water/built/nodata area,
   urban and rural totals (vegetation keeps snow/ice; ours drops it, so the snow area is added back).
a  pixel-to-cell: built-up / water / nodata shares per fully covered temperature cell vs the September
   WorldCover fraction raster (max abs difference; criterion 1e-4).
b  weight sanity: per piece, urban weight / polygon area (EPSG:6933) <= 1.001.
c  regression vs the September release ua_temperature_c (ua_nowater_snowkept weights), all 40 composites.
d  QA rasters for the Columbus window; 2020 JJA TMAX composite clipped to the Columbus UA.
e  benchmark (per window / per tile) and extrapolation to all 135 tiles.
f  snow/ice (WorldCover 70) area per UA.
"""
import json
import math
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds

from uhi_common import UHI_ROOT, Grid, composite_paths, load_config, setup_logging, utc_now, write_json_atomic
from uhi_tables import build_tables
from uhi_weights_worker import UHIContext, iter_windows, kind_masks, partial_path, rasterize_window, window_cells

log = setup_logging("stage2_columbus")
cfg = load_config()
TILE, NAME = "N39W084", "columbus"
QA_BBOX = (-83.30, 39.70, -82.50, 40.40)
COLUMBUS = "19234"
grid = Grid.from_cfg(cfg)
ctx = UHIContext(cfg)
wdir = f"{cfg['paths']['interim']}/test_{NAME}"
qdir = f"{UHI_ROOT}/qa/stage2"
odir = f"{cfg['paths']['output']}/test_{NAME}"
os.makedirs(qdir, exist_ok=True)
os.makedirs(odir, exist_ok=True)
R = {"generated_utc": utc_now(), "tile": TILE}

wcp = os.path.join(ctx.vcfg["paths"]["worldcover_dir"], ctx.vcfg["worldcover"]["file_pattern"].format(tile=TILE))
with rasterio.open(wcp) as src:
    tb = src.bounds
tbox = shapely.box(*tb)
uas = gpd.read_parquet(cfg["urban_areas"]["kept_path"])
pcs = gpd.read_parquet(cfg["pieces"]["path"])
ua_in = uas[uas.within(tbox)]
ua_touch = uas[uas.intersects(tbox)]
pc_in = pcs[pcs.ua_idx.isin(ua_in.ua_idx)]
R["uas_intersecting_tile"] = int(len(ua_touch))
R["uas_fully_inside"] = int(len(ua_in))
R["pieces_of_inside_uas"] = int(len(pc_in))
R["pieces_of_inside_uas_all_within_tile"] = bool(pc_in.within(tbox).all())
col = uas[uas.GEOID20 == COLUMBUS].iloc[0]
R["columbus"] = {"GEOID20": COLUMBUS, "NAME20": col.NAME20, "ua_idx": int(col.ua_idx),
                 "inside_tile": bool(col.geometry.within(tbox)), "bounds": list(col.geometry.bounds),
                 "n_pieces": int((pcs.ua_idx == col.ua_idx).sum())}
log.info("UAs touching tile %d, fully inside %d (pieces %d); Columbus inside: %s",
         len(ua_touch), len(ua_in), len(pc_in), R["columbus"]["inside_tile"])

W = {k: pd.read_parquet(f"{wdir}/uhi_weights_{k}.parquet") for k in
     ("urban", "rural", "ua_all", "ua_built", "ua_nowater_snowkept")}
dpc = pd.read_parquet(f"{wdir}/uhi_diagnostics_piece.parquet").set_index("piece_id")
dua = pd.read_parquet(f"{wdir}/uhi_diagnostics_ua.parquet").set_index("ua_idx")

# ---------------------------------------------------------------- 0 pixel identity with the vegetation product
vp = pd.read_parquet(os.path.join(cfg["paths"]["vegetation_partials_dir"], f"{TILE}.parquet"))
vd = vp[vp.kind == "diag"].set_index("id")
cmp = pd.DataFrame({"veg_valid": vd.valid_m2, "veg_water": vd.water_m2, "veg_built": vd.built_m2,
                    "veg_nodata": vd.nodata_m2}).join(dpc, how="outer")
cmp["new_valid"] = cmp.total_m2 - cmp.wc0_m2


def rel(a, b):
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))


veg_urban = vp[(vp.kind == "urban") & (vp.year == 2020)].groupby("id").area_m2.sum()
veg_rural = vp[(vp.kind == "rural") & (vp.year == 2020)].groupby("id").area_m2.sum()
new_urban_plus_snow = W["urban"].groupby("zone_id").area_m2.sum().reindex(dpc.index).fillna(0) + dpc.wc70_m2
new_rural_plus_snow = W["rural"].groupby("zone_id").area_m2.sum().reindex(dua.index).fillna(0) + dua.wc70_m2
R["check0_vegetation_identity"] = {
    "n_pieces_veg": int(len(vd)), "n_pieces_new": int(len(dpc)),
    "same_piece_set": bool(set(vd.index) == set(dpc.index)),
    "max_rel_diff_valid": rel(cmp.new_valid, cmp.veg_valid), "max_rel_diff_water": rel(cmp.wc80_m2, cmp.veg_water),
    "max_rel_diff_built": rel(cmp.wc50_m2, cmp.veg_built), "max_rel_diff_nodata": rel(cmp.wc0_m2, cmp.veg_nodata),
    "max_rel_diff_urban_total_plus_snow": rel(new_urban_plus_snow.reindex(veg_urban.index), veg_urban),
    "max_rel_diff_rural_total_plus_snow": rel(new_rural_plus_snow.reindex(veg_rural.index), veg_rural),
    "same_ua_set": bool(set(veg_rural.index) == set(dua.index[dua.total_m2 > 0]) | set(veg_rural.index)),
    "veg_partial_meta": {k: v for k, v in json.loads(
        __import__("pyarrow.parquet", fromlist=["x"]).read_schema(
            os.path.join(cfg["paths"]["vegetation_partials_dir"], f"{TILE}.parquet")).metadata[b"urbveg"]).items()
        if k in ("pieces_sha256", "uas_sha256", "window_px", "all_touched", "run")},
    "new_signature": {k: v for k, v in ctx.signature().items() if k in ("pieces_sha256", "uas_sha256", "window_px", "all_touched")}}
log.info("check 0: %s", json.dumps(R["check0_vegetation_identity"]))

# ---------------------------------------------------------------- a pixel-to-cell vs September fraction raster
cd = pd.read_parquet(partial_path(cfg, TILE, "partials_celldiag"))
row, colc = np.divmod(cd.cell_id.to_numpy(), grid.width)
xl, xr = grid.x0 + colc * grid.dx, grid.x0 + (colc + 1) * grid.dx
yt, yb = grid.y0 - row * grid.dy, grid.y0 - (row + 1) * grid.dy
inside = np.ones(len(cd), bool)
for x in (xl, xr):
    for yy in (yt, yb):
        lat = np.degrees(yy / grid.R)
        lon = np.degrees(x / (grid.R * np.cos(yy / grid.R)))
        inside &= (lon > tb.left) & (lon < tb.right) & (lat > tb.bottom) & (lat < tb.top)
full = cd[inside].copy()
with rasterio.open(cfg["paths"]["worldcover_fractions"]) as ds:
    assert Grid.from_raster(cfg["paths"]["worldcover_fractions"]).same_as(grid)
    fr = {i: ds.read(i).reshape(-1)[full.cell_id.to_numpy()] for i in (1, 2, 3, 4)}
ca = grid.dx * grid.dy
tot_rel = full.total_m2.to_numpy() / ca                       # cell coverage as the September code's "tot"
den_sept = np.maximum(tot_rel, 1.0)                            # September: w/max(tot, 1)
res_a = {"n_cells_fully_covered": int(len(full)), "n_cells_any_pixel": int(len(cd)),
         "coverage_ratio_total_over_cell_area": {"min": float(tot_rel.min()), "median": float(np.median(tot_rel)),
                                                 "max": float(tot_rel.max())}}
for name, band, col_ in (("built50", 1, "wc50_m2"), ("water80", 2, "wc80_m2"), ("nodata0", 3, "wc0_m2")):
    new_sept_def = full[col_].to_numpy() / ca / den_sept
    new_share = full[col_].to_numpy() / full.total_m2.to_numpy()
    d1 = np.abs(new_sept_def - fr[band])
    d2 = np.abs(new_share - fr[band])
    res_a[name] = {"max_abs_diff_september_definition": float(d1.max()),
                   "p99_abs_diff_september_definition": float(np.percentile(d1, 99)),
                   "n_gt_1e-4_september_definition": int((d1 > 1e-4).sum()),
                   "max_abs_diff_share_of_cell_pixels": float(d2.max()),
                   "n_gt_1e-4_share_of_cell_pixels": int((d2 > 1e-4).sum())}
res_a["frac_no_tile_max"] = float(fr[4].max())
res_a["pass_1e-4"] = all(res_a[n]["max_abs_diff_september_definition"] <= 1e-4 for n in ("built50", "water80"))
R["check_a_pixel_to_cell"] = res_a
log.info("check a: %s", json.dumps(res_a))

# ---------------------------------------------------------------- b weight sanity
pa_ = pc_in.set_index("piece_id")
a6933 = pc_in.to_crs("EPSG:6933").set_index("piece_id").area
uw = W["urban"].groupby("zone_id").area_m2.sum().reindex(pa_.index).fillna(0.0)
pix = dpc.total_m2.reindex(pa_.index)
ratio = uw / pa_.piece_area_m2
# polygon area on the same sphere as the pixel areas (authalic R, geodetic latitude): separates the
# sphere-vs-ellipsoid convention from pixel-centre rasterization
sph = pc_in.to_crs("+proj=cea +lon_0=0 +lat_ts=0 +R=6371007.181 +units=m +no_defs").set_index("piece_id").area
ratio_sph = pix / sph
ratio_pix = pix / pa_.piece_area_m2
q = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
over = ratio[ratio > 1.001]
small = pa_.piece_area_m2 < 1e4
R["check_b_weight_sanity"] = {
    "n_pieces": int(len(ratio)),
    "piece_area_m2_vs_recomputed_6933_max_rel_diff": float(np.max(np.abs(pa_.piece_area_m2 - a6933) / a6933.clip(lower=1))),
    "ratio_urban_weight_over_polygon_quantiles": dict(zip(map(str, q), ratio.quantile(q).round(6).tolist())),
    "ratio_pixel_area_over_polygon_quantiles": dict(zip(map(str, q), ratio_pix.quantile(q).round(6).tolist())),
    "n_ratio_gt_1.001": int(len(over)),
    "n_ratio_gt_1.001_with_polygon_lt_1e4_m2": int((ratio > 1.001)[small].sum()),
    "n_ratio_gt_1.001_with_polygon_ge_1e4_m2": int((ratio > 1.001)[~small].sum()),
    "pieces_ratio_gt_1.001": [{"piece_id": int(i), "GEOID": pa_.loc[i, "GEOID"], "piece_area_m2": float(pa_.loc[i, "piece_area_m2"]),
                               "urban_weight_m2": float(uw.loc[i]), "ratio": float(ratio.loc[i])}
                              for i in over.sort_values(ascending=False).index[:50]],
    "n_pieces_zero_urban_weight": int((uw == 0).sum()),
    "n_pieces_without_any_pixel": int(pix.isna().sum()),
    "zero_weight_piece_polygon_area_max_m2": float(pa_.piece_area_m2[uw == 0].max()),
    "ratio_pixel_area_over_sphere_polygon_quantiles": dict(zip(map(str, q), ratio_sph.quantile(q).round(6).tolist())),
    "sum_pixel_area_over_sum_sphere_polygon": float(pix.sum() / sph.sum()),
    "sum_pixel_area_over_sum_6933_polygon": float(pix.sum() / pa_.piece_area_m2.sum()),
    "sphere_over_6933_median": float((sph / pa_.piece_area_m2).median())}
pd.DataFrame({"piece_id": ratio.index, "piece_area_m2": pa_.piece_area_m2, "sphere_area_m2": sph, "urban_weight_m2": uw,
              "pixel_area_m2": pix, "ratio_weight": ratio, "ratio_pixel": ratio_pix, "ratio_pixel_vs_sphere": ratio_sph}).to_csv(f"{qdir}/check_b_piece_ratios.csv", index=False)
log.info("check b: %s", json.dumps({k: v for k, v in R["check_b_weight_sanity"].items() if k != "pieces_ratio_gt_1.001"}))

# ---------------------------------------------------------------- tables (Stage 5 method) for UAs inside the tile
pt, ut, pstat, _ = build_tables(cfg, wdir, ua_subset=set(ua_in.ua_idx))
for d, n in ((pt, "tract_piece_uhi_test_columbus"), (ut, "ua_uhi_test_columbus"), (pstat, "piece_diagnostics_test_columbus")):
    d.to_parquet(f"{odir}/{n}.parquet", index=False)
    d.to_csv(f"{odir}/{n}.csv", index=False)

# ---------------------------------------------------------------- c regression vs September release
sep = pd.read_parquet(f"{cfg['paths']['interim']}/september_release/{cfg['paths']['september_release_ua_member']}")
sep = sep[sep.year.isin(cfg["years"])]
m = ut.merge(sep[["UACE20", "year", "season", "temperature_measure", "ua_temperature_c", "quality_status"]]
             .rename(columns={"UACE20": "GEOID20", "quality_status": "sep_status"}),
             on=["GEOID20", "year", "season", "temperature_measure"], how="left")
m["diff"] = m.regr_nowater_snowkept_temperature_c - m.ua_temperature_c
ok = m["diff"].notna()
ad = m.loc[ok, "diff"].abs()
flag = m[ok & (m["diff"].abs() > 0.05)]
# boundary change between the September UA file (TIGER 2025) and the corrected 2020 file
sep_ua = gpd.read_file("/path/to/home/GECC/tl_2025_us_uac20", columns=["UACE20", "geometry"])
sep_ua = sep_ua[sep_ua.UACE20.isin(ua_in.GEOID20)].to_crs("EPSG:6933").set_index("UACE20")
ours = ua_in.to_crs("EPSG:6933").set_index("GEOID20")
bchg = {}
for g in ours.index:
    if g in sep_ua.index:
        a, b = ours.geometry[g], sep_ua.geometry[g]
        bchg[g] = {"area_new_km2": a.area / 1e6, "area_sep_km2": b.area / 1e6,
                   "sym_diff_share": a.symmetric_difference(b).area / a.area}
    else:
        bchg[g] = {"area_new_km2": ours.geometry[g].area / 1e6, "area_sep_km2": None, "sym_diff_share": None}
per_ua = m[ok].groupby(["GEOID20", "NAME20"])["diff"].agg(["mean", "min", "max", lambda s: s.abs().max()]).rename(
    columns={"<lambda_0>": "max_abs"}).reset_index()
per_ua["sym_diff_share"] = per_ua.GEOID20.map(lambda g: bchg.get(g, {}).get("sym_diff_share"))
per_ua.to_csv(f"{qdir}/check_c_regression_per_ua.csv", index=False)
m.to_csv(f"{qdir}/check_c_regression_rows.csv", index=False)
R["check_c_regression"] = {
    "n_rows": int(len(m)), "n_compared": int(ok.sum()),
    "n_new_value_missing_sep": int((m.regr_nowater_snowkept_temperature_c.notna() & m.ua_temperature_c.isna()).sum()),
    "n_sep_value_missing_new": int((m.regr_nowater_snowkept_temperature_c.isna() & m.ua_temperature_c.notna()).sum()),
    "n_ua_not_in_september": int(m.groupby("GEOID20").ua_temperature_c.apply(lambda s: s.isna().all()).sum()),
    "diff_quantiles": dict(zip(map(str, q), m.loc[ok, "diff"].quantile(q).round(6).tolist())),
    "mean_diff": float(m.loc[ok, "diff"].mean()), "mean_abs_diff": float(ad.mean()),
    "n_abs_gt_0.05": int(len(flag)), "uas_abs_gt_0.05": sorted(flag.GEOID20.unique().tolist()),
    "uas_abs_gt_0.05_detail": per_ua[per_ua.max_abs > 0.05].round(4).to_dict("records"),
    "columbus": per_ua[per_ua.GEOID20 == COLUMBUS].round(5).to_dict("records"),
    "corr_maxabs_vs_symdiff": float(per_ua[["max_abs", "sym_diff_share"]].dropna().corr().iloc[0, 1])}
log.info("check c: %s", json.dumps({k: v for k, v in R["check_c_regression"].items() if "detail" not in k}))

# ---------------------------------------------------------------- d QA rasters
with rasterio.open(wcp) as src:
    qw = from_bounds(*QA_BBOX, transform=src.transform).round_offsets().round_lengths()
    r0, c0, H, Wd = int(qw.row_off), int(qw.col_off), int(qw.height), int(qw.width)
    arrs = {"piece_id": np.zeros((H, Wd), np.int32), "ua_idx": np.zeros((H, Wd), np.int32),
            "urban_mask": np.zeros((H, Wd), np.uint8), "rural_mask": np.zeros((H, Wd), np.uint8),
            "cell_id": np.full((H, Wd), -1, np.int32), "worldcover": np.zeros((H, Wd), np.uint8)}
    ua_t = ctx.vctx.uas[ctx.vctx.uas.intersects(tbox)]
    pc_all = ctx.vctx.pieces_in(tuple(tb))
    pc_t = pc_all[pc_all.intersects(tbox)]
    ua_tree, pc_tree = shapely.STRtree(ua_t.geometry.values), shapely.STRtree(pc_t.geometry.values)
    for win in iter_windows(src.width, src.height, ctx.window):
        rr0, cc0 = max(int(win.row_off), r0), max(int(win.col_off), c0)
        rr1, cc1 = min(int(win.row_off + win.height), r0 + H), min(int(win.col_off + win.width), c0 + Wd)
        if rr1 <= rr0 or cc1 <= cc0:
            continue
        rz = rasterize_window(src, win, ua_t, pc_t, ua_tree, pc_tree, ctx.all_touched)
        wc = src.read(1, window=win)
        wt = src.window_transform(win)
        cell = window_cells(grid, wt, int(win.height), int(win.width))
        sl_w = (slice(rr0 - int(win.row_off), rr1 - int(win.row_off)), slice(cc0 - int(win.col_off), cc1 - int(win.col_off)))
        sl_q = (slice(rr0 - r0, rr1 - r0), slice(cc0 - c0, cc1 - c0))
        arrs["worldcover"][sl_q] = wc[sl_w]
        arrs["cell_id"][sl_q] = cell[sl_w]
        if rz is None:
            continue
        _, ua_r, pc_r, pid, uid = rz
        mk = kind_masks(wc, cfg["classes"])
        arrs["piece_id"][sl_q] = pid[pc_r][sl_w]
        arrs["ua_idx"][sl_q] = uid[ua_r][sl_w]
        arrs["urban_mask"][sl_q] = ((pc_r > 0) & mk["urban"])[sl_w]
        arrs["rural_mask"][sl_q] = ((ua_r > 0) & mk["rural"])[sl_w]
    qt = src.window_transform(Window(c0, r0, Wd, H))
    for n, a in arrs.items():
        prof = {"driver": "GTiff", "height": H, "width": Wd, "count": 1, "dtype": a.dtype.name, "crs": src.crs,
                "transform": qt, "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512,
                "nodata": -1 if n == "cell_id" else 0}
        with rasterio.open(f"{qdir}/columbus_{n}_10m.tif.part", "w", **prof) as dst:
            dst.write(a, 1)
        os.replace(f"{qdir}/columbus_{n}_10m.tif.part", f"{qdir}/columbus_{n}_10m.tif")
# composite clipped to the Columbus UA: cells with ua_all weight > 0, others nodata
cu = W["ua_all"][W["ua_all"].zone_id == int(col.ua_idx)]
rw, cl = np.divmod(cu.cell_id.to_numpy(), grid.width)
mp, _ = composite_paths(cfg, 2020, "JJA", "TMAX")
with rasterio.open(mp) as ds:
    win = Window(int(cl.min()), int(rw.min()), int(cl.max() - cl.min() + 1), int(rw.max() - rw.min() + 1))
    a = ds.read(1, window=win)
    keep = np.zeros(a.shape, bool)
    keep[rw - rw.min(), cl - cl.min()] = True
    a = np.where(keep, a, ds.nodata).astype(np.float32)
    prof = {"driver": "GTiff", "height": a.shape[0], "width": a.shape[1], "count": 1, "dtype": "float32",
            "crs": ds.crs, "transform": ds.window_transform(win), "nodata": ds.nodata, "compress": "deflate"}
    with rasterio.open(f"{qdir}/columbus_2020_JJA_TMAX_mean_C_ua_cells.tif", "w", **prof) as dst:
        dst.write(a, 1)
# cell-weight table for the Columbus UA (urban/rural/ua_all) for inspection
R["check_d_qa_rasters"] = {"dir": qdir, "bbox": QA_BBOX, "shape_10m": [H, Wd],
                           "columbus_ua_cells_with_weight": int(len(cu)),
                           "files": sorted(f for f in os.listdir(qdir) if f.endswith(".tif"))}

# ---------------------------------------------------------------- e benchmark and extrapolation
st = json.load(open(os.path.join(cfg["paths"]["logs"], "worker", f"{TILE}.stats.json")))
ws = pd.DataFrame(st["windows"])
tiles = pd.read_csv(cfg["paths"]["worldcover_tiles_manifest"]).tile.tolist()
nwin = {}
uas_geom = ctx.vctx.uas
tree_all = shapely.STRtree(uas_geom.geometry.values)
for t in tiles:
    with rasterio.open(os.path.join(ctx.vcfg["paths"]["worldcover_dir"],
                                    ctx.vcfg["worldcover"]["file_pattern"].format(tile=t))) as src:
        n = 0
        for win in iter_windows(src.width, src.height, ctx.window):
            if tree_all.query(shapely.box(*rasterio.windows.bounds(win, src.transform)), predicate="intersects").size:
                n += 1
        nwin[t] = n
tot_win = sum(nwin.values())
per_win = float(ws.total_s.mean())
R["check_e_benchmark"] = {
    "note": "plain worker run (logs/worker/<tile>.stats.json); --cell-diag runs write <tile>.celldiag.stats.json",
    "tile_wall_s_worker": st["tile_s"], "windows_processed": len(ws), "per_window_s": {
        "mean": per_win, "median": float(ws.total_s.median()), "max": float(ws.total_s.max()),
        "read": float(ws.read_s.mean()), "cell": float(ws.cell_s.mean()), "rasterize": float(ws.rasterize_s.mean()),
        "accumulate": float(ws.accumulate_s.mean())},
    "peak_rss_gb": float(ws.peak_rss_gb.max()), "tile_overhead_s": float(st["tile_s"] - ws.total_s.sum()),
    "n_tiles": len(tiles), "windows_with_ua_all_tiles": tot_win,
    "extrapolated_core_hours_at_columbus_per_window_rate": (tot_win * per_win + len(tiles) * (st["tile_s"] - ws.total_s.sum())) / 3600,
    "extrapolated_node_minutes_64_workers": (tot_win * per_win + len(tiles) * (st["tile_s"] - ws.total_s.sum())) / 64 / 60,
    "largest_tiles_by_windows": sorted(nwin.items(), key=lambda kv: -kv[1])[:10]}
log.info("check e: %s", json.dumps(R["check_e_benchmark"]))

# ---------------------------------------------------------------- f snow/ice per UA
sn = dua.join(uas.set_index("ua_idx")[["GEOID20", "NAME20"]], how="left")
sn = sn[sn.index.isin(ua_in.ua_idx)]
R["check_f_snow_ice"] = {"n_uas": int(len(sn)), "n_uas_with_wc70": int((sn.wc70_m2 > 0).sum()),
                         "total_wc70_m2": float(sn.wc70_m2.sum()),
                         "uas_with_wc70": sn[sn.wc70_m2 > 0][["GEOID20", "NAME20", "wc70_m2", "total_m2"]]
                         .reset_index().to_dict("records")}
sn[["GEOID20", "NAME20", "total_m2", "wc0_m2", "wc50_m2", "wc70_m2", "wc80_m2"]].to_csv(f"{qdir}/check_f_ua_class_areas.csv")

# ---------------------------------------------------------------- Columbus UA summary and status counts
cs = ut[ut.GEOID20 == COLUMBUS][["year", "season", "temperature_measure", "rural_temperature_c", "urban_all_temperature_c",
                                 "uhi_all_c", "urban_built_temperature_c", "uhi_built_c", "rural_effective_cells",
                                 "urban_all_effective_cells", "quality_status"]]
cs.to_csv(f"{qdir}/columbus_ua_uhi.csv", index=False)
cp = pt[pt.GEOID20 == COLUMBUS]
R["tables"] = {"piece_rows": int(len(pt)), "ua_rows": int(len(ut)),
               "piece_status": pt.quality_status.value_counts().to_dict(), "ua_status": ut.quality_status.value_counts().to_dict(),
               "columbus_2020_JJA_TMAX": cs[(cs.year == 2020) & (cs.season == "JJA") & (cs.temperature_measure == "TMAX")]
               .round(4).to_dict("records"),
               "columbus_piece_uhi_quantiles_2020_JJA_TMAX": cp[(cp.year == 2020) & (cp.season == "JJA") &
                                                                (cp.temperature_measure == "TMAX")].uhi_c.quantile(q).round(4).tolist()}
# pooled pieces vs UA (G7 preview)
pp = pt.groupby(["GEOID20", "year", "season", "temperature_measure"])[["urban_sum_wT", "urban_valid_weight_m2"]].sum()
pp["pooled"] = pp.urban_sum_wT / pp.urban_valid_weight_m2
g7 = ut.set_index(["GEOID20", "year", "season", "temperature_measure"])[["urban_all_temperature_c"]].join(pp.pooled)
g7["d"] = g7.pooled - g7.urban_all_temperature_c
R["g7_preview_max_abs_pooled_minus_ua"] = float(g7.d.abs().max())
write_json_atomic(f"{qdir}/stage2_columbus.json", R)
log.info("done; tables: %s", json.dumps({k: v for k, v in R["tables"].items() if "columbus" not in k}))
