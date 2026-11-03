"""Stage 1 worker: one ESA WorldCover tile -> additive partial sums of composite NDVI/EVI per zone.

Follows urbveg_worker.process_tile: same 3000-px windows, windows no UA intersects are skipped, same STRtree
query / safe_clip / features.rasterize (all_touched=False) and the same spherical row areas (row_areas_m2,
R = 6,371,007.181 m). The vegetation pieces and UAs are taken from the vegetation run's own files and queried
from their own trees, so they are burned in exactly the vegetation order; the 527 new pieces / 22 new UAs
(pieces_all / uas_all rows beyond the vegetation files) are appended after them in the same rasterize call.
New pieces never intersect old ones (checked in extend_domains.py), so every existing piece owns exactly the
same 10 m pixels as in the vegetation product.

Per-pixel value: the 30 m composite cell containing the pixel centre (ndvi_common.CompositeLookup;
i = floor(lon/s), j = floor(-lat/s)). Only pixels inside a piece or UA are looked up; a needed pixel whose chunk
was never fetched raises MissingChunkError (an error, not a missing value). Cells with no clear observation are
NaN: they count in total_w but not in valid_w.

Zones (WorldCover masks of the vegetation product; snow/ice 70 kept):
  urban  per piece   piece > 0, WC not in {0, 80}
  rural  per UA      UA > 0,    WC not in {0, 50, 80}
  ua_all per UA      UA > 0,    WC not in {0, 80}
Per zone: total_w = sum of pixel areas; per composite c in NDVI_JJA, NDVI_DJF, EVI_JJA, EVI_DJF:
valid_w_c = sum of areas with a finite value, sum_wv_c = sum area*value, sum_wclear_c = sum area*clear_obs over
valid pixels; distinct_cells = number of distinct 30 m cells with w > 0 (exact within a tile; summing tiles
double-counts cells cut by tile edges, so the reduced value is an upper bound).
Diagnostics: per piece valid/water/built/nodata m2 (as the vegetation diag); per UA pixel/nodata/built/water/snow m2.
Output: data/interim/partials/<tile>.parquet (.part + rename; skipped if a partial with the same signature exists).

Usage: python ndvi_worker.py <tile> [<tile> ...] [--force] [--partials-dir DIR]
"""
import argparse
import json
import math
import os
import resource
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import shapely
from rasterio import features

from ndvi_common import (COMPOSITES, NDVI_ROOT, CompositeGrid, CompositeLookup, comp_name, load_config,
                         setup_logging)

import urbveg_common  # noqa: E402  (path set by ndvi_common)
from urbveg_worker import Context, iter_windows, row_areas_m2, safe_clip  # noqa: E402

CODE_VERSION = "1.0.0"
KINDS = ("urban", "rural", "ua_all")
COMPS = [comp_name(i, s) for i, s in COMPOSITES]          # NDVI_JJA, NDVI_DJF, EVI_JJA, EVI_DJF
OFF = np.int64(1 << 20)


def kind_masks(wc, rules):
    ex = lambda codes: ~np.isin(wc, np.asarray(codes, dtype=wc.dtype))  # noqa: E731
    return {"urban": ex(rules["urban_excluded"]), "rural": ex(rules["rural_excluded"]),
            "ua_all": ex(rules["ua_all_excluded"])}


def cell_keys(zone_ids, ci, rj):
    """int64 key of (zone, global cell column, global cell row); zone < 2**21, |ci|, |rj| < 2**20."""
    return (zone_ids.astype(np.int64) << 42) | ((ci.astype(np.int64) + OFF) << 21) | (rj.astype(np.int64) + OFF)


def accumulate_window(wc, piece, ua, area_row, vals, piece_ids, ua_ids, rules, ci=None, rj=None):
    """Sums for one window.
    wc (H,W) uint8; piece/ua (H,W) local indices (0 = none); area_row (H,) m2; vals: dict with (H,W) arrays
    NDVI_JJA, NDVI_DJF, EVI_JJA, EVI_DJF (float, NaN = missing) and clear_JJA, clear_DJF (int);
    piece_ids/ua_ids map local index -> global id (entry 0 unused); ci (W,), rj (H,) global cell column/row
    (optional; for distinct cells).
    Returns (DataFrame of zone sums, {kind: unique cell keys}, DataFrame of diagnostics)."""
    w = np.broadcast_to(np.asarray(area_row, dtype=np.float64)[:, None], wc.shape)
    piece_ids = np.asarray(piece_ids, dtype=np.int64)
    ua_ids = np.asarray(ua_ids, dtype=np.int64)
    masks = kind_masks(wc, rules)
    rows, keys = [], {}
    for kind in KINDS:
        loc, ids = (piece, piece_ids) if kind == "urban" else (ua, ua_ids)
        m = (loc > 0) & masks[kind]
        if not m.any():
            continue
        z = loc[m].astype(np.int64)
        ww = w[m]
        n = ids.size
        d = {"total_w": np.bincount(z, weights=ww, minlength=n)}
        for c in COMPS:
            v = vals[c][m].astype(np.float64)
            ok = np.isfinite(v)
            clear = vals["clear_" + c.split("_")[1]][m].astype(np.float64)
            d[f"valid_w_{c}"] = np.bincount(z[ok], weights=ww[ok], minlength=n)
            d[f"sum_wv_{c}"] = np.bincount(z[ok], weights=ww[ok] * v[ok], minlength=n)
            d[f"sum_wclear_{c}"] = np.bincount(z[ok], weights=ww[ok] * clear[ok], minlength=n)
        nz = np.nonzero(np.bincount(z, minlength=n))[0]
        nz = nz[nz > 0]
        df = pd.DataFrame({k: v[nz] for k, v in d.items()})
        df.insert(0, "zone_id", ids[nz])
        df.insert(0, "kind", kind)
        rows.append(df)
        if ci is not None:
            # drop pixels equal to their left or upper neighbour (same zone, same cell) before np.unique
            sel = m.copy()
            sl = np.zeros_like(sel)
            sl[:, 1:] = (loc[:, 1:] == loc[:, :-1]) & (ci[1:] == ci[:-1])[None, :] & m[:, :-1]
            su = np.zeros_like(sel)
            su[1:, :] = (loc[1:, :] == loc[:-1, :]) & (rj[1:] == rj[:-1])[:, None] & m[:-1, :]
            sel &= ~(sl | su)
            rr, cc = np.nonzero(sel)
            keys[kind] = np.unique(cell_keys(ids[loc[rr, cc]], ci[cc], rj[rr]))
    diag = []
    inp = piece > 0
    if inp.any():
        pp, ww, wcp = piece[inp].astype(np.int64), w[inp], wc[inp]
        dd = {name: np.bincount(pp[s], weights=ww[s], minlength=piece_ids.size)
              for name, s in [("valid_m2", wcp != rules["nodata"]), ("water_m2", wcp == rules["water"]),
                              ("built_m2", wcp == rules["built"]), ("nodata_m2", wcp == rules["nodata"])]}
        nz = np.nonzero(np.bincount(pp, minlength=piece_ids.size))[0]
        nz = nz[nz > 0]
        df = pd.DataFrame({k: v[nz] for k, v in dd.items()})
        df.insert(0, "zone_id", piece_ids[nz]); df.insert(0, "kind", "diag_piece")
        diag.append(df)
    inu = ua > 0
    if inu.any():
        uu, ww, wcu = ua[inu].astype(np.int64), w[inu], wc[inu]
        dd = {name: np.bincount(uu[s], weights=ww[s], minlength=ua_ids.size)
              for name, s in [("pixel_m2", np.ones(uu.shape, bool)), ("nodata_m2", wcu == rules["nodata"]),
                              ("built_m2", wcu == rules["built"]), ("water_m2", wcu == rules["water"]),
                              ("snow_ice_m2", wcu == 70)]}
        nz = np.nonzero(np.bincount(uu, minlength=ua_ids.size))[0]
        nz = nz[nz > 0]
        df = pd.DataFrame({k: v[nz] for k, v in dd.items()})
        df.insert(0, "zone_id", ua_ids[nz]); df.insert(0, "kind", "diag_ua")
        diag.append(df)
    zs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["kind", "zone_id", "total_w"])
    dg = pd.concat(diag, ignore_index=True) if diag else pd.DataFrame(columns=["kind", "zone_id"])
    return zs, keys, dg


# ---------------------------------------------------------------------------- tile processing
class NDVIContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vcfg = urbveg_common.load_config(cfg["paths"]["urbveg_config"], run="census_tracts_ua")
        assert self.vcfg["run"]["domain_index_col"] == "ua_idx"
        self.vctx = Context(self.vcfg)            # vegetation pieces/UAs: same files, same columns, same order
        dom = cfg["domains"]
        pa_ = gpd.read_parquet(dom["pieces_all"])[["piece_id", "ua_idx", "geometry"]].rename(columns={"ua_idx": "dom_idx"})
        ua_ = gpd.read_parquet(dom["uas_all"])[["ua_idx", "geometry"]].rename(columns={"ua_idx": "dom_idx"})
        nv, nu = dom["n_pieces_veg"], dom["n_uas_veg"]
        assert len(self.vctx.pieces) == nv and len(self.vctx.uas) == nu
        assert (pa_.piece_id.values[:nv] == self.vctx.pieces.piece_id.values).all()
        assert (ua_.dom_idx.values[:nu] == self.vctx.uas.dom_idx.values).all()
        self.new_pieces = pa_.iloc[nv:].reset_index(drop=True)
        self.new_uas = ua_.iloc[nu:].reset_index(drop=True)
        self.pieces_all_sha = urbveg_common.file_sha256(dom["pieces_all"])
        self.uas_all_sha = urbveg_common.file_sha256(dom["uas_all"])
        self.R_area = float(self.vcfg["pipeline"]["authalic_radius_m"])
        self.window = int(cfg["worker"]["window_px"])
        self.all_touched = bool(cfg["worker"]["rasterize_all_touched"])
        if self.window != int(self.vcfg["pipeline"]["window_px"]) or \
                self.all_touched != bool(self.vcfg["pipeline"]["rasterize_all_touched"]):
            raise SystemExit("window_px / all_touched must equal the vegetation run's settings")
        self.rules = cfg["classes"]
        self.grid = CompositeGrid.from_cfg(cfg)

    def signature(self):
        return {"code_version": CODE_VERSION, "pieces_veg_sha256": self.vctx.pieces_sha,
                "uas_veg_sha256": self.vctx.uas_sha, "pieces_all_sha256": self.pieces_all_sha,
                "uas_all_sha256": self.uas_all_sha, "window_px": self.window, "all_touched": self.all_touched,
                "area_radius_m": self.R_area, "rules": self.rules, "grid_s": self.grid.s, "chunk_cells": self.grid.C,
                "composites": self.cfg["paths"]["composites"], "seasons": self.cfg["seasons"], "year": self.cfg["year"]}


def _subset(gdf, tbox):
    t = gdf[gdf.intersects(tbox)]
    return t, shapely.STRtree(t.geometry.values)


def rasterize_window(src, win, sets, all_touched):
    """sets: [(ua_t, ua_tree, pc_t, pc_tree) for the vegetation layer, then the new layer]. Same query, clip and
    rasterize calls as urbveg_worker.process_tile; the shapes of the vegetation layer come first, in its order.
    Returns None if no UA intersects the window."""
    T = src.transform
    wb = rasterio.windows.bounds(win, T)
    wbox = shapely.box(*wb)
    q = [(ua_tree.query(wbox, predicate="intersects"), pc_tree.query(wbox, predicate="intersects"))
         for _, ua_tree, _, pc_tree in sets]
    if sum(ui.size for ui, _ in q) == 0:
        return None
    wt = src.window_transform(win)
    H, W = int(win.height), int(win.width)
    ug, pg, uid, pid = [], [], [np.zeros(1, np.int64)], [np.zeros(1, np.int64)]
    for (ua_t, _, pc_t, _), (ui, pi) in zip(sets, q):
        ug.append(safe_clip(np.asarray(ua_t.geometry.values[ui]), wb))
        pg.append(safe_clip(np.asarray(pc_t.geometry.values[pi]), wb))
        uid.append(ua_t.dom_idx.values[ui].astype(np.int64))
        pid.append(pc_t.piece_id.values[pi].astype(np.int64))
    ug, pg = np.concatenate(ug), np.concatenate(pg)
    uid, pid = np.concatenate(uid), np.concatenate(pid)
    ua_r = features.rasterize(zip(ug, np.arange(1, ug.size + 1)), out_shape=(H, W), transform=wt,
                              fill=0, dtype="int32", all_touched=all_touched)
    pc_r = (features.rasterize(zip(pg, np.arange(1, pg.size + 1)), out_shape=(H, W), transform=wt,
                               fill=0, dtype="int32", all_touched=all_touched) if pg.size else np.zeros((H, W), np.int32))
    return wt, ua_r, pc_r, pid, uid


def process_tile(ctx, tile, qa=None):
    vcfg, vctx, grid = ctx.vcfg, ctx.vctx, ctx.grid
    wcp = os.path.join(vcfg["paths"]["worldcover_dir"], vcfg["worldcover"]["file_pattern"].format(tile=tile))
    lut = CompositeLookup(ctx.cfg["paths"]["composites"], grid, ctx.cfg["seasons"], ctx.cfg["worker"]["chunk_cache"])
    t_tile = time.perf_counter()
    zsum, dsum, wstats = [], [], []
    keys = {k: [] for k in KINDS}
    with rasterio.open(wcp) as src:
        tbox = shapely.box(*src.bounds)
        ua_o, uat_o = _subset(vctx.uas, tbox)
        pcs = vctx.pieces_in(tuple(src.bounds))
        pc_o, pct_o = _subset(pcs, tbox)
        ua_n, uat_n = _subset(ctx.new_uas, tbox)
        pc_n, pct_n = _subset(ctx.new_pieces, tbox)
        sets = [(ua_o, uat_o, pc_o, pct_o), (ua_n, uat_n, pc_n, pct_n)]
        T = src.transform
        lon_c = T.c + (np.arange(src.width) + 0.5) * T.a
        lat_c = T.f + (np.arange(src.height) + 0.5) * T.e
        edge = {"min_lon_edge_dist_deg": float(grid.edge_distance_deg(lon_c, "lon").min()),
                "min_lat_edge_dist_deg": float(grid.edge_distance_deg(lat_c, "lat").min())}
        for win in iter_windows(src.width, src.height, ctx.window):
            t0 = time.perf_counter()
            r = rasterize_window(src, win, sets, ctx.all_touched)
            if r is None:
                continue
            wt, ua_r, pc_r, pid, uid = r
            t_ras = time.perf_counter()
            H, W = int(win.height), int(win.width)
            wc = src.read(1, window=win)
            lons = wt.c + (np.arange(W) + 0.5) * wt.a
            lats = wt.f + (np.arange(H) + 0.5) * wt.e
            need = (ua_r > 0) | (pc_r > 0)
            vals = lut.lookup(lons, lats, need)
            t_val = time.perf_counter()
            area_row = row_areas_m2(wt.f, -wt.e, wt.a, H, ctx.R_area)
            zs, ks, dg = accumulate_window(wc, pc_r, ua_r, area_row, vals, pid, uid, ctx.rules,
                                           grid.col(lons), grid.row(lats))
            zsum.append(zs); dsum.append(dg)
            for k, v in ks.items():
                keys[k].append(v)
            t_acc = time.perf_counter()
            if qa is not None:
                qa(win, wt, wc, pc_r, ua_r, pid, uid, vals)
            wstats.append({"col_off": int(win.col_off), "row_off": int(win.row_off), "h": H, "w": W,
                           "n_ua": int(uid.size - 1), "n_piece": int(pid.size - 1),
                           "rasterize_s": round(t_ras - t0, 3), "values_s": round(t_val - t_ras, 3),
                           "accumulate_s": round(t_acc - t_val, 3), "total_s": round(time.perf_counter() - t0, 3),
                           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)})
    zs = pd.concat(zsum, ignore_index=True) if zsum else pd.DataFrame(columns=["kind", "zone_id", "total_w"])
    zs = zs.groupby(["kind", "zone_id"], as_index=False, sort=True).sum()
    dc = []
    for k, arrs in keys.items():
        if arrs:
            u = np.unique(np.concatenate(arrs))
            z, cnt = np.unique(u >> 42, return_counts=True)
            dc.append(pd.DataFrame({"kind": k, "zone_id": z, "distinct_cells": cnt}))
    if dc:
        zs = zs.merge(pd.concat(dc, ignore_index=True), on=["kind", "zone_id"], how="left")
    dg = pd.concat(dsum, ignore_index=True) if dsum else pd.DataFrame(columns=["kind", "zone_id"])
    dg = dg.groupby(["kind", "zone_id"], as_index=False, sort=True).sum(min_count=1)
    out = pd.concat([zs, dg], ignore_index=True)
    out["zone_id"] = out["zone_id"].astype(np.int64)
    stats = {"tile": tile, "n_windows_total": int(math.ceil(36000 / ctx.window) ** 2),
             "n_windows_processed": len(wstats), "tile_s": round(time.perf_counter() - t_tile, 2),
             "n_uas_in_tile": int(len(ua_o) + len(ua_n)), "n_new_uas_in_tile": int(len(ua_n)),
             "n_pieces_in_tile": int(len(pc_o) + len(pc_n)), "n_new_pieces_in_tile": int(len(pc_n)),
             "n_rows": int(len(out)), "chunks_loaded": len(lut.loaded), **edge,
             "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3), "windows": wstats}
    return out, stats


def partial_path(cfg, tile, pdir=None):
    return os.path.join(pdir or cfg["paths"]["partials_dir"], f"{tile}.parquet")


def partial_is_valid(path, sig):
    if not os.path.exists(path):
        return False
    try:
        m = json.loads((pq.read_schema(path).metadata or {}).get(b"ndvi", b"{}"))
        return m.get("complete") is True and all(m.get(k) == v for k, v in sig.items())
    except Exception:  # noqa: BLE001
        return False


def write_partial(path, df, meta):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), b"ndvi": json.dumps(meta).encode()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(tbl, path + ".part")
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles", nargs="+")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--partials-dir", default=None)
    a = ap.parse_args()
    cfg = load_config()
    log = setup_logging("ndvi_worker", os.path.join(cfg["paths"]["logs"], "worker"))
    ctx = NDVIContext(cfg)
    sig = ctx.signature()
    fails = 0
    for tile in a.tiles:
        path = partial_path(cfg, tile, a.partials_dir)
        if partial_is_valid(path, sig) and not a.force:
            log.info("SKIP %s: valid partial exists", tile)
            continue
        try:
            df, stats = process_tile(ctx, tile)
            if os.path.exists(path):
                os.replace(path, path + f".bak-{time.strftime('%Y%m%dT%H%M%S')}")
            write_partial(path, df, {**sig, "tile": tile, "complete": True, "n_rows": len(df),
                                     "stats": {k: v for k, v in stats.items() if k != "windows"}})
            json.dump(stats, open(os.path.join(cfg["paths"]["logs"], "worker", f"{tile}.stats.json"), "w"), indent=1)
            log.info("OK %s windows=%d t=%.1fs rows=%d chunks=%d peak_rss=%.2fGB edge_min=%.3g/%.3g deg", tile,
                     stats["n_windows_processed"], stats["tile_s"], len(df), stats["chunks_loaded"], stats["peak_rss_gb"],
                     stats["min_lon_edge_dist_deg"], stats["min_lat_edge_dist_deg"])
        except Exception as e:  # noqa: BLE001
            fails += 1
            log.exception("FAIL %s: %s: %s", tile, type(e).__name__, e)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
