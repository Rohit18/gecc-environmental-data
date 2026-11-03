"""Stage 1 weight worker: one ESA WorldCover tile -> per-zone temperature-cell weights (true area, m2).

For each window of the tile (same windows as the vegetation run; windows no kept UA intersects are skipped):
  1. WorldCover (uint8).
  2. piece_id and UA index rasterized exactly as urbveg_worker.process_tile does (same Context: same
     pieces.parquet / uas_kept.parquet, same STRtree query, same safe_clip, same rasterize call and
     all_touched=False), so each piece owns the same 10 m pixels as in the vegetation product.
  3. True pixel area per row: urbveg_worker.row_areas_m2 with the vegetation config's authalic radius.
  4. Temperature cell of each pixel centre: x = R*lon*cos(lat), y = R*lat; col = floor((x-x0)/dx),
     row = floor((y0-y)/dy); cell_id = row*width + col; -1 off the grid (uhi_common.Grid.cell_ids).
  5. np.unique on int64 keys zone_id*Ncells + cell_id, then np.bincount of pixel areas:
       urban               (piece_id, cell)  piece > 0, WC not in {0,70,80}
       rural               (ua_idx, cell)    ua > 0,    WC not in {0,50,70,80}
       ua_all              (ua_idx, cell)    ua > 0,    WC not in {0,70,80}
       ua_built            (ua_idx, cell)    ua > 0,    WC == 50
       ua_nowater_snowkept (ua_idx, cell)    ua > 0,    WC not in {0,80}   (September-release regression only)
     Pixels of a kind that fall off the grid go to "<kind>_outside" rows (cell_id = -1).
     Diagnostics per piece and per UA (cell_id = -1): diag_{piece,ua}_{total,wc0,wc50,wc70,wc80}.
Output: data/interim/partials/<tile>.parquet (kind, zone_id, cell_id, area_m2), written via .part + rename;
skipped when a partial with the same signature exists (resumable). --cell-diag also writes per-cell
all-pixel class areas for every window of the tile (Stage 2a check) to partials_celldiag/<tile>.parquet.

Usage: python uhi_weights_worker.py <tile> [<tile> ...] [--force] [--cell-diag] [--qa-bbox W S E N --qa-name X]
"""
import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import shapely
from rasterio import features

from uhi_common import UHI_ROOT, Grid, load_config, setup_logging

import urbveg_common  # noqa: E402  (path set by uhi_common)
from urbveg_worker import Context, iter_windows, row_areas_m2, safe_clip  # noqa: E402

CODE_VERSION = "1.0.0"
WEIGHT_KINDS = ("urban", "rural", "ua_all", "ua_built", "ua_nowater_snowkept")
DIAG_CLASSES = (0, 50, 70, 80)


def kind_masks(wc, rules):
    """Boolean pixel masks by class rule, independent of zones."""
    ex = lambda codes: ~np.isin(wc, np.asarray(codes, dtype=wc.dtype))  # noqa: E731
    return {"urban": ex(rules["urban_excluded"]), "rural": ex(rules["rural_excluded"]),
            "ua_all": ex(rules["urban_excluded"]), "ua_built": np.isin(wc, np.asarray(rules["built_only"], dtype=wc.dtype)),
            "ua_nowater_snowkept": ex(rules["regression_excluded"])}


def accumulate_weights(wc, piece, ua, cell, area_row, piece_ids, ua_ids, ncells, rules):
    """Weights for one window.
    wc (H,W) uint8; piece/ua (H,W) local indices (0 = none); cell (H,W) int64 cell ids (-1 off grid);
    area_row (H,) m2; piece_ids/ua_ids map local index -> global id (entry 0 unused); ncells = grid size.
    Returns a DataFrame kind, zone_id, cell_id, area_m2 (weights, outside-grid and diagnostic rows)."""
    w = np.broadcast_to(np.asarray(area_row, dtype=np.float64)[:, None], wc.shape)
    piece_ids = np.asarray(piece_ids, dtype=np.int64)
    ua_ids = np.asarray(ua_ids, dtype=np.int64)
    masks = kind_masks(wc, rules)
    out = []
    on = cell >= 0
    for kind in WEIGHT_KINDS:
        loc, ids = (piece, piece_ids) if kind == "urban" else (ua, ua_ids)
        zm = (loc > 0) & masks[kind]
        m = zm & on
        if m.any():
            key = ids[loc[m]] * np.int64(ncells) + cell[m]
            u, inv = np.unique(key, return_inverse=True)
            s = np.bincount(inv.reshape(-1), weights=w[m])
            out.append(pd.DataFrame({"kind": kind, "zone_id": u // ncells, "cell_id": u % ncells, "area_m2": s}))
        mo = zm & ~on
        if mo.any():
            s = np.bincount(loc[mo], weights=w[mo], minlength=ids.size)
            nz = np.nonzero(s)[0]
            out.append(pd.DataFrame({"kind": f"{kind}_outside", "zone_id": ids[nz], "cell_id": np.int64(-1),
                                     "area_m2": s[nz]}))
    for zname, loc, ids in (("piece", piece, piece_ids), ("ua", ua, ua_ids)):
        inz = loc > 0
        if not inz.any():
            continue
        lz, wz, cz = loc[inz], w[inz], wc[inz]
        for name, sel in [("total", None)] + [(f"wc{c}", c) for c in DIAG_CLASSES]:
            mm = np.ones(lz.shape, bool) if sel is None else (cz == sel)
            s = np.bincount(lz[mm], weights=wz[mm], minlength=ids.size)
            nz = np.nonzero(s)[0]
            if nz.size:
                out.append(pd.DataFrame({"kind": f"diag_{zname}_{name}", "zone_id": ids[nz],
                                         "cell_id": np.int64(-1), "area_m2": s[nz]}))
    if not out:
        return pd.DataFrame({"kind": pd.Series(dtype=str), "zone_id": pd.Series(dtype=np.int64),
                             "cell_id": pd.Series(dtype=np.int64), "area_m2": pd.Series(dtype=np.float64)})
    df = pd.concat(out, ignore_index=True)
    df["zone_id"] = df["zone_id"].astype(np.int64)
    df["cell_id"] = df["cell_id"].astype(np.int64)
    return df


def cell_diag(wc, cell, area_row):
    """Per-cell all-pixel areas (every pixel of the window): total, WC 0, 50, 80."""
    w = np.broadcast_to(np.asarray(area_row, dtype=np.float64)[:, None], wc.shape)
    on = cell >= 0
    u, inv = np.unique(cell[on], return_inverse=True)
    inv = inv.reshape(-1)
    res = {"cell_id": u, "total_m2": np.bincount(inv, weights=w[on], minlength=u.size)}
    for c in (0, 50, 80):
        m = (wc[on] == c)
        res[f"wc{c}_m2"] = np.bincount(inv[m], weights=w[on][m], minlength=u.size)
    res["outside_m2"] = np.full(u.size, np.nan)
    df = pd.DataFrame(res)
    off = float(w[~on].sum())
    return df, off


# ---------------------------------------------------------------------------- tile processing
class UHIContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vcfg = urbveg_common.load_config(cfg["paths"]["urbveg_config"], run="census_tracts_ua")
        assert self.vcfg["run"]["domain_index_col"] == "ua_idx"
        self.vctx = Context(self.vcfg)                       # vegetation run's pieces/UAs (same files, same columns)
        self.grid = Grid.from_cfg(cfg)
        self.R_area = float(self.vcfg["pipeline"]["authalic_radius_m"])
        self.window = int(cfg["worker"]["window_px"])
        self.all_touched = bool(cfg["worker"]["rasterize_all_touched"])
        if self.window != int(self.vcfg["pipeline"]["window_px"]) or \
                self.all_touched != bool(self.vcfg["pipeline"]["rasterize_all_touched"]):
            raise SystemExit("window_px / all_touched must equal the vegetation run's settings")
        self.rules = cfg["classes"]

    def signature(self):
        return {"code_version": CODE_VERSION, "pieces_sha256": self.vctx.pieces_sha, "uas_sha256": self.vctx.uas_sha,
                "grid": self.grid.as_dict(), "window_px": self.window, "all_touched": self.all_touched,
                "area_radius_m": self.R_area, "rules": self.rules}


def rasterize_window(src, win, ua_t, pc_t, ua_tree, pc_tree, all_touched):
    """Same steps and calls as urbveg_worker.process_tile (rasterize UAs and pieces for one window).
    Returns None if no kept UA intersects the window."""
    T = src.transform
    wb = rasterio.windows.bounds(win, T)
    wbox = shapely.box(*wb)
    ui = ua_tree.query(wbox, predicate="intersects")
    if ui.size == 0:
        return None
    pi = pc_tree.query(wbox, predicate="intersects")
    wt = src.window_transform(win)
    H, W = int(win.height), int(win.width)
    ug = safe_clip(np.asarray(ua_t.geometry.values[ui]), wb)
    pg = safe_clip(np.asarray(pc_t.geometry.values[pi]), wb)
    ua_r = features.rasterize(zip(ug, np.arange(1, ui.size + 1)), out_shape=(H, W), transform=wt,
                              fill=0, dtype="int32", all_touched=all_touched)
    pc_r = (features.rasterize(zip(pg, np.arange(1, pi.size + 1)), out_shape=(H, W), transform=wt,
                               fill=0, dtype="int32", all_touched=all_touched) if pi.size else np.zeros((H, W), np.int32))
    pid = np.concatenate([[0], pc_t.piece_id.values[pi]]).astype(np.int64)
    uid = np.concatenate([[0], ua_t.dom_idx.values[ui]]).astype(np.int64)
    return wt, ua_r, pc_r, pid, uid


def window_cells(grid, wt, H, W):
    lons = np.radians(wt.c + (np.arange(W) + 0.5) * wt.a)
    lats = np.radians(wt.f + (np.arange(H) + 0.5) * wt.e)
    return grid.cell_ids(lons[None, :], lats[:, None])


def process_tile(ctx, tile, cell_diag_mode=False, qa=None):
    vcfg, vctx, grid = ctx.vcfg, ctx.vctx, ctx.grid
    wcp = os.path.join(vcfg["paths"]["worldcover_dir"], vcfg["worldcover"]["file_pattern"].format(tile=tile))
    t_tile = time.perf_counter()
    parts, cdiag, wstats = [], [], []
    off_grid_all = 0.0
    with rasterio.open(wcp) as src:
        tbox = shapely.box(*src.bounds)
        ua_t = vctx.uas[vctx.uas.intersects(tbox)]
        pcs = vctx.pieces_in(tuple(src.bounds))
        pc_t = pcs[pcs.intersects(tbox)]
        ua_tree, pc_tree = shapely.STRtree(ua_t.geometry.values), shapely.STRtree(pc_t.geometry.values)
        for win in iter_windows(src.width, src.height, ctx.window):
            t0 = time.perf_counter()
            r = rasterize_window(src, win, ua_t, pc_t, ua_tree, pc_tree, ctx.all_touched)
            if r is None and not cell_diag_mode:
                continue
            t_ras = time.perf_counter()
            H, W = int(win.height), int(win.width)
            wc = src.read(1, window=win)
            t_read = time.perf_counter()
            wt = src.window_transform(win)
            cell = window_cells(grid, wt, H, W)
            area_row = row_areas_m2(wt.f, -wt.e, wt.a, H, ctx.R_area)
            t_cell = time.perf_counter()
            if cell_diag_mode:
                cd, off = cell_diag(wc, cell, area_row)
                cdiag.append(cd)
                off_grid_all += off
            if r is None:
                continue
            wt, ua_r, pc_r, pid, uid = r
            t_acc0 = time.perf_counter()
            parts.append(accumulate_weights(wc, pc_r, ua_r, cell, area_row, pid, uid, grid.ncells, ctx.rules))
            t_acc = time.perf_counter()
            if qa is not None:
                qa(win, wt, wc, pc_r, ua_r, pid, uid, cell, ctx.rules)
            t_end = time.perf_counter()
            wstats.append({"col_off": int(win.col_off), "row_off": int(win.row_off), "h": H, "w": W,
                           "n_ua": int(uid.size - 1), "n_piece": int(pid.size - 1),
                           "rasterize_s": round(t_ras - t0, 3), "read_s": round(t_read - t_ras, 3),
                           "cell_s": round(t_cell - t_read, 3), "accumulate_s": round(t_acc - t_acc0, 3),
                           "total_s": round(t_end - t0, 3),
                           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)})
    df = pd.concat(parts, ignore_index=True) if parts else accumulate_weights(
        np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.int32), np.zeros((1, 1), np.int32),
        np.zeros((1, 1), np.int64), np.ones(1), [0], [0], grid.ncells, ctx.rules)
    out = df.groupby(["kind", "zone_id", "cell_id"], as_index=False, sort=True)["area_m2"].sum()
    cd = None
    if cell_diag_mode:
        cd = pd.concat(cdiag, ignore_index=True).groupby("cell_id", as_index=False)[
            ["total_m2", "wc0_m2", "wc50_m2", "wc80_m2"]].sum()
    stats = {"tile": tile, "n_windows_total": int(math.ceil(36000 / ctx.window) ** 2),
             "n_windows_processed": len(wstats), "tile_s": round(time.perf_counter() - t_tile, 2),
             "n_uas_in_tile": int(len(ua_t)), "n_pieces_in_tile": int(len(pc_t)), "n_rows": int(len(out)),
             "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
             "off_grid_all_pixels_m2": off_grid_all if cell_diag_mode else None, "windows": wstats}
    return out, cd, stats


def partial_path(cfg, tile, sub="partials"):
    return os.path.join(cfg["paths"]["interim"], sub, f"{tile}.parquet")


def partial_is_valid(path, sig):
    if not os.path.exists(path):
        return False
    try:
        m = json.loads((pq.read_schema(path).metadata or {}).get(b"uhi", b"{}"))
        return m.get("complete") is True and all(m.get(k) == v for k, v in sig.items())
    except Exception:  # noqa: BLE001
        return False


def write_partial(path, df, meta):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), b"uhi": json.dumps(meta).encode()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(tbl, path + ".part")
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles", nargs="+")
    ap.add_argument("--force", action="store_true", help="recompute even if a valid partial exists (old kept as .bak-*)")
    ap.add_argument("--cell-diag", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    log = setup_logging("uhi_weights_worker", os.path.join(cfg["paths"]["logs"], "worker"))
    ctx = UHIContext(cfg)
    sig = ctx.signature()
    fails = 0
    for tile in a.tiles:
        path = partial_path(cfg, tile)
        cpath = partial_path(cfg, tile, "partials_celldiag")
        need_cd = a.cell_diag and not partial_is_valid(cpath, sig)
        if partial_is_valid(path, sig) and not a.force and not need_cd:
            log.info("SKIP %s: valid partial exists", tile)
            continue
        try:
            df, cd, stats = process_tile(ctx, tile, cell_diag_mode=a.cell_diag)
            meta = {**sig, "tile": tile, "complete": True, "n_rows": len(df),
                    "stats": {k: v for k, v in stats.items() if k != "windows"}}
            if os.path.exists(path) and (a.force or not partial_is_valid(path, sig)):
                os.replace(path, path + f".bak-{time.strftime('%Y%m%dT%H%M%S')}")
            if not os.path.exists(path):
                write_partial(path, df, meta)
            if cd is not None:
                write_partial(cpath, cd, {**meta, "n_rows": len(cd)})
            # benchmark stats come from plain runs only; a --cell-diag run (extra work per window) is kept apart
            sp = os.path.join(cfg["paths"]["logs"], "worker", f"{tile}{'.celldiag' if a.cell_diag else ''}.stats.json")
            json.dump(stats, open(sp, "w"), indent=1)
            log.info("OK %s windows=%d t=%.1fs rows=%d peak_rss=%.2fGB", tile, stats["n_windows_processed"],
                     stats["tile_s"], len(df), stats["peak_rss_gb"])
        except Exception as e:  # noqa: BLE001
            fails += 1
            log.exception("FAIL %s: %s: %s", tile, type(e).__name__, e)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
