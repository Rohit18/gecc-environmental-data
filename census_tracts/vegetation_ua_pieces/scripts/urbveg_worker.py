"""Stage 3 tile worker: one ESA WorldCover tile -> partial class-area sums (Method A).

For each window of the WorldCover tile:
  1. WorldCover; valid = WC != nodata (0).
  2. GLC_FCS30D class by nearest neighbour with integer index math (no warping): each WorldCover pixel
     centre takes the GLC tile whose 5x5 deg NAME-CELL contains it (upper-left naming:
     lon0 = floor(lon/5)*5, lat0 = ceil(lat/5)*5), then col = floor((lon-x0)/dx), row = floor((y0-lat)/dy)
     with that tile's own transform. Missing GLC tile -> class 0 (fill).
  3. Rasterize piece ids and UA indices (all_touched from config; only features intersecting the window).
  4. Pixel area per row on the authalic sphere: A = R^2 * dlon * (sin(lat_top) - sin(lat_bottom)).
  5. Weighted bincount:
       urban (per piece): piece > 0, WC valid, WC != water          -> area by (piece, year, GLC class)
       rural (per UA):    UA > 0, WC valid, WC not in {built, water} -> area by (UA, year, GLC class)
       diag  (per piece): valid / water / built / nodata area, approx. distinct GLC pixels touched
     GLC fill classes stay in the histograms as their own classes.
Windows that no kept UA intersects are skipped (they cannot contribute to any output).
Output: data/interim/partials/<tile>.parquet (written via .part + rename; skipped if it exists and validates).

Usage: python urbveg_worker.py <WC tile> [<WC tile> ...] [--window N] [--force]
"""
import argparse, glob, hashlib, json, math, os, re, resource, sys, time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import shapely
from rasterio import features
from rasterio.windows import Window

from urbveg_common import ROOT, file_sha256, load_config

NCLS = 256


# ---------------------------------------------------------------------------- pure helpers
def row_areas_m2(top_lat, dlat, dlon, nrows, radius):
    """Area (m2) of one pixel in each row; top_lat is the north edge of the first row, dlat > 0 (deg)."""
    tops = np.deg2rad(top_lat - np.arange(nrows) * dlat)
    bots = np.deg2rad(top_lat - (np.arange(nrows) + 1) * dlat)
    return radius ** 2 * math.radians(dlon) * (np.sin(tops) - np.sin(bots))


def glc_tile_name(lon0, lat0):
    ew = "E" if lon0 >= 0 else "W"
    ns = "N" if lat0 >= 0 else "S"
    return f"{ew}{abs(int(lon0))}{ns}{abs(int(lat0))}"


class GLCLookup:
    """Nearest-neighbour GLC class lookup on an arbitrary regular lon/lat grid of pixel centres."""

    def __init__(self, tile_paths, bands, overhang="strict"):
        self.paths = tile_paths          # {tile name: path}
        self.bands = list(bands)         # 1-based band indices, one per year
        self.ds = {}
        # "strict": index straight from the name-cell tile's transform (may hit the tile's last row/col,
        #           the overhang, for centres within 0.000067 deg of a seam).
        # "clamp":  treat each tile's last row and last column as invalid overhang and use the adjacent
        #           row/col of the same tile (the nearest valid GLC pixel to such centres).
        if overhang not in ("strict", "clamp"):
            raise ValueError(overhang)
        self.overhang = overhang

    def _open(self, name):
        if name not in self.ds:
            p = self.paths.get(name)
            self.ds[name] = rasterio.open(p) if p else None
        return self.ds[name]

    def lookup(self, lons, lats, want_pixel_keys=False):
        """lons: (W,) pixel-centre longitudes; lats: (H,) pixel-centre latitudes (north->south).
        Returns classes (n_years, H, W) uint8 and, optionally, (rowkey (H,), colkey (W,)) int64 that
        identify the GLC pixel (tile + row/col) used for each row/column."""
        lon0 = np.floor(lons / 5.0) * 5.0
        lat0 = np.ceil(lats / 5.0) * 5.0
        out = np.zeros((len(self.bands), lats.size, lons.size), dtype=np.uint8)
        rowkey = np.full(lats.size, -1, dtype=np.int64)
        colkey = np.full(lons.size, -1, dtype=np.int64)
        for la in np.unique(lat0):
            rsel = np.nonzero(lat0 == la)[0]
            for lo in np.unique(lon0):
                csel = np.nonzero(lon0 == lo)[0]
                ds = self._open(glc_tile_name(lo, la))
                if ds is None:
                    continue            # missing tile -> class 0
                t = ds.transform
                ci = np.floor((lons[csel] - t.c) / t.a).astype(np.int64)
                ri = np.floor((t.f - lats[rsel]) / -t.e).astype(np.int64)
                if self.overhang == "clamp":
                    ci = np.minimum(ci, ds.width - 2)
                    ri = np.minimum(ri, ds.height - 2)
                if ci.min() < 0 or ci.max() >= ds.width or ri.min() < 0 or ri.max() >= ds.height:
                    raise ValueError(f"index outside GLC tile {glc_tile_name(lo, la)}")
                c0, c1, r0, r1 = ci.min(), ci.max(), ri.min(), ri.max()
                blk = ds.read(self.bands, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
                out[:, rsel[:, None], csel[None, :]] = blk[:, (ri - r0)[:, None], (ci - c0)[None, :]]
                lat_code = int((la + 90) // 5); lon_code = int((lo + 180) // 5)
                rowkey[rsel] = lat_code * 32768 + ri
                colkey[csel] = lon_code * 32768 + ci
        return (out, rowkey, colkey) if want_pixel_keys else out


def accumulate(wc, glc, piece, ua, area_row, n_piece, n_ua, built=50, water=80, nodata=0,
               rowkey=None, colkey=None):
    """Core sums for one window. wc (H,W) uint8; glc (Y,H,W) uint8; piece/ua (H,W) local ids (0 = none);
    area_row (H,) m2. Returns dict of arrays indexed by local id (0..n)."""
    Y = glc.shape[0]
    w = np.broadcast_to(area_row[:, None], wc.shape)
    valid = wc != nodata
    um = (piece > 0) & valid & (wc != water)
    rm = (ua > 0) & valid & (wc != built) & (wc != water)
    res = {"urban": np.zeros((n_piece + 1, Y, NCLS)), "rural": np.zeros((n_ua + 1, Y, NCLS))}
    pu, wu = piece[um].astype(np.int64), w[um]
    ur, wr = ua[rm].astype(np.int64), w[rm]
    for y in range(Y):
        res["urban"][:, y, :] = np.bincount(pu * NCLS + glc[y][um], weights=wu,
                                            minlength=(n_piece + 1) * NCLS).reshape(n_piece + 1, NCLS)
        res["rural"][:, y, :] = np.bincount(ur * NCLS + glc[y][rm], weights=wr,
                                            minlength=(n_ua + 1) * NCLS).reshape(n_ua + 1, NCLS)
    inp = piece > 0
    pp, ww, wcp = piece[inp].astype(np.int64), w[inp], wc[inp]
    for name, m in [("valid_m2", wcp != nodata), ("water_m2", wcp == water),
                    ("built_m2", wcp == built), ("nodata_m2", wcp == nodata)]:
        res[name] = np.bincount(pp[m], weights=ww[m], minlength=n_piece + 1)
    # approximate distinct GLC pixels per piece (valid WC pixels only; dedupe within this window)
    res["glc_px"] = np.zeros(n_piece + 1, dtype=np.int64)
    if rowkey is not None:
        sel = inp & valid & (rowkey >= 0)[:, None] & (colkey >= 0)[None, :]   # skip missing GLC tiles
        # drop pixels identical to their left neighbour (same piece and same GLC column) before unique
        same_left = np.zeros_like(sel)
        same_left[:, 1:] = (piece[:, 1:] == piece[:, :-1]) & (colkey[1:] == colkey[:-1])[None, :]
        sel &= ~same_left
        rr, cc = np.nonzero(sel)
        key = (piece[rr, cc].astype(np.int64) << 44) | (rowkey[rr] << 22) | colkey[cc]
        u = np.unique(key)
        res["glc_px"] = np.bincount((u >> 44).astype(np.int64), minlength=n_piece + 1)
    return res


# ---------------------------------------------------------------------------- tile processing
def safe_clip(geoms, bounds):
    """clip_by_rect; geometries GEOS cannot clip (degenerate zero-area sliver pieces) are passed unclipped
    (the rasterizer clips to the window itself)."""
    try:
        return shapely.clip_by_rect(geoms, *bounds)
    except shapely.errors.GEOSException:
        out = np.empty(len(geoms), dtype=object)
        for i, g in enumerate(geoms):
            try:
                out[i] = shapely.clip_by_rect(g, *bounds)
            except shapely.errors.GEOSException:
                out[i] = g
        return out


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class Context:
    def __init__(self, cfg):
        self.cfg = cfg
        g = cfg["glc_fcs30d"]
        rec = g["zenodo_record"]
        pat = g["records"][rec]["file_pattern_annual"]
        rx = re.compile(re.escape(pat).replace(r"\{tile\}", r"(?P<tile>[EW]\d+[NS]\d+)") + "$")
        self.glc_paths = {}
        for p in glob.glob(os.path.join(cfg["paths"]["glc_dir"], "*.tif")):
            m = rx.match(os.path.basename(p))
            if m:
                self.glc_paths[m["tile"]] = p
        self.years = list(g["years"])
        self.bands = [g["band_year_annual"][y] for y in self.years]
        run = cfg["run"]
        self.run_name = run["name"]
        dom = cfg["layers"]["domains"][run["domain"]]
        idx = run["domain_index_col"]
        self.pieces_path = cfg["pipeline"]["pieces_path"]
        # large piece sets (fishnet) are read per tile with a bbox filter (GeoParquet covering column)
        import pyarrow.parquet as _pq
        self.lazy_pieces = "bbox" in _pq.read_schema(self.pieces_path).names
        self.pieces = None if self.lazy_pieces else gpd.read_parquet(self.pieces_path)[["piece_id", idx, "geometry"]].rename(columns={idx: "dom_idx"})
        self.uas = gpd.read_parquet(dom["kept_path"])[[idx, "geometry"]].rename(columns={idx: "dom_idx"})
        self.pieces_sha = file_sha256(self.pieces_path)
        self.uas_sha = file_sha256(dom["kept_path"])
        wc = cfg["worldcover"]
        self.built, self.water, self.nodata = wc["codes"]["built_up"], wc["codes"]["water"], wc["codes"]["nodata"]

    def signature(self, window):
        return {"years": self.years, "bands": self.bands, "pieces_sha256": self.pieces_sha,
                "uas_sha256": self.uas_sha, "glc_record": self.cfg["glc_fcs30d"]["zenodo_record"],
                "glc_tiles_available": len(self.glc_paths), "window_px": window,
                "all_touched": bool(self.cfg["pipeline"]["rasterize_all_touched"]),
                "glc_overhang": self.cfg["pipeline"].get("glc_overhang", "strict"),
                "run": self.run_name}

    def pieces_in(self, bounds):
        if self.lazy_pieces:
            idx = self.cfg["run"]["domain_index_col"]
            p = gpd.read_parquet(self.pieces_path, bbox=bounds, columns=["piece_id", idx, "geometry"])
            return p.rename(columns={idx: "dom_idx"})
        return self.pieces


def partial_path(cfg, tile):
    return os.path.join(cfg["pipeline"]["partials_dir"], f"{tile}.parquet")


def partial_is_valid(path, sig):
    if not os.path.exists(path):
        return False
    try:
        md = pq.read_schema(path).metadata or {}
        m = json.loads(md.get(b"urbveg", b"{}"))
        keys = ("years", "bands", "pieces_sha256", "uas_sha256", "glc_record", "all_touched", "glc_overhang")
        run_ok = m.get("run", "census_tracts_ua") == sig.get("run", "census_tracts_ua")   # pre-profile partials = census
        return m.get("complete") is True and run_ok and all(m.get(k) == sig[k] for k in keys)
    except Exception:
        return False


def iter_windows(width, height, size):
    for r in range(0, height, size):
        for c in range(0, width, size):
            yield Window(c, r, min(size, width - c), min(size, height - r))


def process_tile(ctx, tile, window_px, qa=None, log=print):
    cfg = ctx.cfg
    wcp = os.path.join(cfg["paths"]["worldcover_dir"], cfg["worldcover"]["file_pattern"].format(tile=tile))
    R = cfg["pipeline"]["authalic_radius_m"]
    at = bool(cfg["pipeline"]["rasterize_all_touched"])
    lut = GLCLookup(ctx.glc_paths, ctx.bands, cfg["pipeline"].get("glc_overhang", "strict"))
    t_tile = time.perf_counter()
    rows, wstats = [], []
    with rasterio.open(wcp) as src:
        tb = src.bounds
        tbox = shapely.box(*tb)
        ua_t = ctx.uas[ctx.uas.intersects(tbox)]
        pcs = ctx.pieces_in(tuple(tb))
        pc_t = pcs[pcs.intersects(tbox)]
        ua_tree, pc_tree = shapely.STRtree(ua_t.geometry.values), shapely.STRtree(pc_t.geometry.values)
        T = src.transform
        for win in iter_windows(src.width, src.height, window_px):
            t0 = time.perf_counter()
            wb = rasterio.windows.bounds(win, T)
            wbox = shapely.box(*wb)
            ui = ua_tree.query(wbox, predicate="intersects")
            if ui.size == 0:
                continue
            pi = pc_tree.query(wbox, predicate="intersects")
            wt = src.window_transform(win)
            H, W = int(win.height), int(win.width)
            wc = src.read(1, window=win)
            t_read = time.perf_counter()
            lons = wt.c + (np.arange(W) + 0.5) * wt.a
            lats = wt.f + (np.arange(H) + 0.5) * wt.e
            glc, rowkey, colkey = lut.lookup(lons, lats, want_pixel_keys=True)
            t_glc = time.perf_counter()
            ug = safe_clip(np.asarray(ua_t.geometry.values[ui]), wb)
            pg = safe_clip(np.asarray(pc_t.geometry.values[pi]), wb)
            ua_r = features.rasterize(zip(ug, np.arange(1, ui.size + 1)), out_shape=(H, W), transform=wt,
                                      fill=0, dtype="int32", all_touched=at)
            pc_r = (features.rasterize(zip(pg, np.arange(1, pi.size + 1)), out_shape=(H, W), transform=wt,
                                       fill=0, dtype="int32", all_touched=at) if pi.size else np.zeros((H, W), np.int32))
            t_ras = time.perf_counter()
            area_row = row_areas_m2(wt.f, -wt.e, wt.a, H, R)
            res = accumulate(wc, glc, pc_r, ua_r, area_row, pi.size, ui.size, ctx.built, ctx.water, ctx.nodata,
                             rowkey, colkey)
            t_acc = time.perf_counter()
            pid = np.concatenate([[0], pc_t.piece_id.values[pi]])
            uid = np.concatenate([[0], ua_t.dom_idx.values[ui]])
            for kind, ids in (("urban", pid), ("rural", uid)):
                a = res[kind]
                li, yi, ci = np.nonzero(a)
                m = li > 0
                rows.append(pd.DataFrame({"kind": kind, "id": ids[li[m]].astype(np.int32),
                                          "year": np.array(ctx.years, dtype=np.int16)[yi[m]],
                                          "glc_class": ci[m].astype(np.int16), "area_m2": a[li[m], yi[m], ci[m]]}))
            d = pd.DataFrame({"kind": "diag", "id": pid[1:].astype(np.int32),
                              **{k: res[k][1:] for k in ("valid_m2", "water_m2", "built_m2", "nodata_m2", "glc_px")}})
            rows.append(d[(d[["valid_m2", "nodata_m2"]].sum(axis=1) > 0)])
            if qa is not None:
                qa(win, wt, wc, glc, pc_r, ua_r, pid, uid)
            t_end = time.perf_counter()
            wstats.append({"col_off": int(win.col_off), "row_off": int(win.row_off), "h": H, "w": W,
                           "n_ua": int(ui.size), "n_piece": int(pi.size),
                           "read_s": round(t_read - t0, 3), "glc_s": round(t_glc - t_read, 3),
                           "rasterize_s": round(t_ras - t_glc, 3), "accumulate_s": round(t_acc - t_ras, 3),
                           "total_s": round(t_end - t0, 3),
                           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)})
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        {"kind": [], "id": [], "year": [], "glc_class": [], "area_m2": []})
    # sum across windows (pieces crossing window edges)
    cls = df[df.kind != "diag"].groupby(["kind", "id", "year", "glc_class"], as_index=False)["area_m2"].sum()
    dg = df[df.kind == "diag"].groupby(["kind", "id"], as_index=False)[
        ["valid_m2", "water_m2", "built_m2", "nodata_m2", "glc_px"]].sum() if (df.kind == "diag").any() else pd.DataFrame()
    out = pd.concat([cls, dg], ignore_index=True)
    for c in ("valid_m2", "water_m2", "built_m2", "nodata_m2", "area_m2"):
        if c not in out:
            out[c] = np.nan
    if "glc_px" not in out:
        out["glc_px"] = np.nan
    stats = {"tile": tile, "n_windows_total": int(math.ceil(36000 / window_px) ** 2),
             "n_windows_processed": len(wstats), "tile_s": round(time.perf_counter() - t_tile, 2),
             "n_uas_in_tile": int(len(ua_t)), "n_pieces_in_tile": int(len(pc_t)), "windows": wstats,
             "glc_tiles_used": sorted(k for k, v in lut.ds.items() if v is not None),
             "glc_tiles_missing": sorted(k for k, v in lut.ds.items() if v is None)}
    return out, stats


def write_partial(path, df, meta):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), b"urbveg": json.dumps(meta).encode()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(tbl, path + ".part")
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles", nargs="+")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recompute even if a valid partial exists (old file kept as .bak)")
    a = ap.parse_args()
    cfg = load_config()
    window = a.window or cfg["pipeline"]["window_px"]
    ctx = Context(cfg)
    wl = f"{ROOT}/logs/worker" if ctx.run_name == "census_tracts_ua" else f"{ROOT}/logs/worker/{ctx.run_name}"
    os.makedirs(wl, exist_ok=True)
    fails = 0
    for tile in a.tiles:
        path = partial_path(cfg, tile)
        sig = ctx.signature(window)
        logf = open(f"{wl}/{tile}.log", "a")
        log = lambda *x: (print(*x, file=logf, flush=True), print(*x, flush=True))
        if partial_is_valid(path, sig) and not a.force:
            log(f"{time.strftime('%FT%T')} SKIP {tile}: valid partial exists"); continue
        if os.path.exists(path):
            os.replace(path, path + f".bak-{time.strftime('%Y%m%dT%H%M%S')}")
        try:
            df, stats = process_tile(ctx, tile, window, log=log)
            write_partial(path, df, {**sig, "tile": tile, "complete": True, "n_rows": len(df),
                                     "stats": {k: v for k, v in stats.items() if k != "windows"}})
            json.dump(stats, open(f"{wl}/{tile}.stats.json", "w"), indent=1)
            log(f"{time.strftime('%FT%T')} OK {tile} windows={stats['n_windows_processed']} "
                f"t={stats['tile_s']}s rows={len(df)} missing_glc={stats['glc_tiles_missing']}")
        except Exception as e:
            fails += 1
            log(f"{time.strftime('%FT%T')} FAIL {tile}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
