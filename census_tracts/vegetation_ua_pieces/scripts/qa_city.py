"""Stage 4c QA rasters for one city bbox, built with the same functions as the worker.

Writes to data/interim/check_city/<name>/: piece_id.tif (int32), urban_mask.tif, rural_mask.tif (uint8),
glc_on_wc_<year>.tif (uint8), worldcover.tif, plus PNG previews of the GLC-on-WC grid around any
5-deg GLC seam inside the bbox, and seam_check.json (row-to-row class agreement across the seam vs
elsewhere, fill share near the seam).
Usage: python qa_city.py --tile N39W084 --bbox -83.30 39.70 -82.50 40.40 --name columbus_oh
"""
import argparse, json, os, struct, zlib

import numpy as np
import rasterio
import shapely
from rasterio import features
from rasterio.windows import from_bounds

from urbveg_common import ROOT, load_config
from urbveg_worker import Context, GLCLookup

ap = argparse.ArgumentParser()
ap.add_argument("--tile", required=True)
ap.add_argument("--bbox", nargs=4, type=float, required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--overhang", default=None, help="strict|clamp (default: config pipeline.glc_overhang)")
a = ap.parse_args()
cfg = load_config()
ctx = Context(cfg)
od = f"{ROOT}/data/interim/check_city/{a.name}" if cfg["run"]["name"] == "census_tracts_ua" else f"{ROOT}/data/interim/check_city/{cfg['run']['name']}/{a.name}"
os.makedirs(od, exist_ok=True)
wcp = os.path.join(cfg["paths"]["worldcover_dir"], cfg["worldcover"]["file_pattern"].format(tile=a.tile))
at = bool(cfg["pipeline"]["rasterize_all_touched"])


def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def palette():
    """GLC class colours from the User Guide (RGB), fill = magenta for visibility."""
    p = np.zeros((256, 3), np.uint8)
    cols = {10: (255, 255, 100), 11: (255, 255, 100), 12: (255, 255, 0), 20: (170, 240, 240), 51: (76, 115, 0),
            52: (0, 100, 0), 61: (170, 200, 0), 62: (0, 160, 0), 71: (0, 80, 0), 72: (0, 60, 0), 81: (40, 100, 0),
            82: (40, 80, 0), 91: (160, 180, 50), 92: (120, 130, 0), 120: (150, 100, 0), 121: (150, 75, 0),
            122: (150, 100, 0), 130: (255, 180, 50), 140: (255, 220, 210), 150: (255, 235, 175), 152: (255, 210, 120),
            153: (255, 235, 175), 181: (0, 168, 132), 182: (115, 255, 223), 183: (158, 187, 215), 184: (130, 130, 130),
            185: (245, 122, 182), 186: (102, 205, 171), 187: (68, 79, 137), 190: (195, 20, 0), 200: (255, 245, 215),
            201: (220, 220, 220), 202: (255, 245, 215), 210: (0, 70, 200), 220: (255, 255, 255), 0: (255, 0, 255),
            250: (255, 0, 255)}
    for k, v in cols.items():
        p[k] = v
    return p


with rasterio.open(wcp) as src:
    win = from_bounds(*a.bbox, transform=src.transform).round_offsets().round_lengths()
    wt = src.window_transform(win)
    H, W = int(win.height), int(win.width)
    wc = src.read(1, window=win)
    prof = src.profile | {"width": W, "height": H, "transform": wt, "tiled": True, "blockxsize": 256,
                          "blockysize": 256, "compress": "deflate"}
lons = wt.c + (np.arange(W) + 0.5) * wt.a
lats = wt.f + (np.arange(H) + 0.5) * wt.e
overhang = a.overhang or cfg["pipeline"].get("glc_overhang", "strict")
glc = GLCLookup(ctx.glc_paths, ctx.bands, overhang).lookup(lons, lats)
wb = rasterio.windows.bounds(win, rasterio.open(wcp).transform)
box = shapely.box(*wb)
pcs = ctx.pieces_in(tuple(wb))
pc = pcs[pcs.intersects(box)]
ua = ctx.uas[ctx.uas.intersects(box)]
pid = features.rasterize(zip(pc.geometry, pc.piece_id), out_shape=(H, W), transform=wt, fill=0, dtype="int32", all_touched=at)
uid = features.rasterize(zip(ua.geometry, ua.dom_idx), out_shape=(H, W), transform=wt, fill=0, dtype="int32", all_touched=at)
valid = wc != ctx.nodata
urban = ((pid > 0) & valid & (wc != ctx.water)).astype(np.uint8)
rural = ((uid > 0) & valid & (wc != ctx.built) & (wc != ctx.water)).astype(np.uint8)
for name, arr, dt, nd in [("worldcover", wc, "uint8", 0), ("piece_id", pid, "int32", 0), ("domain_idx", uid, "int32", 0),
                          ("urban_mask", urban, "uint8", None), ("rural_mask", rural, "uint8", None)] + \
                         [(f"glc_on_wc_{y}", glc[i], "uint8", None) for i, y in enumerate(ctx.years)]:
    with rasterio.open(f"{od}/{name}.tif", "w", **(prof | {"dtype": dt, "nodata": nd, "count": 1})) as o:
        o.write(arr, 1)

# seam diagnostics at every 5-deg parallel / meridian inside the bbox
res = {"overhang": overhang, "bbox": a.bbox, "shape": [H, W], "n_pieces": int(len(pc)), "n_uas": int(len(ua)),
       "glc_fill_share": float(np.isin(glc[0], [0, 250]).mean()), "seams": []}
g = glc[0]
agree_rows = (g[1:] == g[:-1]).mean(axis=1)          # agreement between row i and i+1
for lat5 in range(int(np.ceil(a.bbox[1] / 5) * 5), int(np.floor(a.bbox[3] / 5) * 5) + 1, 5):
    i = int(np.searchsorted(-lats, -lat5))           # first row south of the parallel
    if 3 < i < H - 3:
        near = slice(max(i - 15, 0), min(i + 15, H))
        res["seams"].append({"type": "parallel", "lat": lat5, "row": i,
                             "agree_across_seam": float(agree_rows[i - 1]),
                             "agree_other_rows_median": float(np.median(np.delete(agree_rows, i - 1))),
                             "agree_other_rows_p05": float(np.quantile(np.delete(agree_rows, i - 1), 0.05)),
                             "fill_share_rows_within_15px": float(np.isin(g[near], [0, 250]).mean()),
                             "fill_share_seam_rows": float(np.isin(g[i - 1:i + 1], [0, 250]).mean())})
        pal = palette()
        r0, r1 = max(i - 300, 0), min(i + 300, H)
        c0 = W // 2 - 400; c1 = c0 + 800
        png(f"{od}/glc_on_wc_seam_lat{lat5}.png", pal[g[r0:r1, c0:c1]])
        res["seams"][-1]["png"] = f"{od}/glc_on_wc_seam_lat{lat5}.png"
        res["seams"][-1]["png_window_rows_cols"] = [r0, r1, c0, c1]
        res["seams"][-1]["seam_row_in_png"] = i - r0
for lon5 in range(int(np.ceil(a.bbox[0] / 5) * 5), int(np.floor(a.bbox[2] / 5) * 5) + 1, 5):
    j = int(np.searchsorted(lons, lon5))
    if 3 < j < W - 3:
        agree_cols = (g[:, 1:] == g[:, :-1]).mean(axis=0)
        res["seams"].append({"type": "meridian", "lon": lon5, "col": j, "agree_across_seam": float(agree_cols[j - 1]),
                             "agree_other_cols_median": float(np.median(np.delete(agree_cols, j - 1)))})
png(f"{od}/glc_on_wc_overview.png", palette()[g[::4, ::4]])
json.dump(res, open(f"{od}/seam_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
