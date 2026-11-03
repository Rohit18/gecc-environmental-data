"""Compare GLC_FCS30D v1 (8239305) and v2 (15063683) on the Columbus test tiles, and examine
fill values (0, 250) and the overlap strip between the N-S neighbours W85N40 / W85N45.

Inputs: data/raw/glc_fcs30d/version_compare/{8239305,15063683}/*W85N4[05]*Annual*.tif,
WorldCover N39W084, UA layer from config. Writes manifests/glc_version_compare.json.
"""
import glob, json, time

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
import yaml

ROOT = "/path/to/scratch/urbveg"
cfg = yaml.safe_load(open(f"{ROOT}/config/config.yaml"))
B2020 = cfg["glc_fcs30d"]["band_year_annual"][2020]
VC = f"{ROOT}/data/raw/glc_fcs30d/version_compare"
out = {"band_2020": B2020}


def path(rec, tile):
    return glob.glob(f"{VC}/{rec}/*_{tile}_Annual*.tif")[0]


def counts(a):
    c = np.bincount(a.ravel(), minlength=256)
    return {int(k): int(c[k]) for k in np.nonzero(c)[0]}


# ---------- A. v1 vs v2, same tile ----------
for tile in ["W85N40", "W85N45"]:
    r = {}
    with rasterio.open(path("8239305", tile)) as d1, rasterio.open(path("15063683", tile)) as d2:
        r["same_transform"] = bool(d1.transform.almost_equals(d2.transform, precision=1e-12))
        r["same_shape"] = [d1.shape == d2.shape, d1.count == d2.count]
        per_band = {}
        for b in range(1, d1.count + 1):
            a1, a2 = d1.read(b), d2.read(b)
            per_band[2000 + b - 1] = round(float(np.mean(a1 != a2)), 6)
            if b == B2020:
                diff = a1 != a2
                r["2020_frac_diff"] = float(diff.mean())
                r["2020_n_diff"] = int(diff.sum())
                pairs = (a1[diff].astype(np.int32) * 256 + a2[diff]).astype(np.int64)
                pc = np.bincount(pairs, minlength=65536)
                top = np.argsort(pc)[::-1][:15]
                r["2020_top_changes_v1_to_v2"] = [[int(k // 256), int(k % 256), int(pc[k])] for k in top if pc[k] > 0]
                r["2020_counts_v1"] = counts(a1)
                r["2020_counts_v2"] = counts(a2)
                if tile == "W85N40":
                    keep = {1: a1, 2: a2}
        r["frac_diff_by_year"] = per_band
    out[f"A_{tile}"] = r
    print(tile, "done", flush=True)

# ---------- B. fill values: where do 0 / 250 occur? (2020 band, W85N40) ----------
with rasterio.open(path("15063683", "W85N40")) as g, rasterio.open(
        f"{cfg['paths']['worldcover_dir']}/ESA_WorldCover_10m_2020_v100_N39W084_Map.tif") as w:
    gt = g.transform
    # GLC rows/cols inside the WorldCover tile (lat 39-40, lon -84..-81)
    win = rasterio.windows.from_bounds(-84, 39, -81, 40, transform=gt).round_offsets().round_lengths()
    r0, c0, h, wd = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
    wt = w.transform
    # WorldCover class at each GLC pixel centre (nearest) -> cross-tab fill vs WorldCover
    ys = gt.f + (np.arange(r0, r0 + h) + 0.5) * gt.e
    xs = gt.c + (np.arange(c0, c0 + wd) + 0.5) * gt.a
    wr = np.floor((ys - wt.f) / wt.e).astype(int); wc_ = np.floor((xs - wt.c) / wt.a).astype(int)
    okr = (wr >= 0) & (wr < w.height); okc = (wc_ >= 0) & (wc_ < w.width)
    wcarr = w.read(1)
    for v, a in keep.items():
        sub = a[r0:r0 + h, c0:c0 + wd][np.ix_(okr, okc)]
        wsub = wcarr[np.ix_(wr[okr], wc_[okc])]
        res = {}
        for f in (0, 250):
            m = sub == f
            res[f"n_{f}"] = int(m.sum())
            res[f"worldcover_at_{f}"] = counts(wsub[m]) if m.any() else {}
        res["n_total"] = int(sub.size)
        out[f"B_fill_vs_worldcover_v{v}"] = res
    # fill pixels inside UAs (whole W85N40 tile)
    ua = gpd.read_file(cfg["layers"]["domains"]["urban_areas_2020"]["path"], bbox=tuple(g.bounds)).to_crs(4326)
    m = features.rasterize(((geom, 1) for geom in ua.geometry), out_shape=g.shape, transform=gt, fill=0, dtype="uint8")
    for v, a in keep.items():
        inside = a[m == 1]
        out[f"B_fill_inside_UAs_W85N40_v{v}"] = {"n_ua": int(len(ua)), "n_px": int(inside.size),
                                               "n_0": int((inside == 0).sum()), "n_250": int((inside == 250).sum())}
    # locations of fill pixels (row/col bounding info) for v2
    a = keep[2]
    for f in (0, 250):
        rr, cc = np.nonzero(a == f)
        if rr.size:
            out[f"B_fill_{f}_extent_v2"] = {"n": int(rr.size), "lat_range": [float(gt.f + rr.max() * gt.e), float(gt.f + rr.min() * gt.e)],
                                           "lon_range": [float(gt.c + cc.min() * gt.a), float(gt.c + cc.max() * gt.a)],
                                           "n_in_last_row": int((rr == a.shape[0] - 1).sum()), "n_in_last_col": int((cc == a.shape[1] - 1).sum())}
print("fill done", flush=True)

# ---------- C. overlap strip between W85N45 (north) and W85N40 (south) ----------
for rec in ["8239305", "15063683"]:
    with rasterio.open(path(rec, "W85N45")) as n, rasterio.open(path(rec, "W85N40")) as s:
        res = {"north_bottom": n.bounds.bottom, "south_top": s.bounds.top,
               "overlap_deg": s.bounds.top - n.bounds.bottom,
               "same_column_grid": bool(abs(n.transform.c - s.transform.c) < 1e-12 and abs(n.transform.a - s.transform.a) < 1e-15),
               "north_last_row_span": [n.bounds.bottom, n.bounds.bottom - n.transform.e],
               "south_first_row_span": [s.bounds.top + s.transform.e, s.bounds.top]}
        nl = n.read(B2020, window=((n.height - 2, n.height), (0, n.width)))
        sf = s.read(B2020, window=((0, 2), (0, s.width)))
        valid = (nl[-1] != 0) & (nl[-1] != 250) & (sf[0] != 0) & (sf[0] != 250)
        res["agree_north_last_vs_south_first"] = float(np.mean(nl[-1][valid] == sf[0][valid]))
        res["agree_north_2ndlast_vs_south_first"] = float(np.mean(nl[-2][valid] == sf[0][valid]))
        res["agree_north_last_vs_south_second"] = float(np.mean(nl[-1][valid] == sf[1][valid]))
        res["agree_within_tile_adjacent_rows_south"] = float(np.mean(sf[0][valid] == sf[1][valid]))
        res["n_valid_cols"] = int(valid.sum())
        res["fill_in_north_last_row"] = int(np.isin(nl[-1], [0, 250]).sum())
        res["fill_in_south_first_row"] = int(np.isin(sf[0], [0, 250]).sum())
        out[f"C_overlap_{rec}"] = res
print("overlap done", flush=True)

# ---------- D. where are overlap strips inside the domain? ----------
ext = 18554 * cfg["glc_fcs30d"]["pixel_size_deg"]
ua = gpd.read_file(cfg["layers"]["domains"]["urban_areas_2020"]["path"]).to_crs(4326)
from shapely.geometry import box
from shapely import union_all
strips = []
for lon in range(-180, 181, 5):  # tile with west edge `lon` extends to lon+ext: strip [lon+5, lon+ext]
    strips.append(box(lon + 5, -90, lon + ext, 90))
for lat in range(-90, 91, 5):    # tile with top `lat` extends down to lat-ext: strip [lat-ext, lat-5]
    strips.append(box(-180, lat - ext, 180, lat - 5))
S = union_all(strips)
hit = ua[ua.intersects(S)]
out["D_strips"] = {"strip_width_deg": ext - 5, "n_ua_touching_strips": int(len(hit)),
                   "examples": hit.NAME20.head(25).tolist(),
                   "ua_area_share_in_strips": float(ua.to_crs(6933).intersection(gpd.GeoSeries([S], crs=4326).to_crs(6933).iloc[0]).area.sum()
                                                    / ua.to_crs(6933).area.sum())}
json.dump(out, open(f"{ROOT}/manifests/glc_version_compare.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:20000])
