"""Grid alignment check between one WorldCover tile and one GLC_FCS30D tile, plus small clips
around a test city for visual inspection (data/interim/check_city/).

Reports pixel sizes and their ratio, origins, and the offset of GLC pixel edges from WorldCover
pixel edges, in WorldCover pixels. Writes manifests/alignment_check.json.
Usage: python alignment_check.py --wc N39W084 --glc <path to GLC annual tif> --year 2020
        --city-bbox minx miny maxx maxy --city-name columbus_oh --ua-geoid <GEOID20>
"""
import argparse, json, os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--wc", required=True)
ap.add_argument("--glc", required=True)
ap.add_argument("--year", type=int, default=2020)
ap.add_argument("--city-bbox", nargs=4, type=float, required=True)
ap.add_argument("--city-name", default="city")
ap.add_argument("--ua-geoid", default=None)
a = ap.parse_args()
ROOT = "/path/to/scratch/urbveg"
cfg = yaml.safe_load(open(f"{ROOT}/config/config.yaml"))
wcp = f"{cfg['paths']['worldcover_dir']}/ESA_WorldCover_10m_2020_v100_{a.wc}_Map.tif"
band = cfg["glc_fcs30d"]["band_year_annual"][a.year] if cfg["glc_fcs30d"]["band_year_annual"] else None
if band is None:
    raise SystemExit("band_year_annual not set in config")

with rasterio.open(wcp) as w, rasterio.open(a.glc) as g:
    wt, gt = w.transform, g.transform
    rx, ry = gt.a / wt.a, gt.e / wt.e
    # offset of the GLC origin from the nearest WorldCover pixel edge, in WorldCover pixels
    ox = (gt.c - wt.c) / wt.a
    oy = (gt.f - wt.f) / wt.e
    res = {
        "worldcover": {"path": wcp, "crs": str(w.crs), "res": list(w.res), "origin": [wt.c, wt.f],
                       "transform": list(wt)[:6]},
        "glc": {"path": a.glc, "crs": str(g.crs), "res": list(g.res), "origin": [gt.c, gt.f],
                "transform": list(gt)[:6], "band_for_year": band},
        "both_epsg4326": bool(w.crs.to_epsg() == 4326 and g.crs.to_epsg() == 4326),
        "ratio_x": rx, "ratio_y": ry,
        "glc_origin_offset_in_wc_pixels": [ox, oy],
        "glc_origin_offset_frac_wc_pixel": [ox - round(ox), oy - round(oy)],
        "ratio_is_integer": bool(abs(rx - round(rx)) < 1e-9 and abs(ry - round(ry)) < 1e-9),
    }
    # every GLC edge k*gt.a + gt.c lies on a WC edge iff the ratio is integer and the origin offset is integer
    res["glc_edges_on_wc_edges"] = bool(res["ratio_is_integer"] and abs(ox - round(ox)) < 1e-6 and abs(oy - round(oy)) < 1e-6)
    # fraction of WC pixel centres whose containing GLC pixel is ambiguous (centre within 1e-9 of a GLC edge)
    xs = wt.c + (np.arange(w.width) + 0.5) * wt.a
    fx = (xs - gt.c) / gt.a
    res["wc_centres_on_glc_edge_x"] = int(np.sum(np.abs(fx - np.round(fx)) < 1e-9))

    # --- city clips ---
    od = f"{ROOT}/data/interim/check_city/{a.city_name}"
    os.makedirs(od, exist_ok=True)
    bb = a.city_bbox
    ww = from_bounds(*bb, transform=wt).round_offsets().round_lengths()
    arr = w.read(1, window=ww)
    prof = w.profile | {"width": arr.shape[1], "height": arr.shape[0], "transform": w.window_transform(ww),
                        "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate"}
    with rasterio.open(f"{od}/worldcover2020_{a.city_name}.tif", "w", **prof) as o:
        o.write(arr, 1)
    gw = from_bounds(*bb, transform=gt).round_offsets().round_lengths()
    gw = gw.intersection(Window(0, 0, g.width, g.height))   # part of the city may lie in the neighbouring GLC tile
    garr = g.read(band, window=gw)
    gprof = g.profile | {"count": 1, "width": garr.shape[1], "height": garr.shape[0],
                         "transform": g.window_transform(gw), "tiled": True, "blockxsize": 256,
                         "blockysize": 256, "compress": "deflate"}
    with rasterio.open(f"{od}/glc_fcs30d_{a.year}_{a.city_name}.tif", "w", **gprof) as o:
        o.write(garr, 1)
    res["city"] = {"name": a.city_name, "bbox": bb,
                   "wc_values": {int(k): int(v) for k, v in zip(*np.unique(arr, return_counts=True))},
                   "glc_values": {int(k): int(v) for k, v in zip(*np.unique(garr, return_counts=True))}}

ua = cfg["layers"]["domains"]["urban_areas_2020"]
u = gpd.read_file(ua["path"], bbox=tuple(bb))
if a.ua_geoid:
    u = u[u[ua["id_field"]] == a.ua_geoid]
u.to_file(f"{od}/urban_area_{a.city_name}.gpkg", driver="GPKG")
import glob
tr = []
for p in glob.glob(cfg["layers"]["zones"]["tracts_2020"]["path_glob"]):
    t = gpd.read_file(p, bbox=tuple(bb))
    if len(t):
        tr.append(t)
import pandas as pd
gpd.GeoDataFrame(pd.concat(tr), crs=tr[0].crs).to_file(f"{od}/tracts_2020_{a.city_name}.gpkg", driver="GPKG")
res["city"]["n_tracts_in_bbox"] = int(sum(len(t) for t in tr))
res["city"]["n_ua_in_bbox"] = int(len(u))
json.dump(res, open(f"{ROOT}/manifests/alignment_check_{a.city_name}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
