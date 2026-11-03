"""Inspect the FUA fishnet GeoJSON (from Drive) and compare with the EE asset checks; write GeoParquet.

Checks: feature count, CRS, columns, grid_id uniqueness, eFUA_ID set vs FUA polygons, per-FUA counts vs the
EE aggregate histogram, geometry validity and vertex counts, cell size vs latitude, identity with the cells
downloaded from EE for 4 sample FUAs, coverage of each FUA by its own cells and within-FUA overlaps
(all FUAs). Writes data/interim/fua_fishnet.parquet (EPSG:4326) and manifests/fua_fishnet_check.json.
"""
import json, os, time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

from urbveg_common import ROOT, load_config

t0 = time.time()
cfg = load_config()
SRC = f"{ROOT}/data/raw/boundaries/fua_fishnet/EC_FUA_fishnet_1km_properties_gridID.geojson"
OUT = f"{ROOT}/data/interim/fua_fishnet.parquet"
info = pyogrio.read_info(SRC)
g = gpd.read_file(SRC, engine="pyogrio", use_arrow=True)
print(f"read {len(g)} in {time.time()-t0:.0f}s", flush=True)
res = {"source": SRC, "bytes": os.path.getsize(SRC), "driver_info": {k: str(info[k]) for k in ("crs", "geometry_type", "features", "fields")},
       "n_features": len(g), "columns": [c for c in g.columns if c != "geometry"], "crs": str(g.crs)}
res["dtypes"] = {c: str(t) for c, t in g.dtypes.items() if c != "geometry"}
g["eFUA_ID"] = g.eFUA_ID.astype("int64")
res["grid_id_unique"] = bool(g.grid_id.is_unique)
res["n_null"] = {c: int(g[c].isna().sum()) for c in res["columns"]}
fua = gpd.read_file(cfg["layers"]["domains"]["fua"]["path"], encoding="latin1"); fua["eFUA_ID"] = fua.eFUA_ID.astype("int64")
res["eFUA_ID_set_equal_to_FUA_file"] = set(g.eFUA_ID) == set(fua.eFUA_ID)
h = json.load(open(f"{ROOT}/data/interim/fishnet_check/cells_per_fua.json"))
ee = pd.Series({int(float(k)): v for k, v in h.items()})
cnt = g.groupby("eFUA_ID").size()
res["per_fua_count_equal_to_EE_asset"] = bool(cnt.reindex(ee.index).fillna(-1).astype(int).equals(ee.astype(int)))
res["n_fua_count_mismatch"] = int((cnt.reindex(ee.index).fillna(-1) != ee).sum())
# geometry
v = shapely.is_valid(g.geometry.values)
nv = shapely.get_num_coordinates(g.geometry.values)
res["n_invalid"] = int((~v).sum()); res["coords_per_cell"] = {str(k): int(x) for k, x in pd.Series(nv).value_counts().items()}
res["geom_types"] = g.geom_type.value_counts().to_dict()
b = g.geometry.bounds
w, hgt, lat = b.maxx - b.minx, b.maxy - b.miny, (b.miny + b.maxy) / 2
res["width_deg_quantiles"] = w.quantile([0, .5, 1]).round(9).to_dict()
res["height_over_width_vs_coslat"] = float(np.median((hgt / w) / np.cos(np.radians(lat))))
g["cell_area_m2"] = g.to_crs(cfg["pipeline"]["equal_area_crs"]).area
res["cell_area_km2_by_lat"] = (g.groupby(pd.cut(lat, [-60, -40, -20, 0, 20, 40, 50, 60, 72])).cell_area_m2.median() / 1e6).round(3).rename(str).to_dict()
# identity with EE downloads (4 FUAs)
ident = {}
for fid in (6887, 6515, 2227, 935):
    e = gpd.GeoDataFrame.from_features(json.load(open(f"{ROOT}/data/interim/fishnet_check/cells_{fid}.geojson"))["features"], crs=4326).drop_duplicates("grid_id").set_index("grid_id")
    d = g[g.eFUA_ID == fid].set_index("grid_id")
    common = e.index.intersection(d.index)
    eq = shapely.equals_exact(e.loc[common].geometry.values, d.loc[common].geometry.values, tolerance=1e-9)
    ident[fid] = {"n_ee": len(e), "n_file": len(d), "n_common_ids": len(common), "n_geom_equal_1e-9": int(eq.sum())}
res["identity_with_ee_samples"] = ident
# coverage and within-FUA overlap, all FUAs (area in degrees^2 is enough for ratios)
fg = fua.set_index("eFUA_ID").geometry
cov, ovl = {}, {}
for fid, grp in g.groupby("eFUA_ID"):
    cells = grp.geometry.values
    F = fg.loc[fid]
    inter = shapely.intersection(cells, F)
    s = shapely.area(inter).sum()
    cov[fid] = s / F.area                                    # = 1 if cells cover F without overlap
    ovl[fid] = shapely.area(cells).sum() / shapely.area(shapely.union_all(cells)) - 1 if len(cells) < 20000 else np.nan
cov = pd.Series(cov); ovl = pd.Series(ovl)
res["coverage_sum_cell_cap_fua_over_fua_area_quantiles"] = cov.quantile([0, .001, .01, .5, .99, .999, 1]).round(6).to_dict()
res["n_fua_coverage_off_by_gt_1e-6"] = int((cov.sub(1).abs() > 1e-6).sum())
res["fua_coverage_outliers"] = cov[cov.sub(1).abs() > 1e-6].sort_values().head(20).round(6).to_dict()
res["within_fua_overlap_quantiles"] = ovl.quantile([0, .5, .99, 1]).to_dict()
# cells shared by FUAs: identical geometries under different eFUA_IDs
wkb = shapely.to_wkb(shapely.normalize(shapely.set_precision(g.geometry.values, 1e-9)))
dup = pd.Series(wkb).duplicated(keep=False)
res["n_cells_with_identical_geometry_elsewhere"] = int(dup.sum())
g = g.drop(columns=["cell_area_m2"])
g.to_parquet(OUT + ".part", index=False); os.replace(OUT + ".part", OUT)
res["parquet"] = OUT; res["parquet_bytes"] = os.path.getsize(OUT); res["seconds"] = round(time.time() - t0)
json.dump(res, open(f"{ROOT}/manifests/fua_fishnet_check.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
