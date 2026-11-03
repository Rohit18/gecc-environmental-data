"""Scoping only (no downloads): WorldCover and GLC_FCS30D v1 tiles needed for all 9,031 FUAs (+0.01 deg),
what already exists, sizes, and FUAs outside GLC coverage. Writes manifests/fua_scoping.json and
manifests/fua_scoping_{worldcover,glc}_tiles.csv, manifests/fua_without_full_glc.csv."""
import glob, json, os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from urbveg_common import ROOT, load_config

cfg = load_config()
lay = cfg["layers"]["domains"]["fua"]
f = gpd.read_file(lay["path"], encoding=lay["encoding"]).to_crs(4326)
f["eFUA_ID"] = f.eFUA_ID.astype("int64")
fb = f.copy(); fb["geometry"] = fb.geometry.buffer(0.01, quad_segs=4)
out = {"n_fua": len(f)}

# WorldCover
grid = gpd.read_file(f"{ROOT}/data/raw/worldcover_2020/aux/esa_worldcover_grid.geojson").rename(columns={"ll_tile": "tile"})
lst = pd.read_csv(f"{ROOT}/data/interim/wc_s3_listing.tsv", sep="\t", names=["key", "bytes"])
lst["tile"] = lst.key.str.extract(r"_v100_([NS]\d\d[EW]\d\d\d)_Map\.tif")[0]
wt = gpd.sjoin(grid, fb[["eFUA_ID", "geometry"]], predicate="intersects").groupby("tile").size().rename("n_fua").reset_index()
wt = wt.merge(lst[["tile", "bytes"]], on="tile", how="left")
ex = "/path/to/project/gecc/fua_uhi/worldcover_2020/tiles"
wt["existing_fua_uhi"] = [os.path.exists(f"{ex}/ESA_WorldCover_10m_2020_v100_{t}_Map.tif") for t in wt.tile]
wt["in_urbveg"] = [os.path.exists(f"{ROOT}/data/raw/worldcover_2020/tiles/ESA_WorldCover_10m_2020_v100_{t}_Map.tif") for t in wt.tile]
wt.to_csv(f"{ROOT}/manifests/fua_scoping_worldcover_tiles.csv", index=False)
out["worldcover"] = {"n_tiles": len(wt), "total_GB": round(wt.bytes.sum() / 1e9, 2),
                     "n_missing_on_s3": int(wt.bytes.isna().sum()),
                     "n_existing_on_cfs_fua_uhi": int(wt.existing_fua_uhi.sum()),
                     "n_not_on_cfs": int((~wt.existing_fua_uhi & ~wt.in_urbveg).sum()),
                     "GB_to_download": round(wt.loc[~wt.existing_fua_uhi & ~wt.in_urbveg, "bytes"].sum() / 1e9, 2)}

# GLC v1
m = pd.read_csv(f"{ROOT}/manifests/glc_zip_members_8239305.csv")
m = m[m.member.str.contains("_Annual")].copy()
p = m.member.str.extract(r"_(?P<ew>[EW])(?P<lon>\d+)(?P<ns>[NS])(?P<lat>\d+)_Annual")
m["tile"] = p.ew + p.lon + p.ns + p.lat
m["lon0"] = p.lon.astype(int) * p.ew.map({"E": 1, "W": -1}); m["lat0"] = p.lat.astype(int) * p.ns.map({"N": 1, "S": -1})
ext = 18554 * cfg["glc_fcs30d"]["pixel_size_deg"]
gt = gpd.GeoDataFrame(m, geometry=[box(x, y - ext, x + ext, y) for x, y in zip(m.lon0, m.lat0)], crs=4326)
gs = gpd.sjoin(gt, fb[["eFUA_ID", "geometry"]], predicate="intersects").groupby(["tile", "compress_size", "file_size"]).size().rename("n_fua").reset_index()
have = {os.path.basename(x).split("_")[3] for x in glob.glob(f"{cfg['paths']['glc_dir']}/*.tif")}
gs["downloaded"] = gs.tile.isin(have)
gs.to_csv(f"{ROOT}/manifests/fua_scoping_glc_tiles.csv", index=False)
out["glc_v1"] = {"n_tiles": len(gs), "GB_on_disk_total": round(gs.file_size.sum() / 1e9, 2),
                 "n_already_downloaded": int(gs.downloaded.sum()),
                 "GB_to_download": round(gs.loc[~gs.downloaded, "compress_size"].sum() / 1e9, 2)}
# GLC coverage (name-cells of existing tiles)
cells = shapely.union_all([box(x, y - 5, x + 5, y) for x, y in zip(m.lon0, m.lat0)])
inside = f.geometry.within(cells); touch = f.geometry.intersects(cells)
cov_share = shapely.area(shapely.intersection(f.geometry.values, cells)) / shapely.area(f.geometry.values)
f["glc_cover_share"] = cov_share
bad = f[cov_share < 1 - 1e-9][["eFUA_ID", "eFUA_nm", "glc_cover_share"]].sort_values("glc_cover_share")
bad.to_csv(f"{ROOT}/manifests/fua_without_full_glc.csv", index=False)
out["glc_coverage"] = {"n_fua_fully_covered": int(inside.sum()), "n_fua_partly": int((touch & ~inside).sum()),
                       "n_fua_not_covered": int((~touch).sum()),
                       "examples_not_covered": bad[bad.glc_cover_share == 0].eFUA_nm.head(25).tolist(),
                       "examples_partly": bad[bad.glc_cover_share > 0].head(10)[["eFUA_nm", "glc_cover_share"]].round(3).values.tolist()}
# US FUAs (for reference): FUAs intersecting kept UAs
ua = gpd.read_parquet(cfg["layers"]["domains"]["urban_areas_2020"]["kept_path"])
us = gpd.sjoin(f[["eFUA_ID", "eFUA_nm", "geometry"]], ua[["GEOID20", "geometry"]], predicate="intersects")
out["n_fua_intersecting_kept_US_UAs"] = int(us.eFUA_ID.nunique())
json.dump(out, open(f"{ROOT}/manifests/fua_scoping.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
