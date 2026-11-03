"""Describe fishnet cells that appear under more than one FUA (identical geometry, different grid_id):
how many FUAs each is listed under, the share of the cell inside each listing FUA, and the share outside all FUAs.
Also: boundary cells listed under a single FUA that extend outside it. Writes manifests/fua_fishnet_shared_cells.json"""
import json
import geopandas as gpd, numpy as np, pandas as pd, shapely
from urbveg_common import ROOT, load_config
cfg = load_config()
g = gpd.read_parquet(cfg["layers"]["zones"]["fua_fishnet_1km"]["path"], columns=["grid_id", "eFUA_ID", "geometry"])
g["eFUA_ID"] = g.eFUA_ID.astype("int64")
fua = gpd.read_file(cfg["layers"]["domains"]["fua"]["path"], encoding="latin1"); fua["eFUA_ID"] = fua.eFUA_ID.astype("int64")
fg = fua.set_index("eFUA_ID").geometry
g["key"] = shapely.to_wkb(shapely.normalize(shapely.set_precision(g.geometry.values, 1e-9)))
n = g.groupby("key").eFUA_ID.transform("size")
sh = g[n > 1].copy()
sh["share_in_own_fua"] = shapely.area(shapely.intersection(sh.geometry.values, fg.loc[sh.eFUA_ID].values)) / shapely.area(sh.geometry.values)
grp = sh.groupby("key")
tot_in = grp.share_in_own_fua.sum()
res = {"n_cells_total": len(g), "n_listings_shared": int(len(sh)), "n_physical_shared_cells": int(grp.ngroups),
       "listings_per_shared_cell": grp.size().value_counts().to_dict(),
       "n_shared_cells_same_fua_twice": int((grp.eFUA_ID.nunique() < grp.size()).sum()),
       "share_in_own_fua_quantiles": sh.share_in_own_fua.quantile([0, .1, .5, .9, 1]).round(4).to_dict(),
       "sum_of_shares_over_listings_quantiles": tot_in.quantile([0, .01, .5, .99, 1]).round(6).to_dict(),
       "n_shared_cells_partly_outside_all_listing_fuas": int((tot_in < 1 - 1e-6).sum())}
single = g[n == 1]
inside = shapely.area(shapely.intersection(single.geometry.values, fg.loc[single.eFUA_ID].values)) / shapely.area(single.geometry.values)
res["single_listed_cells"] = {"n": len(single), "n_fully_inside": int((inside > 1 - 1e-9).sum()),
                              "n_partly_outside": int(((inside > 1e-9) & (inside <= 1 - 1e-9)).sum()),
                              "n_zero_area_inside": int((inside <= 1e-9).sum())}
json.dump(res, open(f"{ROOT}/manifests/fua_fishnet_shared_cells.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
