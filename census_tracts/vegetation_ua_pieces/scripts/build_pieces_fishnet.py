"""Pieces for run profile fua_fishnet: each fishnet cell intersected with ITS OWN FUA (eFUA_ID attribute),
option B (2026-10-06). cell_uid identifies the physical cell: listings of the same cell under
neighbouring FUAs (identical geometry at 1e-9 deg) share a cell_uid; n_listings = number of kept FUAs listing it.
piece_id = 1..N ordered by (eFUA_ID, grid_id); dom_idx from the kept FUA layer (0 = none).
Areas in EPSG:6933: cell_area_m2, piece_area_m2, overlap_share = piece/cell. Listings whose intersection with
their FUA has no area (touch only) produce no piece (counted). Per-FUA check: sum of piece areas vs FUA area.
Writes data/interim/fua/pieces.parquet (GeoParquet with bbox covering column, small row groups) and
manifests/fua_pieces_summary.json. Refuses to overwrite unless --force-rebuild.
"""
import json, os, sys, time
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from urbveg_common import ROOT, load_config

cfg = load_config(run="fua_fishnet")
R = cfg["run"]
out_path = R["pieces_path"]
if os.path.exists(out_path) and "--force-rebuild" not in sys.argv:
    sys.exit(f"{out_path} exists; not overwriting (use --force-rebuild)")
t0 = time.time()
EA = cfg["pipeline"]["equal_area_crs"]
zl = cfg["layers"]["zones"][R["zone"]]
fua = gpd.read_parquet(cfg["layers"]["domains"][R["domain"]]["kept_path"])[["dom_idx", "eFUA_ID", "geometry"]]
cells = gpd.read_parquet(zl["path"], columns=["grid_id", "eFUA_ID", "geometry"])
cells["eFUA_ID"] = cells.eFUA_ID.astype("int64")
n_all = len(cells)
cells = cells[cells.eFUA_ID.isin(fua.eFUA_ID)].sort_values(["eFUA_ID", "grid_id"]).reset_index(drop=True)
print(f"{n_all} cells, {len(cells)} in kept FUAs ({time.time()-t0:.0f}s)", flush=True)
key = shapely.to_wkb(shapely.normalize(shapely.set_precision(cells.geometry.values, 1e-9)))
codes, uniq = pd.factorize(pd.Series(key), sort=False)
cells["cell_uid"] = (codes + 1).astype(np.int64)
cells["n_listings"] = cells.groupby("cell_uid").cell_uid.transform("size").astype(np.int16)
cells["cell_area_m2"] = cells.to_crs(EA).area
FG = dict(zip(fua.eFUA_ID, fua.geometry))
for g in FG.values():
    shapely.prepare(g)
GEOM = cells.geometry.values
BOUNDS = cells.groupby("eFUA_ID").indices        # eFUA_ID -> row positions


def one(fid):
    rows = BOUNDS[fid]; F = FG[fid]; c = GEOM[rows]
    inside = shapely.contains(F, c)
    out = np.empty(len(rows), dtype=object)
    out[inside] = c[inside]
    if (~inside).any():
        x = shapely.intersection(c[~inside], F)
        # keep polygonal parts only
        parts, pi = shapely.get_parts(x, return_index=True)
        keep = shapely.get_type_id(parts) == 3
        polys = pd.Series(list(parts[keep])).groupby(pi[keep]).agg(lambda s: shapely.union_all(list(s)) if len(s) > 1 else s.iloc[0])
        res = np.full((~inside).sum(), None, dtype=object)
        res[polys.index.values] = polys.values
        out[~inside] = res
    return rows, out


with Pool(int(os.environ.get("NPROC", 64))) as p:
    res = p.map(one, list(BOUNDS), chunksize=8)
geom = np.empty(len(cells), dtype=object)
for rows, g in res:
    geom[rows] = g
has = np.array([g is not None and not shapely.is_empty(g) for g in geom])
pcs = gpd.GeoDataFrame(cells.drop(columns="geometry")[has].reset_index(drop=True), geometry=list(geom[has]), crs=4326)
pcs["piece_area_m2"] = pcs.to_crs(EA).area
n_touch = int((~has).sum()) + int((pcs.piece_area_m2 <= 0).sum())
pcs = pcs[pcs.piece_area_m2 > 0].reset_index(drop=True)
pcs = pcs.merge(fua[["eFUA_ID", "dom_idx"]], on="eFUA_ID", how="left")
pcs.insert(0, "piece_id", np.arange(1, len(pcs) + 1, dtype=np.int32))
pcs["overlap_share"] = pcs.piece_area_m2 / pcs.cell_area_m2
pcs = pcs[["piece_id", "grid_id", "cell_uid", "n_listings", "eFUA_ID", "dom_idx", "piece_area_m2", "cell_area_m2",
           "overlap_share", "geometry"]]
# per-FUA closure check
fa = fua.set_index("eFUA_ID").to_crs(EA).area
pa = pcs.groupby("eFUA_ID").piece_area_m2.sum()
rel = (pa.reindex(fa.index).fillna(0) / fa - 1)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
pcs.to_parquet(out_path + ".part", index=False, write_covering_bbox=True, row_group_size=20000)
os.replace(out_path + ".part", out_path)
summ = {"n_cells_input": n_all, "n_cells_in_kept_fuas": int(len(cells)), "n_listings_without_area_overlap": n_touch,
        "n_pieces": int(len(pcs)), "n_fuas_with_pieces": int(pcs.eFUA_ID.nunique()), "n_fuas_kept": int(len(fua)),
        "n_physical_cells": int(pcs.cell_uid.nunique()), "n_pieces_from_shared_cells": int((pcs.n_listings > 1).sum()),
        "overlap_share_bins": pd.cut(pcs.overlap_share, [0, 1e-3, 0.01, 0.1, 0.5, 0.99, 0.999999, 1.000001], include_lowest=True)
                                 .value_counts(sort=False).astype(int).rename(str).to_dict(),
        "n_overlap_share_lt_0.01": int((pcs.overlap_share < 0.01).sum()),
        "fua_area_closure_rel_quantiles": rel.quantile([0, .001, .5, .999, 1]).round(9).to_dict(),
        "n_fua_closure_off_gt_1e-6": int((rel.abs() > 1e-6).sum()),
        "sum_piece_area_km2": float(pcs.piece_area_m2.sum() / 1e6), "sum_fua_area_km2": float(fa.sum() / 1e6),
        "cell_area_km2_quantiles": (pcs.cell_area_m2.quantile([0, .5, 1]) / 1e6).round(4).to_dict(),
        "parquet_bytes": os.path.getsize(out_path), "seconds": round(time.time() - t0)}
json.dump(summ, open(f"{ROOT}/manifests/fua_pieces_summary.json", "w"), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
