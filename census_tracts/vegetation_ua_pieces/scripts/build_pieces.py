"""Stage 2: pieces = tract ∩ kept Urban Area, one per (tract GEOID, UA GEOID20) pair.

piece_id = 1..N ordered by (GEOID, GEOID20); 0 means "no piece". ua_idx comes from uas_kept.parquet
(1..N by GEOID20; 0 = no UA). Areas in config pipeline.equal_area_crs (EPSG:6933):
piece_area_m2, tract_area_m2, overlap_share = piece/tract.
Tract coordinates are NAD83 (EPSG:4269) relabelled to EPSG:4326 with pyproj's default (null) transform.
Writes data/interim/pieces.parquet (GeoParquet, EPSG:4326) and manifests/pieces_summary.json.
Idempotent: refuses to overwrite an existing pieces.parquet unless --force-rebuild is given
(the old file is then kept as pieces.parquet.bak-<timestamp>).
"""
import json, os, sys, time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from urbveg_common import ROOT, load_config, read_domain, read_tracts

cfg = load_config()
out_path = cfg["pipeline"]["pieces_path"]
if os.path.exists(out_path) and "--force-rebuild" not in sys.argv:
    sys.exit(f"{out_path} exists; not overwriting (use --force-rebuild)")
t0 = time.time()
EA = cfg["pipeline"]["equal_area_crs"]
zl = cfg["layers"]["zones"]["tracts_2020"]
ua = read_domain(cfg["layers"]["domains"]["urban_areas_2020"])[["ua_idx", "GEOID20", "NAME20", "geometry"]]
tr = read_tracts(cfg)[[zl["id_field"], zl["state_field"], "geometry"]].to_crs("EPSG:4326")
tr = tr.rename(columns={zl["id_field"]: "GEOID", zl["state_field"]: "STATEFP"})
tr["tract_area_m2"] = tr.to_crs(EA).area
print(f"read {len(tr)} tracts, {len(ua)} UAs in {time.time()-t0:.0f}s", flush=True)

# candidate pairs by bbox, then exact intersection
cand = gpd.sjoin(tr, ua, predicate="intersects", how="inner")
print(f"{len(cand)} candidate pairs", flush=True)
tg = tr.set_index("GEOID").geometry
ug = ua.set_index("GEOID20").geometry
inter = shapely.intersection(tg.loc[cand.GEOID].values, ug.loc[cand.GEOID20].values)
# keep polygonal parts only (drop shared-edge lines/points)
inter = shapely.get_parts(inter, return_index=True)
parts, idx = inter
poly = shapely.get_type_id(parts) == 3   # Polygon
pcs = gpd.GeoDataFrame({"row": idx[poly]}, geometry=parts[poly], crs="EPSG:4326")
pcs = pcs.dissolve(by="row").reset_index()
pcs = pcs.join(cand.reset_index(drop=True)[["GEOID", "STATEFP", "tract_area_m2", "GEOID20", "ua_idx"]], on="row")
pcs["piece_area_m2"] = pcs.to_crs(EA).area
n_line_only = len(cand) - len(pcs)
pcs = pcs[pcs.piece_area_m2 > 0].drop(columns="row")
pcs = pcs.sort_values(["GEOID", "GEOID20"]).reset_index(drop=True)
assert not pcs.duplicated(["GEOID", "GEOID20"]).any()
pcs.insert(0, "piece_id", np.arange(1, len(pcs) + 1, dtype=np.int32))
pcs["overlap_share"] = pcs.piece_area_m2 / pcs.tract_area_m2
pcs = pcs[["piece_id", "GEOID", "STATEFP", "GEOID20", "ua_idx", "piece_area_m2", "tract_area_m2", "overlap_share", "geometry"]]
pcs.to_parquet(out_path + ".part", index=False)
os.replace(out_path + ".part", out_path)

n_ua_per_tract = pcs.groupby("GEOID").size()
q = pcs.overlap_share.quantile([0, .01, .05, .1, .25, .5, .75, .9, 1]).round(6).to_dict()
summ = {
    "n_pieces": int(len(pcs)), "n_candidate_pairs_bbox": int(len(cand)),
    "n_pairs_without_area_overlap": int(n_line_only),
    "n_tracts_with_pieces": int(n_ua_per_tract.size), "n_tracts_total": int(len(tr)),
    "n_tracts_in_more_than_one_ua": int((n_ua_per_tract > 1).sum()),
    "max_uas_per_tract": int(n_ua_per_tract.max()),
    "n_uas_with_pieces": int(pcs.GEOID20.nunique()), "n_uas_kept": int(len(ua)),
    "overlap_share_quantiles": {str(k): v for k, v in q.items()},
    "overlap_share_bins": pd.cut(pcs.overlap_share, [0, 1e-4, 1e-3, 0.01, 0.05, 0.25, 0.5, 0.75, 0.99, 0.999999, 1.000001],
                                 include_lowest=True).value_counts(sort=False).astype(int).rename(str).to_dict(),
    "n_overlap_share_gt_1": int((pcs.overlap_share > 1 + 1e-9).sum()),
    "piece_area_m2_quantiles": {str(k): round(v, 1) for k, v in pcs.piece_area_m2.quantile([0, .01, .05, .5, .95, 1]).to_dict().items()},
    "n_pieces_lt_100m2": int((pcs.piece_area_m2 < 100).sum()),
    "n_pieces_lt_10000m2": int((pcs.piece_area_m2 < 1e4).sum()),
    "sum_piece_area_km2": float(pcs.piece_area_m2.sum() / 1e6),
    "sum_ua_area_km2": float(ua.to_crs(EA).area.sum() / 1e6),
    "seconds": round(time.time() - t0, 1),
}
json.dump(summ, open(f"{ROOT}/manifests/pieces_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
