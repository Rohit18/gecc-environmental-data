"""Stage 1 step 2: extend the vegetation domains to all 2,644 UAs of the corrected 2020 UA file.

uas_all.parquet: the 2,622 kept UAs unchanged (same rows, columns, ua_idx, geometry) followed by the 22 UAs the
  vegetation run dropped (HI 15, GU 2, MP 1, AS 1, VI 3), with ua_idx 2,623..2,644 ordered by GEOID20. Built from
  the corrected file exactly as prepare_domains.py does (read, relabel NAD83 -> EPSG:4326).
pieces_all.parquet: the 75,383 vegetation pieces unchanged, followed by the pieces of the 22 new UAs, built with
  build_pieces.py's logic (bbox sjoin, exact intersection, polygon parts only, dissolve per pair, EPSG:6933
  areas, zero-area pairs dropped), piece_id 75,384.. ordered by (GEOID, GEOID20). Tracts: the vegetation run's 52
  files + the 4 island-area files of this subproject (56 files).
Never overwrites an existing output (use --force-rebuild; the old file is kept as .bak-<timestamp>).
Checks and counts -> qa/stage1_domains.json, qa/stage1_new_pieces_by_state.csv.
"""
import glob
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ndvi_common import load_config, setup_logging, utc_now, write_json_atomic

cfg = load_config()
dom = cfg["domains"]
log = setup_logging("extend_domains")
QA = cfg["paths"]["qa"]
EA = "EPSG:6933"     # vegetation config pipeline.equal_area_crs
force = "--force-rebuild" in sys.argv
for p in (dom["uas_all"], dom["pieces_all"]):
    if os.path.exists(p):
        if not force:
            sys.exit(f"{p} exists; not overwriting (use --force-rebuild)")
        os.replace(p, p + f".bak-{time.strftime('%Y%m%dT%H%M%S')}")
t0 = time.time()
rep = {"utc": utc_now()}

# ------------------------------------------------------------------ UAs
kept = gpd.read_parquet(dom["uas_kept_veg"])
assert len(kept) == dom["n_uas_veg"] and kept.ua_idx.tolist() == list(range(1, len(kept) + 1))
src = gpd.read_file(dom["ua_source"])
assert len(src) == dom["n_uas_all"] and src.GEOID20.is_unique
new = src[~src.GEOID20.isin(kept.GEOID20)].sort_values("GEOID20").reset_index(drop=True)
new.insert(0, "ua_idx", range(len(kept) + 1, len(kept) + 1 + len(new)))
new = new.to_crs("EPSG:4326")[kept.columns.tolist()]
uas_all = gpd.GeoDataFrame(pd.concat([kept, new], ignore_index=True), crs=kept.crs)
assert len(uas_all) == dom["n_uas_all"] and uas_all.GEOID20.is_unique and uas_all.ua_idx.is_unique
# old rows: the corrected file re-read now must give the same records as uas_kept (same source file)
src4326 = src.to_crs("EPSG:4326").set_index("GEOID20")
chk = src4326.loc[kept.GEOID20]
same_geom = shapely.to_wkb(chk.geometry.values) == shapely.to_wkb(kept.geometry.values)
attr_cols = [c for c in kept.columns if c not in ("ua_idx", "GEOID20", "geometry")]
same_attr = (chk[attr_cols].reset_index(drop=True) == kept[attr_cols].reset_index(drop=True)).all(axis=1).to_numpy()
new_suffix = new.NAME20.str.rsplit(", ", n=1).str[1]
rep["uas"] = {"n_all": len(uas_all), "n_old": len(kept), "n_new": len(new),
              "new_by_suffix": new_suffix.value_counts().to_dict(),
              "new_ua_idx_range": [int(new.ua_idx.min()), int(new.ua_idx.max())],
              "old_rows_equal_source_geometry_wkb": int(same_geom.sum()),
              "old_rows_equal_source_attributes": int(same_attr.sum()),
              "new_uas": new[["ua_idx", "GEOID20", "NAME20"]].to_dict("records")}
log.info("UAs: %s", {k: v for k, v in rep["uas"].items() if k != "new_uas"})
assert same_geom.all() and same_attr.all()
uas_all.to_parquet(dom["uas_all"] + ".part", index=False)
os.replace(dom["uas_all"] + ".part", dom["uas_all"])

# ------------------------------------------------------------------ tracts (56 files)
files = sorted(f for g in dom["tract_globs"] for f in glob.glob(g))
parts = [gpd.read_file(p) for p in files]
tr_counts = {os.path.basename(p): len(x) for p, x in zip(files, parts)}
tr = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
assert all(x.crs == parts[0].crs for x in parts)
for fips, n in dom["new_tract_fips"].items():
    got = tr_counts[f"tl_2020_{fips}_tract.shp"]
    assert got == n, (fips, got, n)
rep["tracts"] = {"n_files": len(files), "n_tracts": len(tr), "crs": str(parts[0].crs),
                 "new_file_counts": {f: tr_counts[f"tl_2020_{f}_tract.shp"] for f in dom["new_tract_fips"]},
                 "expected_new_file_counts": dom["new_tract_fips"], "GEOID_unique": bool(tr.GEOID.is_unique),
                 "invalid_geometries_new_files": int((~tr[tr.STATEFP.isin(list(dom["new_tract_fips"]))].is_valid).sum())}
log.info("tracts: %s", rep["tracts"])
tr = tr[["GEOID", "STATEFP", "geometry"]].to_crs("EPSG:4326")
tr["tract_area_m2"] = tr.to_crs(EA).area

# ------------------------------------------------------------------ new pieces (build_pieces.py logic)
ua = new[["ua_idx", "GEOID20", "NAME20", "geometry"]]
cand = gpd.sjoin(tr, ua, predicate="intersects", how="inner")
tg = tr.set_index("GEOID").geometry
ug = ua.set_index("GEOID20").geometry
inter = shapely.intersection(tg.loc[cand.GEOID].values, ug.loc[cand.GEOID20].values)
prt, idx = shapely.get_parts(inter, return_index=True)
poly = shapely.get_type_id(prt) == 3
pcs = gpd.GeoDataFrame({"row": idx[poly]}, geometry=prt[poly], crs="EPSG:4326")
pcs = pcs.dissolve(by="row").reset_index()
pcs = pcs.join(cand.reset_index(drop=True)[["GEOID", "STATEFP", "tract_area_m2", "GEOID20", "ua_idx"]], on="row")
pcs["piece_area_m2"] = pcs.to_crs(EA).area
n_line_only = len(cand) - len(pcs)
pcs = pcs[pcs.piece_area_m2 > 0].drop(columns="row")
pcs = pcs.sort_values(["GEOID", "GEOID20"]).reset_index(drop=True)
assert not pcs.duplicated(["GEOID", "GEOID20"]).any()
old = gpd.read_parquet(dom["pieces_veg"])
assert len(old) == dom["n_pieces_veg"] and old.piece_id.tolist() == list(range(1, len(old) + 1))
pcs.insert(0, "piece_id", np.arange(len(old) + 1, len(old) + 1 + len(pcs), dtype=np.int32))
pcs["overlap_share"] = pcs.piece_area_m2 / pcs.tract_area_m2
pcs = pcs[old.columns.tolist()].astype({c: old[c].dtype for c in old.columns if c != "geometry"})
allp = gpd.GeoDataFrame(pd.concat([old, pcs], ignore_index=True), crs=old.crs)
allp.to_parquet(dom["pieces_all"] + ".part", index=False)
os.replace(dom["pieces_all"] + ".part", dom["pieces_all"])

# ------------------------------------------------------------------ checks (re-read from disk)
ra = gpd.read_parquet(dom["pieces_all"])
ro = gpd.read_parquet(dom["pieces_veg"])
head = ra.iloc[:len(ro)].reset_index(drop=True)
cols = ["piece_id", "GEOID", "STATEFP", "GEOID20", "ua_idx", "piece_area_m2", "tract_area_m2", "overlap_share"]
eq_cols = {c: bool((head[c].to_numpy() == ro[c].to_numpy()).all()) for c in cols}
eq_dtypes = {c: str(head[c].dtype) == str(ro[c].dtype) for c in cols}
eq_wkb = bool((shapely.to_wkb(head.geometry.values) == shapely.to_wkb(ro.geometry.values)).all())
newp = ra.iloc[len(ro):]
ua_with = set(newp.GEOID20)
tree = shapely.STRtree(ro.geometry.values)
hits = tree.query(newp.geometry.values, predicate="intersects")
q = newp.overlap_share.quantile([0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]).round(6)
by_state = newp.assign(state=newp.STATEFP.map({"15": "HI", "60": "AS", "66": "GU", "69": "MP", "78": "VI"})).groupby(
    ["STATEFP", "state"]).agg(n_pieces=("piece_id", "size"), n_tracts=("GEOID", "nunique"), n_uas=("GEOID20", "nunique"),
                              area_km2=("piece_area_m2", lambda s: s.sum() / 1e6),
                              n_sliver=("overlap_share", lambda s: int((s < dom["sliver_overlap_share_lt"]).sum())),
                              n_lt_100m2=("piece_area_m2", lambda s: int((s < 100).sum()))).reset_index()
by_state.to_csv(f"{QA}/stage1_new_pieces_by_state.csv", index=False)
ua_area = new.to_crs(EA).area
rep["pieces"] = {
    "n_all": len(ra), "n_old": len(ro), "n_new": len(newp),
    "new_piece_id_range": [int(newp.piece_id.min()), int(newp.piece_id.max())],
    "n_candidate_pairs_bbox_new": int(len(cand)), "n_pairs_without_area_overlap_new": int(n_line_only),
    "old_rows_equal": {"columns": eq_cols, "dtypes": eq_dtypes, "geometry_wkb": eq_wkb},
    "every_new_ua_has_piece": len(ua_with) == len(new), "new_uas_without_piece": sorted(set(new.GEOID20) - ua_with),
    "new_pieces_intersecting_old_pieces": int(hits.shape[1]),
    "new_by_state": by_state.to_dict("records"),
    "new_overlap_share_quantiles": {str(k): float(v) for k, v in q.items()},
    "new_n_sliver_overlap_lt_0.01": int((newp.overlap_share < dom["sliver_overlap_share_lt"]).sum()),
    "new_n_overlap_gt_1": int((newp.overlap_share > 1 + 1e-9).sum()),
    "new_piece_area_closure": {"sum_piece_km2": float(newp.piece_area_m2.sum() / 1e6),
                               "sum_new_ua_km2": float(ua_area.sum() / 1e6)},
    "new_tracts_with_piece": int(newp.GEOID.nunique()),
    "new_tracts_in_more_than_one_ua": int((newp.groupby("GEOID").size() > 1).sum()),
    "invalid_new_piece_geometries": int((~newp.is_valid).sum()),
}
rep["seconds"] = round(time.time() - t0, 1)
write_json_atomic(f"{QA}/stage1_domains.json", rep)
log.info("pieces: %s", {k: v for k, v in rep["pieces"].items() if k != "new_by_state"})
print(by_state.to_string())
ok = all(eq_cols.values()) and all(eq_dtypes.values()) and eq_wkb and rep["pieces"]["every_new_ua_has_piece"] \
    and rep["pieces"]["new_pieces_intersecting_old_pieces"] == 0
log.info("CHECKS %s", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
