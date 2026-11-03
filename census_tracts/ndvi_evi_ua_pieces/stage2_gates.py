"""Stage 2 gates G1-G6 on the full run. Evaluated in order; stops at the first failure (nothing is changed to pass).

G1 coverage: every tile ran (142 partials valid, none failed after retry), no MissingChunkError in the worker log
   (a needed pixel without a fetched chunk raises), and every chunk-season of the chunk list is in the manifest.
G2 ownership (rel 1e-9): for all 75,383 original pieces, diagnostics valid/water/built/nodata = vegetation
   piece_diagnostics; urban total_w = vegetation urban class area summed over classes (2020); for all 2,622
   original UAs, rural total_w = vegetation rural class area summed over classes.
G3 baselines: all 2,644 UAs have rural total_w > 0. UAs with rural valid_w = 0 in a season are listed with the
   reason from the fetch manifest (no scenes in any of the UA's chunks, or scenes but no clear observation).
G4 rows: piece rows = pieces x 2, UA rows = 2,644 x 2, unique keys; the first 75,383 piece_ids join one-to-one
   with urban_rural_veg on (piece_id, GEOID, GEOID20).
G5 values: values in [-1, 1]; valid_w <= total_w; delta == urban - rural exactly; every null in a non-ok row and
   every ok row complete; per UA and composite, pooled pieces vs ua_all: relative total-weight difference <= 1e-4
   and |pooled value - ua_all value| <= 1e-4. Every UA x composite whose pooled weight differs from ua_all by
   more than 1e-9 (relative) is listed with the pixel area involved.
G6 additivity: reducing the partials in reverse tile order gives identical tables (max abs diff < 1e-12).
   Also reported (information): the same with plain order-dependent summation.
Output: qa/stage2_gates.json, qa/stage2_g5_pooling_exceptions.csv, qa/stage2_g3_no_rural_value.csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from ndvi_common import SEASON_SHORT, load_config, utc_now, write_json_atomic
from ndvi_tables import build, reduce_partials

cfg = load_config()
QA = cfg["paths"]["qa"]
OUT = os.path.join(cfg["paths"]["output"], "census_tracts_ua_ndvi")
VEG = "/path/to/scratch/urbveg/data/output/census_tracts_ua"
NV_P, NV_U = cfg["domains"]["n_pieces_veg"], cfg["domains"]["n_uas_veg"]
REL = 1e-9
res = {"utc": utc_now(), "gates": {}}


def finish(ok_all):
    res["pass"] = bool(ok_all)
    write_json_atomic(os.path.join(QA, "stage2_gates.json"), res)
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if ok_all else 2)


def gate(name, ok, **info):
    res["gates"][name] = {"pass": bool(ok), **info}
    print(f"{name}: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        res["stopped_at"] = name
        finish(False)


def relcmp(a, b):
    idx = a.index.union(b.index)
    x = a.reindex(idx).fillna(0.0).to_numpy(np.float64)
    y = b.reindex(idx).fillna(0.0).to_numpy(np.float64)
    m = np.maximum(np.abs(x), np.abs(y))
    d = np.abs(x - y)
    bad = d > REL * m
    return {"n": int(idx.size), "n_mismatch": int(bad.sum()), "max_rel_diff": float(np.max(np.where(m > 0, d / m, 0))),
            "examples": [int(i) for i in idx[bad][:10]]}


tiles = pd.read_csv(cfg["worldcover"]["tiles_manifest_all"]).tile.tolist()
pdir = cfg["paths"]["partials_dir"]
tp = pd.read_parquet(os.path.join(OUT, "tract_piece_ndvi_evi_2020.parquet"))
tu = pd.read_parquet(os.path.join(OUT, "ua_ndvi_evi_2020.parquet"))
dg = pd.read_parquet(os.path.join(OUT, "piece_diagnostics_ndvi.parquet"))
red = pd.read_parquet(os.path.join(OUT, "reduced_sums.parquet"))
uas = pd.read_parquet(cfg["domains"]["uas_all"], columns=["ua_idx", "GEOID20", "NAME20"])
pieces = pd.read_parquet(cfg["domains"]["pieces_all"], columns=["piece_id", "GEOID", "GEOID20", "ua_idx"])
STATUS = ["quality_status"] if "quality_status" in tp else ["quality_status_ndvi", "quality_status_evi"]

# ---------------------------------------------------------------- G1
logs = sorted(glob.glob(os.path.join(cfg["paths"]["logs"], "run_all_*.json")), key=os.path.getmtime)
ra = json.load(open(logs[-1]))
chunks = pd.read_csv(os.path.join(cfg["paths"]["interim"], "chunks", "all_uas.csv"))
man = pd.read_json(os.path.join(cfg["paths"]["composites"], "manifest.jsonl"), lines=True)
man = man[man.C == cfg["grid"]["chunk_cells"]].drop_duplicates(["I", "J", "season"], keep="last")
need = [(int(i), int(j), s_) for i, j in zip(chunks.I, chunks.J) for s_ in cfg["seasons"]]
have = set(zip(man.I, man.J, man.season))
missing = [k for k in need if k not in have]
gate("G1_coverage", not ra["failed_after_retry"] and not ra["missing_chunk_errors"] and ra["partials_valid"] == len(tiles)
     and not missing, run_all_log=os.path.basename(logs[-1]), tiles=len(tiles), partials_valid=ra["partials_valid"],
     failed_after_retry=ra["failed_after_retry"], missing_chunk_errors=ra["missing_chunk_errors"],
     chunk_seasons_needed=len(need), chunk_seasons_missing=len(missing))

# ---------------------------------------------------------------- G2
vd = pd.read_parquet(os.path.join(VEG, "piece_diagnostics.parquet")).set_index("piece_id")
od = dg[dg.piece_id <= NV_P].set_index("piece_id")
g2 = {f"diag_{c}": relcmp(od[c], vd[c]) for c in ("valid_m2", "water_m2", "built_m2", "nodata_m2")}
vu = pd.read_parquet(os.path.join(VEG, "urban_class_area.parquet"))
vu = vu[vu.year == 2020].groupby("piece_id").area_m2.sum()
ou = red[(red.kind == "urban") & (red.zone_id <= NV_P)].set_index("zone_id").total_w
g2["urban_total_w"] = relcmp(ou, vu)
vr = pd.read_parquet(os.path.join(VEG, "rural_class_area.parquet"))
vr = vr[vr.year == 2020].groupby("GEOID20").area_m2.sum()
vr.index = vr.index.map(uas.set_index("GEOID20").ua_idx)
orr = red[(red.kind == "rural") & (red.zone_id <= NV_U)].set_index("zone_id").total_w
g2["rural_total_w"] = relcmp(orr, vr)
g2["n_original_pieces_compared"] = int(len(od))
g2["n_original_uas_compared"] = int(orr.index.union(vr.index).size)
gate("G2_ownership", all(v["n_mismatch"] == 0 for k, v in g2.items() if isinstance(v, dict)) and len(od) == NV_P, **g2)

# ---------------------------------------------------------------- G3
rt = red[red.kind == "rural"].set_index("zone_id")
rtot = rt.total_w.reindex(uas.ua_idx).fillna(0)
ch_ua = chunks.assign(ua=chunks.ua_idx.astype(str).str.split(";")).explode("ua")
ch_ua["ua"] = ch_ua.ua.astype(int)
sc = ch_ua.merge(man[["I", "J", "season", "scene_count", "nan_share_ndvi"]], on=["I", "J"])
nov = []
for season in cfg["seasons"]:
    sh = SEASON_SHORT[season]
    for ix in ("NDVI", "EVI"):
        v = rt[f"valid_w_{ix}_{sh}"].reindex(uas.ua_idx).fillna(0)
        for u in v.index[v.to_numpy() <= 0]:
            s = sc[(sc.ua == u) & (sc.season == season)]
            r = uas.set_index("ua_idx").loc[u]
            nov.append({"ua_idx": int(u), "GEOID20": r.GEOID20, "NAME20": r.NAME20, "season": season, "index": ix,
                        "n_chunks": len(s), "max_scene_count": int(s.scene_count.max()) if len(s) else None,
                        "reason": "no scenes" if len(s) and s.scene_count.max() == 0 else "scenes present, no clear observation"})
nov = pd.DataFrame(nov)
nov.to_csv(os.path.join(QA, "stage2_g3_no_rural_value.csv"), index=False)
gate("G3_baselines", bool((rtot > 0).all()), n_uas=len(uas), n_uas_rural_total_w_zero=int((rtot <= 0).sum()),
     n_ua_season_index_without_rural_value=len(nov),
     uas_without_rural_value=(nov[nov["index"] == "NDVI"][["GEOID20", "NAME20", "season", "max_scene_count", "reason"]]
                              .to_dict("records") if len(nov) else []))

# ---------------------------------------------------------------- G4
ns = len(cfg["seasons"])
uv = pd.read_parquet(os.path.join(VEG, "urban_rural_veg.parquet"), columns=["piece_id", "GEOID", "GEOID20"])
g4 = {"piece_rows": len(tp), "expected_piece_rows": len(pieces) * ns, "ua_rows": len(tu), "expected_ua_rows": len(uas) * ns,
      "piece_keys_unique": bool(not tp.duplicated(["piece_id", "season"]).any()),
      "ua_keys_unique": bool(not tu.duplicated(["GEOID20", "season"]).any())}
okj = True
for season in cfg["seasons"]:
    t = tp[(tp.season == season) & (tp.piece_id <= NV_P)][["piece_id", "GEOID", "GEOID20"]]
    j = t.merge(uv, on=["piece_id", "GEOID", "GEOID20"], how="outer", indicator=True)
    g4[f"join_{season}"] = j["_merge"].value_counts().to_dict()
    okj &= bool((j["_merge"] == "both").all() and len(j) == NV_P and not t.piece_id.duplicated().any())
g4["veg_join_one_to_one"] = okj
gate("G4_rows", g4["piece_rows"] == g4["expected_piece_rows"] and g4["ua_rows"] == g4["expected_ua_rows"]
     and g4["piece_keys_unique"] and g4["ua_keys_unique"] and okj, **g4)

# ---------------------------------------------------------------- G5
g5 = {}
vals = [c for c in tp.columns if c.split("_")[0] in ("urban", "rural") and c.endswith(("_ndvi", "_evi"))]
uvals = [c for c in tu.columns if c.startswith(("rural_", "ua_all_")) and c.endswith(("_ndvi", "_evi"))]
g5["values_in_range"] = bool(all(tp[c].dropna().between(-1, 1).all() for c in vals) and
                             all(tu[c].dropna().between(-1, 1).all() for c in uvals))
vl = [(tp, f"{x}_urban_valid_weight_m2", f"{x}_urban_total_weight_m2") for x in ("ndvi", "evi")] + \
     [(tu, f"{z}_{x}_valid_weight_m2", f"{z}_{x}_total_weight_m2") for z in ("rural", "ua_all") for x in ("ndvi", "evi")]
g5["valid_le_total"] = bool(all((d[a] <= d[b]).all() for d, a, b in vl))
dex = {}
for x in ("ndvi", "evi"):
    d = tp[f"delta_{x}"].to_numpy(); e = (tp[f"urban_{x}"] - tp[f"rural_{x}"]).to_numpy()
    dex[x] = bool(np.array_equal(d, e, equal_nan=True))
    du = tu[f"delta_all_{x}"].to_numpy(); eu = (tu[f"ua_all_{x}"] - tu[f"rural_{x}"]).to_numpy()
    dex[f"ua_{x}"] = bool(np.array_equal(du, eu, equal_nan=True))
g5["delta_exact"] = dex
nn = {}
for x in ("ndvi", "evi"):
    st = tp[STATUS[0] if len(STATUS) == 1 else f"quality_status_{x}"]
    cols = [f"urban_{x}", f"rural_{x}", f"delta_{x}"]
    nul = tp[cols].isna().any(axis=1)
    nn[x] = {"null_rows_non_ok": bool((st[nul] != "ok").all()), "ok_rows_complete": bool(tp.loc[st == "ok", cols].notna().all().all()),
             "status_null": int(st.isna().sum())}
g5["nulls_have_status"] = nn
# pooling: pieces vs ua_all
ex, pool_ok = [], True
U = red[red.kind == "urban"].set_index("zone_id")
A = red[red.kind == "ua_all"].set_index("zone_id")
pu = pieces.set_index("piece_id").ua_idx
Ug = U.join(pu.rename("ua_idx"), how="left").groupby("ua_idx").sum(numeric_only=True)
Ug = Ug.reindex(uas.ua_idx).fillna(0)
Aa = A.reindex(uas.ua_idx).fillna(0)
maxw, maxv = 0.0, 0.0
for season in cfg["seasons"]:
    sh = SEASON_SHORT[season]
    for ix in ("NDVI", "EVI"):
        c = f"{ix}_{sh}"
        wp, wa = Ug.total_w.to_numpy(), Aa.total_w.to_numpy()
        relw = np.where(np.maximum(wp, wa) > 0, np.abs(wp - wa) / np.maximum(wp, wa), 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            vp = Ug[f"sum_wv_{c}"].to_numpy() / Ug[f"valid_w_{c}"].to_numpy()
            va = Aa[f"sum_wv_{c}"].to_numpy() / Aa[f"valid_w_{c}"].to_numpy()
        dv = np.abs(vp - va)
        both = np.isfinite(vp) & np.isfinite(va)
        one = np.isfinite(vp) != np.isfinite(va)
        maxw = max(maxw, float(relw.max())); maxv = max(maxv, float(np.nanmax(np.where(both, dv, 0))))
        bad = (relw > 1e-4) | (both & (dv > 1e-4)) | one
        pool_ok &= not bad.any()
        for k in np.nonzero((relw > REL) | bad)[0]:
            ex.append({"ua_idx": int(uas.ua_idx.iloc[k]), "GEOID20": uas.GEOID20.iloc[k], "NAME20": uas.NAME20.iloc[k],
                       "composite": c, "pieces_total_w": wp[k], "ua_all_total_w": wa[k], "pixel_area_diff_m2": wp[k] - wa[k],
                       "rel_weight_diff": relw[k], "pooled_value": vp[k], "ua_all_value": va[k],
                       "abs_value_diff": dv[k] if both[k] else None, "exceeds_gate": bool(bad[k])})
ex = pd.DataFrame(ex)
ex.to_csv(os.path.join(QA, "stage2_g5_pooling_exceptions.csv"), index=False)
g5["pooling"] = {"max_rel_weight_diff": maxw, "max_abs_value_diff": maxv,
                 "n_ua_composite_listed": len(ex), "n_uas_listed": int(ex.ua_idx.nunique()) if len(ex) else 0,
                 "n_exceeding_gate": int(ex.exceeds_gate.sum()) if len(ex) else 0,
                 "pixel_area_diff_m2_abs_max": float(ex.pixel_area_diff_m2.abs().max()) if len(ex) else 0.0,
                 "pixel_area_diff_m2_abs_sum_one_composite": float(ex[ex.composite == "NDVI_JJA"].pixel_area_diff_m2.abs().sum()) if len(ex) else 0.0}
ok5 = g5["values_in_range"] and g5["valid_le_total"] and all(dex.values()) and \
    all(v["null_rows_non_ok"] and v["ok_rows_complete"] and v["status_null"] == 0 for v in nn.values()) and pool_ok
gate("G5_values", ok5, **g5)

# ---------------------------------------------------------------- G6
paths = [f"{pdir}/{t}.parquet" for t in tiles]
red_rev = reduce_partials(paths[::-1], canonical=True)
pg = pd.read_parquet(cfg["domains"]["pieces_all"])
import geopandas as gpd  # noqa: E402
pg = gpd.read_parquet(cfg["domains"]["pieces_all"])
ug = gpd.read_parquet(cfg["domains"]["uas_all"])
split = len(STATUS) == 2
tp2, tu2, dg2, _ = build(cfg, red_rev, pg, ug, split_status=split)


def maxdiff(a, b):
    m = 0.0; same_obj = True
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and not pd.api.types.is_bool_dtype(a[c]):
            x, y = a[c].to_numpy(np.float64), b[c].to_numpy(np.float64)
            d = np.abs(x - y)[~(np.isnan(x) & np.isnan(y))]
            if (np.isnan(x) != np.isnan(y)).any():
                return float("inf"), False
            m = max(m, float(d.max()) if d.size else 0.0)
        else:
            same_obj &= bool(a[c].astype(str).equals(b[c].astype(str)))
    return m, same_obj


g6 = {}
for n, a, b in (("tract_piece", tp, tp2), ("ua", tu, tu2), ("diag", dg, dg2)):
    md, so = maxdiff(a, b)
    g6[n] = {"max_abs_diff": md, "non_numeric_equal": so}
red_nc = reduce_partials(paths[::-1], canonical=False)
tp3, tu3, _, _ = build(cfg, red_nc, pg, ug, split_status=split)
info = {}
for n, a, b in (("tract_piece", tp, tp3), ("ua", tu, tu3)):
    vc = [c for c in a.columns if c.endswith(("_ndvi", "_evi", "coverage", "clear_obs_mean"))]
    wc = [c for c in a.columns if c.endswith(("_m2", "_sum_wv"))]
    info[n] = {"values_max_abs_diff": maxdiff(a[vc], b[vc])[0], "weights_max_abs_diff_m2": maxdiff(a[wc], b[wc])[0]}
g6["order_dependent_summation_information"] = info
gate("G6_additivity", all(v["max_abs_diff"] < 1e-12 and v["non_numeric_equal"] for k, v in g6.items() if k != "order_dependent_summation_information"), **g6)
finish(True)
