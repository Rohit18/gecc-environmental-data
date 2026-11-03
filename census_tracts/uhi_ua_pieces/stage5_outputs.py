"""Stage 5: release tables, validation gates G1-G8 (qa/validation.json) and summary statistics.

Outputs (data/output/, parquet + csv, long format, IDs as strings with leading zeros):
  tract_piece_uhi_2015_2024   one row per piece x year x season x measure
  ua_uhi_2015_2024            one row per UA x year x season x measure
  piece_diagnostics           static per piece (weights, distinct cells, outside-grid area, class areas,
                              pixel_area_ratio vs EPSG:6933 and vs the same-sphere polygon area)
QA: qa/composite_grid_deviations.csv, qa/ua_uhi_full_detail.parquet (all UA columns incl. the September-rule
regression value), qa/validation.json, qa/summary_stats.{json,csv}, qa/g6_regression_*.csv.
Run only after Stage 4 passed (refuses otherwise).
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from uhi_common import UHI_ROOT, load_config, setup_logging, utc_now, write_json_atomic
from uhi_tables import build_tables

log = setup_logging("stage5_outputs")
cfg = load_config()
gate = json.load(open(f"{UHI_ROOT}/qa/stage4/stage4_gate.json"))
if gate["decision"] != "PASS":
    sys.exit(f"STOP: Stage 4 gate decision is {gate['decision']}")
I, O, Q = cfg["paths"]["interim"], cfg["paths"]["output"], cfg["paths"]["qa"]
os.makedirs(O, exist_ok=True)

pt, ut, st, dev = build_tables(cfg, I)
dev.drop(columns=["path"]).assign(file=dev.path.map(os.path.basename)).to_csv(f"{Q}/composite_grid_deviations.csv", index=False)

PIECE_COLS = ["piece_id", "GEOID", "GEOID20", "NAME20", "year", "season", "temperature_measure",
              "urban_temperature_c", "rural_temperature_c", "uhi_c", "urban_sum_wT", "urban_valid_weight_m2",
              "urban_total_weight_m2", "urban_effective_cells", "urban_distinct_cells", "urban_valid_coverage",
              "overlap_share", "piece_area_m2", "tract_area_m2", "piece_pixel_area_m2", "piece_water_m2",
              "piece_snow_ice_m2", "piece_built_m2", "piece_nodata_m2", "min_valid_days", "max_valid_days",
              "expected_days", "observed_days", "missing_dates", "period_complete", "quality_status"]
UA_COLS = ["GEOID20", "NAME20", "year", "season", "temperature_measure",
           "rural_temperature_c", "rural_sum_wT", "rural_valid_weight_m2", "rural_total_weight_m2",
           "rural_valid_coverage", "rural_effective_cells", "rural_distinct_cells",
           "urban_all_temperature_c", "uhi_all_c", "urban_all_sum_wT", "urban_all_valid_weight_m2",
           "urban_all_total_weight_m2", "urban_all_valid_coverage", "urban_all_effective_cells", "urban_all_distinct_cells",
           "urban_built_temperature_c", "uhi_built_c", "urban_built_sum_wT", "urban_built_valid_weight_m2",
           "urban_built_total_weight_m2", "urban_built_valid_coverage", "urban_built_effective_cells",
           "urban_built_distinct_cells",
           "ua_pixel_area_m2", "ua_water_m2", "ua_snow_ice_m2", "ua_built_m2", "ua_nodata_m2",
           "rural_min_valid_days", "rural_max_valid_days", "urban_all_min_valid_days", "urban_all_max_valid_days",
           "urban_built_min_valid_days", "urban_built_max_valid_days",
           "expected_days", "observed_days", "missing_dates", "period_complete", "quality_status", "built_status"]
DIAG_COLS = ["piece_id", "GEOID", "GEOID20", "NAME20", "overlap_share", "piece_area_m2", "piece_area_sphere_m2",
             "tract_area_m2", "owns_pixels", "pixel_area_m2", "valid_m2", "nodata_m2", "built_m2", "snow_ice_m2",
             "water_m2", "urban_weight_m2", "urban_distinct_cells", "outside_grid_urban_m2", "pixel_area_ratio",
             "pixel_area_ratio_sphere"]
for c in ("min_valid_days", "max_valid_days"):
    pt[c] = pt[c].astype("Int64")
for c in [c for c in UA_COLS if c.endswith("_valid_days")]:
    ut[c] = ut[c].astype("Int64")
P, U, D = pt[PIECE_COLS], ut[UA_COLS], st[DIAG_COLS]


def write(df, name):
    for ext in ("parquet", "csv"):
        p = f"{O}/{name}.{ext}"
        (df.to_parquet if ext == "parquet" else df.to_csv)(p + ".part", index=False)
        os.replace(p + ".part", p)
    log.info("wrote %s (%d rows)", name, len(df))


write(P, "tract_piece_uhi_2015_2024")
write(U, "ua_uhi_2015_2024")
write(D, "piece_diagnostics")
ut.to_parquet(f"{Q}/ua_uhi_full_detail.parquet", index=False)

# ------------------------------------------------------------------------------------------------ gates
V = {"generated_utc": utc_now()}
n_p, n_u, nk = len(st), cfg["urban_areas"]["expected_n"], 40
pk = ["piece_id", "year", "season", "temperature_measure"]
uk = ["GEOID20", "year", "season", "temperature_measure"]
V["G1_row_counts"] = {"piece_rows": len(P), "expected_piece_rows": n_p * nk, "ua_rows": len(U), "expected_ua_rows": n_u * nk,
                      "n_pieces": n_p, "piece_keys_unique": bool(not P.duplicated(pk).any()),
                      "ua_keys_unique": bool(not U.duplicated(uk).any()),
                      "id_strings": {"GEOID_len11": bool(P.GEOID.str.len().eq(11).all()),
                                     "GEOID20_len5": bool(P.GEOID20.str.len().eq(5).all() and U.GEOID20.str.len().eq(5).all())}}
V["G1_row_counts"]["pass"] = bool(len(P) == n_p * nk and len(U) == n_u * nk and V["G1_row_counts"]["piece_keys_unique"]
                                  and V["G1_row_counts"]["ua_keys_unique"] and all(V["G1_row_counts"]["id_strings"].values()))


def g2(df, val, a, b, status):
    both = df[a].notna() & df[b].notna()
    exact = bool((df.loc[both & df[val].notna(), val] == (df.loc[both & df[val].notna(), a] - df.loc[both & df[val].notna(), b])).all())
    ok = df[status].str.startswith("ok")
    return {"exact_difference": exact, "value_iff_status_ok": bool((df[val].notna() == ok).all()),
            "n_value": int(df[val].notna().sum()), "n_ok": int(ok.sum()),
            "n_value_without_ok": int((df[val].notna() & ~ok).sum()), "n_ok_without_value": int((ok & df[val].isna()).sum()),
            "null_rows_status_counts": df.loc[df[val].isna(), status].value_counts().to_dict()}


V["G2_uhi_definition"] = {"piece": g2(P, "uhi_c", "urban_temperature_c", "rural_temperature_c", "quality_status"),
                          "ua_all": g2(U, "uhi_all_c", "urban_all_temperature_c", "rural_temperature_c", "quality_status"),
                          "ua_built": g2(U, "uhi_built_c", "urban_built_temperature_c", "rural_temperature_c", "built_status")}
V["G2_uhi_definition"]["pass"] = all(v["exact_difference"] and v["value_iff_status_ok"]
                                     for k, v in V["G2_uhi_definition"].items() if k != "pass")

u1 = U.drop_duplicates("GEOID20").set_index("GEOID20")
rz = u1[~(u1.rural_total_weight_m2 > 0)]
rz_status = U[U.GEOID20.isin(rz.index)].quality_status.str.replace("+incomplete_period", "", regex=False).value_counts().to_dict()
V["G3_rural_weight"] = {"n_uas_rural_weight_le_0": int(len(rz)), "uas": rz.index.tolist(), "their_statuses": rz_status,
                        "pass": bool(len(rz) == 0 or set(rz_status) <= {"ua_no_valid_temperature", "rural_no_valid_temperature",
                                                                         "outside_temperature_grid"})}
pr = U[U.NAME20.str.endswith(", PR")]
prp = P[P.NAME20.str.endswith(", PR")]
V["G4_puerto_rico"] = {"n_uas": int(pr.GEOID20.nunique()), "ua_status": pr.quality_status.value_counts().to_dict(),
                       "n_pieces": int(prp.piece_id.nunique()), "piece_status": prp.quality_status.value_counts().to_dict()}
V["G4_puerto_rico"]["pass"] = bool(pr.GEOID20.nunique() == cfg["urban_areas"]["puerto_rico_expected_n"]
                                   and pr.quality_status.str.startswith("ua_no_valid_temperature").all()
                                   and prp.quality_status.str.startswith(("ua_no_valid_temperature", "piece_no_pixels")).all())
inc = U[~U.period_complete][["year", "season", "temperature_measure", "missing_dates"]].drop_duplicates()
V["G5_incomplete_periods"] = {"incomplete": inc.to_dict("records"),
                              "pass": bool(sorted(map(tuple, inc[["year", "season", "missing_dates"]].drop_duplicates().values.tolist()))
                                           == [(2016, "DJF_same_year", "2016-12-31"), (2020, "DJF_same_year", "2020-12-31"),
                                               (2024, "DJF_same_year", "2024-12-31")]
                                           and len(inc) == 6
                                           and bool(P[~P.period_complete].quality_status.str.endswith("+incomplete_period").all())
                                           and bool(~P[P.period_complete].quality_status.str.contains("incomplete").any()))}

# G6 regression vs September (review flag)
sep = pd.read_parquet(f"{I}/september_release/{cfg['paths']['september_release_ua_member']}")
sep = sep[sep.year.isin(cfg["years"])].rename(columns={"UACE20": "GEOID20", "quality_status": "sep_status"})
m = ut.merge(sep[["GEOID20", "year", "season", "temperature_measure", "ua_temperature_c", "sep_status"]], on=uk, how="left")
m["diff"] = m.regr_nowater_snowkept_temperature_c - m.ua_temperature_c
ok = m["diff"].notna()
q = [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]
per = m[ok].groupby(["GEOID20", "NAME20"])["diff"].agg(["mean", "min", "max"]).reset_index()
per["max_abs"] = per[["min", "max"]].abs().max(axis=1)
# likely reasons: boundary change (TIGER 2025 vs corrected 2020) and cell-weight differences (water / nodata)
flag = per[per.max_abs > 0.05].copy()
if len(flag):
    sep_ua = gpd.read_file("/path/to/home/GECC/tl_2025_us_uac20", columns=["UACE20", "geometry"])
    sep_ua = sep_ua[sep_ua.UACE20.isin(flag.GEOID20)].to_crs("EPSG:6933").set_index("UACE20")
    ours = gpd.read_parquet(cfg["urban_areas"]["kept_path"])
    ours = ours[ours.GEOID20.isin(flag.GEOID20)].to_crs("EPSG:6933").set_index("GEOID20")
    flag["boundary_sym_diff_share"] = [ours.geometry[g].symmetric_difference(sep_ua.geometry[g]).area / ours.geometry[g].area
                                       if g in sep_ua.index else np.nan for g in flag.GEOID20]
    sw = pd.read_parquet("/path/to/project/gecc/census_tract_uhi/geometry/zone_cell_weights.parquet",
                         columns=["zone_type", "UACE20", "weight_urban"], filters=[("zone_type", "==", "ua")])
    ca = float(cfg["temperature_grid"]["dx"]) * float(cfg["temperature_grid"]["dy"])
    sw_tot = sw.groupby("UACE20").weight_urban.sum() * ca
    one = ut[(ut.year == 2020) & (ut.season == "JJA") & (ut.temperature_measure == "TMAX")].set_index("GEOID20")
    flag["weight_new_km2"] = flag.GEOID20.map(one.regr_nowater_snowkept_total_weight_m2) / 1e6
    flag["weight_september_km2"] = flag.GEOID20.map(sw_tot) / 1e6
    flag["water_share_of_ua_pixels"] = flag.GEOID20.map(one.ua_water_m2 / one.ua_pixel_area_m2)
    flag["new_valid_coverage_2020JJA"] = flag.GEOID20.map(one.regr_nowater_snowkept_valid_coverage)

    def reason(r):
        rs = []
        if r.boundary_sym_diff_share > 0.01:
            rs.append(f"UA boundary change {r.boundary_sym_diff_share:.1%}")
        if abs(r.weight_new_km2 / r.weight_september_km2 - 1) > 0.05:
            rs.append(f"cell weights differ ({r.weight_new_km2:.1f} vs {r.weight_september_km2:.1f} km2; water share "
                      f"{r.water_share_of_ua_pixels:.1%}): September scaled cells by the cell-wide water share")
        if r.new_valid_coverage_2020JJA < 0.99:
            rs.append(f"no-temperature cells ({1 - r.new_valid_coverage_2020JJA:.1%} of weight)")
        return "; ".join(rs) or "small UA / other exact-weight differences"
    flag["likely_reason"] = flag.apply(reason, axis=1)
per.to_csv(f"{Q}/g6_regression_per_ua.csv", index=False)
flag.to_csv(f"{Q}/g6_regression_flagged_uas.csv", index=False)
V["G6_regression_september"] = {
    "n_rows": int(len(m)), "n_compared": int(ok.sum()), "n_uas_compared": int(m[ok].GEOID20.nunique()),
    "n_uas_not_in_september": int(len(set(ut.GEOID20) - set(sep.GEOID20))),
    "uas_not_in_september": sorted(set(ut.GEOID20) - set(sep.GEOID20)),
    "n_rows_new_value_sep_null": int((m.regr_nowater_snowkept_temperature_c.notna() & m.ua_temperature_c.isna() & m.sep_status.notna()).sum()),
    "n_rows_sep_value_new_null": int((m.regr_nowater_snowkept_temperature_c.isna() & m.ua_temperature_c.notna()).sum()),
    "diff_quantiles": dict(zip(map(str, q), m.loc[ok, "diff"].quantile(q).round(5).tolist())),
    "mean_abs_diff": float(m.loc[ok, "diff"].abs().mean()),
    "n_rows_abs_gt_0.05": int((m.loc[ok, "diff"].abs() > 0.05).sum()), "n_uas_abs_gt_0.05": int(len(flag)),
    "flagged": flag.round(4).to_dict("records"), "review_flag_only": True}

# G7 pooled pieces vs UA
pp = P.groupby(uk)[["urban_sum_wT", "urban_valid_weight_m2"]].sum()
pp["pooled_c"] = pp.urban_sum_wT / pp.urban_valid_weight_m2
g7 = U.set_index(uk)[["urban_all_temperature_c", "urban_all_valid_weight_m2"]].join(pp)
g7["diff"] = g7.pooled_c - g7.urban_all_temperature_c
g7["weight_not_in_pieces_m2"] = g7.urban_all_valid_weight_m2 - g7.urban_valid_weight_m2
bad7 = g7[g7["diff"].abs() > 0.01]
V["G7_pooled_pieces_vs_ua"] = {"n_compared": int(g7["diff"].notna().sum()), "max_abs_diff": float(g7["diff"].abs().max()),
                               "n_abs_gt_0.01": int(len(bad7)),
                               "max_abs_weight_not_in_pieces_m2": float(g7.weight_not_in_pieces_m2.abs().max()),
                               "rel_weight_not_in_pieces_max": float((g7.weight_not_in_pieces_m2.abs() / g7.urban_all_valid_weight_m2).max()),
                               "pass": bool(len(bad7) == 0)}
# G8 plausibility
tcols_p = ["urban_temperature_c", "rural_temperature_c"]
tcols_u = ["rural_temperature_c", "urban_all_temperature_c", "urban_built_temperature_c"]
ext = {c: [float(P[c].min()), float(P[c].max())] for c in tcols_p}
ext.update({f"ua_{c}": [float(U[c].min()), float(U[c].max())] for c in tcols_u})
uhi = {"piece_uhi_c": [float(P.uhi_c.min()), float(P.uhi_c.max())], "uhi_all_c": [float(U.uhi_all_c.min()), float(U.uhi_all_c.max())],
       "uhi_built_c": [float(U.uhi_built_c.min()), float(U.uhi_built_c.max())]}
ext_rows = {}
for name, df, col in (("piece_uhi_min", P, "uhi_c"), ("ua_uhi_all_max", U, "uhi_all_c")):
    i = df[col].idxmin() if name.endswith("min") else df[col].idxmax()
    ext_rows[name] = df.loc[i, [c for c in ("piece_id", "GEOID", "GEOID20", "NAME20", "year", "season", "temperature_measure", col)
                                if c in df]].to_dict()
for name, df, col in (("piece_uhi_max", P, "uhi_c"), ("ua_uhi_all_min", U, "uhi_all_c")):
    i = df[col].idxmax() if name.endswith("max") else df[col].idxmin()
    ext_rows[name] = df.loc[i, [c for c in ("piece_id", "GEOID", "GEOID20", "NAME20", "year", "season", "temperature_measure", col)
                                if c in df]].to_dict()
V["G8_plausibility"] = {"temperature_ranges": ext, "uhi_ranges": uhi, "extreme_rows": ext_rows,
                        "pass": bool(all(-60 <= a and b <= 60 for a, b in ext.values()) and all(abs(a) < 10 and abs(b) < 10 for a, b in uhi.values()))}
V["all_hard_gates_pass"] = all(V[g]["pass"] for g in ("G1_row_counts", "G2_uhi_definition", "G3_rural_weight", "G4_puerto_rico",
                                                       "G5_incomplete_periods", "G7_pooled_pieces_vs_ua", "G8_plausibility"))
V["status_counts"] = {"piece": P.quality_status.value_counts().to_dict(), "ua": U.quality_status.value_counts().to_dict(),
                      "ua_built": U.built_status.value_counts().to_dict()}
V["composite_grid_deviation_max"] = {c: float(dev[c].abs().max()) for c in ("dx0_m", "dy0_m", "ddx_m", "ddy_m")}
V["composite_grid_all_ok"] = bool(dev.ok.all())
write_json_atomic(f"{Q}/validation.json", V)
log.info("gates: %s", {g: V[g].get("pass") for g in V if g.startswith("G")})

# ------------------------------------------------------------------------------------------------ summary stats
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
rows = []
for (s, mm), g in P.groupby(["season", "temperature_measure"]):
    for lvl, d, col in (("piece", g, "uhi_c"),):
        v = d[col].dropna()
        rows.append({"level": lvl, "value": col, "season": s, "measure": mm, "n": len(v), **{f"p{int(x * 100)}": v.quantile(x) for x in qs},
                     "share_positive": float((v > 0).mean())})
for (s, mm), g in U.groupby(["season", "temperature_measure"]):
    for col in ("uhi_all_c", "uhi_built_c"):
        v = g[col].dropna()
        rows.append({"level": "ua", "value": col, "season": s, "measure": mm, "n": len(v), **{f"p{int(x * 100)}": v.quantile(x) for x in qs},
                     "share_positive": float((v > 0).mean())})
S = pd.DataFrame(rows)
S.to_csv(f"{Q}/summary_stats.csv", index=False)
q2 = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
one = P[(P.year == 2020) & (P.season == "JJA") & (P.temperature_measure == "TMAX") & P.quality_status.str.startswith("ok")]
support = {"piece_effective_cells": dict(zip(map(str, q2), one.urban_effective_cells.quantile(q2).round(4).tolist())),
           "piece_distinct_cells": dict(zip(map(str, q2), one.urban_distinct_cells.quantile(q2).round(1).tolist())),
           "pieces_effective_cells_lt": {str(t): int((one.urban_effective_cells < t).sum()) for t in (0.1, 0.25, 0.5, 1, 2, 5)},
           "pieces_distinct_cells_le": {str(t): int((one.urban_distinct_cells <= t).sum()) for t in (1, 2, 3, 5)},
           "n_pieces_ok_2020_JJA_TMAX": int(len(one))}
uone = U[(U.year == 2020) & (U.season == "JJA") & (U.temperature_measure == "TMAX") & U.quality_status.str.startswith("ok")]
support["ua_rural_effective_cells"] = dict(zip(map(str, q2), uone.rural_effective_cells.quantile(q2).round(3).tolist()))
support["ua_urban_all_effective_cells"] = dict(zip(map(str, q2), uone.urban_all_effective_cells.quantile(q2).round(3).tolist()))
support["note"] = "support is static across composites except where cells lack valid temperature; shown for 2020 JJA TMAX"
write_json_atomic(f"{Q}/summary_stats.json", {"generated_utc": utc_now(), "uhi_distributions": S.round(4).to_dict("records"),
                                              "support": support})
log.info("summary written")
