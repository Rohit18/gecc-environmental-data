"""Stage 1 sanity checks on the draft tables (test tiles) and test-area summaries.

Checks: values in [-1, 1]; valid_w <= total_w; every null value sits in a row whose status is not ok and every
ok row has all values; Columbus (19234) JJA rural baseline greener than its built-heavy pieces.
Summaries per test UA (Columbus + the largest new UA of HI, GU, MP, AS, VI) and season: UA values and coverage,
piece value / delta / coverage distributions, status counts, composite depth.
Output: qa/stage1_checks.json, qa/stage1_test_area_summary.csv
"""
import json
import os

import numpy as np
import pandas as pd

from ndvi_common import load_config, utc_now, write_json_atomic

cfg = load_config()
D = os.path.join(cfg["paths"]["output"], "stage1_test")
tp = pd.read_parquet(f"{D}/tract_piece_ndvi_evi_2020.parquet")
tu = pd.read_parquet(f"{D}/ua_ndvi_evi_2020.parquet")
dg = pd.read_parquet(f"{D}/piece_diagnostics_ndvi.parquet")
TEST_UAS = {"19234": "Columbus, OH", "39889": "Honolulu, HI", "22811": "Dededo--Apotgan--Tamuning, GU",
            "32506": "Garapan, MP", "86369": "Tafuna--Pago Pago, AS", "16750": "Christiansted--Frederiksted, VI"}
out = {"utc": utc_now(), "checks": {}}
vcols = [c for c in tp.columns if c.split("_")[0] in ("urban", "rural", "delta") and c.endswith(("ndvi", "evi"))]
ucols = [c for c in tu.columns if c.startswith(("rural_", "ua_all_", "delta_all_")) and c.endswith(("ndvi", "evi"))]
chk = out["checks"]
chk["values_in_range"] = {c: bool(tp[c].dropna().between(-1, 1).all()) for c in vcols if not c.startswith("delta")}
chk["values_in_range"].update({c: bool(tu[c].dropna().between(-1, 1).all()) for c in ucols if not c.startswith("delta")})
chk["delta_in_range"] = bool(all(tp[c].dropna().between(-2, 2).all() for c in vcols if c.startswith("delta")))
tol = 1e-9
chk["valid_le_total"] = bool(all((tp[f"{x}_urban_valid_weight_m2"] <= tp[f"{x}_urban_total_weight_m2"] * (1 + tol)).all()
                                 for x in ("ndvi", "evi")) and
                             all((tu[f"{z}_{x}_valid_weight_m2"] <= tu[f"{z}_{x}_total_weight_m2"] * (1 + tol)).all()
                                 for z in ("rural", "ua_all") for x in ("ndvi", "evi")))
st = tp["quality_status"]
null_any = tp[vcols].isna().any(axis=1)
chk["null_rows_have_non_ok_status"] = bool((st[null_any] != "ok").all())
chk["ok_rows_complete"] = bool(tp.loc[st == "ok", vcols].notna().all().all())
chk["null_urban_only_with_status"] = tp.loc[tp.urban_ndvi.isna(), "quality_status"].value_counts().to_dict()
chk["status_counts"] = {s: g.quality_status.value_counts().to_dict() for s, g in tp.groupby("season")}
chk["ndvi_evi_status_differ_rows"] = 0 if "quality_status" in tp else None
chk["ua_status_counts"] = {s: g.quality_status.value_counts().to_dict() for s, g in tu.groupby("season")}
# Columbus: built-heavy pieces (WorldCover built-up >= 50% of the piece's non-water valid area)
c = tp[(tp.GEOID20 == "19234") & (tp.season == "JJA") & (tp.quality_status == "ok")].merge(
    dg[["piece_id", "built_m2", "valid_m2", "water_m2"]], on="piece_id")
c["built_share"] = c.built_m2 / (c.valid_m2 - c.water_m2)
bh = c[c.built_share >= 0.5]
chk["columbus_jja"] = {"rural_ndvi": float(c.rural_ndvi.iloc[0]), "rural_evi": float(c.rural_evi.iloc[0]),
                       "n_built_heavy_pieces": len(bh), "built_heavy_urban_ndvi_median": float(bh.urban_ndvi.median()),
                       "share_built_heavy_below_rural_ndvi": float((bh.urban_ndvi < bh.rural_ndvi).mean()),
                       "share_built_heavy_below_rural_evi": float((bh.urban_evi < bh.rural_evi).mean()),
                       "spearman_built_share_vs_urban_ndvi": float(c[["built_share", "urban_ndvi"]].corr("spearman").iloc[0, 1]),
                       "pass": bool(bh.urban_ndvi.median() < c.rural_ndvi.iloc[0])}
out["pass"] = bool(all(chk["values_in_range"].values()) and chk["delta_in_range"] and chk["valid_le_total"]
                   and chk["null_rows_have_non_ok_status"] and chk["ok_rows_complete"] and chk["columbus_jja"]["pass"])

# ------------------------------------------------------------------ test-area summaries
rows = []
q = lambda s, p: float(s.quantile(p)) if s.notna().any() else None  # noqa: E731
for g20, name in TEST_UAS.items():
    for season in cfg["seasons"]:
        u = tu[(tu.GEOID20 == g20) & (tu.season == season)].iloc[0]
        p = tp[(tp.GEOID20 == g20) & (tp.season == season)]
        pn = p[~p.sliver_flag]
        r = {"GEOID20": g20, "NAME20": name, "season": season, "ua_complete_in_test_tiles": bool(u.test_ua_complete),
             "n_pieces": len(p), "n_pieces_non_sliver": len(pn)}
        for x in ("ndvi", "evi"):
            r[f"rural_{x}"] = u[f"rural_{x}"]; r[f"rural_{x}_cov"] = u[f"rural_{x}_valid_coverage"]
            r[f"ua_all_{x}"] = u[f"ua_all_{x}"]; r[f"ua_all_{x}_cov"] = u[f"ua_all_{x}_valid_coverage"]
            r[f"delta_all_{x}"] = u[f"delta_all_{x}"]
        r["rural_clear_obs_mean"] = u["rural_ndvi_clear_obs_mean"]
        r["piece_urban_ndvi_median"] = q(pn.urban_ndvi, .5)
        r["piece_delta_ndvi_p5"] = q(pn.delta_ndvi, .05); r["piece_delta_ndvi_median"] = q(pn.delta_ndvi, .5)
        r["piece_delta_ndvi_p95"] = q(pn.delta_ndvi, .95)
        r["piece_delta_evi_median"] = q(pn.delta_evi, .5)
        r["piece_urban_cov_median"] = q(pn.ndvi_urban_valid_coverage, .5)
        r["piece_urban_cov_p5"] = q(pn.ndvi_urban_valid_coverage, .05)
        r["pieces_cov_lt_0.5"] = int((pn.ndvi_urban_valid_coverage < 0.5).sum())
        r["piece_clear_obs_median"] = q(pn.urban_clear_obs_mean, .5)
        r["status_counts"] = json.dumps(p.quality_status.value_counts().to_dict())
        rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(cfg["paths"]["qa"], "stage1_test_area_summary.csv"), index=False)
out["test_area_summary"] = summ.round(4).to_dict("records")
# scene counts per test-area chunk and season (from the fetch manifest)
man = pd.read_json(os.path.join(cfg["paths"]["composites"], "manifest.jsonl"), lines=True)
man = man[man.C == cfg["grid"]["chunk_cells"]]
ch = pd.read_csv(os.path.join(cfg["paths"]["interim"], "chunks", "stage1_test_tiles.csv"))
uas = pd.read_parquet(cfg["domains"]["uas_all"], columns=["ua_idx", "GEOID20"]).set_index("GEOID20").ua_idx
sc = {}
for g20, name in TEST_UAS.items():
    keys = ch[ch.ua_idx.astype(str).str.split(";").map(lambda l: str(uas[g20]) in l)][["I", "J"]]
    m = man.merge(keys, on=["I", "J"])
    sc[name] = {s: {"n_chunks": len(g), "scene_count_min": int(g.scene_count.min()), "scene_count_max": int(g.scene_count.max()),
                    "nan_share_ndvi_mean": round(float(g.nan_share_ndvi.mean()), 3)} for s, g in m.groupby("season")}
out["scene_counts"] = sc
write_json_atomic(os.path.join(cfg["paths"]["qa"], "stage1_checks.json"), out)
print(json.dumps({k: v for k, v in out.items() if k != "test_area_summary"}, indent=1))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(summ.drop(columns=["status_counts"]).round(3).to_string())
print(summ[["NAME20", "season", "status_counts"]].to_string())
