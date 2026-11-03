"""Stage 2 summary statistics by season and index, for all UAs and for the 22 new UAs separately.

Per group x season x index:
  quantiles (p1, p5, p25, p50, p75, p95, p99) of piece urban values and deltas (rows with status ok, all and
  non-sliver) and of UA rural baselines;
  shares of piece urban valid coverage < 0.5 and < 0.9 (pieces that own pixels) and of UA rural coverage;
  counts by quality status; sliver count.
Extremes: the 10 most negative and 10 most positive non-sliver deltas per season x index, with coverage, distinct
cells, piece area, water share and clear-obs mean.
Output: qa/summary_stats.csv (long: group, season, index, statistic, value), qa/summary_stats.json (+ extremes),
qa/summary_extreme_deltas.csv
"""
import json
import os

import numpy as np
import pandas as pd

from ndvi_common import load_config, utc_now, write_json_atomic

cfg = load_config()
OUT = os.path.join(cfg["paths"]["output"], "census_tracts_ua_ndvi")
tp = pd.read_parquet(os.path.join(OUT, "tract_piece_ndvi_evi_2020.parquet"))
tu = pd.read_parquet(os.path.join(OUT, "ua_ndvi_evi_2020.parquet"))
dg = pd.read_parquet(os.path.join(OUT, "piece_diagnostics_ndvi.parquet"))
tp = tp.merge(dg[["piece_id", "owns_pixels", "water_m2", "pixel_area_m2"]], on="piece_id")
tp["water_share"] = tp.water_m2 / tp.pixel_area_m2
Q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
rows, ext = [], []


def st(x, ix):
    return x["quality_status"] if "quality_status" in x else x[f"quality_status_{ix}"]


for group, p_sel, u_sel in (("all", tp.piece_id > 0, tu.ua_idx > 0),
                            ("new_uas", ~tp.in_vegetation_product, ~tu.in_vegetation_product)):
    for season in cfg["seasons"]:
        P = tp[p_sel & (tp.season == season)]
        Uu = tu[u_sel & (tu.season == season)]
        add = lambda ix, k, v: rows.append({"group": group, "season": season, "index": ix, "statistic": k, "value": v})  # noqa: E731
        add("-", "n_pieces", len(P)); add("-", "n_uas", len(Uu)); add("-", "n_sliver", int(P.sliver_flag.sum()))
        for ix in ("ndvi", "evi"):
            ok = P[st(P, ix) == "ok"]
            okn = ok[~ok.sliver_flag]
            for name, s in (("urban", ok[f"urban_{ix}"]), ("delta", ok[f"delta_{ix}"]), ("urban_nonsliver", okn[f"urban_{ix}"]),
                            ("delta_nonsliver", okn[f"delta_{ix}"]), ("rural_ua", Uu[f"rural_{ix}"].dropna())):
                add(ix, f"{name}_n", int(s.notna().sum()))
                add(ix, f"{name}_mean", float(s.mean()) if len(s) else None)
                for q in Q:
                    add(ix, f"{name}_p{int(q * 100)}", float(s.quantile(q)) if len(s) else None)
            pc = P.loc[P.owns_pixels, f"{ix}_urban_valid_coverage"].fillna(0)
            add(ix, "piece_cov_lt_0.5_share", float((pc < 0.5).mean()) if len(pc) else None)
            add(ix, "piece_cov_lt_0.9_share", float((pc < 0.9).mean()) if len(pc) else None)
            rc = Uu[f"rural_{ix}_valid_coverage"].fillna(0)
            add(ix, "ua_rural_cov_lt_0.5_share", float((rc < 0.5).mean()) if len(rc) else None)
            add(ix, "ua_rural_cov_lt_0.9_share", float((rc < 0.9).mean()) if len(rc) else None)
            for k, v in st(P, ix).value_counts().items():
                add(ix, f"status_{k}", int(v))
            if group == "all":
                cand = okn.dropna(subset=[f"delta_{ix}"])
                for side, sub in (("most_negative", cand.nsmallest(10, f"delta_{ix}")), ("most_positive", cand.nlargest(10, f"delta_{ix}"))):
                    for r in sub.itertuples():
                        ext.append({"season": season, "index": ix, "side": side, "piece_id": r.piece_id, "GEOID": r.GEOID,
                                    "GEOID20": r.GEOID20, "NAME20": r.NAME20, "delta": getattr(r, f"delta_{ix}"),
                                    "urban": getattr(r, f"urban_{ix}"), "rural": getattr(r, f"rural_{ix}"),
                                    "urban_valid_coverage": getattr(r, f"{ix}_urban_valid_coverage"),
                                    "rural_valid_coverage": getattr(r, f"{ix}_rural_valid_coverage"),
                                    "urban_distinct_cells": r.urban_distinct_cells, "urban_clear_obs_mean": r.urban_clear_obs_mean,
                                    "piece_area_m2": r.piece_area_m2, "overlap_share": r.overlap_share,
                                    "water_share": r.water_share})
S = pd.DataFrame(rows)
E = pd.DataFrame(ext)
S.to_csv(os.path.join(cfg["paths"]["qa"], "summary_stats.csv"), index=False)
E.to_csv(os.path.join(cfg["paths"]["qa"], "summary_extreme_deltas.csv"), index=False)
js = {"utc": utc_now(), "statistics": {}, "extreme_deltas": E.to_dict("records")}
for (g, s, ix), d in S.groupby(["group", "season", "index"]):
    js["statistics"].setdefault(g, {}).setdefault(s, {})[ix] = dict(zip(d.statistic, d.value))
write_json_atomic(os.path.join(cfg["paths"]["qa"], "summary_stats.json"), js)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
w = S.pivot_table(index=["statistic"], columns=["group", "season", "index"], values="value", aggfunc="first")
print(w.to_string())
print(E.round(3).to_string())
