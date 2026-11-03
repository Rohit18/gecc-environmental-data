"""Stage 5 finalize: vegetation fractions and urban - rural difference.

Reads glc_fcs30d.vegetation_classes from config and REFUSES to run if the list is empty.
Excludes GLC fill classes (config glc_fcs30d.fill_values) from numerator and denominator.
Outputs <run output_dir>/urban_rural_veg.parquet (and .csv): per piece and year
  piece_id, <zone id>, <domain id>, [extra], year, urban_veg_m2, urban_denom_m2, urban_frac,
  rural_veg_m2, rural_denom_m2, rural_frac, diff = urban_frac - rural_frac,
  piece_area_m2, overlap_share, sliver_flag (flag only; thresholds from config finalize.sliver_flag)
Usage: python finalize.py [--indir data/output/test_columbus]
"""
import argparse, json, os, sys

import pandas as pd

from finalize_core import fractions
from urbveg_common import ROOT, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--indir", default=None)
a = ap.parse_args()
cfg = load_config()
veg = cfg["glc_fcs30d"].get("vegetation_classes") or []
if not veg:
    sys.exit("REFUSING: config glc_fcs30d.vegetation_classes is empty; set the vegetation class list first.")
known = set(pd.read_csv(cfg["glc_fcs30d"]["class_table"]).code)
unknown = sorted(set(veg) - known)
if unknown:
    sys.exit(f"REFUSING: vegetation classes not in the GLC class table: {unknown}")
fill = cfg["glc_fcs30d"]["fill_values"]
if set(veg) & set(fill):
    sys.exit("REFUSING: vegetation classes include fill values")
indir = a.indir or cfg["pipeline"]["output_dir"]
R = cfg["run"]; Z, D, EX = R["zone_id_col"], R["domain_id_col"], list(R.get("extra_piece_cols") or [])
u = pd.read_parquet(os.path.join(indir, "urban_class_area.parquet"))
r = pd.read_parquet(os.path.join(indir, "rural_class_area.parquet")).rename(columns={D: "ua"})
dg = pd.read_parquet(os.path.join(indir, "piece_diagnostics.parquet"))
uf, rf = fractions(u[["piece_id", "year", "glc_class", "area_m2"]], r, veg, fill)
uf = uf.rename(columns={"veg_m2": "urban_veg_m2", "denom_m2": "urban_denom_m2", "frac": "urban_frac"})
rf = rf.rename(columns={"ua": D, "veg_m2": "rural_veg_m2", "denom_m2": "rural_denom_m2", "frac": "rural_frac"})
out = dg[["piece_id", Z, D, *EX, "piece_area_m2", "overlap_share"]].merge(uf, on="piece_id", how="left")
years = cfg["glc_fcs30d"]["years"]
out["year"] = out.year.fillna(years[0] if len(years) == 1 else pd.NA)   # pieces with no valid pixels
out = out.merge(rf, on=[D, "year"], how="left")
out["diff"] = out.urban_frac - out.rural_frac
th = cfg["finalize"]["sliver_flag"]
out["sliver_flag"] = out.overlap_share < th["overlap_share_lt"]
if th.get("piece_area_m2_lt") is not None:
    out["sliver_flag"] |= out.piece_area_m2 < th["piece_area_m2_lt"]
cols = ["piece_id", Z, D, *EX, "year", "urban_veg_m2", "urban_denom_m2", "urban_frac", "rural_veg_m2",
        "rural_denom_m2", "rural_frac", "diff", "piece_area_m2", "overlap_share", "sliver_flag"]
out = out[cols].sort_values([Z, D, "year"])
p = os.path.join(indir, "urban_rural_veg")
out.to_parquet(p + ".parquet", index=False)
out.to_csv(p + ".csv", index=False)
meta = {"run": R["name"], "vegetation_classes": veg, "fill_excluded": fill, "sliver_thresholds": th, "n_rows": len(out),
        "n_sliver_flagged": int(out.sliver_flag.sum()), "n_missing_urban_frac": int(out.urban_frac.isna().sum())}
json.dump(meta, open(p + ".meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
