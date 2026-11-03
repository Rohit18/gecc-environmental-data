"""Stage 5 reduce: sum per-tile partials into the output tables and run post-reduce checks.

Outputs (run profile output_dir; column names from the profile, e.g. census: GEOID/GEOID20/tract_area_m2,
fua_fishnet: grid_id/eFUA_ID/cell_area_m2 + cell_uid, n_listings):
  urban_class_area.parquet  piece_id, <zone id>, <domain id>, year, glc_class, area_m2
  rural_class_area.parquet  <domain id>, year, glc_class, area_m2
  piece_diagnostics.parquet piece_id, <zone id>, <domain id>, [extra], piece_area_m2, <zone area>, overlap_share,
                            valid_m2, water_m2, built_m2, nodata_m2, glc_px (approx; cross-window double counts)
  reduce_checks.json
Refuses to run if any expected tile partial is missing or invalid (unless --tiles is given for a test reduce,
which writes to data/output/test_<name>/ instead).
Usage: python reduce.py [--tiles N39W084 ... --name columbus]
"""
import argparse, json, os, sys

import geopandas as gpd
import numpy as np
import pandas as pd

from urbveg_common import ROOT, load_config
from urbveg_worker import Context, partial_is_valid, partial_path

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", nargs="*")
ap.add_argument("--name", default=None)
ap.add_argument("--fill-flag-share", type=float, default=0.01)
ap.add_argument("--area-tolerance", type=float, default=0.005, help="relative tolerance valid vs polygon area")
a = ap.parse_args()
cfg = load_config()
ctx = Context(cfg)
sig = ctx.signature(cfg["pipeline"]["window_px"])
R = cfg["run"]; Z, D, DN, IDX, ZA, EX = (R["zone_id_col"], R["domain_id_col"], R["domain_name_col"], R["domain_index_col"],
                                         R["zone_area_col"], list(R.get("extra_piece_cols") or []))
tiles = a.tiles or pd.read_csv(R["worldcover_tiles"]).tile.tolist()
outdir = cfg["pipeline"]["output_dir"] if not a.tiles else os.path.join(cfg["pipeline"]["output_dir"], f"test_{a.name or 'subset'}")
bad = [t for t in tiles if not partial_is_valid(partial_path(cfg, t), sig)]
if bad:
    sys.exit(f"STOP: {len(bad)} tiles without a valid partial: {bad[:20]}")
os.makedirs(outdir, exist_ok=True)

df = pd.concat([pd.read_parquet(partial_path(cfg, t)) for t in tiles], ignore_index=True)
attrs = pd.read_parquet(cfg["pipeline"]["pieces_path"], columns=["piece_id", Z, D, *EX, "piece_area_m2", ZA, "overlap_share"])
uas = pd.read_parquet(cfg["layers"]["domains"][R["domain"]]["kept_path"], columns=[IDX, D, DN])

u = (df[df.kind == "urban"].groupby(["id", "year", "glc_class"], as_index=False).area_m2.sum()
     .rename(columns={"id": "piece_id"}).merge(attrs[["piece_id", Z, D]], on="piece_id", how="left"))
u = u[["piece_id", Z, D, "year", "glc_class", "area_m2"]]
r = (df[df.kind == "rural"].groupby(["id", "year", "glc_class"], as_index=False).area_m2.sum()
     .rename(columns={"id": IDX}).merge(uas, on=IDX, how="left"))
r = r[[D, "year", "glc_class", "area_m2"]]
dcols = ["valid_m2", "water_m2", "built_m2", "nodata_m2", "glc_px"]
d = df[df.kind == "diag"].groupby("id")[dcols].sum().rename_axis("piece_id").reset_index()
pdg = attrs[["piece_id", Z, D, *EX, "piece_area_m2", ZA, "overlap_share"]].merge(d, on="piece_id", how="left")
if a.tiles:   # test reduce: keep only pieces touched by these tiles
    pdg = pdg[pdg.piece_id.isin(d.piece_id)]
pdg[dcols] = pdg[dcols].fillna(0)
pdg["glc_px"] = pdg.glc_px.astype(np.int64)

for name, t in (("urban_class_area", u), ("rural_class_area", r), ("piece_diagnostics", pdg)):
    t.to_parquet(os.path.join(outdir, f"{name}.parquet.part"), index=False)
    os.replace(os.path.join(outdir, f"{name}.parquet.part"), os.path.join(outdir, f"{name}.parquet"))

# ---------- checks ----------
fill = cfg["glc_fcs30d"]["fill_values"]
years = cfg["glc_fcs30d"]["years"]
ua_set = uas[D] if not a.tiles else pd.Series(r[D].unique())
rural_tot = r.groupby(D).area_m2.sum()
no_rural = sorted(map(str, set(ua_set) - set(rural_tot[rural_tot > 0].index)))
pdg["valid_over_polygon"] = np.where(pdg.piece_area_m2 > 0, (pdg.valid_m2 + pdg.nodata_m2) / pdg.piece_area_m2, np.nan)
over = pdg[(pdg.valid_m2 + pdg.nodata_m2) > pdg.piece_area_m2 * (1 + a.area_tolerance) + 200]   # +~2 px absolute slack
uf = u.assign(fill=u.glc_class.isin(fill)).groupby([D, "year", "fill"]).area_m2.sum().unstack(fill_value=0)
uf = uf.rename(columns={False: "nonfill", True: "fill"}).reindex(columns=["nonfill", "fill"], fill_value=0)
rf = r.assign(fill=r.glc_class.isin(fill)).groupby([D, "year", "fill"]).area_m2.sum().unstack(fill_value=0)
rf = rf.rename(columns={False: "nonfill", True: "fill"}).reindex(columns=["nonfill", "fill"], fill_value=0)
fs = pd.DataFrame({"urban_fill_share": uf.fill / (uf.fill + uf.nonfill), "rural_fill_share": rf.fill / (rf.fill + rf.nonfill)})
flag = fs[(fs.urban_fill_share > a.fill_flag_share) | (fs.rural_fill_share > a.fill_flag_share)]
fs.reset_index().to_csv(os.path.join(outdir, "ua_fill_share.csv"), index=False)
chk = {
    "tiles": len(tiles), "years": years,
    "n_urban_rows": len(u), "n_rural_rows": len(r), "n_pieces_in_diagnostics": len(pdg),
    "n_pieces_with_valid_area": int((pdg.valid_m2 > 0).sum()),
    "n_uas_expected": int(len(ua_set)), "n_uas_without_rural_area": len(no_rural), "uas_without_rural_area": no_rural[:50],
    "valid_vs_polygon_tolerance": {"relative": a.area_tolerance, "absolute_m2": 200},
    "n_pieces_valid_gt_polygon": int(len(over)),
    "valid_over_polygon_ratio_quantiles": pdg.valid_over_polygon.quantile([.01, .5, .99, 1]).round(5).to_dict(),
    "pieces_valid_gt_polygon_examples": over[["piece_id", Z, D, "piece_area_m2", "valid_m2", "nodata_m2"]].head(20).to_dict("records"),
    "fill_flag_share": a.fill_flag_share, "n_ua_year_flagged_fill": int(len(flag)),
    "flagged_fill_examples": flag.reset_index().head(30).to_dict("records"),
    "conventions": {
        "pixel_area": "authalic sphere R=6371007.181 m, A = R^2*dlon*(sin(lat_top)-sin(lat_bottom)), geodetic latitude; "
                      "same convention as GECC census-tract/FUA UHI WorldCover fractions (cos-lat weights, same R)",
        "polygon_area": f"EPSG:6933 (piece_area_m2, {ZA}), as in GECC UHI",
        "run": R["name"],
        "glc_assignment": "nearest neighbour by integer index; name-cell tile; glc_overhang=" + str(cfg["pipeline"].get("glc_overhang")),
        "rasterize_all_touched": bool(cfg["pipeline"]["rasterize_all_touched"]),
        "glc_fill_classes_kept_in_tables": cfg["glc_fcs30d"]["fill_values"],
        "area_check": "report_only"},
    "total_urban_area_km2": float(u.area_m2.sum() / 1e6 / len(years)),
    "total_rural_area_km2": float(r.area_m2.sum() / 1e6 / len(years)),
}
json.dump(chk, open(os.path.join(outdir, "reduce_checks.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in chk.items() if not k.endswith("examples")}, indent=1, default=str))
