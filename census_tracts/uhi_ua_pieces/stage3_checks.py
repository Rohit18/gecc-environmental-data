"""Stage 3 checks on the reduced full-run weights (data/interim/). Writes qa/stage3/*.

1. every kept UA has rural weight > 0 (list any that do not)
2. outside_grid_m2 per zone and kind (expected 0; list any)
3. per-piece pixel area / polygon area: report-only (pieces outside 0.999-1.001 listed in a CSV), against the
   EPSG:6933 polygon area (piece_area_m2) and against the polygon area on the same sphere as the pixel areas.
   Hard check: total pixel area of all pieces / total polygon area within 1e-4 of 1 (both areas reported).
4. exact equality with the vegetation run's piece diagnostics: valid, water, built-up, nodata area per piece.
5. snow/ice (WorldCover 70) area per UA.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from uhi_common import UHI_ROOT, load_config, setup_logging, utc_now, write_json_atomic
from uhi_tables import SPHERE_CEA

log = setup_logging("stage3_checks")
cfg = load_config()
I = cfg["paths"]["interim"]
qd = f"{UHI_ROOT}/qa/stage3"
os.makedirs(qd, exist_ok=True)
R = {"generated_utc": utc_now()}
uas = gpd.read_parquet(cfg["urban_areas"]["kept_path"])
pcs = gpd.read_parquet(cfg["pieces"]["path"])
dua = pd.read_parquet(f"{I}/uhi_diagnostics_ua.parquet").set_index("ua_idx")
dpc = pd.read_parquet(f"{I}/uhi_diagnostics_piece.parquet").set_index("piece_id")
import json  # noqa: E402
R["reduce_report"] = {k: v for k, v in json.load(open(f"{I}/reduce_report.json")).items() if k != "tiles"}

# 1 rural weight
u = uas.set_index("ua_idx")[["GEOID20", "NAME20"]].join(dua, how="left")
bad = u[~(u.weight_rural_m2 > 0)]
R["check1_rural_weight"] = {"n_uas": int(len(u)), "n_uas_in_diagnostics": int(u.total_m2.notna().sum()),
                            "n_rural_weight_le_0": int(len(bad)),
                            "uas": bad.reset_index()[["ua_idx", "GEOID20", "NAME20", "total_m2", "wc50_m2", "wc80_m2",
                                                      "weight_rural_m2"]].to_dict("records")}
# 2 outside grid
og = {}
for name, d, kinds in (("piece", dpc, ["urban"]), ("ua", dua, ["rural", "ua_all", "ua_built", "ua_nowater_snowkept"])):
    for k in kinds:
        c = f"outside_grid_{k}_m2"
        og[f"{name}_{k}"] = {"total_m2": float(d[c].sum()), "n_zones_gt0": int((d[c] > 0).sum()),
                             "zones": d[d[c] > 0][[c]].reset_index().head(50).to_dict("records")}
R["check2_outside_grid"] = og

# 3 area ratio
p = pcs.set_index("piece_id")
sph = pcs.to_crs(SPHERE_CEA).set_index("piece_id").area
pix = dpc.total_m2.reindex(p.index).fillna(0.0)
ar = pd.DataFrame({"GEOID": p.GEOID, "GEOID20": p.GEOID20, "piece_area_m2": p.piece_area_m2, "piece_area_sphere_m2": sph,
                   "pixel_area_m2": pix})
ar["pixel_area_ratio"] = ar.pixel_area_m2 / ar.piece_area_m2
ar["pixel_area_ratio_sphere"] = ar.pixel_area_m2 / ar.piece_area_sphere_m2
ar["lat"] = p.geometry.representative_point().y
q = [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]
own = ar.pixel_area_m2 > 0
out6933 = ar[own & ((ar.pixel_area_ratio < 0.999) | (ar.pixel_area_ratio > 1.001))]
outsph = ar[own & ((ar.pixel_area_ratio_sphere < 0.999) | (ar.pixel_area_ratio_sphere > 1.001))]
out6933.to_csv(f"{qd}/pieces_pixel_area_ratio_outside_0999_1001_epsg6933.csv")
outsph.to_csv(f"{qd}/pieces_pixel_area_ratio_outside_0999_1001_sphere.csv")
ar.to_parquet(f"{qd}/piece_area_ratios.parquet")
lat_bins = pd.cut(ar.lat, [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 72])
tot6933 = float(ar.pixel_area_m2.sum() / ar.piece_area_m2.sum())
totsph = float(ar.pixel_area_m2.sum() / ar.piece_area_sphere_m2.sum())
R["check3_area_ratio"] = {
    "n_pieces": int(len(ar)), "n_pieces_owning_pixels": int(own.sum()), "n_pieces_no_pixels": int((~own).sum()),
    "no_pixel_pieces_max_polygon_area_m2": float(ar.piece_area_m2[~own].max()) if (~own).any() else None,
    "ratio_epsg6933_quantiles": dict(zip(map(str, q), ar.pixel_area_ratio[own].quantile(q).round(6).tolist())),
    "ratio_sphere_quantiles": dict(zip(map(str, q), ar.pixel_area_ratio_sphere[own].quantile(q).round(6).tolist())),
    "n_outside_0999_1001_epsg6933": int(len(out6933)), "n_outside_0999_1001_sphere": int(len(outsph)),
    "median_ratio_epsg6933_by_lat": {str(k): round(float(v), 5) for k, v in ar[own].groupby(lat_bins, observed=True).pixel_area_ratio.median().items()},
    "median_ratio_sphere_by_lat": {str(k): round(float(v), 5) for k, v in ar[own].groupby(lat_bins, observed=True).pixel_area_ratio_sphere.median().items()},
    "extremes_sphere": outsph.assign(d=(outsph.pixel_area_ratio_sphere - 1).abs()).sort_values("d", ascending=False)
                             .head(20).drop(columns="d").reset_index().round(6).to_dict("records"),
    "total_ratio_epsg6933": tot6933, "total_ratio_sphere": totsph,
    # Rule: the governing hard check uses the polygon area on the same sphere as the pixel
    # areas (it tests for lost pixels). The EPSG:6933 ratio is information only: it reflects spherical pixel areas
    # with geodetic latitude (vegetation-run convention), R^2 (1 - e^2 sin^2 phi)^2 / (a^2 (1 - e^2)) - 1 =
    # +0.32% at 18N, 0 at 35.3N, -0.11% at 40N, -0.58% at 61N.
    "hard_check": {"governing_basis": "same_sphere_polygon_area", "tolerance": 1e-4, "ratio": totsph,
                   "pass": abs(totsph - 1) <= 1e-4},
    "info_only_total_ratio_epsg6933": tot6933}

log.info("check3: %s", {k: v for k, v in R["check3_area_ratio"].items() if k != "extremes_sphere"})

# 4 equality with vegetation piece diagnostics
veg = pd.read_parquet(os.path.join(cfg["paths"]["urbveg_root"], "data/output/piece_diagnostics.parquet")).set_index("piece_id")
d = dpc.reindex(veg.index)
new = pd.DataFrame({"valid_m2": d.total_m2.fillna(0) - d.wc0_m2.fillna(0), "water_m2": d.wc80_m2.fillna(0),
                    "built_m2": d.wc50_m2.fillna(0), "nodata_m2": d.wc0_m2.fillna(0)}, index=veg.index)
eq = {"n_pieces_veg": int(len(veg)), "n_pieces_new_diag": int(len(dpc)),
      "same_piece_ids": bool(set(veg.index) == set(p.index)), "piece_ids_in_new_not_veg": sorted(set(dpc.index) - set(veg.index))[:20]}
diffrows = []
for c in ("valid_m2", "water_m2", "built_m2", "nodata_m2"):
    a, b = new[c].to_numpy(), veg[c].fillna(0).to_numpy()
    ne = a != b
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    eq[c] = {"n_not_exactly_equal": int(ne.sum()), "max_abs_diff_m2": float(np.abs(a - b).max()),
             "max_rel_diff": float(rel[ne].max()) if ne.any() else 0.0}
    if ne.any():
        diffrows.append(pd.DataFrame({"piece_id": veg.index[ne], "column": c, "new": a[ne], "veg": b[ne], "diff": (a - b)[ne]}))
dd = pd.concat(diffrows) if diffrows else pd.DataFrame(columns=["piece_id", "column", "new", "veg", "diff"])
dd.to_csv(f"{qd}/vegetation_equality_differences.csv", index=False)
if len(dd):
    # pieces crossing tile edges are summed over several tiles; summation order can differ by 1 ulp
    ntile = None
    pt = {}
    for t in pd.read_csv(cfg["paths"]["worldcover_tiles_manifest"]).tile:
        x = pd.read_parquet(os.path.join(I, "partials", f"{t}.parquet"), filters=[("kind", "==", "diag_piece_total")])
        for pid in x.zone_id:
            pt[pid] = pt.get(pid, 0) + 1
    dd["n_tiles_of_piece"] = dd.piece_id.map(pt)
    eq["differing_pieces_n_tiles"] = dd.drop_duplicates("piece_id").n_tiles_of_piece.value_counts().to_dict()
    eq["pieces_in_more_than_one_tile"] = int(sum(v > 1 for v in pt.values()))
R["check4_vegetation_equality"] = eq
log.info("check4: %s", eq)

# 5 snow/ice per UA
sn = u[["GEOID20", "NAME20", "total_m2", "wc70_m2"]].copy()
sn["wc70_share"] = sn.wc70_m2 / sn.total_m2
sn.to_csv(f"{qd}/ua_snow_ice_area.csv")
R["check5_snow_ice"] = {"n_uas_with_wc70": int((sn.wc70_m2 > 0).sum()), "total_wc70_m2": float(sn.wc70_m2.sum()),
                        "uas": sn[sn.wc70_m2 > 0].sort_values("wc70_m2", ascending=False).reset_index().round(6).to_dict("records")}
log.info("check5: %d UAs with snow/ice, total %.0f m2", R["check5_snow_ice"]["n_uas_with_wc70"], R["check5_snow_ice"]["total_wc70_m2"])
write_json_atomic(f"{qd}/stage3_checks.json", R)
if not R["check3_area_ratio"]["hard_check"]["pass"]:
    raise SystemExit("STOP: hard check failed: total pixel area / same-sphere polygon area not within 1e-4 of 1")
