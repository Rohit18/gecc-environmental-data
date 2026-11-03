"""Stage 4: out-of-range gate, all 10 years. Writes qa/stage4/stage4_gate.json and exits 3 (STOP) if any
year built WITHOUT the -95..65 C rule has an out-of-range cell inside a zone (cell with weight > 0 in any
weight table), or exits 4 if a year's records are missing (rescan needed).

extreme_value_cells_<year>.parquet (September extract_and_inventory.py) holds cell_id = row*width + col on the
archive grid plus the cell-centre lon/lat; the mapping to our cell_id is verified by recomputing cell_id from
lon/lat with Grid.cell_ids for every record.
Also (item 7): the cell(s) holding the 2020 composite minimum of -104.7 C: composite, location, nearest kept UA
and distance, and whether the gate covers them.
"""
import glob
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

from uhi_common import UHI_ROOT, Grid, composite_paths, load_config, setup_logging, utc_now, write_json_atomic

log = setup_logging("stage4_gate")
cfg = load_config()
grid = Grid.from_cfg(cfg)
I = cfg["paths"]["interim"]
qd = f"{UHI_ROOT}/qa/stage4"
os.makedirs(qd, exist_ok=True)
kinds = ("urban", "rural", "ua_all", "ua_built", "ua_nowater_snowkept")
W = {k: pd.read_parquet(f"{I}/uhi_weights_{k}.parquet") for k in kinds}
zone_cells = np.unique(np.concatenate([w.cell_id[w.area_m2 > 0].to_numpy() for w in W.values()]))
inv = pd.read_csv(f"{UHI_ROOT}/qa/stage0_composites.csv")
uas = gpd.read_parquet(cfg["urban_areas"]["kept_path"]).set_index("ua_idx")
pcs = pd.read_parquet(cfg["pieces"]["path"], columns=["piece_id", "GEOID", "GEOID20"]).set_index("piece_id")


def first(globs, year):
    for g in globs:
        h = sorted(glob.glob(g.format(year=year)))
        if h:
            return h[0]
    return None


R = {"generated_utc": utc_now(), "n_zone_cells": int(zone_cells.size), "years": {}}
stop, missing = [], []
for y in cfg["years"]:
    rule = inv[inv.year == y].exclude_implausible_daily.fillna(False).astype(bool)
    rec = {"rule_on_all_composites": bool(rule.all()), "rule_on_any_composite": bool(rule.any())}
    p = first(cfg["paths"]["extreme_cells_globs"], y)
    rp = first(cfg["paths"]["inventory_report_globs"], y)
    rep = json.load(open(rp)).get("extreme_value_report", {}) if rp else {}
    rec.update({"records_path": p, "inventory_report": rp,
                "report_total_out_of_range_obs": rep.get("total_extreme_pixel_observations")})
    if p is None:
        if rep.get("total_extreme_pixel_observations") == 0:
            rec.update({"status": "no_out_of_range_values_in_year", "affected_zone_cells": 0})
        else:
            rec["status"] = "RECORDS_MISSING"
            missing.append(y)
        R["years"][y] = rec
        continue
    ex = pd.read_parquet(p)
    mapped = grid.cell_ids(np.radians(ex.lon.to_numpy()), np.radians(ex.lat.to_numpy()))
    rec["mapping_check"] = {"n_records": int(len(ex)), "n_match": int((mapped == ex.cell_id.to_numpy()).sum()),
                            "n_mismatch": int((mapped != ex.cell_id.to_numpy()).sum())}
    hit = ex[ex.cell_id.isin(zone_cells)]
    rec["n_cells_in_year"] = int(len(ex))
    rec["n_obs_in_year"] = int(ex.n_files.sum())
    rec["affected_zone_cells"] = int(len(hit))
    rec["affected_obs"] = int(hit.n_files.sum())
    if len(hit):
        zones = []
        for k, w in W.items():
            z = w[w.cell_id.isin(hit.cell_id) & (w.area_m2 > 0)]
            for zid, g in z.groupby("zone_id"):
                name = (pcs.loc[zid, "GEOID"] + "/" + pcs.loc[zid, "GEOID20"]) if k == "urban" else \
                    uas.loc[zid, "GEOID20"] + " " + uas.loc[zid, "NAME20"]
                zones.append({"kind": k, "zone_id": int(zid), "zone": name, "cells": g.cell_id.tolist(),
                              "weight_m2_in_affected_cells": float(g.area_m2.sum())})
        rec["zones"] = zones
        rec["affected_cells"] = hit.round(5).to_dict("records")
        # composite values at the affected cells (were they excluded / do they look implausible?)
        vals = {}
        for s in cfg["seasons"]:
            for m in cfg["measures"]:
                with rasterio.open(composite_paths(cfg, y, s, m)[0]) as ds:
                    a = ds.read(1).reshape(-1)
                vals[f"{s}_{m}"] = a[hit.cell_id.to_numpy()].round(3).tolist()
        rec["composite_values_at_affected_cells"] = vals
        if not rec["rule_on_all_composites"]:
            stop.append(y)
    rec["status"] = "affected" if len(hit) else "clean"
    R["years"][y] = rec
    log.info("%d: rule=%s cells=%d obs=%d affected_zone_cells=%d mapping_mismatch=%s", y, rec["rule_on_all_composites"],
             rec["n_cells_in_year"], rec["n_obs_in_year"], rec["affected_zone_cells"], rec["mapping_check"]["n_mismatch"])

# ---- item 7: the -104.7 C cell(s) in 2020
ua6933 = uas.to_crs("EPSG:6933")
item7 = []
for s in cfg["seasons"]:
    for m in cfg["measures"]:
        mp = composite_paths(cfg, 2020, s, m)[0]
        with rasterio.open(mp) as ds:
            a = ds.read(1).reshape(-1)
            nod = ds.nodata
        v = a[a != nod]
        mn = float(v.min())
        if mn > -100:
            continue
        cells = np.flatnonzero(np.isclose(a, mn, atol=0.05) & (a != nod))
        lon, lat = grid.cell_centre_lonlat_deg(cells)
        ex = pd.read_parquet(first(cfg["paths"]["extreme_cells_globs"], 2020))
        for c, lo, la in zip(cells, lon, lat):
            pt = gpd.GeoSeries([shapely.Point(lo, la)], crs="EPSG:4326").to_crs("EPSG:6933").iloc[0]
            dist = ua6933.distance(pt)
            j = dist.idxmin()
            item7.append({"composite": os.path.basename(mp), "value_c": round(float(a[c]), 3), "cell_id": int(c),
                          "row": int(c // grid.width), "col": int(c % grid.width), "lon": round(float(lo), 5),
                          "lat": round(float(la), 5), "nearest_kept_ua": uas.loc[j, "GEOID20"] + " " + uas.loc[j, "NAME20"],
                          "distance_km_epsg6933": round(float(dist[j]) / 1000, 2),
                          "in_extreme_value_cells_2020": bool(c in set(ex.cell_id)),
                          "n_out_of_range_days_recorded": int(ex.set_index("cell_id").n_files.get(c, 0)),
                          "cell_has_zone_weight": bool(c in set(zone_cells))})
R["item7_min_2020"] = item7
log.info("item 7: %s", item7)
R["stop_years_affected_without_rule"] = stop
R["years_records_missing"] = missing
R["decision"] = "STOP" if stop or missing else "PASS"
write_json_atomic(f"{qd}/stage4_gate.json", R)
log.info("gate decision: %s (stop years %s, missing %s)", R["decision"], stop, missing)
sys.exit(3 if stop else 4 if missing else 0)
