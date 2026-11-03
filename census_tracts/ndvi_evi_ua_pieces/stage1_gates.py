"""Stage 1 mask/ownership gates: NDVI worker partial vs the vegetation partial of the same WorldCover tile.

Exact within rel 1e-9 (|a - b| <= 1e-9 * max(|a|, |b|)):
  G1 per piece: urban total_w = sum over all GLC classes of the vegetation urban class areas (year 2020)
  G2 per UA:    rural total_w = sum over all GLC classes of the vegetation rural class areas
  G3 per piece: diag valid/water/built/nodata m2 = vegetation diag
Also (extra): G4 per UA: ua_all total_w = UHI ua_nowater_snowkept weights (same mask WC not in {0, 80}) summed
over cells (+ outside rows), from the UHI partial of the same tile.
Zones present in one partial but not the other count as mismatches unless the other side is 0.
New pieces / UAs (ids above the vegetation range) are excluded from the comparison and counted separately.
Output: qa/stage1_gates.json (+ qa/stage1_gate_mismatches_<tile>.csv when any).
Usage: python stage1_gates.py <tile> [<tile> ...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from ndvi_common import load_config, utc_now, write_json_atomic

REL = 1e-9


def compare(a, b, name):
    """a, b: Series indexed by zone id. Returns summary and mismatch frame."""
    idx = a.index.union(b.index)
    x = a.reindex(idx).fillna(0.0).to_numpy(np.float64)
    y = b.reindex(idx).fillna(0.0).to_numpy(np.float64)
    tol = REL * np.maximum(np.abs(x), np.abs(y))
    d = np.abs(x - y)
    bad = d > tol
    rel = np.where(np.maximum(np.abs(x), np.abs(y)) > 0, d / np.maximum(np.abs(x), np.abs(y)), 0.0)
    mm = pd.DataFrame({"gate": name, "zone_id": idx[bad], "ndvi": x[bad], "reference": y[bad]})
    return {"n_zones": int(idx.size), "n_ndvi_only": int((~a.index.isin(b.index)).sum()),
            "n_reference_only": int((~b.index.isin(a.index)).sum()), "n_mismatch": int(bad.sum()),
            "max_rel_diff": float(rel.max()) if rel.size else 0.0, "sum_ndvi": float(x.sum()),
            "sum_reference": float(y.sum())}, mm


def main():
    cfg = load_config()
    nv_p, nv_u = cfg["domains"]["n_pieces_veg"], cfg["domains"]["n_uas_veg"]
    out = {"utc": utc_now(), "rel_tol": REL, "tiles": {}}
    allpass = True
    for tile in sys.argv[1:]:
        nd = pd.read_parquet(os.path.join(cfg["paths"]["partials_dir"], f"{tile}.parquet"))
        vp = os.path.join(cfg["paths"]["vegetation_partials_dir"], f"{tile}.parquet")
        if not os.path.exists(vp):
            out["tiles"][tile] = {"vegetation_partial": "none (tile not in the vegetation run)"}
            continue
        vg = pd.read_parquet(vp)
        vg = vg[(vg.kind == "diag") | (vg.year == 2020)]
        res, mms = {}, []
        sel = lambda k, lim: nd[(nd.kind == k) & (nd.zone_id <= lim)].set_index("zone_id")  # noqa: E731
        u = sel("urban", nv_p)
        r = sel("rural", nv_u)
        dp = sel("diag_piece", nv_p)
        res["new_pieces_in_partial"] = int(((nd.kind == "urban") & (nd.zone_id > nv_p)).sum())
        res["new_uas_in_partial"] = int(((nd.kind == "rural") & (nd.zone_id > nv_u)).sum())
        vu = vg[vg.kind == "urban"].groupby("id").area_m2.sum()
        vr = vg[vg.kind == "rural"].groupby("id").area_m2.sum()
        vd = vg[vg.kind == "diag"].set_index("id")
        for name, a, b in [("G1_urban_total_w", u.total_w, vu), ("G2_rural_total_w", r.total_w, vr)] + \
                [(f"G3_diag_{c}", dp[c], vd[c]) for c in ("valid_m2", "water_m2", "built_m2", "nodata_m2")]:
            s, mm = compare(a, b, name)
            res[name] = s
            mms.append(mm)
        up = os.path.join(cfg["paths"]["uhi_partials_dir"], f"{tile}.parquet")
        if os.path.exists(up):
            uh = pd.read_parquet(up, filters=[("kind", "in", ["ua_nowater_snowkept", "ua_nowater_snowkept_outside"])])
            s, mm = compare(sel("ua_all", nv_u).total_w, uh.groupby("zone_id").area_m2.sum(), "G4_ua_all_vs_uhi")
            res["G4_ua_all_vs_uhi"] = s
            mms.append(mm)
        gates = [k for k in res if k.startswith("G")]
        res["pass"] = all(res[k]["n_mismatch"] == 0 for k in gates)
        allpass &= res["pass"]
        mm = pd.concat(mms, ignore_index=True)
        if len(mm):
            mm.to_csv(os.path.join(cfg["paths"]["qa"], f"stage1_gate_mismatches_{tile}.csv"), index=False)
        out["tiles"][tile] = res
    out["pass"] = bool(allpass)
    write_json_atomic(os.path.join(cfg["paths"]["qa"], "stage1_gates.json"), out)
    print(json.dumps(out, indent=1))
    sys.exit(0 if allpass else 2)


if __name__ == "__main__":
    main()
