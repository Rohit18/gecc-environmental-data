"""Pure functions for the Stage 5 statistics (also used by the Columbus test and the unit tests).

For one composite: T = mean[cell_id]; valid = T != nodata. Per zone: sum_wT = sum(w*T) and valid_w = sum(w)
over valid cells; total_w = sum(w) over all cells; min/max valid days over cells with w > 0; distinct cells
with w > 0. temperature = sum_wT / valid_w (null when valid_w == 0). Effective cells = valid_w / (dx*dy).
min/max valid days are taken over cells with w > 0 AND valid temperature (September convention
2026-10-07); null when the zone has no such cell.
"""
import numpy as np
import pandas as pd


def zone_stats(w, T, days, nodata):
    """w: DataFrame zone_id, cell_id, area_m2 (one kind, cell_id >= 0); T, days: flat arrays indexed by cell_id.
    Returns DataFrame indexed by zone_id with sum_wT, valid_w, total_w, temperature, min_days, max_days,
    distinct_cells."""
    w = w[w.area_m2 > 0]
    zid = w["zone_id"].to_numpy()
    cid = w["cell_id"].to_numpy()
    a = w["area_m2"].to_numpy(dtype=np.float64)
    t = np.asarray(T)[cid].astype(np.float64)
    valid = t != nodata
    zones, inv = np.unique(zid, return_inverse=True)
    n = zones.size
    out = pd.DataFrame(index=pd.Index(zones, name="zone_id"))
    out["sum_wT"] = np.bincount(inv, weights=np.where(valid, a * t, 0.0), minlength=n)
    out["valid_w"] = np.bincount(inv, weights=np.where(valid, a, 0.0), minlength=n)
    out["total_w"] = np.bincount(inv, weights=a, minlength=n)
    out["distinct_cells"] = np.bincount(inv, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = out["sum_wT"].to_numpy() / out["valid_w"].to_numpy()
    out["temperature"] = np.where(out["valid_w"].to_numpy() > 0, tm, np.nan)
    if days is not None:
        d = np.asarray(days)[cid].astype(np.int64)
        big = np.iinfo(np.int64).max
        mn = np.full(n, big)
        mx = np.full(n, -1)
        np.minimum.at(mn, inv[valid], d[valid])
        np.maximum.at(mx, inv[valid], d[valid])
        out["min_days"] = np.where(mn == big, np.nan, mn)
        out["max_days"] = np.where(mx < 0, np.nan, mx)
    return out
