"""Stage 0, item 4b: year-to-year differences of the seasonal composites inside the kept UAs' bounding boxes.

Cells: temperature cells whose centre lies inside the lon/lat bounding box of at least one kept Urban Area.
For each season x measure and each consecutive pair (2015->2016 ... 2023->2024), over cells valid in both
years: mean, median, std of (T[y+1] - T[y]); and, if the September WorldCover fraction raster exists, the
Pearson and Spearman correlation of the difference with the cell's built-up fraction (frac_class50).
Report only; nothing is changed based on it. Writes qa/stage0_continuity.{json,csv}. Run on a compute node.
"""
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import stats

from uhi_common import UHI_ROOT, Grid, composite_paths, load_config, setup_logging, utc_now, write_json_atomic

log = setup_logging("stage0_continuity")
cfg = load_config()
grid = Grid.from_raster(composite_paths(cfg, cfg["years"][0], "JJA", "TMAX")[0])

# ---- cells whose centre lies in any kept UA bbox (exact per-row test on the sinusoidal grid)
uas = gpd.read_parquet(cfg["urban_areas"]["kept_path"])
b = uas.geometry.bounds.to_numpy()
mask = np.zeros((grid.height, grid.width), dtype=bool)
yc = grid.y0 - (np.arange(grid.height) + 0.5) * grid.dy           # cell-centre y per row
latc = yc / grid.R                                                  # radians
for lon0, lat0, lon1, lat1 in b:
    r = np.nonzero((latc >= np.radians(lat0)) & (latc <= np.radians(lat1)))[0]
    if r.size == 0:
        continue
    cosl = np.cos(latc[r])
    # centre lon of col c in row r: (x0 + (c+0.5)dx) / (R cos lat); inside [lon0, lon1]
    c0 = np.ceil((grid.R * cosl * np.radians(lon0) - grid.x0) / grid.dx - 0.5).astype(int)
    c1 = np.floor((grid.R * cosl * np.radians(lon1) - grid.x0) / grid.dx - 0.5).astype(int)
    for rr, a0, a1 in zip(r, np.clip(c0, 0, grid.width), np.clip(c1, -1, grid.width - 1)):
        if a1 >= a0:
            mask[rr, a0:a1 + 1] = True
sel = np.flatnonzero(mask)
log.info("cells in kept-UA bboxes: %d", sel.size)

frac_path = cfg["paths"]["worldcover_fractions"]
built = None
if os.path.exists(frac_path):
    with rasterio.open(frac_path) as ds:
        assert Grid.from_raster(frac_path).same_as(grid), "fraction raster grid differs"
        f50 = ds.read(1).reshape(-1)[sel]
        built = np.where(f50 == ds.nodata, np.nan, f50)


def load(y, s, m):
    p = composite_paths(cfg, y, s, m)[0]
    with rasterio.open(p) as ds:
        a = ds.read(1).reshape(-1)[sel].astype(np.float64)
        a[a == ds.nodata] = np.nan
    return a


rows, yearly = [], []
for s in cfg["seasons"]:
    for m in cfg["measures"]:
        T = {y: load(y, s, m) for y in cfg["years"]}
        for y in cfg["years"]:
            v = T[y][np.isfinite(T[y])]
            yearly.append({"season": s, "measure": m, "year": y, "n_valid": int(v.size),
                           "mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std())})
        for y0, y1 in zip(cfg["years"][:-1], cfg["years"][1:]):
            d = T[y1] - T[y0]
            ok = np.isfinite(d)
            rec = {"season": s, "measure": m, "pair": f"{y0}->{y1}", "y0": y0, "y1": y1, "n": int(ok.sum()),
                   "mean": float(d[ok].mean()), "median": float(np.median(d[ok])), "std": float(d[ok].std()),
                   "p5": float(np.percentile(d[ok], 5)), "p95": float(np.percentile(d[ok], 95)),
                   "mean_abs": float(np.abs(d[ok]).mean())}
            if built is not None:
                ok2 = ok & np.isfinite(built)
                rec["n_corr"] = int(ok2.sum())
                rec["pearson_built"] = float(stats.pearsonr(d[ok2], built[ok2])[0])
                rec["spearman_built"] = float(stats.spearmanr(d[ok2], built[ok2])[0])
                # mean difference in cells with >= 50% built-up minus cells with < 5% built-up
                hi, lo = ok2 & (built >= 0.5), ok2 & (built < 0.05)
                rec["mean_diff_built_ge50"] = float(d[hi].mean()) if hi.any() else None
                rec["mean_diff_built_lt05"] = float(d[lo].mean()) if lo.any() else None
                rec["built_contrast"] = (rec["mean_diff_built_ge50"] - rec["mean_diff_built_lt05"]
                                         if hi.any() and lo.any() else None)
            rows.append(rec)
            log.info("%s %s %s: %s", s, m, rec["pair"], json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                                    for k, v in rec.items() if k not in ("season", "measure", "pair")}))
df = pd.DataFrame(rows)
df.to_csv(f"{UHI_ROOT}/qa/stage0_continuity.csv", index=False)
pd.DataFrame(yearly).to_csv(f"{UHI_ROOT}/qa/stage0_continuity_yearly.csv", index=False)

# does 2020->2021 stand out? z-score of each statistic against the other 8 pairs of the same composite type
flags = []
for (s, m), g in df.groupby(["season", "measure"]):
    other = g[g.pair != "2020->2021"]
    tgt = g[g.pair == "2020->2021"].iloc[0]
    for k in ("mean", "median", "std", "pearson_built", "built_contrast"):
        if k not in g or other[k].isna().all():
            continue
        z = (tgt[k] - other[k].mean()) / other[k].std(ddof=1)
        rank = int((g[k].rank(ascending=False)[g.pair == "2020->2021"]).iloc[0])
        flags.append({"season": s, "measure": m, "stat": k, "value_2020_2021": float(tgt[k]),
                      "others_mean": float(other[k].mean()), "others_sd": float(other[k].std(ddof=1)),
                      "others_min": float(other[k].min()), "others_max": float(other[k].max()),
                      "z": float(z), "rank_desc_of_9": rank,
                      "outside_others_range": bool(tgt[k] < other[k].min() or tgt[k] > other[k].max())})
write_json_atomic(f"{UHI_ROOT}/qa/stage0_continuity.json",
                  {"generated_utc": utc_now(), "n_cells_in_bboxes": int(sel.size),
                   "fraction_raster_used": built is not None, "pairs": rows, "yearly": yearly,
                   "stand_out_2020_2021": flags})
for f in flags:
    log.info("2020->2021 %s %s %s: value %.4f others %.4f +- %.4f [%.4f, %.4f] z=%.2f outside=%s",
             f["season"], f["measure"], f["stat"], f["value_2020_2021"], f["others_mean"], f["others_sd"],
             f["others_min"], f["others_max"], f["z"], f["outside_others_range"])
