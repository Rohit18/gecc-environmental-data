#!/usr/bin/env python3
"""Independent recomputation of tract LAI statistics on a stratified sample.

Shares no code path with 03_tract_stats.py beyond reading the same VRT:
  * cell coverage fractions from GEOS polygon-cell intersection areas
    (exact for every cell within one cell of the tract boundary; cells farther
    inside or outside are 1 or 0), not from exactextract;
  * statistics from a sort-based weighted percentile over individual cells,
    not from the 256-bin histogram.
It also records alternative count definitions for the validation report:
  * touched      valid cells with intersection area > 0 (the canonical definition)
  * all_touched  GDAL rasterize(all_touched=True) of the polygon, valid cells only
  * centre       GDAL rasterize(all_touched=False), i.e. cell centre inside, valid only
  * weighted     sum of coverage fractions over valid cells

Sample (seeded): the smallest tracts, coastal tracts (water >= 50% of Area_Land +
Area_Water), random tracts, some 20-200 km^2 tracts, and (optionally) every
tract in the Earth Engine reference sample.

Example:
    python 04_independent_check.py --manifest-dir $WORK/manifest --vrt $LAI_VRT \
        --out $WORK/qa/independent_check.parquet --workers 32
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from lai_common import DIVISOR, RES, gdal_env, write_json

SEED = 20260921


def weighted_quantile_first(v: np.ndarray, w: np.ndarray, p: float) -> float:
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    cum = np.cumsum(w)
    # first value whose cumulative weight >= p * total (same rule, different code path)
    target = p * cum[-1] * (1 - 1e-12)
    i = int(np.argmax(cum >= target))
    return float(v[i])


def dilate8(m: np.ndarray) -> np.ndarray:
    """One-cell dilation with the 8-neighbourhood."""
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def one_tract(job) -> dict:
    geoid, wkb, vrt = job
    for k, val in gdal_env().items():
        os.environ.setdefault(k, val)
    import rasterio
    import shapely
    from rasterio import features
    from rasterio.windows import Window

    poly = shapely.from_wkb(wkb)
    with rasterio.open(vrt) as ds:
        inv = ~ds.transform
        minx, miny, maxx, maxy = poly.bounds
        c0, r0 = inv * (minx, maxy)
        c1, r1 = inv * (maxx, miny)
        c0, r0 = int(np.floor(c0)) - 1, int(np.floor(r0)) - 1
        c1, r1 = int(np.floor(c1)) + 2, int(np.floor(r1)) + 2
        win = Window(c0, r0, c1 - c0, r1 - r0)
        a = ds.read(1, window=win, boundless=True, fill_value=ds.nodata)
        wt = ds.window_transform(win)
        nodata = ds.nodata
    h, w = a.shape
    shape = (h, w)
    centre = features.rasterize([(poly, 1)], out_shape=shape, transform=wt, all_touched=False, dtype="uint8").astype(bool)
    alltch = features.rasterize([(poly, 1)], out_shape=shape, transform=wt, all_touched=True, dtype="uint8").astype(bool)
    edge = features.rasterize([(poly.boundary, 1)], out_shape=shape, transform=wt, all_touched=True, dtype="uint8")
    edge = dilate8(edge.astype(bool))

    cov = centre.astype(np.float64)            # cells away from the boundary are fully in or out
    rr, cc = np.nonzero(edge)
    x0 = wt.c + cc * wt.a
    y1 = wt.f + rr * wt.e
    boxes = shapely.box(x0, y1 + wt.e, x0 + wt.a, y1)
    cell_area = RES * RES
    cov[rr, cc] = shapely.area(shapely.intersection(boxes, poly)) / cell_area
    cov = np.clip(cov, 0.0, 1.0)

    valid = a != nodata
    sel = valid & (cov > 0)
    v = a[sel].astype(np.int64)
    wts = cov[sel]
    out = {"GEOID": geoid, "ind_touched": int(sel.sum()), "ind_weighted_count": float(wts.sum()),
           "ind_all_touched": int((alltch & valid).sum()), "ind_centre": int((centre & valid).sum()),
           "ind_nodata_cells_touched": int(((~valid) & (cov > 0)).sum())}
    if v.size:
        mu = np.average(v, weights=wts)
        out.update({
            "ind_mean": mu / DIVISOR, "ind_stdDev": np.sqrt(np.average((v - mu) ** 2, weights=wts)) / DIVISOR,
            "ind_min": v.min() / DIVISOR, "ind_max": v.max() / DIVISOR,
            "ind_p25": weighted_quantile_first(v, wts, 0.25) / DIVISOR,
            "ind_median": weighted_quantile_first(v, wts, 0.50) / DIVISOR,
            "ind_p75": weighted_quantile_first(v, wts, 0.75) / DIVISOR,
        })
    return out


def pick_sample(tr: pd.DataFrame, n_each: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    tot = tr.Area_Land + tr.Area_Water
    water = (tr.Area_Water / tot.where(tot > 0)).fillna(0)
    groups = {
        "tiny": tr.nsmallest(n_each, "geometry_area_m2").index,
        "coastal": tr[(water >= 0.5) & (tr.geometry_area_m2 < 50e6)].index,
        "random": tr[tr.geometry_area_m2 < 50e6].index,
        "medium_20_200km2": tr[(tr.geometry_area_m2 >= 20e6) & (tr.geometry_area_m2 < 200e6)].index,
    }
    picks = []
    for name, idx in groups.items():
        idx = np.asarray(idx)
        k = min(len(idx), n_each if name != "medium_20_200km2" else max(n_each // 4, 1))
        chosen = idx if name == "tiny" else rng.choice(idx, size=k, replace=False)
        picks.append(pd.DataFrame({"idx": chosen, "stratum": name}))
    p = pd.concat(picks).drop_duplicates("idx")
    return tr.loc[p.idx].assign(stratum=p.stratum.values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--vrt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-stratum", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--include-geoids-from", help="parquet with a GEOID column (e.g. the EE reference sample); "
                    "these tracts are added as stratum 'ee_reference'")
    a = ap.parse_args()
    import geopandas as gpd
    import shapely

    t0 = time.time()
    tr = gpd.read_parquet(Path(a.manifest_dir) / "tracts_conus_4326.parquet")
    s = pick_sample(tr, a.n_per_stratum)
    if a.include_geoids_from and Path(a.include_geoids_from).exists():
        extra = set(pd.read_parquet(a.include_geoids_from, columns=["GEOID"]).GEOID) - set(s.GEOID)
        s = pd.concat([s, tr[tr.GEOID.isin(extra)].assign(stratum="ee_reference")])
    jobs = [(g, shapely.to_wkb(geom), a.vrt) for g, geom in zip(s.GEOID, s.geometry)]
    with get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(one_tract, jobs, chunksize=2)
    df = pd.DataFrame(res).merge(s[["GEOID", "stratum", "geometry_area_m2", "Area_Land", "Area_Water"]], on="GEOID")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out.with_suffix(".tmp"), index=False)
    os.replace(out.with_suffix(".tmp"), out)
    write_json(out.with_suffix(".json"), {"n": len(df), "strata": df.stratum.value_counts().to_dict(),
                                           "seed": SEED, "seconds": round(time.time() - t0, 1)})
    print(f"independent check: {len(df)} tracts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
