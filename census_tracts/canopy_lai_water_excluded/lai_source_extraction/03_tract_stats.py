#!/usr/bin/env python3
"""Per-shard LAI statistics for CONUS 2020 Census Tracts (exactextract, native grid).

For each tract, one exactextract pass over the native-grid VRT returns:
  touched = count(coverage_weight=none)  valid cells with coverage > 0  -> lai_native_count
  wcount  = count                        coverage-weighted valid cells   -> QA only
  unique, frac                           distinct raw values and their coverage-weighted
                                         share (the 256-bin weighted histogram)
  exact_mean/stdev/min/max               exactextract's own statistics   -> QA cross-check

Only the raster's declared nodata is excluded; raw 0 is valid. Statistics come
from lai_common.histogram_stats and are divided by 10 there.

Resumable: each shard writes shard_NNNN.parquet atomically plus shard_NNNN.done.json
holding the input hash (shard GEOIDs, geometry WKB hash, tile manifest, VRT,
code version, ops) and the parquet sha256. A shard is skipped only if all of
those still match; anything else is recomputed.

Example:
    python 03_tract_stats.py --manifest-dir $WORK/manifest --vrt $LAI_VRT --out-dir $WORK/shards --workers 64
    python 03_tract_stats.py ... --shards 0,5,17     # just these shards
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from lai_common import (OPS, SCRIPT_VERSION, gdal_env, histogram_stats, shard_input_hash, shard_is_complete,
                        shard_paths, sha256_file, sha256_text, write_json)


def run_shard(job: tuple) -> str:
    shard, manifest_dir, vrt, out_dir, input_hash, max_cells = job
    for k, v in gdal_env().items():
        os.environ.setdefault(k, v)
    import geopandas as gpd
    from exactextract import exact_extract
    from osgeo import gdal
    gdal.UseExceptions()

    sid = int(shard["shard_id"])
    out_dir = Path(out_dir)
    if shard_is_complete(out_dir, shard, input_hash):
        return f"shard {sid}: complete, skipped"
    t0 = time.time()
    tracts = gpd.read_parquet(Path(manifest_dir) / "tracts_conus_4326.parquet",
                              columns=["GEOID", "shard", "geometry"], filters=[("shard", "==", sid)])
    tracts = tracts.sort_values("GEOID").reset_index(drop=True)
    if len(tracts) != int(shard["n_tracts"]) or sha256_text("\n".join(tracts.GEOID)) != shard["geoid_list_sha256"]:
        raise RuntimeError(f"shard {sid}: tract set does not match the shard manifest")

    res = exact_extract(vrt, tracts[["GEOID", "geometry"]], OPS, include_cols=["GEOID"], output="pandas",
                        strategy="feature-sequential", max_cells_in_memory=max_cells)
    res = res.set_index("GEOID").loc[tracts.GEOID]

    rows = []
    for geoid, r in res.iterrows():
        u = np.asarray(r["unique"]) if r["unique"] is not None else np.array([], np.int64)
        f = np.asarray(r["frac"]) if r["frac"] is not None else np.array([], np.float64)
        touched = float(r["touched"]) if pd.notna(r["touched"]) else 0.0
        st = histogram_stats(u, f, touched)
        if touched and not touched.is_integer():
            raise RuntimeError(f"{geoid}: touched count {touched} is not an integer")
        rows.append({
            "GEOID": geoid, **st,
            "qa_weighted_count": float(r["wcount"]) if pd.notna(r["wcount"]) else 0.0,
            "qa_n_distinct_values": int(u.size),
            "qa_exactextract_mean": float(r["exact_mean"]) / 10.0 if pd.notna(r["exact_mean"]) else np.nan,
            "qa_exactextract_stdev": float(r["exact_stdev"]) / 10.0 if pd.notna(r["exact_stdev"]) else np.nan,
            "qa_exactextract_min": float(r["exact_min"]) / 10.0 if pd.notna(r["exact_min"]) else np.nan,
            "qa_exactextract_max": float(r["exact_max"]) / 10.0 if pd.notna(r["exact_max"]) else np.nan,
            "qa_hist_values": u.astype(np.int16).tolist(),
            "qa_hist_frac": f.astype(np.float64).tolist(),
        })
    df = pd.DataFrame(rows)
    df["lai_native_count"] = df["lai_native_count"].astype("int64")
    df["shard"] = sid

    pq, done = shard_paths(out_dir, sid)
    tmp = pq.with_name(pq.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, pq)
    secs = time.time() - t0
    write_json(done, {"shard_id": sid, "n_rows": len(df), "input_sha256": input_hash,
                      "parquet_sha256": sha256_file(pq), "seconds": round(secs, 1), "host": socket.gethostname(),
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "script_version": SCRIPT_VERSION,
                      "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return (f"shard {sid}: {len(df)} tracts, {int((df.lai_native_count == 0).sum())} without data, "
            f"{secs:.0f}s, est_load {int(shard['est_load_px']):,} px")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--vrt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shards", help="comma-separated shard ids (default: all)")
    ap.add_argument("--max-cells-in-memory", type=int, default=30_000_000)
    a = ap.parse_args()

    mdir, out = Path(a.manifest_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rv = json.loads((mdir / "raster_validation.json").read_text())
    tv = json.loads((mdir / "tract_validation.json").read_text())
    if not (rv.get("passed") and tv.get("passed")):
        sys.exit("raster or tract validation did not pass; run 01/02 first")
    if os.path.realpath(a.vrt) != rv["vrt"]["path"] or sha256_file(a.vrt) != rv["vrt"]["sha256"]:
        sys.exit("VRT differs from the one validated in raster_validation.json")
    tile_sha, vrt_sha = rv["tile_manifest_sha256"], rv["vrt"]["sha256"]

    shards = pd.read_csv(mdir / "shard_manifest.csv", dtype={"first_geoid": str, "last_geoid": str}).to_dict("records")
    if a.shards:
        want = {int(s) for s in a.shards.split(",")}
        shards = [s for s in shards if int(s["shard_id"]) in want]
    shards.sort(key=lambda s: -int(s["est_load_px"]))   # biggest first
    jobs = [(s, str(mdir), a.vrt, str(out), shard_input_hash(s, tile_sha, vrt_sha), a.max_cells_in_memory)
            for s in shards]
    todo = [j for j in jobs if not shard_is_complete(out, j[0], j[4])]
    print(f"{time.strftime('%F %T')} shards selected={len(jobs)} already complete={len(jobs) - len(todo)} "
          f"to run={len(todo)} workers={a.workers}", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(min(a.workers, max(len(todo), 1)), maxtasksperchild=4) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_shard, todo), 1):
            print(f"{time.strftime('%F %T')} [{i}/{len(todo)}] {msg}", flush=True)
    print(f"{time.strftime('%F %T')} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
