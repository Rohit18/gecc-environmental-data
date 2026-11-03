#!/usr/bin/env python3
"""Tile manifest and raster validation for the CONUS LAI tiles.

Reads every tile once: records size + sha256, checks CRS, cell size, grid
alignment, dtype and declared nodata, and builds a histogram of every valid
raw value. Then checks that the VRT lists exactly the manifest tiles and sits
on the native grid.

Fails (exit 1) unless:
  * every tile is EPSG:4326, 1 band, integer, declared nodata, cell size RES,
    2048x2048, with its origin on the native grid at its (row, col) position;
  * no valid value is outside 0..254 and 255 never occurs as a valid value
    (so the data equal the uint8 source, where 255 is the only possible sentinel);
  * the VRT has the native cell size, an on-grid origin and exactly the manifest files.

Example:
    python 01_raster_manifest.py --tiles-dir $LAI_TILES --tile-list $LAI_TILE_LIST \
        --vrt $LAI_VRT --out-dir $WORK/manifest --workers 64
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio

from lai_common import (GRID_CRS, ORIGIN_X, ORIGIN_Y, RES, TILE_PX, UINT8_NODATA, VALID_MAX_RAW, on_grid,
                        sha256_file, tile_path, write_json)

MANIFEST_FIELDS = ["relpath", "row", "col", "size_bytes", "sha256", "dtype", "nodata", "width", "height",
                   "x_origin", "y_origin", "n_nodata", "n_valid", "min_valid", "max_valid", "problems"]


def scan_tile(job: tuple[str, int, int]) -> tuple[dict, np.ndarray, int, int]:
    tiles_dir, row, col = job
    path = tile_path(tiles_dir, row, col)
    rec = {"relpath": str(path.relative_to(Path(tiles_dir).parent)), "row": row, "col": col}
    problems = []
    if not path.exists():
        rec.update(problems="missing")
        return rec, np.zeros(256, np.int64), 0, 0
    rec["size_bytes"] = path.stat().st_size
    rec["sha256"] = sha256_file(path)
    with rasterio.open(path) as ds:
        t = ds.transform
        rec.update(dtype=ds.dtypes[0], nodata=ds.nodata, width=ds.width, height=ds.height,
                   x_origin=repr(t.c), y_origin=repr(t.f))
        if ds.crs is None or ds.crs.to_string() != GRID_CRS:
            problems.append(f"crs={ds.crs}")
        if ds.count != 1:
            problems.append(f"bands={ds.count}")
        if not np.issubdtype(np.dtype(ds.dtypes[0]), np.integer):
            problems.append(f"dtype={ds.dtypes[0]}")
        if ds.nodata is None:
            problems.append("no declared nodata")
        if abs(t.a - RES) > 1e-15 or abs(t.e + RES) > 1e-15 or t.b != 0 or t.d != 0:
            problems.append(f"cell size {t.a},{t.e}")
        if abs(t.c - (ORIGIN_X + col * TILE_PX * RES)) > 1e-9 or abs(t.f - (ORIGIN_Y - row * TILE_PX * RES)) > 1e-9:
            problems.append("origin not at native grid position")
        if (ds.width, ds.height) != (TILE_PX, TILE_PX):
            problems.append(f"size {ds.width}x{ds.height}")
        a = ds.read(1)
    nd = a == ds.nodata if ds.nodata is not None else np.zeros(a.shape, bool)
    valid = a[~nd]
    rec["n_nodata"] = int(nd.sum())
    rec["n_valid"] = int(valid.size)
    rec["min_valid"] = int(valid.min()) if valid.size else ""
    rec["max_valid"] = int(valid.max()) if valid.size else ""
    below = int((valid < 0).sum())
    above = int((valid > VALID_MAX_RAW).sum())   # includes 255 appearing as a valid value
    in_range = valid[(valid >= 0) & (valid <= 255)].astype(np.int64)
    hist = np.bincount(in_range, minlength=256)[:256]
    if below or above:
        problems.append(f"{below} valid values < 0, {above} > {VALID_MAX_RAW}")
    rec["problems"] = "; ".join(problems)
    return rec, hist, below, above


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--tile-list", required=True, help='text file of "row col" lines')
    ap.add_argument("--vrt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc = np.loadtxt(a.tile_list, dtype=int, ndmin=2)
    jobs = sorted({(a.tiles_dir, int(r), int(c)) for r, c in rc}, key=lambda j: (j[1], j[2]))
    with Pool(a.workers) as pool:
        results = pool.map(scan_tile, jobs, chunksize=16)

    hist = np.zeros(256, np.int64)
    below = above = 0
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    w.writeheader()
    for rec, h, b, ab in results:
        hist += h
        below += b
        above += ab
        w.writerow({k: rec.get(k, "") for k in MANIFEST_FIELDS})
    manifest_path = out / "tile_manifest.csv"
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(buf.getvalue())
    os.replace(tmp, manifest_path)
    manifest_sha = sha256_file(manifest_path)

    recs = [r for r, *_ in results]
    bad_tiles = [(r["relpath"], r["problems"]) for r in recs if r.get("problems")]
    dtypes = sorted({r.get("dtype") for r in recs if r.get("dtype")})
    nodatas = sorted({r.get("nodata") for r in recs if r.get("nodata") is not None})

    # VRT: native grid, same files as the manifest.
    with rasterio.open(a.vrt) as ds:
        t = ds.transform
        vrt_files = {os.path.realpath(f) for f in ds.files if not f.endswith(".vrt")}
        vrt = {
            "path": os.path.realpath(a.vrt), "sha256": sha256_file(a.vrt), "crs": ds.crs.to_string(),
            "dtype": ds.dtypes[0], "nodata": ds.nodata, "width": ds.width, "height": ds.height,
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "cell_size_native": abs(t.a - RES) < 1e-15 and abs(t.e + RES) < 1e-15,
            "origin_on_native_grid": on_grid(t.c, ORIGIN_X, RES) and on_grid(t.f, ORIGIN_Y, RES),
            "n_source_files": len(vrt_files),
        }
    manifest_files = {os.path.realpath(tile_path(a.tiles_dir, r["row"], r["col"])) for r in recs}
    vrt["source_files_equal_manifest"] = vrt_files == manifest_files

    valid_total = int(hist.sum())
    observed = np.flatnonzero(hist)
    checks = {
        "tiles_listed": len(jobs),
        "tiles_missing": sum(1 for r in recs if r.get("problems") == "missing"),
        "tiles_with_problems": len(bad_tiles),
        "tile_problems_first_20": bad_tiles[:20],
        "tile_dtypes": dtypes,
        "tile_declared_nodata": nodatas,
        "uint8_type": dtypes == ["uint8"],
        "uint8_representable": below == 0 and above == 0,
        "valid_values_below_0": below,
        "valid_values_above_254": above,
        "valid_value_255_count": int(hist[255]),
        "nodata_sentinel_255_used": nodatas == [float(UINT8_NODATA)],
        "valid_pixels": valid_total,
        "nodata_pixels": int(sum(r.get("n_nodata", 0) or 0 for r in recs)),
        "raw_value_min": int(observed.min()) if observed.size else None,
        "raw_value_max": int(observed.max()) if observed.size else None,
        "raw_zero_pixels": int(hist[0]),
        "raw_value_histogram": {int(v): int(hist[v]) for v in observed},
        "tile_bytes": int(sum(r.get("size_bytes", 0) or 0 for r in recs)),
        "vrt": vrt,
        "tile_manifest": str(manifest_path),
        "tile_manifest_sha256": manifest_sha,
        "seconds": round(time.time() - t0, 1),
    }
    failures = []
    if checks["tiles_missing"] or bad_tiles:
        failures.append("tile problems (see tile_problems_first_20)")
    if len(nodatas) != 1:
        failures.append(f"tiles do not share one declared nodata: {nodatas}")
    if not checks["uint8_representable"]:
        failures.append("valid values outside 0..254")
    if checks["valid_value_255_count"]:
        failures.append("255 occurs as a valid value")
    if vrt["crs"] != GRID_CRS or not vrt["cell_size_native"] or not vrt["origin_on_native_grid"]:
        failures.append("VRT not on the native grid")
    if not vrt["source_files_equal_manifest"]:
        failures.append("VRT source files differ from the tile manifest")
    if len(nodatas) == 1 and vrt["nodata"] != nodatas[0]:
        failures.append(f"VRT nodata {vrt['nodata']} differs from tile nodata {nodatas[0]}")
    checks["failures"] = failures
    checks["passed"] = not failures
    write_json(out / "raster_validation.json", checks)
    print({k: v for k, v in checks.items() if k not in ("raw_value_histogram", "tile_problems_first_20")}, flush=True)
    if failures:
        sys.exit(f"raster validation FAILED: {failures}")


if __name__ == "__main__":
    main()
