"""Aggregate ESA WorldCover 2020 v100 to class fractions on the temperature grid.

The temperature rasters are on the MODIS sinusoidal grid (a sphere), while
WorldCover tiles are 10 m geographic.  A sinusoidal cell is a sheared
parallelogram in lon/lat, badly so at high latitude far from the central
meridian, which rules out warp-based averaging.  Instead every WorldCover pixel
centre is projected into the temperature grid and accumulated into the cell that
contains it, weighted by its true area (proportional to cos(latitude)).  The
pixel size is read from each tile, never assumed.

Output bands (float32, nodata -1):
  1 frac_class50           built-up
  2 frac_class80           permanent water
  3 frac_worldcover_nodata class 0 inside a tile
  4 frac_no_tile           cell area with no WorldCover tile at all
  5 valid_fraction_urban   1 - class80 - unknown          (urban mean)
  6 valid_fraction_rural   1 - class50 - class80 - unknown (rural ring mean)

Usage:
  python build_worldcover_fractions.py --tiles DIR --template SAMPLE.tif \
      --out FRACTIONS.tif [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS as RCRS

import common

log = common.setup_logging("build_worldcover_fractions")

FRACTION_NODATA = -1.0
BAND_NAMES = ["frac_class50", "frac_class80", "frac_worldcover_nodata",
              "frac_no_tile", "valid_fraction_urban", "valid_fraction_rural"]

_GRID = None


def _init(grid):
    global _GRID
    _GRID = grid


def _tile_window(path: Path, grid: dict) -> tuple:
    """Bounding window in target cells covered by this tile's footprint."""
    R = grid["R"]
    with rasterio.open(path) as ds:
        b = ds.bounds
    lons = np.linspace(b.left, b.right, 200)
    lats = np.linspace(b.bottom, b.top, 200)
    LON, LAT = np.meshgrid(lons, lats)
    x = R * np.radians(LON) * np.cos(np.radians(LAT))
    y = R * np.radians(LAT)
    cols = (x - grid["x0"]) / grid["dx"]
    rows = (grid["y0"] - y) / grid["dy_abs"]
    c0 = int(math.floor(cols.min())) - 2
    c1 = int(math.ceil(cols.max())) + 2
    r0 = int(math.floor(rows.min())) - 2
    r1 = int(math.ceil(rows.max())) + 2
    return (max(0, r0), min(grid["height"], r1),
            max(0, c0), min(grid["width"], c1))


def _process_tile(path_str: str) -> dict | None:
    """Return per-cell weighted class sums for one WorldCover tile."""
    grid = _GRID
    path = Path(path_str)
    r0, r1, c0, c1 = _tile_window(path, grid)
    if r1 <= r0 or c1 <= c0:
        return None
    nrl, ncl = r1 - r0, c1 - c0
    tot = np.zeros(nrl * ncl, dtype=np.float64)
    w50 = np.zeros_like(tot)
    w80 = np.zeros_like(tot)
    w00 = np.zeros_like(tot)

    R = grid["R"]
    with rasterio.open(path) as ds:
        T = ds.transform
        dlon = abs(T.a)
        dlat = abs(T.e)
        left, top = T.c, T.f
        # Area of one WorldCover pixel relative to one temperature cell.
        unit = (R * math.radians(dlon)) * (R * math.radians(dlat)) / \
               (grid["dx"] * grid["dy_abs"])
        lon = np.radians(left + (np.arange(ds.width) + 0.5) * dlon)
        band_rows = max(1, int(2e7 // ds.width))
        for start in range(0, ds.height, band_rows):
            n = min(band_rows, ds.height - start)
            arr = ds.read(1, window=((start, start + n), (0, ds.width)))
            lat = np.radians(top - (np.arange(start, start + n) + 0.5) * dlat)
            coslat = np.cos(lat)
            x = (R * coslat)[:, None] * lon[None, :]
            y = (R * lat)[:, None]
            cols = np.floor((x - grid["x0"]) / grid["dx"]).astype(np.int64)
            rows = np.floor((grid["y0"] - y) / grid["dy_abs"]).astype(np.int64)
            rows = np.broadcast_to(rows, cols.shape)
            good = ((rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1))
            if not good.any():
                continue
            idx = ((rows - r0) * ncl + (cols - c0))[good]
            wpix = np.broadcast_to((coslat * unit)[:, None], cols.shape)[good]
            cls = arr[good]
            tot += np.bincount(idx, weights=wpix, minlength=tot.size)
            for code, acc in ((common.WC_BUILT, w50), (common.WC_WATER, w80),
                              (common.WC_NODATA, w00)):
                m = cls == code
                if m.any():
                    acc += np.bincount(idx[m], weights=wpix[m], minlength=acc.size)
    return {"r0": r0, "r1": r1, "c0": c0, "c1": c1, "tile": path.name,
            "tot": tot, "w50": w50, "w80": w80, "w00": w00}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", required=True, help="directory of WorldCover tiles")
    ap.add_argument("--template", required=True, help="a raster on the target grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tiles = sorted(Path(args.tiles).glob("ESA_WorldCover_10m_2020_v100_*_Map.tif"))
    if not tiles:
        raise SystemExit(f"no WorldCover tiles in {args.tiles}")
    out = Path(args.out)

    with rasterio.open(args.template) as ds:
        T = ds.transform
        crs = ds.crs
        grid = {"x0": T.c, "y0": T.f, "dx": abs(T.a), "dy_abs": abs(T.e),
                "width": ds.width, "height": ds.height,
                "R": RCRS.from_wkt(crs.to_wkt()).to_dict().get("R")
                     or crs.to_dict().get("R")
                     or 6371007.181}
    # Take the sphere radius from the CRS itself rather than a constant.
    from pyproj import CRS as PCRS
    grid["R"] = PCRS.from_wkt(crs.to_wkt()).ellipsoid.semi_major_metre
    log.info("target grid %dx%d dx=%.6f R=%.3f", grid["width"], grid["height"],
             grid["dx"], grid["R"])

    params = {"grid": {k: grid[k] for k in ("x0", "y0", "dx", "dy_abs", "width",
                                            "height", "R")},
              "classes": {"built": common.WC_BUILT, "water": common.WC_WATER,
                          "nodata": common.WC_NODATA},
              "method": "pixel-centre accumulation, cos(lat) area weights",
              "bands": BAND_NAMES}
    inputs = [common.fingerprint(t, full_hash=False) for t in tiles]
    if not args.force and common.stage_complete(out, inputs, params):
        log.info("fractions already complete: %s", out)
        return

    tot = np.zeros((grid["height"], grid["width"]), dtype=np.float32)
    w50 = np.zeros_like(tot)
    w80 = np.zeros_like(tot)
    w00 = np.zeros_like(tot)

    log.info("aggregating %d tiles on %d workers", len(tiles), args.workers)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(grid,)) as ex:
        for res in ex.map(_process_tile, [str(t) for t in tiles]):
            done += 1
            if res is None:
                continue
            sl = (slice(res["r0"], res["r1"]), slice(res["c0"], res["c1"]))
            shape = (res["r1"] - res["r0"], res["c1"] - res["c0"])
            tot[sl] += res["tot"].reshape(shape).astype(np.float32)
            w50[sl] += res["w50"].reshape(shape).astype(np.float32)
            w80[sl] += res["w80"].reshape(shape).astype(np.float32)
            w00[sl] += res["w00"].reshape(shape).astype(np.float32)
            if done % 10 == 0:
                log.info("  %d/%d tiles", done, len(tiles))

    covered = tot > 0
    denom = np.maximum(tot, 1.0)
    f50 = np.where(covered, w50 / denom, FRACTION_NODATA).astype(np.float32)
    f80 = np.where(covered, w80 / denom, FRACTION_NODATA).astype(np.float32)
    f00 = np.where(covered, w00 / denom, FRACTION_NODATA).astype(np.float32)
    no_tile = np.where(covered, np.maximum(0.0, 1.0 - tot), FRACTION_NODATA
                       ).astype(np.float32)
    valid_urban = np.where(covered,
                           np.clip(1.0 - f80 - f00 - np.maximum(0.0, 1.0 - tot),
                                   0.0, 1.0), FRACTION_NODATA).astype(np.float32)
    valid_rural = np.where(covered, np.clip(valid_urban - f50, 0.0, 1.0),
                           FRACTION_NODATA).astype(np.float32)

    profile = {"driver": "GTiff", "height": grid["height"], "width": grid["width"],
               "count": 6, "dtype": "float32", "crs": crs,
               "transform": T, "nodata": FRACTION_NODATA, "tiled": True,
               "blockxsize": 512, "blockysize": 512, "compress": "zstd",
               "predictor": 2, "BIGTIFF": "IF_SAFER"}
    with common.atomic_path(out) as tmp:
        with rasterio.open(tmp, "w", **profile) as dst:
            for i, arr in enumerate([f50, f80, f00, no_tile, valid_urban,
                                     valid_rural], start=1):
                dst.write(arr, i)
                dst.set_band_description(i, BAND_NAMES[i - 1])
            dst.update_tags(source="ESA WorldCover 2020 v100 Map band",
                            method=params["method"],
                            urban_excluded_classes=str(common.URBAN_EXCLUDED_CLASSES),
                            rural_excluded_classes=str(common.RURAL_EXCLUDED_CLASSES),
                            n_tiles=str(len(tiles)))

    stats = {
        "n_tiles": len(tiles),
        "n_cells_with_worldcover": int(covered.sum()),
        "coverage_ratio_min": float(np.min(tot[covered])) if covered.any() else None,
        "coverage_ratio_median": float(np.median(tot[covered])) if covered.any() else None,
        "coverage_ratio_max": float(np.max(tot[covered])) if covered.any() else None,
        "mean_frac_class50": float(np.mean(f50[covered])) if covered.any() else None,
        "mean_frac_class80": float(np.mean(f80[covered])) if covered.any() else None,
        "n_cells_all_water": int(((f80 > 0.999) & covered).sum()),
        "n_cells_any_built": int(((f50 > 0) & covered).sum()),
    }
    common.write_sidecar(out, inputs, params, stats)
    log.info("wrote %s; %s", out, json.dumps(stats))


if __name__ == "__main__":
    main()
