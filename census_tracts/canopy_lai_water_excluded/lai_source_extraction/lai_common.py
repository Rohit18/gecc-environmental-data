"""Shared constants and helpers for the 2020 Census Tract LAI workflow (CONUS).

The LAI raster is used on its own grid only: EPSG:4326, 8.983152841195215e-05
degree cells, origin (-180.00001488697754, 90.00005235925298). Nothing here
resamples, aggregates or reprojects the raster; tract polygons are brought to
EPSG:4326 instead.

Stored cell values are LAI x 10 (0-254 when valid). Statistics are computed from
a coverage-weighted histogram of those raw integers and divided by 10 at the end.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "nersc-1.0.0"

# Native grid of projects/your-ee-project/assets/LAI_2020_Global_Mosaic_uint8
# (identical to LAI_Global_Mosaic_2020, from which it was cast).
GRID_CRS = "EPSG:4326"
RES = 8.983152841195215e-05
ORIGIN_X = -180.00001488697754
ORIGIN_Y = 90.00005235925298
TILE_PX = 2048
GRID_TOL_DEG = 1e-9          # tolerance for "on the native grid" checks
UINT8_NODATA = 255           # documented sentinel if a uint8 copy is supplied
VALID_MAX_RAW = 254          # largest raw value representable as valid uint8

# Everything outside the 48 contiguous states + DC.
NON_CONUS_STFIPS = {"02", "15", "60", "66", "69", "72", "78"}

EE_SOURCE = "projects/your-ee-project/assets/LAI_2020_Global_Mosaic_uint8"
COUNT_DEFINITION = ("valid native pixels touched by the tract "
                    "(exactextract coverage fraction > 0, count with coverage_weight=none)")

ORIGINAL_COLUMNS = [
    "GEOID", "StFIPS", "CoFIPS", "TrFIPS", "Area_Land", "Area_Water", "Pop", "geometry_area_m2",
    "lai_native_mean", "lai_native_median", "lai_native_stdDev", "lai_native_count",
    "lai_native_min", "lai_native_max", "lai_native_p25", "lai_native_p75",
    "boundary_year", "lai_year", "batch", "n_batches", "batch_seed", "analysis_scale_m",
    "analysis_crs", "lai_units", "statistical_support", "missing_value_policy", "lai_source",
    "lai_input_band", "lai_input_divisor", "lai_storage_units", "script_version",
]
PROVENANCE_COLUMNS = ["processing_platform", "exactextract_version", "tile_manifest_sha256", "count_definition"]
OUTPUT_COLUMNS = ORIGINAL_COLUMNS + PROVENANCE_COLUMNS

STAT_COLUMNS = ["lai_native_mean", "lai_native_median", "lai_native_stdDev", "lai_native_count",
                "lai_native_min", "lai_native_max", "lai_native_p25", "lai_native_p75"]

# Per-row constants. analysis_scale_m is the nominal scale of the native grid
# (RES degrees = 10.000 m of arc at the equator), recorded for Earth Engine
# compatibility; it is not a resampling target.
CONSTANT_COLUMNS = {
    "boundary_year": 2020,
    "lai_year": 2020,
    "analysis_scale_m": 10.0,
    "analysis_crs": GRID_CRS,
    "lai_units": "m2_leaf_per_m2_ground",
    "statistical_support": "valid pixels in the supplied LAI mosaic",
    "missing_value_policy": "null for masked pixels; source zeros remain valid",
    "lai_source": EE_SOURCE,
    "lai_input_band": "LAI_uint8",
    "lai_input_divisor": 10,
    "lai_storage_units": "LAI_times_10_uint8",
    "script_version": SCRIPT_VERSION,
    "processing_platform": "NERSC Perlmutter CPU (Slurm)",
    "count_definition": COUNT_DEFINITION,
}
DIVISOR = 10.0
QUANTILE_REL_TOL = 1e-12     # cumulative weight >= p * total * (1 - tol); guards float round-off at exact ties


def sha256_file(path: str | os.PathLike, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: str | os.PathLike, obj) -> None:
    """Strict JSON: NaN/inf become null, numpy scalars become Python values."""
    text = json.dumps(_finite(obj), indent=2, default=_json_default, allow_nan=False)
    atomic_write_text(path, text + "\n")


def _finite(o):
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_finite(v) for v in o.tolist()]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def on_grid(value: float, origin: float, res: float, tol: float = GRID_TOL_DEG) -> bool:
    """True if value lies on a cell edge of the native grid."""
    k = (value - origin) / res
    return abs(k - round(k)) * res < tol


def histogram_stats(values, weights, touched_count) -> dict:
    """LAI statistics from a coverage-weighted histogram of raw integer values.

    values:  distinct valid raw values (LAI x 10) of cells with coverage > 0
    weights: coverage-weighted share of each value (exactextract `frac`, or raw sums)
    touched_count: number of valid cells with coverage > 0 (the canonical count)

    Returns native-unit statistics; all None (count 0) when no valid cell is touched.
    Quantiles take the first raw value whose cumulative weight >= p * total weight,
    without interpolation. The standard deviation is the population form.
    """
    v = np.asarray(values, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    keep = w > 0
    v, w = v[keep], w[keep]
    out = dict.fromkeys(STAT_COLUMNS)
    out["lai_native_count"] = int(round(touched_count)) if touched_count is not None else 0
    if out["lai_native_count"] == 0 or v.size == 0:
        out["lai_native_count"] = 0
        return out
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    total = w.sum()
    mean_raw = float(np.sum(w * v) / total)
    var_raw = float(np.sum(w * (v - mean_raw) ** 2) / total)   # = E(x^2) - E(x)^2, computed stably
    cum = np.cumsum(w)

    def quantile(p: float) -> int:
        idx = int(np.searchsorted(cum, p * total * (1.0 - QUANTILE_REL_TOL), side="left"))
        return int(v[min(idx, v.size - 1)])

    out.update({
        "lai_native_mean": mean_raw / DIVISOR,
        "lai_native_stdDev": float(np.sqrt(max(var_raw, 0.0))) / DIVISOR,
        "lai_native_min": int(v[0]) / DIVISOR,
        "lai_native_max": int(v[-1]) / DIVISOR,
        "lai_native_p25": quantile(0.25) / DIVISOR,
        "lai_native_median": quantile(0.50) / DIVISOR,
        "lai_native_p75": quantile(0.75) / DIVISOR,
    })
    return out


def tile_path(tiles_dir: str | os.PathLike, row: int, col: int) -> Path:
    return Path(tiles_dir) / f"r{row:04d}" / f"r{row:04d}_c{col:04d}.tif"


def gdal_env() -> dict:
    """GDAL settings for many-worker reads of a 25k-source VRT."""
    return {
        "GDAL_CACHEMAX": os.environ.get("GDAL_CACHEMAX", "1024"),
        "GDAL_MAX_DATASET_POOL_SIZE": os.environ.get("GDAL_MAX_DATASET_POOL_SIZE", "2000"),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VRT_SHARED_SOURCE": "0",
    }


# ---- shard bookkeeping shared by 03_tract_stats.py and 06_merge_validate.py ----

OPS = ["touched=count(coverage_weight=none)", "wcount=count", "unique", "frac",
       "exact_mean=mean", "exact_stdev=stdev", "exact_min=min", "exact_max=max"]


def shard_input_hash(shard: dict, tile_manifest_sha: str, vrt_sha: str) -> str:
    return sha256_text(json.dumps({
        "shard_id": int(shard["shard_id"]), "geoid_list_sha256": shard["geoid_list_sha256"],
        "geometry_sha256": shard["geometry_sha256"], "tile_manifest_sha256": tile_manifest_sha,
        "vrt_sha256": vrt_sha, "script_version": SCRIPT_VERSION, "ops": OPS,
    }, sort_keys=True))


def shard_paths(out_dir: Path, sid: int) -> tuple[Path, Path]:
    return out_dir / f"shard_{sid:04d}.parquet", out_dir / f"shard_{sid:04d}.done.json"


def shard_is_complete(out_dir: Path, shard: dict, input_hash: str) -> bool:
    pq, done = shard_paths(out_dir, int(shard["shard_id"]))
    if not (pq.exists() and done.exists()):
        return False
    try:
        d = json.loads(done.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (d.get("input_sha256") == input_hash and d.get("n_rows") == int(shard["n_tracts"])
            and d.get("parquet_sha256") == sha256_file(pq))
