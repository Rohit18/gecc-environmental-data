"""Shared helpers: checksums, atomic outputs, resumable stage sidecars, logging.

Every stage writes <output>.meta.json next to its output.  A stage may skip an
existing output only when the sidecar's recorded input fingerprints, parameter
hash, and output checksum all still match.  Nothing is ever overwritten in
place: outputs are written to a temporary name in the same directory and
atomically renamed.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

CODE_VERSION = "1.0.0"

# ---------------------------------------------------------------- constants --
# ESA WorldCover v100 class codes.
WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP = 10, 20, 30, 40
WC_BUILT, WC_BARE, WC_SNOW_ICE, WC_WATER = 50, 60, 70, 80
WC_WETLAND, WC_MANGROVE, WC_MOSS = 90, 95, 100
WC_NODATA = 0
WC_VALID_CLASSES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)

# Locked masking rule: the urban mean drops permanent water only; the rural ring
# additionally drops built-up.  Class 70 (snow and ice) is deliberately kept.
URBAN_EXCLUDED_CLASSES = (WC_WATER,)
RURAL_EXCLUDED_CLASSES = (WC_BUILT, WC_WATER)

SEASONS = ("JJA", "DJF_same_year")
MEASURES = ("TMAX", "TMIN")
SEASON_MONTHS = {"JJA": (6, 7, 8), "DJF_same_year": (1, 2, 12)}

TEMPERATURE_SCALE_C = 0.1  # raw integer units are 0.1 degrees Celsius

# Plausible Celsius range for a daily near-surface air temperature field.  One
# definition, shared by the inventory gate and the composite step, so the cells
# the gate counts as defective are exactly the observations the composite drops.
PLAUSIBLE_DAILY_C = (-95.0, 65.0)

# Project decision of 2026-09-20: a daily value outside PLAUSIBLE_DAILY_C is
# treated as missing in the seasonal composite, exactly like source nodata.
# Before this such values were left in place; the retroactive zone gate then
# found them inside sampled zones for one Urban Area (Bethel, AK, 2023
# DJF_same_year TMAX), where they biased the released UHI by -0.33 C.
EXCLUDE_IMPLAUSIBLE_DAILY = True


def plausible_raw_bounds() -> tuple:
    """PLAUSIBLE_DAILY_C in raw 0.1 C units, computed one way everywhere."""
    return (PLAUSIBLE_DAILY_C[0] / TEMPERATURE_SCALE_C,
            PLAUSIBLE_DAILY_C[1] / TEMPERATURE_SCALE_C)
COMPOSITE_NODATA = -9999.0
COUNT_NODATA = 65535

# Area/geometry conventions.
EQUAL_AREA_CRS = "EPSG:6933"  # ellipsoidal equal-area, valid for all US areas
BUFFER_DISTANCE_M = 10000.0
BUFFER_QUAD_SEGS = 32
# GEOS buffers simplify their input by 0.01 * distance before offsetting, so the
# realized outward distance is within a few metres of 10 km rather than exact.
# The realized distance is measured per UA and gated against this tolerance; it
# is never assumed.
BUFFER_DISTANCE_TOL_M = 50.0
# Segment length used to densify geometries before any change of projection, so
# that a straight line in one CRS stays within tolerance of the curve it should
# be in the other CRS.
SEGMENTIZE_DEG = 0.002
SEGMENTIZE_M = 100.0

# A ring must not overlap any Census UA.  Anything at or below this total area
# is floating-point residue from the overlay, not a real overlap.
RING_OVERLAP_TOL_M2 = 1.0
# Below this an intersection is numerical noise rather than a positive-area
# tract-UA overlap.  Real slivers above it are kept and flagged, never dropped.
POSITIVE_AREA_TOL_M2 = 1e-6

# A tract-UA pair counts as an association only when the intersection covers at
# least this fraction of the tract.  Project decision of 2026-09-16, taken after
# the pilot found 2,740 tracts whose largest overlap was under 1% of the tract
# (the smallest was 1 mm2), nearly all boundary-vintage slivers between the 2020
# tract file and the TIGER 2025 Urban Area release.  Every positive-area pair is
# still retained in the QA crosswalk; the threshold governs association,
# primary selection and release eligibility only.
MIN_OVERLAP_FRACTION_OF_TRACT = 0.01

# Support thresholds (documented; they flag, they never enlarge a zone).
LOW_SUPPORT_EFFECTIVE_CELLS = 5.0
LOW_SUPPORT_COVERAGE_FRACTION = 0.05


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- checksums --
def hash_file(path, algo: str = "sha256", chunk: int = 1 << 24) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_file(path) -> str:
    return hash_file(path, "sha256")


def md5_file(path) -> str:
    return hash_file(path, "md5")


def fingerprint(path, full_hash: bool = True) -> dict:
    """Identify an input file.  Large archives use a recorded verification
    sidecar instead of re-hashing 19 GB on every stage."""
    path = Path(path)
    st = path.stat()
    fp = {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    sidecar = Path(str(path) + ".verified.json")
    if sidecar.exists():
        rec = json.loads(sidecar.read_text())
        fp["md5"] = rec.get("md5")
        fp["verified_sidecar"] = str(sidecar)
    elif full_hash:
        fp["sha256"] = sha256_file(path)
    return fp


def fingerprints(paths, full_hash: bool = True) -> list:
    return [fingerprint(p, full_hash) for p in sorted(str(p) for p in paths)]


def params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------- atomic outputs --
@contextmanager
def atomic_path(final):
    """Yield a temporary path in the same directory; rename it on success."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{final.name}.tmp{os.getpid()}"
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"stage produced no output at {tmp}")
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(path, obj) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str))


def sidecar_for(output) -> Path:
    return Path(str(output) + ".meta.json")


def software_versions() -> dict:
    vers = {"python": sys.version.split()[0], "platform": platform.platform(),
            "code_version": CODE_VERSION}
    for mod in ("numpy", "pandas", "pyarrow", "geopandas", "shapely", "pyogrio",
                "pyproj", "rasterio", "exactextract", "matplotlib"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = None
    try:
        import shapely
        import rasterio
        import pyproj
        vers["geos"] = shapely.geos_version_string
        vers["gdal"] = rasterio.__gdal_version__
        vers["proj"] = pyproj.proj_version_str
    except Exception:
        pass
    return vers


def stage_complete(output, inputs: list, params: dict, verify_output: bool = True) -> bool:
    """True when the existing output still matches its recorded inputs."""
    output = Path(output)
    side = sidecar_for(output)
    if not output.exists() or not side.exists():
        return False
    try:
        rec = json.loads(side.read_text())
    except Exception:
        return False
    if rec.get("params_hash") != params_hash(params):
        return False
    if rec.get("code_version") != CODE_VERSION:
        return False
    recorded = rec.get("inputs", [])
    if len(recorded) != len(inputs):
        return False
    for old, new in zip(recorded, inputs):
        for key in ("path", "bytes", "sha256", "md5"):
            if key in old and key in new and old[key] != new[key]:
                return False
        # Without a content hash, size alone cannot see an edit that keeps the
        # byte count - one float replaced in a parquet does exactly that - so
        # fall back to the modification time.  Found 2026-09-20 when the 2023
        # re-run changed Bethel's UA value and the tract assignment was reused.
        hashed = any(old.get(k) and new.get(k) for k in ("sha256", "md5"))
        if not hashed and old.get("mtime_ns") != new.get("mtime_ns"):
            return False
    if verify_output:
        want = rec.get("output_sha256")
        if want and want != sha256_file(output):
            return False
    return True


def write_sidecar(output, inputs: list, params: dict, extra: dict | None = None,
                  hash_output: bool = True) -> dict:
    output = Path(output)
    rec = {
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output) if hash_output else None,
        "written_utc": utc_now(),
        "code_version": CODE_VERSION,
        "params": params,
        "params_hash": params_hash(params),
        "inputs": inputs,
        "software": software_versions(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if extra:
        rec.update(extra)
    write_json_atomic(sidecar_for(output), rec)
    return rec


# ------------------------------------------------------------------ logging --
def setup_logging(name: str, log_dir=None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s",
                            "%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_rss_children_mb() -> float:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0


@contextmanager
def stage_timer(logger, stage: str, report_path=None):
    t0 = time.time()
    logger.info("stage start: %s", stage)
    try:
        yield
    finally:
        rec = {
            "stage": stage,
            "seconds": round(time.time() - t0, 2),
            "peak_rss_mb_self": round(peak_rss_mb(), 1),
            "peak_rss_mb_children": round(peak_rss_children_mb(), 1),
            "finished_utc": utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        logger.info("stage end: %s %.1f s peak_rss_self=%.0f MB children=%.0f MB",
                    stage, rec["seconds"], rec["peak_rss_mb_self"],
                    rec["peak_rss_mb_children"])
        if report_path:
            p = Path(report_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if p.exists():
                try:
                    existing = json.loads(p.read_text())
                except Exception:
                    existing = []
            existing.append(rec)
            write_json_atomic(p, existing)


# -------------------------------------------------------------- environment --
def env_paths() -> dict:
    """Resolve the workflow directories from config.sh's environment."""
    def need(name, default=None):
        v = os.environ.get(name, default)
        if v is None:
            raise SystemExit(f"{name} is not set; source config.sh first")
        return v

    scratch = need("GECC_UHI_SCRATCH",
                   os.path.join(os.environ.get("PSCRATCH", ""), "gecc_census_tract_uhi"))
    return {
        "code": Path(need("GECC_UHI_CODE", str(Path(__file__).resolve().parent))),
        "cfs": Path(need("GECC_UHI_CFS")),
        "scratch": Path(scratch),
        "ua_dir": Path(need("GECC_UHI_UA_DIR")),
        "tract_dir": Path(need("GECC_UHI_TRACT_DIR")),
        "archives": Path(need("GECC_UHI_ARCHIVES")),
        "worldcover": Path(need("GECC_UHI_WORLDCOVER")),
        "manifests": Path(need("GECC_UHI_MANIFESTS")),
        "extracted": Path(need("GECC_UHI_EXTRACTED")),
        "work": Path(need("GECC_UHI_WORK")),
    }


def du_bytes(path) -> int:
    """Disk usage in bytes, tolerant of a missing directory."""
    p = Path(path)
    if not p.exists():
        return 0
    out = subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True)
    try:
        return int(out.stdout.split()[0])
    except Exception:
        return 0


def copy_verified(src, dst, logger=None) -> dict:
    """Copy to CFS and verify by checksum; never clobber a matching file."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256_file(src)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        if sha256_file(dst) == src_hash:
            return {"src": str(src), "dst": str(dst), "bytes": src.stat().st_size,
                    "sha256": src_hash, "action": "already_present"}
    with atomic_path(dst) as tmp:
        shutil.copy2(src, tmp)
    got = sha256_file(dst)
    if got != src_hash:
        raise RuntimeError(f"checksum mismatch after copy: {src} -> {dst}")
    if logger:
        logger.info("persisted %s -> %s", src, dst)
    return {"src": str(src), "dst": str(dst), "bytes": src.stat().st_size,
            "sha256": src_hash, "action": "copied"}


# ----------------------------------------------------------------- calendar --
def season_dates(year: int, season: str) -> list:
    """Dates of a season, from the calendar; leap years are never hard-coded.

    JJA is June 1 - August 31 of `year`.  DJF_same_year is January 1 -
    end of February plus December 1-31 of the *same* calendar year.
    """
    if season not in SEASON_MONTHS:
        raise ValueError(f"unknown season {season}")
    months = SEASON_MONTHS[season]
    out = []
    d = _dt.date(year, 1, 1)
    end = _dt.date(year, 12, 31)
    while d <= end:
        if d.month in months:
            out.append(d)
        d += _dt.timedelta(days=1)
    return out


def doy_to_date(year: int, doy: int) -> _dt.date:
    return _dt.date(year, 1, 1) + _dt.timedelta(days=doy - 1)


def date_to_doy(d: _dt.date) -> int:
    return d.timetuple().tm_yday
