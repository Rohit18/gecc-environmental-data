"""Shared helpers for the tract NDVI/EVI (Method A) scripts: config, logging, composite grid, chunk store.

Only numpy/yaml/stdlib are imported here, so the Earth Engine fetcher (conda env s1aef-train, which has no
geospatial stack) and the worker (urbveg env) can both use this module.

Composite grid (GECC v1 reduction grid): EPSG:4326, cell s = 30 / (2*pi*6378137/360) deg, origin (0, 0).
  column i covers lon [i*s, (i+1)*s)        i = floor(lon / s)
  row    j covers lat (-(j+1)*s, -j*s]      j = floor(-lat / s)
Chunks: C x C cells, key (I, J) = (floor(i/C), floor(j/C)); chunk (I, J) holds cells i in [I*C, (I+1)*C),
j in [J*C, (J+1)*C), stored as arrays [j - J*C, i - I*C] (row-major, north to south, west to east).
"""
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import yaml

NDVI_ROOT = "/path/to/scratch/urbveg/ndvi"
URBVEG_SCRIPTS = "/path/to/scratch/urbveg/scripts"
if URBVEG_SCRIPTS not in sys.path:
    sys.path.insert(0, URBVEG_SCRIPTS)

SEASON_SHORT = {"JJA": "JJA", "DJF_same_year": "DJF"}
COMPOSITES = [("NDVI", "JJA"), ("NDVI", "DJF_same_year"), ("EVI", "JJA"), ("EVI", "DJF_same_year")]


def comp_name(index, season):
    return f"{index}_{SEASON_SHORT[season]}"


def load_config(path=f"{NDVI_ROOT}/config/ndvi.yaml"):
    return yaml.safe_load(open(path))


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(name, log_dir=f"{NDVI_ROOT}/logs"):
    os.makedirs(log_dir, exist_ok=True)
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        lg.addHandler(fh)
        lg.addHandler(sh)
    return lg


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- composite grid
class CompositeGrid:
    def __init__(self, s, C):
        self.s = float(s)
        self.C = int(C)

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg["grid"]["cell_deg"], cfg["grid"]["chunk_cells"])

    def col(self, lon):
        return np.floor(np.asarray(lon, dtype=np.float64) / self.s).astype(np.int64)

    def row(self, lat):
        return np.floor(-np.asarray(lat, dtype=np.float64) / self.s).astype(np.int64)

    def chunk_of(self, idx):
        return np.floor_divide(idx, self.C)

    def chunk_bounds(self, I, J):
        """(west, south, east, north) in degrees of chunk (I, J)."""
        s, C = self.s, self.C
        return (I * C * s, -(J + 1) * C * s, (I + 1) * C * s, -J * C * s)

    def affine(self, I, J):
        """computePixels affineTransform for chunk (I, J)."""
        s, C = self.s, self.C
        return {"scaleX": s, "shearX": 0, "translateX": I * C * s, "shearY": 0, "scaleY": -s, "translateY": -J * C * s}

    def edge_distance_deg(self, coords, axis):
        """Distance (deg) from each coordinate to the nearest cell edge along lon (axis='lon') or lat."""
        v = np.asarray(coords, dtype=np.float64) / self.s
        if axis == "lat":
            v = -v
        f = v - np.floor(v)
        return np.minimum(f, 1.0 - f) * self.s


def chunk_name(I, J):
    return f"i{int(I):+05d}_j{int(J):+05d}"


def chunk_path(root, season, I, J):
    return os.path.join(root, season, f"{chunk_name(I, J)}.npz")


def save_chunk(path, ndvi, evi, clear_obs, meta):
    """Atomic write (.part + rename) of one chunk; returns sha256 of the written file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, NDVI=ndvi.astype(np.float32), EVI=evi.astype(np.float32),
                            clear_obs=clear_obs.astype(np.int16), meta=np.array(json.dumps(meta)))
    sha = sha256_file(tmp)
    os.replace(tmp, path)
    return sha


class MissingChunkError(RuntimeError):
    pass


class CompositeLookup:
    """Nearest-neighbour lookup of composite values for 10 m pixel centres (grid above).

    lookup(lons (W,), lats (H,), need (H,W) bool) -> dict {"NDVI_JJA": (H,W) float32, ..., "clear_JJA": (H,W) int16,
    ...} with NaN / 0 where need is False. A needed pixel whose chunk file does not exist raises MissingChunkError."""

    def __init__(self, root, grid, seasons, cache=24):
        self.root, self.grid, self.seasons = root, grid, list(seasons)
        self.cache = OrderedDict()
        self.maxcache = cache
        self.loaded = set()

    def _chunk(self, I, J):
        k = (int(I), int(J))
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        d = {}
        for season in self.seasons:
            p = chunk_path(self.root, season, I, J)
            if not os.path.exists(p):
                raise MissingChunkError(f"chunk {chunk_name(I, J)} season {season} not fetched: {p}")
            with np.load(p) as z:
                sh = SEASON_SHORT[season]
                d[f"NDVI_{sh}"] = z["NDVI"]
                d[f"EVI_{sh}"] = z["EVI"]
                d[f"clear_{sh}"] = z["clear_obs"]
        self.cache[k] = d
        self.loaded.add(k)
        if len(self.cache) > self.maxcache:
            self.cache.popitem(last=False)
        return d

    def keys(self):
        out = []
        for season in self.seasons:
            sh = SEASON_SHORT[season]
            out += [f"NDVI_{sh}", f"EVI_{sh}", f"clear_{sh}"]
        return out

    def lookup(self, lons, lats, need):
        g = self.grid
        ci, rj = g.col(lons), g.row(lats)
        CI, RJ = g.chunk_of(ci), g.chunk_of(rj)
        H, W = rj.size, ci.size
        out = {k: (np.zeros((H, W), np.int16) if k.startswith("clear") else np.full((H, W), np.nan, np.float32))
               for k in self.keys()}
        for J in np.unique(RJ):
            rsel = np.nonzero(RJ == J)[0]
            for I in np.unique(CI):
                csel = np.nonzero(CI == I)[0]
                sub = need[rsel[:, None], csel[None, :]]
                if not sub.any():
                    continue
                d = self._chunk(I, J)
                rr = (rj[rsel] - J * g.C)[:, None]
                cc = (ci[csel] - I * g.C)[None, :]
                for k in out:
                    out[k][rsel[:, None], csel[None, :]] = d[k][rr, cc]
        return out
