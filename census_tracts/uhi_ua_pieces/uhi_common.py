"""Shared helpers for the tract UHI (Method A) scripts: config, logging, temperature grid, cell ids.

The temperature grid parameters are read from the composites (never assumed); uhi.yaml records the values
measured in Stage 0 and every script re-checks them against the files it reads.
"""
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
import warnings

import numpy as np
import yaml

warnings.filterwarnings("ignore", message="You will likely lose important projection information")

UHI_ROOT = "/path/to/scratch/urbveg/uhi"
URBVEG_SCRIPTS = "/path/to/scratch/urbveg/scripts"
if URBVEG_SCRIPTS not in sys.path:
    sys.path.insert(0, URBVEG_SCRIPTS)


def load_config(path=f"{UHI_ROOT}/config/uhi.yaml"):
    return yaml.safe_load(open(path))


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(name, log_dir=f"{UHI_ROOT}/logs"):
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


def composite_paths(cfg, year, season, measure):
    d = cfg["paths"]["composites_dir"].format(year=year)
    stem = f"{year}_{season}_{measure}"
    return os.path.join(d, f"{stem}_mean_C.tif"), os.path.join(d, f"{stem}_valid_days.tif")


def composite_keys(cfg):
    return [(y, s, m) for y in cfg["years"] for s in cfg["seasons"] for m in cfg["measures"]]


# ----------------------------------------------------------------------------- temperature grid
class Grid:
    """Temperature grid (MODIS sinusoidal sphere). x0/y0 = outer corner of the top-left cell."""

    def __init__(self, x0, y0, dx, dy, width, height, R):
        self.x0, self.y0, self.dx, self.dy = float(x0), float(y0), float(dx), float(dy)
        self.width, self.height, self.R = int(width), int(height), float(R)
        self.ncells = self.width * self.height

    @classmethod
    def from_cfg(cls, cfg):
        g = cfg["temperature_grid"]
        return cls(g["x0"], g["y0"], g["dx"], g["dy"], g["width"], g["height"], g["sphere_radius_m"])

    @classmethod
    def from_raster(cls, path):
        import rasterio
        from pyproj import CRS
        with rasterio.open(path) as ds:
            T = ds.transform
            R = CRS.from_wkt(ds.crs.to_wkt()).ellipsoid.semi_major_metre
            return cls(T.c, T.f, abs(T.a), abs(T.e), ds.width, ds.height, R)

    def as_dict(self):
        return {"x0": self.x0, "y0": self.y0, "dx": self.dx, "dy": self.dy, "width": self.width,
                "height": self.height, "sphere_radius_m": self.R}

    def same_as(self, other, rtol=0.0):
        a, b = self.as_dict(), other.as_dict()
        return all((a[k] == b[k]) if rtol == 0 else abs(a[k] - b[k]) <= rtol * max(abs(a[k]), 1) for k in a)

    def deviation(self, path, origin_tol_m=0.01, cell_tol_m=1e-6):
        """Numeric comparison of a raster's grid with this (reference) grid; CRS name text is ignored.
        Returns a record of the actual deviations and 'ok' (same size; same sphere radius, central meridian,
        false easting/northing, sinusoidal projection; origin within origin_tol_m; cell size within cell_tol_m)."""
        import rasterio
        from pyproj import CRS
        with rasterio.open(path) as ds:
            T, w, h, crs = ds.transform, ds.width, ds.height, CRS.from_wkt(ds.crs.to_wkt())
        d = crs.to_dict()
        rec = {"path": path, "width": w, "height": h, "proj": d.get("proj"),
               "sphere_radius_m": crs.ellipsoid.semi_major_metre, "inverse_flattening": crs.ellipsoid.inverse_flattening,
               "lon_0": d.get("lon_0", 0.0), "x_0": d.get("x_0", 0.0), "y_0": d.get("y_0", 0.0),
               "rotation": [T.b, T.d],
               "dx0_m": T.c - self.x0, "dy0_m": T.f - self.y0, "ddx_m": abs(T.a) - self.dx, "ddy_m": abs(T.e) - self.dy,
               "crs_name": crs.name}
        rec["ok"] = bool(w == self.width and h == self.height and rec["proj"] == "sinu"
                         and rec["sphere_radius_m"] == self.R and rec["inverse_flattening"] == 0
                         and rec["lon_0"] == 0 and rec["x_0"] == 0 and rec["y_0"] == 0 and T.b == 0 and T.d == 0
                         and abs(rec["dx0_m"]) <= origin_tol_m and abs(rec["dy0_m"]) <= origin_tol_m
                         and abs(rec["ddx_m"]) <= cell_tol_m and abs(rec["ddy_m"]) <= cell_tol_m)
        return rec

    def cell_ids(self, lon_rad, lat_rad):
        """Cell of each point (radians; broadcastable arrays). Same conversion as the September
        build_worldcover_fractions.py: x = R*lon*cos(lat), y = R*lat; col = floor((x-x0)/dx),
        row = floor((y0-y)/dy). Returns int64 row*width+col, -1 off the grid."""
        lat_rad = np.asarray(lat_rad, dtype=np.float64)
        lon_rad = np.asarray(lon_rad, dtype=np.float64)
        x = (self.R * np.cos(lat_rad)) * lon_rad
        y = self.R * lat_rad
        col = np.floor((x - self.x0) / self.dx).astype(np.int64)
        row = np.floor((self.y0 - y) / self.dy).astype(np.int64)
        row = np.broadcast_to(row, col.shape) if row.shape != col.shape else row
        col = np.broadcast_to(col, row.shape) if col.shape != row.shape else col
        ok = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        return np.where(ok, row * self.width + col, -1).astype(np.int64)

    def cell_centre_lonlat_deg(self, cell_id):
        row, col = np.divmod(np.asarray(cell_id, dtype=np.int64), self.width)
        x = self.x0 + (col + 0.5) * self.dx
        y = self.y0 - (row + 0.5) * self.dy
        lat = y / self.R
        lon = x / (self.R * np.cos(lat))
        return np.degrees(lon), np.degrees(lat)
