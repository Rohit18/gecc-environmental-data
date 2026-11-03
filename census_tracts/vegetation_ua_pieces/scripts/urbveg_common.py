"""Shared helpers for urbveg scripts: config loading and domain/zone layer readers."""
import glob
import os

import geopandas as gpd
import pandas as pd
import yaml

ROOT = "/path/to/scratch/urbveg"


def load_config(path=f"{ROOT}/config/config.yaml", run=None):
    """Config with the selected run profile applied: cfg["run"] (with "name"), and cfg["pipeline"]
    pieces_path / partials_dir / output_dir taken from the profile. Run = argument, else env URBVEG_RUN,
    else config default_run."""
    cfg = yaml.safe_load(open(path))
    name = run or os.environ.get("URBVEG_RUN") or cfg.get("default_run")
    if name and "runs" in cfg:
        r = dict(cfg["runs"][name]); r["name"] = name
        cfg["run"] = r
        for k in ("pieces_path", "partials_dir", "output_dir"):
            cfg["pipeline"][k] = r[k]
    return cfg


def file_sha256(p, chunk=1 << 22):
    """sha256 of a file, cached in <p>.sha256.json (reused while size and mtime are unchanged)."""
    import hashlib, json
    st = os.stat(p); side = p + ".sha256.json"
    try:
        c = json.load(open(side))
        if c["size"] == st.st_size and c["mtime_ns"] == st.st_mtime_ns:
            return c["sha256"]
    except Exception:
        pass
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    try:
        json.dump({"sha256": h.hexdigest(), "size": st.st_size, "mtime_ns": st.st_mtime_ns}, open(side, "w"))
    except OSError:
        pass
    return h.hexdigest()


def enabled_domains(cfg):
    return {k: v for k, v in cfg["layers"]["domains"].items() if v.get("enabled")}


def read_domain(lay):
    """Kept (filtered) layer if the domain defines kept_path, else the raw file. Returns EPSG:4326."""
    if lay.get("kept_path"):
        return gpd.read_parquet(lay["kept_path"])
    g = gpd.read_file(lay["path"], encoding=lay.get("encoding"))
    return g.to_crs("EPSG:4326")


def read_tracts(cfg, bbox=None):
    z = cfg["layers"]["zones"]["tracts_2020"]
    parts = [gpd.read_file(p, bbox=bbox) for p in sorted(glob.glob(z["path_glob"]))]
    parts = [p for p in parts if len(p)]
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
