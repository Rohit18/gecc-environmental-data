"""Build the 1,000 m equatorial sampling grid and tract coverage index."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from affine import Affine

import math, os
from filter_sampling_index import filter_index
# Final NERSC run settings.
STEP = float(os.environ.get("GRID_STEP_DEG", 0.008983152841195215))
ORIGIN_LON = float(os.environ.get("GRID_ORIGIN_LON", 0.0)); ORIGIN_LAT = float(os.environ.get("GRID_ORIGIN_LAT", 0.0))
TAG = os.environ.get("GRID_TAG", "final")
# window: whole cells of this grid covering lon -180..-64, lat 18..72
LON0 = ORIGIN_LON + math.floor((-180.0 - ORIGIN_LON) / STEP) * STEP
LON1 = ORIGIN_LON + math.ceil((-64.0 - ORIGIN_LON) / STEP) * STEP
LAT_TOP = ORIGIN_LAT + math.ceil((72.0 - ORIGIN_LAT) / STEP) * STEP
LAT_BOTTOM = ORIGIN_LAT + math.floor((18.0 - ORIGIN_LAT) / STEP) * STEP
NCOL = int(round((LON1 - LON0) / STEP)); NROW = int(round((LAT_TOP - LAT_BOTTOM) / STEP))

def _chunk(args):
    template, wkb_list, ids = args
    from exactextract import exact_extract
    import shapely
    g = gpd.GeoDataFrame({"i": ids}, geometry=shapely.from_wkb(wkb_list), crs="EPSG:4326")
    r = exact_extract(template, g, ["cell_id", "coverage"], include_cols=["i"], output="pandas")
    t, c, w = [], [], []
    for i, cid, cov in zip(r["i"].values, r["cell_id"].values, r["coverage"].values):
        if cid is None or len(cid) == 0:
            continue
        cid = np.asarray(cid, np.int64); cov = np.asarray(cov, np.float64)
        keep = cov > 0
        t.append(np.full(keep.sum(), i, np.int32)); c.append(cid[keep]); w.append(cov[keep].astype(np.float32))
    if not t:
        return np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(t), np.concatenate(c), np.concatenate(w)

def main(tracts_path, native_meta_json, work, workers=64):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise FileExistsError("Use a new empty index directory; existing artifacts are protected")
    template = work / f"grid_template_{TAG}.tif"
    if not template.exists():
        with rasterio.open(template, "w", driver="GTiff", width=NCOL, height=NROW, count=1, dtype="uint8",
                           crs="EPSG:4326", transform=Affine(STEP, 0, LON0, 0, -STEP, LAT_TOP),
                           tiled=True, compress="deflate") as ds:
            pass
    g = gpd.read_file(tracts_path, columns=["GEOID"]).reset_index(drop=True)
    g[["GEOID"]].to_csv(work / "grid_index_geoids.csv", index=False)
    b = g.geometry.bounds
    wrap = (b.maxx - b.minx) > 180
    if wrap.any():
        from shapely.ops import transform
        g.loc[wrap, "geometry"] = g.loc[wrap, "geometry"].apply(
            lambda geom: transform(lambda x, y, z=None: (np.where(np.asarray(x) < 0, np.asarray(x) + 360, x), y), geom))
    ids = np.arange(len(g))
    parts = np.array_split(ids, workers * 4)
    wkb = g.geometry.to_wkb().values
    tasks = [(str(template), list(wkb[p]), p) for p in parts]
    T, C, W = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for t, c, w in ex.map(_chunk, tasks):
            T.append(t); C.append(c); W.append(w)
    tract = np.concatenate(T); cell = np.concatenate(C); cov = np.concatenate(W)
    meta = json.loads(Path(native_meta_json).read_text())
    rr, cc = np.divmod(cell, NCOL)
    lon = LON0 + (cc + 0.5) * STEP; lat = LAT_TOP - (rr + 0.5) * STEP
    R = 6371007.181
    x = R * np.radians(lon) * np.cos(np.radians(lat)); y = R * np.radians(lat)
    a, _, x0, _, e, y0 = meta["transform"]
    col = np.floor((x - x0) / a).astype(np.int64); row = np.floor((y - y0) / e).astype(np.int64)
    inside = (col >= 0) & (col < meta["width"]) & (row >= 0) & (row < meta["height"])
    native_idx = np.where(inside, row * meta["width"] + col, -1)
    np.savez(work / f"coverage_index_{TAG}.npz", tract=tract, cell=cell, coverage=cov, native_idx=native_idx)
    if TAG == "final":
        filter_index(work / f"coverage_index_{TAG}.npz", work / "sampling_index_final.npz")
    info = {"grid_tag": TAG, "step_deg": STEP, "origin": [ORIGIN_LON, ORIGIN_LAT], "window": [LON0, LAT_BOTTOM, LON1, LAT_TOP], "shape": [NROW, NCOL], "n_tract_cell_pairs": int(len(tract)), "tracts_touching_any_cell": int(np.unique(tract).size),
            "pairs_outside_source_grid": int((~inside).sum()), "antimeridian_shifted": int(wrap.sum())}
    Path(work, f"coverage_index_{TAG}_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 64)
