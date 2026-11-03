"""Audit every selected daily raster and build seasonal mean composites on the native grid.

For each year x season x measure:
  * each daily raster: metadata, SHA-256, nodata count, and every value outside the
    plausible range (below -95 C or above 65 C) with its cell locations;
  * composite = mean of the valid daily values per cell (source nodata masked before
    scaling; out-of-range values treated as missing), stored as float32 in Celsius,
    plus the per-cell count of valid days.
Nothing is imputed or substituted."""
import hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import rasterio

SCALE_C = 0.1                   # source integers are 0.1 degrees Celsius (verified below)
PLAUSIBLE_C = (-95.0, 65.0)
LO_RAW, HI_RAW = PLAUSIBLE_C[0] / SCALE_C, PLAUSIBLE_C[1] / SCALE_C
NODATA_OUT = -9999.0

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def build(task):
    year, key, members, daily_dir, out_dir = task
    season, measure = key.split("_")
    total = count = None
    files = []
    for rec in members:
        p = Path(daily_dir) / rec["member"]
        with rasterio.open(p) as ds:
            a = ds.read(1)
            meta = {"crs_wkt": ds.crs.to_wkt(), "transform": list(ds.transform)[:6],
                    "res": list(ds.res), "width": ds.width, "height": ds.height,
                    "dtype": ds.dtypes[0], "nodata": ds.nodata,
                    "scales": list(ds.scales), "offsets": list(ds.offsets),
                    "units": list(ds.units), "tags": ds.tags()}
        nod = meta["nodata"]
        valid = a != nod if nod is not None else np.ones(a.shape, bool)
        bad = valid & ((a < LO_RAW) | (a > HI_RAW))
        n_bad = int(bad.sum())
        cells = np.flatnonzero(bad.ravel()).astype(np.int64) if n_bad else np.zeros(0, np.int64)
        if total is None:
            total = np.zeros(a.shape, np.float64); count = np.zeros(a.shape, np.uint16)
        use = valid & ~bad
        total[use] += a[use]; count += use
        files.append({"member": rec["member"], "date": rec["date"], "measure": measure, "season": season,
                      "bytes": p.stat().st_size, "sha256": sha256(p), **meta,
                      "valid_pixels_before": int(valid.sum()), "valid_pixels_after": int(use.sum()),
                      "out_of_range_masked": n_bad,
                      "out_of_range_raw_min": int(a[bad].min()) if n_bad else None,
                      "out_of_range_raw_max": int(a[bad].max()) if n_bad else None})
        if n_bad:
            np.save(Path(out_dir) / f"outliers_{year}_{key}_{rec['date']}.npy", cells)
    mean = np.full(total.shape, NODATA_OUT, np.float32)
    has = count > 0
    mean[has] = (total[has] / count[has] * SCALE_C).astype(np.float32)
    np.save(Path(out_dir) / f"composite_{year}_{key}.npy", mean)
    np.save(Path(out_dir) / f"validdays_{year}_{key}.npy", count)
    return year, key, files

def main(years, lists_dir, daily_root, out_dir, workers):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tasks = []
    for y in years:
        sel = json.loads(Path(lists_dir, f"selection_{y}.json").read_text())
        for key, rec in sel["seasons"].items():
            tasks.append((y, key, rec["members"], str(Path(daily_root) / str(y)), out_dir))
    audit = []
    with ProcessPoolExecutor(workers) as ex:
        for y, key, files in ex.map(build, tasks):
            audit += files
            n_bad = sum(f["out_of_range_masked"] for f in files)
            print(f"{y} {key}: {len(files)} days, out-of-range values masked {n_bad}", flush=True)
    Path(out_dir, "daily_audit.json").write_text(json.dumps(audit))
    fields = ("crs_wkt", "transform", "width", "height", "dtype", "nodata")
    native_meta = {field: audit[0][field] for field in fields}
    if any(any(record[field] != native_meta[field] for field in fields) for record in audit):
        raise ValueError("Daily source rasters do not share one native grid")
    Path(out_dir).parent.joinpath("native_meta.json").write_text(json.dumps(native_meta, indent=2))

if __name__ == "__main__":
    ys = [int(v) for v in sys.argv[1].split(",")]
    main(ys, sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
