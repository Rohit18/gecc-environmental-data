"""Verify two WorldCover tiles: CRS, pixel size, transform/origin, dtype, nodata, unique values,
and that the tile name equals the lower-left corner of the bounds. Full-tile read (compute node).
Writes manifests/worldcover_verify.json. Usage: python verify_worldcover.py TILE1 TILE2
"""
import json, sys, time

import numpy as np
import rasterio
import yaml

ROOT = "/path/to/scratch/urbveg"
cfg = yaml.safe_load(open(f"{ROOT}/config/config.yaml"))
out = []
for tile in sys.argv[1:]:
    p = f"{cfg['paths']['worldcover_dir']}/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
    t = time.time()
    with rasterio.open(p) as ds:
        a = ds.read(1)
        lat = int(tile[1:3]) * (1 if tile[0] == "N" else -1)
        lon = int(tile[4:7]) * (1 if tile[3] == "E" else -1)
        cnt = np.bincount(a.ravel(), minlength=256)
        out.append({"tile": tile, "path": p, "crs": str(ds.crs), "res": list(ds.res),
                    "res_times_12000": [r * 12000 for r in ds.res],
                    "transform": list(ds.transform)[:6], "width": ds.width, "height": ds.height,
                    "bounds": list(ds.bounds), "dtype": ds.dtypes[0], "nodata": ds.nodata,
                    "count": ds.count, "block_shapes": ds.block_shapes, "compression": str(ds.compression),
                    "overviews": ds.overviews(1),
                    "name_is_lower_left": bool(abs(ds.bounds.left - lon) < 1e-9 and abs(ds.bounds.bottom - lat) < 1e-9),
                    "values_present": {int(v): int(cnt[v]) for v in np.nonzero(cnt)[0]},
                    "read_and_count_s": round(time.time() - t, 1)})
json.dump(out, open(f"{ROOT}/manifests/worldcover_verify.json", "w"), indent=1)
print(json.dumps(out, indent=1))
