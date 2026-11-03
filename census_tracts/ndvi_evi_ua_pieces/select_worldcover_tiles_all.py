"""Stage 1 step 3: WorldCover 2020 tiles for all 2,644 UAs, with the vegetation run's selection logic
(scripts/select_worldcover_tiles.py: bucket grid file, UA buffered by 0.01 deg in geographic coordinates with
quad_segs=4, sjoin intersects, sizes from the S3 listing), then verification of every tile not in the
vegetation run's 135: present on disk with the S3 size, opens, EPSG:4326, 1/12000 deg pixels, 36000 x 36000,
uint8, nodata 0, name = lower-left corner, values present (full read; scripts/verify_worldcover.py checks).
Writes data/interim/worldcover_tiles_all.csv and qa/stage1_worldcover_tiles.json.
"""
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from ndvi_common import load_config, utc_now, write_json_atomic

cfg = load_config()
R = "/path/to/scratch/urbveg"
vcfg = __import__("urbveg_common").load_config(cfg["paths"]["urbveg_config"], run="census_tracts_ua")
lay = vcfg["layers"]["domains"]["urban_areas_2020"]
grid = gpd.read_file(f"{R}/data/raw/worldcover_2020/aux/esa_worldcover_grid.geojson").to_crs("EPSG:4326")
grid = grid.rename(columns={"ll_tile": "tile"})
g = gpd.read_parquet(cfg["domains"]["uas_all"])[["GEOID20", "ua_idx", "geometry"]]
g["geometry"] = g.geometry.buffer(lay.get("tile_selection_buffer_deg", 0.0), quad_segs=4)
j = gpd.sjoin(grid[["tile", "geometry"]], g, predicate="intersects", how="inner")
sel = j.groupby("tile").agg(n_domain_features=("GEOID20", "size"),
                            max_ua_idx=("ua_idx", "max")).reset_index()
new_ua = j[j.ua_idx > cfg["domains"]["n_uas_veg"]].groupby("tile").GEOID20.apply(lambda s: ";".join(sorted(s)))
sel["new_uas"] = sel.tile.map(new_ua).fillna("")
lst = pd.read_csv(f"{R}/data/interim/wc_s3_listing.tsv", sep="\t", names=["key", "s3_bytes"])
lst["tile"] = lst.key.str.extract(r"_v100_([NS]\d\d[EW]\d\d\d)_Map\.tif")[0]
sel = sel.merge(lst[["tile", "s3_bytes"]], on="tile", how="left")
sel["fname"] = "ESA_WorldCover_10m_2020_v100_" + sel.tile + "_Map.tif"
sel["local_path"] = vcfg["paths"]["worldcover_dir"] + "/" + sel.fname
veg = pd.read_csv(cfg["worldcover"]["tiles_manifest_veg"])
sel["in_vegetation_run"] = sel.tile.isin(veg.tile)

recs = []
for r in sel[~sel.in_vegetation_run].itertuples():
    rec = {"tile": r.tile, "path": r.local_path, "exists": os.path.exists(r.local_path), "s3_bytes": r.s3_bytes}
    if rec["exists"]:
        rec["realpath"] = os.path.realpath(r.local_path)
        rec["bytes"] = os.path.getsize(r.local_path)
        rec["size_match"] = rec["bytes"] == r.s3_bytes
        with rasterio.open(r.local_path) as ds:
            a = ds.read(1)
            lat = int(r.tile[1:3]) * (1 if r.tile[0] == "N" else -1)
            lon = int(r.tile[4:7]) * (1 if r.tile[3] == "E" else -1)
            cnt = np.bincount(a.ravel(), minlength=256)
            rec.update({"crs": str(ds.crs), "res_times_12000": [x * 12000 for x in ds.res],
                        "width": ds.width, "height": ds.height, "dtype": ds.dtypes[0], "nodata": ds.nodata,
                        "name_is_lower_left": bool(abs(ds.bounds.left - lon) < 1e-9 and abs(ds.bounds.bottom - lat) < 1e-9),
                        "values_present": {int(v): int(cnt[v]) for v in np.nonzero(cnt)[0]}})
        rec["pass"] = bool(rec["size_match"] and rec["crs"] == "EPSG:4326" and rec["width"] == rec["height"] == 36000
                           and all(abs(x - 1) < 1e-9 for x in rec["res_times_12000"]) and rec["dtype"] == "uint8"
                           and rec["nodata"] == 0 and rec["name_is_lower_left"])
    else:
        rec["pass"] = False
    recs.append(rec)
sel.to_csv(cfg["worldcover"]["tiles_manifest_all"], index=False)
out = {"utc": utc_now(), "n_tiles_all": len(sel), "n_in_vegetation_run": int(sel.in_vegetation_run.sum()),
       "n_extra": int((~sel.in_vegetation_run).sum()), "vegetation_tiles_not_selected": sorted(set(veg.tile) - set(sel.tile)),
       "extra_tiles": recs, "all_extra_pass": all(r["pass"] for r in recs),
       "missing_on_s3": sel.loc[sel.s3_bytes.isna(), "tile"].tolist()}
write_json_atomic(f"{cfg['paths']['qa']}/stage1_worldcover_tiles.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "extra_tiles"}, indent=1))
for r in recs:
    print(r["tile"], r.get("bytes"), r.get("size_match"), r["pass"], sel.set_index("tile").new_uas[r["tile"]], r.get("values_present"))
