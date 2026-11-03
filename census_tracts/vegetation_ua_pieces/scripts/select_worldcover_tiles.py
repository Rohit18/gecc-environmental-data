"""Select ESA WorldCover 2020 tiles intersecting the union of all enabled domain layers.

Domain layers and their buffers come from config/config.yaml (layers.domains, enabled: true).
Tile rectangles come from the bucket's grid file (esa_worldcover_grid.geojson, 3x3 deg, ll_tile key);
sizes from the S3 listing. Also records whether a size-matching copy already exists on CFS.
Writes manifests/worldcover_tiles.csv. Logs to logs/select_worldcover_tiles.log.
"""
import glob, json, logging, os, time

import geopandas as gpd
import pandas as pd
import yaml

from urbveg_common import read_domain

ROOT = "/path/to/scratch/urbveg"
cfg = yaml.safe_load(open(f"{ROOT}/config/config.yaml"))
import sys
# optional: --domains a,b (default: enabled domains)  --out <csv> (default manifests/worldcover_tiles.csv)
_dom = sys.argv[sys.argv.index("--domains") + 1].split(",") if "--domains" in sys.argv else None
_out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
logging.basicConfig(filename=f"{ROOT}/logs/select_worldcover_tiles.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger()
t0 = time.time()

EXISTING_DIRS = [  # found by the Phase 1 search; read-only
    "/path/to/project/gecc/census_tract_uhi/worldcover_2020/tiles",
    "/path/to/project/gecc/fua_uhi/worldcover_2020/tiles",
]

grid = gpd.read_file(f"{ROOT}/data/raw/worldcover_2020/aux/esa_worldcover_grid.geojson").to_crs("EPSG:4326")
grid = grid.rename(columns={"ll_tile": "tile"})

hits = []
for name, lay in cfg["layers"]["domains"].items():
    if (_dom is None and not lay.get("enabled")) or (_dom is not None and name not in _dom):
        continue
    g = read_domain(lay)[[lay["id_field"], "geometry"]]   # kept layer when configured
    # 0.01 deg buffer in geographic coordinates, as specified (selection only)
    g["geometry"] = g.geometry.buffer(lay.get("tile_selection_buffer_deg", 0.0), quad_segs=4)
    j = gpd.sjoin(grid[["tile", "geometry"]], g, predicate="intersects", how="inner")
    c = j.groupby("tile").size().rename("n_features").reset_index()
    c["domain"] = name
    hits.append(c)
    log.info("domain %s: %d features -> %d tiles", name, len(g), len(c))
hits = pd.concat(hits)
sel = hits.groupby("tile").agg(domains=("domain", lambda s: ";".join(sorted(set(s)))),
                               n_domain_features=("n_features", "sum")).reset_index()

lst = pd.read_csv(f"{ROOT}/data/interim/wc_s3_listing.tsv", sep="\t", names=["key", "s3_bytes"])
lst["tile"] = lst.key.str.extract(r"_v100_([NS]\d\d[EW]\d\d\d)_Map\.tif")[0]
sel = sel.merge(lst, on="tile", how="left")
sel = sel.merge(grid.assign(bounds=grid.geometry.bounds.round(6).values.tolist())[["tile", "bounds"]], on="tile")
sel["fname"] = "ESA_WorldCover_10m_2020_v100_" + sel.tile + "_Map.tif"
sel["s3_uri"] = "s3://esa-worldcover/" + sel.key

def existing(fn, size):
    for d in EXISTING_DIRS:
        p = f"{d}/{fn}"
        if os.path.exists(p):
            return p, os.path.getsize(p), os.path.getsize(p) == size, os.path.exists(p + ".verified.json")
    return None, None, False, False

ex = [existing(f, s) for f, s in zip(sel.fname, sel.s3_bytes)]
sel[["existing_path", "existing_bytes", "existing_size_match", "existing_has_verified_json"]] = pd.DataFrame(ex, index=sel.index)
sel["local_path"] = cfg["paths"]["worldcover_dir"] + "/" + sel.fname
sel["action"] = sel.apply(lambda r: "missing_on_s3" if pd.isna(r.s3_bytes)
                          else ("symlink_existing" if r.existing_size_match else "download"), axis=1)
sel = sel.drop(columns="key")
sel.to_csv(_out or f"{ROOT}/manifests/worldcover_tiles.csv", index=False)
summ = {"n_tiles": len(sel), "total_bytes": int(sel.s3_bytes.sum()),
        "actions": sel.action.value_counts().to_dict(),
        "bytes_to_download": int(sel.loc[sel.action == "download", "s3_bytes"].sum()),
        "seconds": round(time.time() - t0, 1)}
log.info(json.dumps(summ)); print(json.dumps(summ, indent=1))
