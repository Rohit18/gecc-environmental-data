"""Select GLC_FCS30D tiles intersecting the union of enabled domain layers. No tile data downloaded.

1. Parses tile names from manifests/glc_zip_members_<record>.csv.
2. Reads the GeoTIFF header of one tile remotely (/vsizip//vsicurl/, a few KB) to verify how the
   tile name maps to its bounds (which corner, tile size). Selection fails if the name does not match.
3. Intersects the tile rectangles with the buffered domain layers from config/config.yaml.
Writes manifests/glc_tiles_<record>.csv and manifests/glc_header_check_<record>.json.
Usage: python select_glc_tiles.py <zenodo_record>
"""
import json, logging, re, sys, time

import geopandas as gpd
import pandas as pd
import rasterio
import yaml

from urbveg_common import read_domain
from shapely.geometry import box

ROOT = "/path/to/scratch/urbveg"
record = sys.argv[1]
# optional: --domains a,b (default: enabled domains)  --out <csv> (default manifests/glc_tiles_<record>.csv)
_dom = sys.argv[sys.argv.index("--domains") + 1].split(",") if "--domains" in sys.argv else None
_out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
cfg = yaml.safe_load(open(f"{ROOT}/config/config.yaml"))
logging.basicConfig(filename=f"{ROOT}/logs/select_glc_tiles_{record}.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger()

m = pd.read_csv(f"{ROOT}/manifests/glc_zip_members_{record}.csv")
m = m[m.member.str.endswith(".tif")].copy()
pat = r"GLC_FCS30D_(?P<period>\d{8})_(?P<ew>[EW])(?P<lon>\d+)(?P<ns>[NS])(?P<lat>\d+)_(?P<kind>Annual|5years)"
p = m.member.str.extract(pat)
m = pd.concat([m, p], axis=1)
bad = m[m.period.isna()]
if len(bad):
    log.warning("%d members do not match the name pattern: %s", len(bad), bad.member.head().tolist())
m = m.dropna(subset=["period"])
m["lon_name"] = m.lon.astype(int) * m.ew.map({"E": 1, "W": -1})
m["lat_name"] = m.lat.astype(int) * m.ns.map({"N": 1, "S": -1})
m["tile"] = m.ew + m.lon + m.ns + m.lat

# --- verify naming convention from one header (prefer a CONUS tile) ---
def vsi(zipname, member):
    return f"/vsizip//vsicurl/https://zenodo.org/records/{record}/files/{zipname}/{member}"

checks = []
for tile in ["W85N40", "W75N45", "W70N45", "W75N40"]:  # last two are E and S neighbours of W75N45
    r = m[(m.tile == tile) & (m.kind == "Annual")]
    if r.empty:
        continue
    r = r.iloc[0]
    t = time.time()
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_USE_HEAD="NO"):
        with rasterio.open(vsi(r.zip, r.member)) as ds:
            checks.append({"tile": tile, "member": r.member, "bounds": list(ds.bounds), "crs": str(ds.crs),
                           "res": list(ds.res), "width": ds.width, "height": ds.height, "count": ds.count,
                           "dtypes": list(set(ds.dtypes)), "nodata": ds.nodata,
                           "transform": list(ds.transform)[:6], "descriptions": list(ds.descriptions),
                           "tags": ds.tags(), "lon_name": int(r.lon_name), "lat_name": int(r.lat_name),
                           "header_read_s": round(time.time() - t, 1)})
json.dump(checks, open(f"{ROOT}/manifests/glc_header_check_{record}.json", "w"), indent=1)
c = checks[0]
L, Bm, R, T = c["bounds"]
ext_x, ext_y = c["width"] * c["res"][0], c["height"] * c["res"][1]   # tile extent from header (not assumed)
if abs(L - c["lon_name"]) < 1e-6 and abs(T - c["lat_name"]) < 1e-6:
    convention = "name = upper-left corner (west lon, north lat)"
    m["minx"], m["maxy"] = m.lon_name, m.lat_name
    m["maxx"], m["miny"] = m.minx + ext_x, m.maxy - ext_y
elif abs(L - c["lon_name"]) < 1e-6 and abs(Bm - c["lat_name"]) < 1e-6:
    convention = "name = lower-left corner (west lon, south lat)"
    m["minx"], m["miny"] = m.lon_name, m.lat_name
    m["maxx"], m["maxy"] = m.minx + ext_x, m.miny + ext_y
else:
    raise SystemExit(f"STOP: tile name {c['tile']} does not match header bounds {c['bounds']}; ask user")
for cc in checks[1:]:  # second tile must agree with the derived convention (ASSUMES all tiles same size)
    e = m[(m.tile == cc["tile"]) & (m.kind == "Annual")].iloc[0]
    assert all(abs(a - b) < 1e-9 for a, b in zip(cc["bounds"], [e.minx, e.miny, e.maxx, e.maxy])), cc
size = 5
# names step by 5 deg but tiles span width*res; gap between adjacent tiles (no GLC pixels there)
seam_gap_deg = size - ext_x
log.info("convention: %s; extent %s deg", convention, ext_x)

tiles = m.pivot_table(index=["tile", "minx", "miny", "maxx", "maxy"], columns="kind",
                      values=["zip", "member", "compress_size", "file_size"], aggfunc="first")
tiles.columns = [f"{a}_{b}" for a, b in tiles.columns]
tiles = tiles.reset_index()
tiles["geometry"] = [box(*r) for r in tiles[["minx", "miny", "maxx", "maxy"]].values]
tiles = gpd.GeoDataFrame(tiles, crs="EPSG:4326")

hits = []
for name, lay in cfg["layers"]["domains"].items():
    if (_dom is None and not lay.get("enabled")) or (_dom is not None and name not in _dom):
        continue
    g = read_domain(lay)[[lay["id_field"], "geometry"]]   # kept layer when configured
    g["geometry"] = g.geometry.buffer(lay.get("tile_selection_buffer_deg", 0.0), quad_segs=4)
    j = gpd.sjoin(tiles[["tile", "geometry"]], g, predicate="intersects", how="inner")
    cnt = j.groupby("tile").size().rename("n_features").reset_index()
    cnt["domain"] = name
    hits.append(cnt)
hits = pd.concat(hits)
sel = hits.groupby("tile").agg(domains=("domain", lambda s: ";".join(sorted(set(s)))),
                               n_domain_features=("n_features", "sum")).reset_index()
sel = tiles.drop(columns="geometry").merge(sel, on="tile")
sel["local_path_annual"] = cfg["paths"]["glc_dir"] + "/" + sel.member_Annual.str.split("/").str[-1]
sel.to_csv(_out or f"{ROOT}/manifests/glc_tiles_{record}.csv", index=False)  # Phase 1 versions kept in manifests/phase1/
summ = {"record": record, "convention": convention, "tile_name_step_deg": size, "tile_extent_deg": [ext_x, ext_y],
        "seam_gap_deg": seam_gap_deg, "seam_gap_glc_px": seam_gap_deg / c["res"][0], "seam_gap_wc_px": seam_gap_deg * 12000, "n_tiles_total": int(len(tiles)),
        "n_tiles_selected": int(len(sel)),
        "annual_download_bytes_compressed": int(sel.compress_size_Annual.sum()),
        "annual_on_disk_bytes": int(sel.file_size_Annual.sum()),
        "annual_missing": int(sel.member_Annual.isna().sum()),
        "zips_touched": sorted(sel.zip_Annual.dropna().unique().tolist())}
log.info(json.dumps(summ)); print(json.dumps(summ, indent=1))
