"""Kept FUA layer for the fua_fishnet run: drop FUAs with no GLC_FCS30D tile (name-cell coverage of the record's
tiles), check the count against expected_n_kept, assign dom_idx = 1..N ordered by eFUA_ID (0 = none).
Writes kept_path (GeoParquet, EPSG:4326) and manifests/dropped_fuas.csv."""
import json, os, sys

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box

from urbveg_common import ROOT, load_config

cfg = load_config()
lay = cfg["layers"]["domains"]["fua"]
f = gpd.read_file(lay["path"], encoding=lay["encoding"]).to_crs(4326)
f["eFUA_ID"] = f.eFUA_ID.astype("int64")
rec = cfg["glc_fcs30d"]["zenodo_record"]
m = pd.read_csv(f"{ROOT}/manifests/glc_zip_members_{rec}.csv")
p = m.member.str.extract(r"_(?P<ew>[EW])(?P<lon>\d+)(?P<ns>[NS])(?P<lat>\d+)_Annual").dropna()
cells = shapely.union_all([box(int(a) * (1 if e == "E" else -1), int(b) * (1 if n == "N" else -1) - 5,
                               int(a) * (1 if e == "E" else -1) + 5, int(b) * (1 if n == "N" else -1))
                           for e, a, n, b in p[["ew", "lon", "ns", "lat"]].values])
share = shapely.area(shapely.intersection(f.geometry.values, cells)) / shapely.area(f.geometry.values)
f["glc_cover_share"] = share
drop = f[share < 1 - 1e-9]
if (drop.glc_cover_share > 0).any():
    sys.exit(f"STOP: partially covered FUAs: {drop[drop.glc_cover_share > 0].eFUA_nm.tolist()}")
kept = f[share >= 1 - 1e-9].sort_values("eFUA_ID").reset_index(drop=True)
if len(kept) != lay["expected_n_kept"]:
    sys.exit(f"STOP: kept {len(kept)} != expected {lay['expected_n_kept']}")
kept.insert(0, "dom_idx", range(1, len(kept) + 1))
kept = kept.drop(columns="glc_cover_share")
os.makedirs(os.path.dirname(lay["kept_path"]), exist_ok=True)
kept.to_parquet(lay["kept_path"] + ".part", index=False); os.replace(lay["kept_path"] + ".part", lay["kept_path"])
drop.drop(columns="geometry").to_csv(f"{ROOT}/manifests/dropped_fuas.csv", index=False)
print(json.dumps({"n_input": len(f), "n_dropped": len(drop), "dropped": drop.eFUA_nm.tolist(), "n_kept": len(kept)}))
