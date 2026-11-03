"""Stage 0/2 helper: build the kept Urban Area layer from config.

Reads layers.domains.urban_areas_2020 (corrected TIGER file), drops UAs whose NAME20 state suffix
(text after the last ", ", split on "--") contains any of drop_state_suffixes, checks the kept count
against expected_n_kept (stops if different), assigns ua_idx = 1..N ordered by GEOID20 (0 = no UA),
and writes kept_path (GeoParquet, EPSG:4326) and manifests/dropped_uas.csv.
Geometry: NAD83 (EPSG:4269) coordinates are relabelled as EPSG:4326 via pyproj's default transform
(a null shift; ~1-2 m), see REPORT.md.
"""
import json, os, sys

import geopandas as gpd
import yaml

ROOT = "/path/to/scratch/urbveg"
cfg = yaml.safe_load(open(f"{ROOT}/config/config.yaml"))
lay = cfg["layers"]["domains"]["urban_areas_2020"]
g = gpd.read_file(lay["path"])
idf, nmf = lay["id_field"], lay["name_field"]
suffix = g[nmf].str.rsplit(", ", n=1).str[1]
states = suffix.str.split("--")
drop_set = set(lay["drop_state_suffixes"])
drop = states.map(lambda s: bool(set(s) & drop_set))
mixed = states.map(lambda s: bool(set(s) & drop_set) and bool(set(s) - drop_set))
if mixed.any():
    sys.exit(f"STOP: multi-state UAs mixing dropped and kept states: {g.loc[mixed, nmf].tolist()}")
dropped = g[drop].copy()
dropped["state_suffix"] = suffix[drop]
kept = g[~drop].sort_values(idf).reset_index(drop=True)
summary = {"n_input": len(g), "n_dropped": int(drop.sum()), "n_kept": len(kept),
           "dropped_by_suffix": dropped.state_suffix.value_counts().to_dict(),
           "expected_n_kept": lay["expected_n_kept"]}
print(json.dumps(summary, indent=1))
if len(kept) != lay["expected_n_kept"]:
    sys.exit(f"STOP: kept {len(kept)} != expected {lay['expected_n_kept']}")
kept.insert(0, "ua_idx", range(1, len(kept) + 1))
kept = kept.to_crs("EPSG:4326")
os.makedirs(os.path.dirname(lay["kept_path"]), exist_ok=True)
kept.to_parquet(lay["kept_path"] + ".part", index=False)
os.replace(lay["kept_path"] + ".part", lay["kept_path"])
dropped.drop(columns="geometry").to_csv(f"{ROOT}/manifests/dropped_uas.csv", index=False)
json.dump(summary, open(f"{ROOT}/manifests/uas_kept_summary.json", "w"), indent=1)
