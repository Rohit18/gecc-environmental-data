"""Stage 1 step 1: inventory of the vegetation inputs this subproject depends on (read-only) and of the new
domain files. Records git HEAD of the urbveg repo, sha256 and counts. Output: qa/stage1_inventory.json"""
import subprocess
import geopandas as gpd
import pandas as pd
from ndvi_common import load_config, sha256_file, utc_now, write_json_atomic
cfg = load_config()
d = cfg["domains"]
head = subprocess.run(["git", "-C", "/path/to/scratch/urbveg", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
files = {"pieces.parquet": d["pieces_veg"], "uas_kept.parquet": d["uas_kept_veg"],
         "worldcover_tiles.csv": cfg["worldcover"]["tiles_manifest_veg"],
         "uas_all.parquet": d["uas_all"], "pieces_all.parquet": d["pieces_all"],
         "worldcover_tiles_all.csv": cfg["worldcover"]["tiles_manifest_all"]}
out = {"utc": utc_now(), "urbveg_git_head": head, "files": {k: {"path": p, "sha256": sha256_file(p)} for k, p in files.items()}}
out["counts"] = {"pieces": len(pd.read_parquet(d["pieces_veg"], columns=["piece_id"])),
                 "uas_kept": len(pd.read_parquet(d["uas_kept_veg"], columns=["ua_idx"])),
                 "worldcover_tiles": len(pd.read_csv(cfg["worldcover"]["tiles_manifest_veg"])),
                 "pieces_all": len(pd.read_parquet(d["pieces_all"], columns=["piece_id"])),
                 "uas_all": len(pd.read_parquet(d["uas_all"], columns=["ua_idx"])),
                 "worldcover_tiles_all": len(pd.read_csv(cfg["worldcover"]["tiles_manifest_all"]))}
out["counts_as_expected"] = out["counts"]["pieces"] == 75383 and out["counts"]["uas_kept"] == 2622 and out["counts"]["worldcover_tiles"] == 135
write_json_atomic(f"{cfg['paths']['qa']}/stage1_inventory.json", out)
import json; print(json.dumps(out, indent=1))
