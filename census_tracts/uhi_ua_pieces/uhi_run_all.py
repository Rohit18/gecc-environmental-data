"""Stage 3 driver: run uhi_weights_worker.py over all kept-UA WorldCover tiles with a process pool on one node.
Tiles whose partial exists with the current signature are skipped (resumable); failed tiles are retried once
automatically; if every tile succeeded, uhi_reduce.py runs. Each tile is its own subprocess (isolated memory);
per-tile logs in logs/worker/. Run log: logs/run_all_<jobid>.json.
Usage: python uhi_run_all.py [--procs N] [--no-reduce]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from uhi_common import UHI_ROOT, load_config
from uhi_weights_worker import UHIContext, partial_is_valid, partial_path

ap = argparse.ArgumentParser()
ap.add_argument("--procs", type=int, default=32)
ap.add_argument("--no-reduce", action="store_true")
a = ap.parse_args()
cfg = load_config()
sig = UHIContext(cfg).signature()
tiles = pd.read_csv(cfg["paths"]["worldcover_tiles_manifest"]).tile.tolist()
todo = [t for t in tiles if not partial_is_valid(partial_path(cfg, t), sig)]
print(f"{len(tiles)} tiles, {len(tiles) - len(todo)} already done, {len(todo)} to run", flush=True)
py = sys.executable
sdir = os.path.join(UHI_ROOT, "scripts")


def run(t):
    t0 = time.time()
    p = subprocess.run([py, os.path.join(sdir, "uhi_weights_worker.py"), t], cwd=sdir, capture_output=True, text=True)
    return t, p.returncode, round(time.time() - t0, 1), p.stderr[-2000:]


log = {"start": time.strftime("%FT%T"), "n_tiles": len(tiles), "attempts": []}
for attempt in (1, 2):           # second pass = automatic single retry of failed tiles
    if not todo:
        break
    with ThreadPoolExecutor(a.procs) as ex:
        res = list(ex.map(run, todo))
    for t, rc, s, _ in res:
        print(f"attempt {attempt} {t} rc={rc} {s}s", flush=True)
    log["attempts"].append([{"tile": t, "rc": rc, "s": s, "stderr_tail": err if rc else ""} for t, rc, s, err in res])
    todo = [t for t, rc, _, _ in res if rc != 0 or not partial_is_valid(partial_path(cfg, t), sig)]
log["failed_after_retry"] = todo
log["end"] = time.strftime("%FT%T")
os.makedirs(cfg["paths"]["logs"], exist_ok=True)
json.dump(log, open(os.path.join(cfg["paths"]["logs"], f"run_all_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w"), indent=1)
if todo:
    sys.exit(f"FAILED tiles after retry: {todo}")
if not a.no_reduce:
    subprocess.run([py, os.path.join(sdir, "uhi_reduce.py")], cwd=sdir, check=True)
