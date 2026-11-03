"""Stage 2 driver: run ndvi_worker.py over every WorldCover tile that intersects a UA (142) with a process pool on
one node (one subprocess per tile), skip tiles whose partial exists with the current signature, retry failed
tiles once, then report. Mirrors ../../scripts/run_all.py.
Log: logs/run_all_<jobid>.json (per attempt: tile, return code, seconds, stderr tail; failed_after_retry;
missing_chunk_errors = tiles whose log shows MissingChunkError, gate G1).
Usage: python ndvi_run_all.py [--procs N]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ndvi_common import load_config
from ndvi_worker import NDVIContext, partial_is_valid, partial_path

ap = argparse.ArgumentParser()
ap.add_argument("--procs", type=int, default=32)
a = ap.parse_args()
cfg = load_config()
ctx = NDVIContext(cfg)
sig = ctx.signature()
tiles = pd.read_csv(cfg["worldcover"]["tiles_manifest_all"]).tile.tolist()
todo = [t for t in tiles if not partial_is_valid(partial_path(cfg, t), sig)]
print(f"{len(tiles)} tiles, {len(tiles) - len(todo)} already done, {len(todo)} to run", flush=True)
py = sys.executable
here = os.path.dirname(os.path.abspath(__file__))


def run(t):
    t0 = time.time()
    p = subprocess.run([py, "-W", "ignore", os.path.join(here, "ndvi_worker.py"), t], cwd=here, capture_output=True, text=True)
    return t, p.returncode, round(time.time() - t0, 1), p.stdout[-3000:] + p.stderr[-3000:]


log = {"start": time.strftime("%FT%T"), "n_tiles": len(tiles), "procs": a.procs, "attempts": []}
for attempt in (1, 2):           # second pass = automatic single retry of failed tiles
    if not todo:
        break
    with ThreadPoolExecutor(a.procs) as ex:
        res = list(ex.map(run, todo))
    for t, rc, s, _ in res:
        print(f"attempt {attempt} {t} rc={rc} {s}s", flush=True)
    log["attempts"].append([{"tile": t, "rc": rc, "s": s, "out_tail": out if rc else ""} for t, rc, s, out in res])
    todo = [t for t, rc, _, _ in res if rc != 0 or not partial_is_valid(partial_path(cfg, t), sig)]
log["failed_after_retry"] = todo
log["end"] = time.strftime("%FT%T")
wl = open(os.path.join(cfg["paths"]["logs"], "worker", "ndvi_worker.log")).read()
log["missing_chunk_errors"] = sorted({ln.split("FAIL ")[1].split(":")[0] for ln in wl.splitlines()
                                      if "FAIL" in ln and "MissingChunkError" in ln and log["start"][:10] in ln})
log["partials_valid"] = sum(partial_is_valid(partial_path(cfg, t), sig) for t in tiles)
stats = [json.load(open(p)) for p in glob.glob(os.path.join(cfg["paths"]["logs"], "worker", "*.stats.json"))]
log["worker_seconds_total"] = round(sum(s["tile_s"] for s in stats if s["tile"] in tiles), 1)
log["peak_rss_gb_max"] = max(s["peak_rss_gb"] for s in stats if s["tile"] in tiles)
json.dump(log, open(os.path.join(cfg["paths"]["logs"], f"run_all_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in log.items() if k != "attempts"}, indent=1))
if todo:
    sys.exit(f"FAILED tiles after retry: {todo}")
