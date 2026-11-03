"""Stage 5 driver: run the tile worker over all tiles with a process pool on one node, skip tiles whose
partial already exists and validates, retry failed tiles once, then (if everything succeeded) run reduce.
Each tile runs in its own subprocess (isolated memory; failures logged per tile in logs/worker/<tile>.log).
Usage: python run_all.py [--procs N] [--no-reduce]
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from urbveg_common import ROOT, load_config
from urbveg_worker import Context, partial_is_valid, partial_path

ap = argparse.ArgumentParser()
ap.add_argument("--procs", type=int, default=32)
ap.add_argument("--no-reduce", action="store_true")
a = ap.parse_args()
cfg = load_config()
ctx = Context(cfg)
sig = ctx.signature(cfg["pipeline"]["window_px"])
tiles = pd.read_csv(cfg["run"]["worldcover_tiles"]).tile.tolist()
todo = [t for t in tiles if not partial_is_valid(partial_path(cfg, t), sig)]
print(f"{len(tiles)} tiles, {len(tiles) - len(todo)} already done, {len(todo)} to run", flush=True)
py = os.path.join(cfg["project"]["env_prefix"], "bin", "python")


def run(t):
    t0 = time.time()
    p = subprocess.run([py, os.path.join(ROOT, "scripts", "urbveg_worker.py"), t], cwd=os.path.join(ROOT, "scripts"),
                       capture_output=True, text=True)
    return t, p.returncode, round(time.time() - t0, 1), p.stderr[-2000:]


log = {"start": time.strftime("%FT%T"), "attempts": []}
for attempt in (1, 2):           # second pass = automatic single retry of failed tiles
    if not todo:
        break
    with ThreadPoolExecutor(a.procs) as ex:
        res = list(ex.map(run, todo))
    for t, rc, s, err in res:
        print(f"attempt {attempt} {t} rc={rc} {s}s", flush=True)
    log["attempts"].append([{"tile": t, "rc": rc, "s": s, "stderr_tail": err if rc else ""} for t, rc, s, err in res])
    todo = [t for t, rc, _, _ in res if rc != 0 or not partial_is_valid(partial_path(cfg, t), sig)]
log["failed_after_retry"] = todo
log["end"] = time.strftime("%FT%T")
log["run"] = cfg["run"]["name"]
json.dump(log, open(f"{ROOT}/logs/run_all_{cfg['run']['name']}_{os.environ.get('SLURM_JOB_ID', 'local')}.json", "w"), indent=1)
if todo:
    sys.exit(f"FAILED tiles after retry: {todo}")
if not a.no_reduce:
    subprocess.run([py, os.path.join(ROOT, "scripts", "reduce.py")], cwd=os.path.join(ROOT, "scripts"), check=True)
