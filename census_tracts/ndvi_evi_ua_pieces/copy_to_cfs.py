"""Copy to CFS (/path/to/project/urbveg/ndvi/) with sha256 verification. Never overwrites: a destination file
with identical content is skipped; one with different content is first renamed to <name>.bak-<UTC stamp>.

  --composites  data/composites/<season>/*.npz + manifest.jsonl -> composites/; each copied chunk is verified
                against the sha256 in the fetch manifest (latest record per chunk x season).
  --outputs     data/output/census_tracts_ua_ndvi, data/interim/{uas_all, pieces_all, worldcover_tiles_all.csv,
                chunks/all_uas.csv}, data/raw (4 island-area tract files), config, scripts, tests, qa, logs,
                README.md, REPORT.md. Stage 1 test tables (data/output/stage1_test) are not copied (their checks are in qa/).
Writes manifests/cfs_copy_<mode>_<stamp>.csv (path, bytes, sha256, action) in the destination and in logs/.
Usage: python copy_to_cfs.py --composites | --outputs [--threads 8]
"""
import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import json

from ndvi_common import NDVI_ROOT, load_config, setup_logging, sha256_file

ap = argparse.ArgumentParser()
ap.add_argument("--composites", action="store_true")
ap.add_argument("--outputs", action="store_true")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
log = setup_logging("copy_to_cfs")
cfg = load_config()
DST = "/path/to/project/urbveg/ndvi"
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def walk(items, skip_dirs=()):
    out = []
    for it in items:
        src = os.path.join(NDVI_ROOT, it)
        if os.path.isfile(src):
            out.append(it)
            continue
        for root, dirs, fs in os.walk(src):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache") + tuple(skip_dirs)]
            out += [os.path.relpath(os.path.join(root, f), NDVI_ROOT) for f in fs
                    if not f.endswith(".part") and ".bak-" not in f]
    return sorted(out)


def copy_one(rel, dst_rel, expect=None):
    s, d = os.path.join(NDVI_ROOT, rel), os.path.join(DST, dst_rel)
    h = sha256_file(s)
    if expect is not None and h != expect:
        return {"path": dst_rel, "bytes": os.path.getsize(s), "sha256": h, "action": "SOURCE_SHA_MISMATCH"}
    action = "copied"
    if os.path.exists(d):
        if sha256_file(d) == h:
            return {"path": dst_rel, "bytes": os.path.getsize(s), "sha256": h, "action": "identical_skipped"}
        os.replace(d, f"{d}.bak-{stamp}")
        action = "replaced_old_kept_as_bak"
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d + ".part")
    if sha256_file(d + ".part") != h:
        return {"path": dst_rel, "bytes": os.path.getsize(s), "sha256": h, "action": "COPY_SHA_MISMATCH"}
    os.replace(d + ".part", d)
    return {"path": dst_rel, "bytes": os.path.getsize(s), "sha256": h, "action": action}


def run(mode, jobs):
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        rows = list(ex.map(lambda j: copy_one(*j), jobs))
    bad = [r for r in rows if r["action"].isupper() or "MISMATCH" in r["action"]]
    for p in (os.path.join(DST, "manifests"), os.path.join(cfg["paths"]["logs"])):
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, f"cfs_copy_{mode}_{stamp}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "action"])
            w.writeheader(); w.writerows(rows)
    acts = {}
    for r in rows:
        acts[r["action"]] = acts.get(r["action"], 0) + 1
    log.info("%s: %d files, %.2f GB, actions=%s, bad=%d, %.0fs", mode, len(rows), sum(r["bytes"] for r in rows) / 1e9,
             acts, len(bad), time.time() - t0)
    return bad


bad = []
if a.composites:
    root = cfg["paths"]["composites"]
    rec = {}
    for line in open(os.path.join(root, "manifest.jsonl")):
        r = json.loads(line)
        if r.get("C") == cfg["grid"]["chunk_cells"]:
            rec[(r["season"], r["key"])] = r["sha256"]
    jobs = []
    for (season, key), sha in sorted(rec.items()):
        rel = os.path.relpath(os.path.join(root, season, key + ".npz"), NDVI_ROOT)
        jobs.append((rel, os.path.join("composites", season, key + ".npz"), sha))
    jobs.append((os.path.relpath(os.path.join(root, "manifest.jsonl"), NDVI_ROOT), "composites/manifest.jsonl", None))
    if os.path.exists(os.path.join(root, "failures.jsonl")):
        jobs.append((os.path.relpath(os.path.join(root, "failures.jsonl"), NDVI_ROOT), "composites/failures.jsonl", None))
    bad += run("composites", jobs)
if a.outputs:
    items = ["data/output/census_tracts_ua_ndvi", "data/interim/uas_all.parquet", "data/interim/pieces_all.parquet",
             "data/interim/worldcover_tiles_all.csv", "data/interim/chunks/all_uas.csv", "data/raw", "config", "scripts",
             "tests", "qa", "logs", "README.md", "REPORT.md"]
    files = walk(items)
    bad += run("outputs", [(f, f, None) for f in files])
if bad:
    log.error("FAILED: %s", bad[:20])
    sys.exit(1)
