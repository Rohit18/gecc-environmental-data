"""Fetch seasonal NDVI/EVI composite chunks from Earth Engine with ee.data.computePixels (high-volume endpoint).

Runs in conda env s1aef-train (earthengine-api). For each chunk (I, J) of a chunk list (select_chunks.py) and
each season: region = the chunk rectangle (planar EPSG:4326); image = seasonal_composite(region, 2020, season)
with NDVI/EVI unmasked to -9999 (float32) and clear_obs unmasked to 0 (int16); grid = C x C cells,
affineTransform {s, I*C*s, -s, -J*C*s}, crsCode EPSG:4326. Stored as <composites>/<season>/<chunk>.npz with
NDVI/EVI float32 (NaN = missing), clear_obs int16. Writes are atomic (.part + rename); existing chunks are
skipped (resumable, idempotent). Manifest: <composites>/manifest.jsonl (one record per chunk x season: key,
season, sha256, request seconds, NaN share, scene count, ...). Failures: <composites>/failures.jsonl.

Retries with exponential backoff on 429 / resource-exhausted / transient server errors. Stops (no further
requests) on computation time-outs, quota exhaustion or memory-limit errors, and reports.

Usage: python fetch_composites.py --chunks <csv> [--seasons JJA,DJF_same_year] [--C N] [--out DIR]
                                  [--concurrency 8] [--limit N]
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ndvi_common import (CompositeGrid, chunk_name, chunk_path, load_config, save_chunk, setup_logging,  # noqa: E402
                         sha256_file, utc_now)

RETRY = ("too many requests", "resource_exhausted", "resource exhausted", "rate limit", "internal error",
         "backend error", "service unavailable", "connection", "temporarily", "deadline")
RETRY_CODES = re.compile(r"\b(429|500|502|503|504)\b")
STOP = ("timed out", "timeout", "quota", "restricted", "memory limit", "capacity exceeded")
STOP_EVENT = threading.Event()
GATE = threading.Semaphore(0)          # limits concurrent EE requests; permits added in main()
RETRY_EVENTS = {"retry": 0}            # 429 / resource / transient errors seen (each retried attempt counts)
RETRY_LOCK = threading.Lock()


def classify(msg):
    m = msg.lower()
    if any(k in m for k in STOP) and "deadline" not in m:
        return "stop"
    if any(k in m for k in RETRY) or RETRY_CODES.search(m):
        return "retry"
    return "fail"


def fetch_one(ee, ec, grid, I, J, season, year, cfg_ee, out_root):
    if STOP_EVENT.is_set():
        return {"status": "skipped_stop", "I": I, "J": J, "season": season}
    region = ec.chunk_region(grid.chunk_bounds(I, J))
    img, _, n = ec.fetch_image(region, year, season, cfg_ee["fill_float"], cfg_ee["fill_clear_obs"])
    req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
           "grid": {"dimensions": {"width": grid.C, "height": grid.C}, "affineTransform": grid.affine(I, J),
                    "crsCode": "EPSG:4326"}}
    attempts, last, retry_errors = 0, None, []
    while attempts < cfg_ee["max_retries"]:
        attempts += 1
        if STOP_EVENT.is_set():
            return {"status": "skipped_stop", "I": I, "J": J, "season": season}
        try:
            with GATE:
                t0 = time.perf_counter()
                arr = ee.data.computePixels(req)
                t1 = time.perf_counter()
                scenes = int(ee.data.computeValue(n))
                t2 = time.perf_counter()
            break
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            kind = classify(last)
            if kind == "stop":
                STOP_EVENT.set()
                return {"status": "stop", "I": I, "J": J, "season": season, "error": last[:2000], "attempts": attempts}
            if kind == "retry":
                with RETRY_LOCK:
                    RETRY_EVENTS["retry"] += 1
                retry_errors.append(last[:300])
            if kind == "fail" or attempts >= cfg_ee["max_retries"]:
                return {"status": "fail", "I": I, "J": J, "season": season, "error": last[:2000], "attempts": attempts,
                        "retry_errors": retry_errors}
            time.sleep(min(300, 2 ** attempts + random.random() * 2))
    else:
        return {"status": "fail", "I": I, "J": J, "season": season, "error": (last or "")[:2000], "attempts": attempts}
    if arr.shape != (grid.C, grid.C):
        return {"status": "fail", "I": I, "J": J, "season": season, "error": f"shape {arr.shape}", "attempts": attempts}
    ndvi = arr["NDVI"].astype(np.float32)
    evi = arr["EVI"].astype(np.float32)
    co = arr["clear_obs"].astype(np.int16)
    ff = np.float32(cfg_ee["fill_float"])
    nd_fill, ev_fill = ndvi == ff, evi == ff
    ndvi[nd_fill] = np.nan
    evi[ev_fill] = np.nan
    # any finite value outside [-1, 1] would mean a fill value other than -9999: report it
    bad = int((np.abs(ndvi[~nd_fill]) > 1).sum() + (np.abs(evi[~ev_fill]) > 1).sum())
    meta = {"I": I, "J": J, "season": season, "year": year, "C": grid.C, "s": grid.s, "affine": grid.affine(I, J),
            "fill_float": cfg_ee["fill_float"], "fill_clear_obs": cfg_ee["fill_clear_obs"], "utc": utc_now()}
    path = chunk_path(out_root, season, I, J)
    if os.path.exists(path):                      # stale file without a matching manifest record: keep it aside
        os.replace(path, path + f".bak-{time.strftime('%Y%m%dT%H%M%S')}")
    sha = save_chunk(path, ndvi, evi, co, meta)
    return {"status": "ok", "key": chunk_name(I, J), "I": I, "J": J, "season": season, "C": grid.C, "sha256": sha,
            "bytes": os.path.getsize(path), "request_s": round(t1 - t0, 3), "count_request_s": round(t2 - t1, 3),
            "nan_share_ndvi": float(nd_fill.mean()), "nan_share_evi": float(ev_fill.mean()),
            "ndvi_evi_nan_differ": int((nd_fill != ev_fill).sum()),
            "clear_obs_zero_where_valid": int(((co == 0) & ~nd_fill).sum()), "out_of_range_values": bad,
            "scene_count": scenes, "attempts": attempts, "retry_errors": retry_errors, "utc": utc_now()}


def done_set(out_root, C, log):
    """(I, J, season) whose chunk file exists and whose sha256 equals the latest manifest record."""
    mp = os.path.join(out_root, "manifest.jsonl")
    rec = {}
    if os.path.exists(mp):
        for line in open(mp):
            r = json.loads(line)
            if r.get("C") == C:
                rec[(r["I"], r["J"], r["season"])] = r["sha256"]
    done, bad = set(), 0
    for (I, J, s), sha in rec.items():
        p = chunk_path(out_root, s, I, J)
        if os.path.exists(p) and sha256_file(p) == sha:
            done.add((I, J, s))
        else:
            bad += 1
    log.info("manifest records=%d verified=%d mismatched_or_missing=%d", len(rec), len(done), bad)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--C", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="raise to this after --raise-after requests if no 429/resource retries were seen")
    ap.add_argument("--raise-after", type=int, default=1000)
    ap.add_argument("--max-fail-share", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    cee = cfg["earth_engine"]
    grid = CompositeGrid(cfg["grid"]["cell_deg"], a.C or cfg["grid"]["chunk_cells"])
    out_root = a.out or cfg["paths"]["composites"]
    seasons = a.seasons.split(",") if a.seasons else cfg["seasons"]
    conc = a.concurrency or cee["concurrency"]
    maxc = max(conc, a.max_concurrency or conc)
    log = setup_logging("fetch_composites")
    import ee
    import ee_composite as ec
    ec.init_ee(cee["project"], cee["url"])
    chunks = [(int(r["I"]), int(r["J"])) for r in csv.DictReader(open(a.chunks))]
    done = done_set(out_root, grid.C, log)
    todo = [(I, J, s) for I, J in chunks for s in seasons if (I, J, s) not in done]
    if a.limit:
        todo = todo[:a.limit]
    total = len(chunks) * len(seasons)
    log.info("chunks=%d seasons=%s C=%d chunk-seasons=%d already_verified=%d todo=%d concurrency=%d (max %d) out=%s",
             len(chunks), seasons, grid.C, total, total - len(todo), len(todo), conc, maxc, out_root)
    os.makedirs(out_root, exist_ok=True)
    man = open(os.path.join(out_root, "manifest.jsonl"), "a")
    fails = open(os.path.join(out_root, "failures.jsonl"), "a")
    for _ in range(conc):
        GATE.release()
    cur = conc
    t0 = time.time()
    n = {"ok": 0, "fail": 0, "stop": 0, "skipped_stop": 0}
    acc = {s: {"n": 0, "req": 0.0, "nan": 0.0} for s in seasons}
    step = max(1, len(todo) // 10)
    max_fail = a.max_fail_share * len(todo)
    with ThreadPoolExecutor(maxc) as ex:
        futs = [ex.submit(fetch_one, ee, ec, grid, I, J, s, cfg["year"], cee, out_root) for I, J, s in todo]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001  (graph-building errors)
                r = {"status": "fail", "error": f"{type(e).__name__}: {e}"[:2000]}
            n[r["status"]] = n.get(r["status"], 0) + 1
            if r["status"] == "ok":
                man.write(json.dumps(r) + "\n"); man.flush()
                a_ = acc[r["season"]]; a_["n"] += 1; a_["req"] += r["request_s"]; a_["nan"] += r["nan_share_ndvi"]
            elif r["status"] != "skipped_stop":
                fails.write(json.dumps({**r, "utc": utc_now()}) + "\n"); fails.flush()
                log.warning("%s %s", r["status"], r)
            if n["fail"] > max_fail and not STOP_EVENT.is_set():
                log.error("STOP: %d failures > %.1f%% of %d requests", n["fail"], 100 * a.max_fail_share, len(todo))
                STOP_EVENT.set()
            finished = n["ok"] + n["fail"] + n["stop"]
            if cur < maxc and finished >= a.raise_after:
                if RETRY_EVENTS["retry"] == 0 and n["fail"] == 0 and n["stop"] == 0:
                    for _ in range(maxc - cur):
                        GATE.release()
                    log.info("CONCURRENCY raised %d -> %d after %d requests (0 retry events, 0 failures)", cur, maxc, finished)
                    cur = maxc
                else:
                    log.info("CONCURRENCY kept at %d: %d retry events, %d failures in the first %d requests",
                             cur, RETRY_EVENTS["retry"], n["fail"], finished)
                    maxc = cur
            done_n = sum(n.values())
            if done_n % step == 0 or done_n == len(todo):
                per = {s: {"done": v["n"], "mean_request_s": round(v["req"] / v["n"], 2) if v["n"] else None,
                           "mean_nan_share_ndvi": round(v["nan"] / v["n"], 4) if v["n"] else None} for s, v in acc.items()}
                log.info("progress %d/%d (%.0f%%) ok=%d fail=%d retry_events=%d concurrency=%d elapsed=%.0fs by_season=%s",
                         done_n, len(todo), 100 * done_n / max(1, len(todo)), n["ok"], n["fail"], RETRY_EVENTS["retry"],
                         cur, time.time() - t0, json.dumps(per))
    log.info("DONE %s retry_events=%d wall=%.1fs", n, RETRY_EVENTS["retry"], time.time() - t0)
    if STOP_EVENT.is_set():
        log.error("STOPPED: time-out / quota / memory error or failure share exceeded; see failures.jsonl")
        sys.exit(3)
    sys.exit(1 if n["fail"] else 0)


if __name__ == "__main__":
    main()
