"""Stage 1 benchmark summary and full-run extrapolation (all 2,644 UAs, both seasons).

Fetch: from data/composites/manifest.jsonl (the Stage 1 test-tile fetch, C = config chunk_cells) and the
fetch log wall time; chunk count from data/interim/chunks/all_uas.csv.
Worker: windows that intersect any UA in all 142 WorldCover tiles (same 3000-px windows) x measured seconds per
window on the test tiles, plus measured per-tile overhead.
Output: qa/stage1_benchmark.json
"""
import glob
import json
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ndvi_common import load_config, utc_now, write_json_atomic

cfg = load_config()
C = cfg["grid"]["chunk_cells"]
man = pd.read_json(os.path.join(cfg["paths"]["composites"], "manifest.jsonl"), lines=True)
man = man[man.C == C]
fails = os.path.join(cfg["paths"]["composites"], "failures.jsonl")
n_fail = sum(1 for _ in open(fails)) if os.path.exists(fails) else 0
log = open(os.path.join(cfg["paths"]["logs"], "fetch_composites.log")).read()
walls = [(int(m.group(1)), float(m.group(2))) for m in
         re.finditer(r"DONE \{'ok': (\d+), 'fail': \d+.*?\} wall=([\d.]+)s", log)]
ok_n, wall = max(walls)                                     # the Stage 1 test fetch (largest run)
chunks_all = pd.read_csv(os.path.join(cfg["paths"]["interim"], "chunks", "all_uas.csv"))
N = len(chunks_all) * len(cfg["seasons"])
conc = cfg["earth_engine"]["concurrency"]
q = lambda s: {k: round(float(v), 3) for k, v in s.quantile([.05, .5, .95, 1]).items()}  # noqa: E731
fetch = {
    "chunk_cells": C, "n_chunk_seasons_measured": int(len(man)), "n_failures_logged": n_fail,
    "request_s_quantiles": q(man.request_s), "request_s_mean": round(float(man.request_s.mean()), 3),
    "count_request_s_mean": round(float(man.count_request_s.mean()), 3),
    "bytes_mean": int(man.bytes.mean()), "scene_count_quantiles": q(man.scene_count),
    "by_season": {s: {"request_s_mean": round(float(g.request_s.mean()), 3), "nan_share_ndvi_mean": round(float(g.nan_share_ndvi.mean()), 4),
                      "scene_count_median": float(g.scene_count.median())} for s, g in man.groupby("season")},
    "measured_run": {"ok": ok_n, "wall_s": wall, "concurrency": conc, "throughput_per_s": round(ok_n / wall, 3)},
    "full_run": {"n_chunks": int(len(chunks_all)), "n_chunk_seasons": int(N),
                 "request_seconds_total_h": round(N * (man.request_s.mean() + man.count_request_s.mean()) / 3600, 2),
                 "expected_wall_h_at_measured_throughput": round(N / (ok_n / wall) / 3600, 2),
                 "storage_gb": round(N * man.bytes.mean() / 1e9, 2)},
}

# ------------------------------------------------------------------ worker
stats = {os.path.basename(p)[:-11]: json.load(open(p)) for p in glob.glob(os.path.join(cfg["paths"]["logs"], "worker", "*.stats.json"))}
w = pd.DataFrame([dict(x, tile=t) for t, s in stats.items() for x in s["windows"]])
tl = pd.read_csv(cfg["worldcover"]["tiles_manifest_all"])
ua = gpd.read_parquet(cfg["domains"]["uas_all"])
tree = shapely.STRtree(ua.geometry.values)
win = cfg["worker"]["window_px"]
nw = {}
for t in tl.tile:
    lat = int(t[1:3]) * (1 if t[0] == "N" else -1)
    lon = int(t[4:7]) * (1 if t[3] == "E" else -1)
    k = np.arange(0, 36000, win) / 12000
    boxes = [shapely.box(lon + c, lat + 3 - r - win / 12000, lon + c + win / 12000, lat + 3 - r) for r in k for c in k]
    nw[t] = int(np.unique(tree.query(boxes, predicate="intersects")[0]).size)
tile_over = np.mean([s["tile_s"] - sum(x["total_s"] for x in s["windows"]) for s in stats.values()])
per_win = float(w.total_s.mean())
worker = {"tiles_measured": {t: {"windows": s["n_windows_processed"], "tile_s": s["tile_s"], "peak_rss_gb": s["peak_rss_gb"],
                                 "pieces": s["n_pieces_in_tile"], "uas": s["n_uas_in_tile"], "chunks_loaded": s["chunks_loaded"]}
                             for t, s in stats.items()},
          "seconds_per_window_mean": round(per_win, 3), "seconds_per_window_p95": round(float(w.total_s.quantile(.95)), 3),
          "per_tile_overhead_s": round(float(tile_over), 1),
          "full_run": {"n_tiles": len(tl), "n_windows": int(sum(nw.values())),
                       "core_hours": round((sum(nw.values()) * per_win + len(tl) * tile_over) / 3600, 2),
                       "peak_rss_gb_max_measured": max(s["peak_rss_gb"] for s in stats.values())}}
out = {"utc": utc_now(), "fetch": fetch, "worker": worker}
write_json_atomic(os.path.join(cfg["paths"]["qa"], "stage1_benchmark.json"), out)
print(json.dumps(out, indent=1))
