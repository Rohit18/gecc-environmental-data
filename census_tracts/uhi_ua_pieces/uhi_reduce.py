"""Reduce per-tile weight partials into weight tables (zones cross tile edges, so sums are taken across tiles).

Outputs (data/interim/ for the full run; data/interim/test_<name>/ with --tiles):
  uhi_weights_{urban,rural,ua_all,ua_built,ua_nowater_snowkept}.parquet   zone_id, cell_id, area_m2
  uhi_diagnostics_piece.parquet   piece_id, total/wc0/wc50/wc70/wc80 area, outside-grid area per kind
  uhi_diagnostics_ua.parquet      ua_idx, same columns
  reduce_report.json
Refuses to run if any expected tile lacks a valid partial (full run). Idempotent: rewrites outputs atomically.
Usage: python uhi_reduce.py [--tiles N39W084 ... --name columbus]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from uhi_common import load_config, setup_logging, utc_now, write_json_atomic
from uhi_weights_worker import WEIGHT_KINDS, UHIContext, partial_is_valid, partial_path

log = setup_logging("uhi_reduce")
ap = argparse.ArgumentParser()
ap.add_argument("--tiles", nargs="*")
ap.add_argument("--name", default=None)
a = ap.parse_args()
cfg = load_config()
ctx = UHIContext(cfg)
sig = ctx.signature()
tiles = a.tiles or pd.read_csv(cfg["paths"]["worldcover_tiles_manifest"]).tile.tolist()
outdir = cfg["paths"]["interim"] if not a.tiles else os.path.join(cfg["paths"]["interim"], f"test_{a.name or 'subset'}")
bad = [t for t in tiles if not partial_is_valid(partial_path(cfg, t), sig)]
if bad:
    sys.exit(f"STOP: {len(bad)} tiles without a valid partial: {bad[:20]}")
os.makedirs(outdir, exist_ok=True)

df = pd.concat([pd.read_parquet(partial_path(cfg, t)) for t in tiles], ignore_index=True)
log.info("%d tiles, %d partial rows", len(tiles), len(df))
df = df.groupby(["kind", "zone_id", "cell_id"], as_index=False, sort=True)["area_m2"].sum()


def atomic_parquet(d, path):
    d.to_parquet(path + ".part", index=False)
    os.replace(path + ".part", path)


rep = {"generated_utc": utc_now(), "tiles": tiles, "n_tiles": len(tiles), "signature": sig, "kinds": {}}
for k in WEIGHT_KINDS:
    w = df[df.kind == k][["zone_id", "cell_id", "area_m2"]].reset_index(drop=True)
    assert (w.cell_id >= 0).all() and (w.area_m2 > 0).all()
    atomic_parquet(w, os.path.join(outdir, f"uhi_weights_{k}.parquet"))
    rep["kinds"][k] = {"rows": int(len(w)), "zones": int(w.zone_id.nunique()), "cells": int(w.cell_id.nunique()),
                       "total_area_m2": float(w.area_m2.sum())}
    log.info("%s: %s", k, rep["kinds"][k])

for zname, idcol, kinds in (("piece", "piece_id", ["urban"]),
                            ("ua", "ua_idx", ["rural", "ua_all", "ua_built", "ua_nowater_snowkept"])):
    d = df[df.kind.str.startswith(f"diag_{zname}_")]
    wide = d.pivot_table(index="zone_id", columns="kind", values="area_m2", aggfunc="sum")
    wide.columns = [c.replace(f"diag_{zname}_", "") + "_m2" for c in wide.columns]
    for c in ("total_m2", "wc0_m2", "wc50_m2", "wc70_m2", "wc80_m2"):
        if c not in wide:
            wide[c] = np.nan
    # class areas: absent = no pixel of that class in the zone (true zero area, not a missing value)
    for c in ("wc0_m2", "wc50_m2", "wc70_m2", "wc80_m2"):
        wide[c] = wide[c].fillna(0.0)
    for k in kinds:
        o = df[df.kind == f"{k}_outside"].groupby("zone_id").area_m2.sum()
        wide[f"outside_grid_{k}_m2"] = o.reindex(wide.index).fillna(0.0)
        wsum = df[df.kind == k].groupby("zone_id").area_m2.sum()
        wide[f"weight_{k}_m2"] = wsum.reindex(wide.index).fillna(0.0)
        wide[f"distinct_cells_{k}"] = df[df.kind == k].groupby("zone_id").size().reindex(wide.index).fillna(0).astype(int)
    wide = wide.reset_index().rename(columns={"zone_id": idcol})
    atomic_parquet(wide, os.path.join(outdir, f"uhi_diagnostics_{zname}.parquet"))
    rep[f"diag_{zname}"] = {"zones": int(len(wide)),
                            "outside_grid_total_m2": {k: float(wide[f"outside_grid_{k}_m2"].sum()) for k in kinds},
                            "zones_with_outside_grid": {k: int((wide[f"outside_grid_{k}_m2"] > 0).sum()) for k in kinds}}
    log.info("diag %s: %s", zname, rep[f"diag_{zname}"])
write_json_atomic(os.path.join(outdir, "reduce_report.json"), rep)
