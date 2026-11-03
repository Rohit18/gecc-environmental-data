"""Composite chunks to fetch: every C x C chunk of the composite grid that intersects a UA buffered by
grid.ua_buffer_cells cells (planar degrees, as the tile selection buffer).

Usage:
  python select_chunks.py --out <csv> [--C 1024] [--uas GEOID20,...] [--tiles N39W084,...]
    --uas    only these UAs (default: all 2,644)
    --tiles  restrict to the parts of the (buffered) UAs inside these WorldCover tiles, i.e. exactly what a
             worker run on these tiles can look up (Stage 1 test runs)
Output CSV: I, J, west, south, east, north, n_uas, ua_idx (';'-joined).
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ndvi_common import CompositeGrid, load_config


def chunks_for_geoms(grid, geoms, ids):
    """{(I, J): set(ids)} for chunks intersecting each geometry."""
    out = {}
    for g, uid in zip(geoms, ids):
        if g is None or g.is_empty:
            continue
        w, s, e, n = g.bounds
        I0, I1 = grid.chunk_of(grid.col(w)), grid.chunk_of(grid.col(e))
        J0, J1 = grid.chunk_of(grid.row(n)), grid.chunk_of(grid.row(s))
        II, JJ = np.meshgrid(np.arange(I0, I1 + 1), np.arange(J0, J1 + 1))
        II, JJ = II.ravel(), JJ.ravel()
        boxes = shapely.box(II * grid.C * grid.s, -(JJ + 1) * grid.C * grid.s, (II + 1) * grid.C * grid.s, -JJ * grid.C * grid.s)
        shapely.prepare(g)
        hit = shapely.intersects(g, boxes)
        for I, J in zip(II[hit], JJ[hit]):
            out.setdefault((int(I), int(J)), set()).add(int(uid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=int, default=None)
    ap.add_argument("--uas", default=None)
    ap.add_argument("--tiles", default=None)
    a = ap.parse_args()
    cfg = load_config()
    grid = CompositeGrid(cfg["grid"]["cell_deg"], a.C or cfg["grid"]["chunk_cells"])
    ua = gpd.read_parquet(cfg["domains"]["uas_all"])[["ua_idx", "GEOID20", "geometry"]]
    if a.uas:
        ua = ua[ua.GEOID20.isin(a.uas.split(","))]
    geoms = ua.geometry.buffer(cfg["grid"]["ua_buffer_cells"] * grid.s).values
    ids = ua.ua_idx.values
    if a.tiles:
        gs, ii = [], []
        for t in a.tiles.split(","):
            lat = int(t[1:3]) * (1 if t[0] == "N" else -1)
            lon = int(t[4:7]) * (1 if t[3] == "E" else -1)
            clip = shapely.clip_by_rect(geoms, lon, lat, lon + 3, lat + 3)
            keep = ~shapely.is_empty(clip)
            gs.append(clip[keep]); ii.append(ids[keep])
        geoms, ids = np.concatenate(gs), np.concatenate(ii)
    ch = chunks_for_geoms(grid, geoms, ids)
    rows = []
    for (I, J), u in sorted(ch.items()):
        w, s, e, n = grid.chunk_bounds(I, J)
        rows.append({"I": I, "J": J, "west": w, "south": s, "east": e, "north": n, "n_uas": len(u),
                     "ua_idx": ";".join(map(str, sorted(u)))})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"C={grid.C}: {len(df)} chunks for {len(set(ids))} UAs -> {a.out}")


if __name__ == "__main__":
    main()
