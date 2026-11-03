#!/usr/bin/env python3
"""Select the CONUS 2020 Census Tracts, validate them, and write the shard manifest.

CONUS = the 48 contiguous states + DC (StFIPS not in 02, 15, 60, 66, 69, 72, 78).
No clipping to any other boundary.

Steps
  1. Read the tract shapefile with GEOID/StFIPS/CoFIPS/TrFIPS as text.
  2. Check: GEOID is 11 digits, unique, equal to StFIPS+CoFIPS+TrFIPS;
     geometries non-empty and valid; 49 state codes.
  3. Reproject geometries to EPSG:4326 (the raster is never reprojected) and
     record the transformation and the largest coordinate shift it caused.
  4. geometry_area_m2: geodesic area on the WGS84 ellipsoid (pyproj.Geod).
  5. Check that every tile a tract polygon intersects is in the tile manifest,
     so no part of a tract falls outside the supplied raster.
  6. Balanced shards: tracts in GEOID order, cut into --n-shards groups of
     roughly equal estimated work (bounding-box pixels + a per-tract constant).

Outputs in --out-dir:
  tracts_conus_4326.parquet   GeoParquet, one row per tract, with `shard`
  shard_manifest.csv          shard_id, n_tracts, GEOID range, states, load, hashes
  tract_validation.json

Example:
    python 02_prepare_tracts.py --tracts $TRACTS --tile-manifest $WORK/manifest/tile_manifest.csv \
        --out-dir $WORK/manifest --n-shards 128
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely import STRtree

from lai_common import (GRID_CRS, NON_CONUS_STFIPS, ORIGIN_X, ORIGIN_Y, RES, TILE_PX, sha256_file, sha256_text,
                        write_json)

ATTR_COLS = ["GEOID", "StFIPS", "CoFIPS", "TrFIPS", "Area_Land", "Area_Water", "Pop"]
PER_TRACT_OVERHEAD_PX = 50_000


def geodesic_area_m2(geoms: np.ndarray) -> np.ndarray:
    geod = pyproj.Geod(ellps="WGS84")
    return np.array([abs(geod.geometry_area_perimeter(g)[0]) for g in geoms])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracts", required=True, help="2020 Census Tract polygons (shapefile)")
    ap.add_argument("--tile-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-shards", type=int, default=128)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src = Path(a.tracts)
    src_files = {p.name: sha256_file(p) for p in sorted(src.parent.glob(src.stem + ".*"))}
    gdf = gpd.read_file(src)
    for c in ["GEOID", "StFIPS", "CoFIPS", "TrFIPS"]:
        gdf[c] = gdf[c].astype(str)
    n_all = len(gdf)
    missing_attrs = [c for c in ATTR_COLS if c not in gdf.columns]
    if "Pop" in missing_attrs:
        gdf["Pop"] = pd.Series(pd.NA, index=gdf.index, dtype="Int64")
        missing_attrs.remove("Pop")
    if missing_attrs:
        sys.exit(f"tract file lacks columns {missing_attrs}")
    conus = gdf[~gdf.StFIPS.isin(NON_CONUS_STFIPS)].copy()

    geoid = conus.GEOID
    checks = {
        "source": str(src), "source_sha256": src_files, "source_crs": gdf.crs.to_wkt() if gdf.crs else None,
        "features_in_source": n_all, "conus_tracts": len(conus), "conus_states": int(conus.StFIPS.nunique()),
        "excluded_non_conus": n_all - len(conus),
        "excluded_by_stfips": gdf[gdf.StFIPS.isin(NON_CONUS_STFIPS)].StFIPS.value_counts().sort_index().to_dict(),
        "geoid_not_11_digits": int((~geoid.str.fullmatch(r"\d{11}")).sum()),
        "duplicate_geoids": int(geoid.duplicated().sum()),
        "geoid_ne_st_co_tr": int((geoid != conus.StFIPS + conus.CoFIPS + conus.TrFIPS).sum()),
        "empty_geometries": int(conus.geometry.is_empty.sum()),
        "invalid_geometries": int((~conus.geometry.is_valid).sum()),
        "geometry_types": conus.geom_type.value_counts().to_dict(),
        "pop_column_present": "Pop" in gdf.columns and gdf["Pop"].notna().any(),
        "pop_nulls": int(conus["Pop"].isna().sum()),
    }

    # Reproject polygons (never the raster) and record what the transformation did.
    tf = pyproj.Transformer.from_crs(gdf.crs, GRID_CRS, always_xy=True)
    before = shapely.get_coordinates(conus.geometry.values)
    conus = conus.to_crs(GRID_CRS)
    after = shapely.get_coordinates(conus.geometry.values)
    checks["reprojection"] = {"to": GRID_CRS, "transformer": tf.description,
                              "max_abs_coordinate_shift_deg": float(np.abs(after - before).max())}

    conus["geometry_area_m2"] = geodesic_area_m2(conus.geometry.values)

    # Tile coverage: every tile a polygon intersects must exist.
    man = pd.read_csv(a.tile_manifest, usecols=["row", "col", "problems"], keep_default_na=False)
    have = set(zip(man.row, man.col))
    b = conus.geometry.bounds
    c0 = np.floor((b.minx - ORIGIN_X) / (TILE_PX * RES)).astype(int)
    c1 = np.floor((b.maxx - ORIGIN_X) / (TILE_PX * RES)).astype(int)
    r0 = np.floor((ORIGIN_Y - b.maxy) / (TILE_PX * RES)).astype(int)
    r1 = np.floor((ORIGIN_Y - b.miny) / (TILE_PX * RES)).astype(int)
    cand = sorted({(r, c) for ra, rb, ca, cb in zip(r0, r1, c0, c1) for r in range(ra, rb + 1) for c in range(ca, cb + 1)})
    boxes = np.array([shapely.box(ORIGIN_X + c * TILE_PX * RES, ORIGIN_Y - (r + 1) * TILE_PX * RES,
                                  ORIGIN_X + (c + 1) * TILE_PX * RES, ORIGIN_Y - r * TILE_PX * RES) for r, c in cand])
    tree = STRtree(conus.geometry.values)
    tile_i, tract_i = tree.query(boxes, predicate="intersects")
    needed = {cand[i] for i in np.unique(tile_i)}
    absent = sorted(needed - have)
    uncovered = sorted(set(conus.GEOID.values[tract_i[np.isin(tile_i, [cand.index(t) for t in absent])]])) if absent else []
    checks["tiles_needed_by_tracts"] = len(needed)
    checks["tiles_needed_but_absent"] = len(absent)
    checks["tracts_not_fully_covered_by_tiles"] = len(uncovered)

    # Balanced shards in GEOID order (keeps neighbouring tracts together).
    conus = conus.sort_values("GEOID").reset_index(drop=True)
    b = conus.geometry.bounds
    load = ((b.maxx - b.minx) / RES + 1) * ((b.maxy - b.miny) / RES + 1) + PER_TRACT_OVERHEAD_PX
    conus["est_load_px"] = load.round().astype("int64")
    cum = np.cumsum(conus.est_load_px.values, dtype=np.float64)
    conus["shard"] = np.minimum((cum - 1) * a.n_shards // cum[-1], a.n_shards - 1).astype(int)

    rows = []
    for sid, g in conus.groupby("shard", sort=True):
        wkb = b"".join(shapely.to_wkb(g.geometry.values, hex=False, output_dimension=2))
        rows.append({
            "shard_id": int(sid), "n_tracts": len(g), "first_geoid": g.GEOID.iloc[0], "last_geoid": g.GEOID.iloc[-1],
            "states": ",".join(sorted(g.StFIPS.unique())), "est_load_px": int(g.est_load_px.sum()),
            "geoid_list_sha256": sha256_text("\n".join(g.GEOID)),
            "geometry_sha256": hashlib.sha256(wkb).hexdigest(),
        })
    shards = pd.DataFrame(rows)
    checks["n_shards"] = len(shards)
    checks["shard_load_px_min_median_max"] = [int(shards.est_load_px.min()), int(shards.est_load_px.median()),
                                              int(shards.est_load_px.max())]
    checks["largest_tract_load_px"] = int(conus.est_load_px.max())
    checks["seconds"] = round(time.time() - t0, 1)

    failures = [k for k in ("geoid_not_11_digits", "duplicate_geoids", "geoid_ne_st_co_tr", "empty_geometries",
                            "invalid_geometries", "tiles_needed_but_absent", "tracts_not_fully_covered_by_tiles")
                if checks[k]]
    if checks["conus_states"] != 49:
        failures.append(f"expected 49 CONUS state codes, found {checks['conus_states']}")
    checks["failures"] = failures
    checks["passed"] = not failures

    keep = ATTR_COLS + ["geometry_area_m2", "est_load_px", "shard", "geometry"]
    tmp = out / "tracts_conus_4326.parquet.tmp"
    conus[keep].to_parquet(tmp, index=False)
    os.replace(tmp, out / "tracts_conus_4326.parquet")
    buf = io.StringIO()
    shards.to_csv(buf, index=False, lineterminator="\n")
    (out / "shard_manifest.csv.tmp").write_text(buf.getvalue())
    os.replace(out / "shard_manifest.csv.tmp", out / "shard_manifest.csv")
    checks["tracts_parquet_sha256"] = sha256_file(out / "tracts_conus_4326.parquet")
    checks["shard_manifest_sha256"] = sha256_file(out / "shard_manifest.csv")
    write_json(out / "tract_validation.json", checks)
    print({k: v for k, v in checks.items() if k not in ("source_crs",)}, flush=True)
    if failures:
        sys.exit(f"tract validation FAILED: {failures}")


if __name__ == "__main__":
    main()
