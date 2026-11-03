"""Reduce worker partials and build the output tables (Stage 2 schema; Stage 1 runs it on the test tiles only).

Reduce: concatenate partials (data/interim/partials/<tile>.parquet), sum all columns by (kind, zone_id).
All sums are additive across windows and tiles; distinct_cells summed over tiles is an upper bound.

Tables (each as .parquet and .csv; GEOID / GEOID20 are zero-padded strings; piece_id / ua_idx integers as in the
vegetation and UHI products):
  tract_piece_ndvi_evi_2020  one row per piece x season
  ua_ndvi_evi_2020           one row per UA x season
  piece_diagnostics_ndvi     one row per piece (static)
Values: value = sum_wv / valid_w (null when valid_w == 0); coverage = valid_w / total_w; delta = urban - rural with
the piece's own UA baseline. A null is never a zero; every null has a quality status. Status precedence:
piece_no_pixels, ua_no_valid_value (rural baseline has no valid value), urban_no_valid_value, ok. Statuses are
computed per index; if NDVI and EVI statuses never differ a single quality_status column is written, otherwise
quality_status_ndvi and quality_status_evi.

Usage: python ndvi_tables.py [--tiles T1,T2,...] [--partials-dir DIR] [--out DIR] [--test]
  --test: restrict rows to pieces/UAs that intersect the given tiles and add test_ua_complete (False when the
          UA extends into tiles that were not processed, i.e. its rural baseline is partial).
"""
import argparse
import glob
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ndvi_common import SEASON_SHORT, load_config, setup_logging, utc_now, write_json_atomic

IDX = {"NDVI": "ndvi", "EVI": "evi"}


def reduce_partials(paths, canonical=True):
    """Sum partial rows by (kind, zone_id). canonical=True sorts rows by (kind, zone_id, tile) first, so the
    result does not depend on the order in which partials are listed (bit-identical); canonical=False sums in
    the given order (used only to measure the floating-point order effect in gate G6)."""
    df = pd.concat([pd.read_parquet(p).assign(_tile=os.path.basename(p)) for p in paths], ignore_index=True)
    if canonical:
        df = df.sort_values(["kind", "zone_id", "_tile"], kind="mergesort")
    return df.drop(columns="_tile").groupby(["kind", "zone_id"], as_index=False, sort=True).sum(min_count=1)


def validity_differ(red, seasons):
    """Zones where NDVI and EVI valid weight differ in any season (a pixel valid for one index only).
    Returns {kind: number of zones}."""
    out = {}
    for kind in ("urban", "rural", "ua_all"):
        z = red[red.kind == kind]
        m = np.zeros(len(z), bool)
        for season in seasons:
            sh = SEASON_SHORT[season]
            m |= z[f"valid_w_NDVI_{sh}"].fillna(0).to_numpy() != z[f"valid_w_EVI_{sh}"].fillna(0).to_numpy()
        out[kind] = int(m.sum())
    return out


def zone_frame(red, kind):
    z = red[red.kind == kind].set_index("zone_id")
    return z.drop(columns="kind").dropna(axis=1, how="all")


def safe_div(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / b, np.nan)


def status(no_pixels, rural_valid, urban_valid):
    s = np.full(no_pixels.shape, "ok", dtype=object)
    s[urban_valid <= 0] = "urban_no_valid_value"
    s[rural_valid <= 0] = "ua_no_valid_value"
    s[no_pixels] = "piece_no_pixels"
    return s


def build(cfg, red, pieces, uas, test_tiles=None, ua_complete=None, split_status=None):
    year = cfg["year"]
    nv_p, nv_u = cfg["domains"]["n_pieces_veg"], cfg["domains"]["n_uas_veg"]
    U = zone_frame(red, "urban").reindex(pieces.piece_id)
    Rr = zone_frame(red, "rural").reindex(uas.ua_idx)
    A = zone_frame(red, "ua_all").reindex(uas.ua_idx)
    DP = zone_frame(red, "diag_piece").reindex(pieces.piece_id)
    DU = zone_frame(red, "diag_ua").reindex(uas.ua_idx)
    ua_by_idx = uas.set_index("ua_idx")
    pix = (DP["valid_m2"].fillna(0) + DP["nodata_m2"].fillna(0)).to_numpy()
    no_pixels = pix <= 0
    sliver = (pieces.overlap_share < cfg["domains"]["sliver_overlap_share_lt"]).to_numpy()
    base = pd.DataFrame({"piece_id": pieces.piece_id.to_numpy(), "GEOID": pieces.GEOID.to_numpy(),
                         "GEOID20": pieces.GEOID20.to_numpy(),
                         "NAME20": ua_by_idx.loc[pieces.ua_idx, "NAME20"].to_numpy()})
    rows, stat_differ = [], 0
    for season in cfg["seasons"]:
        sh = SEASON_SHORT[season]
        d = base.copy()
        d["year"] = year
        d["season"] = season
        st = {}
        utot = U["total_w"].fillna(0).to_numpy()
        for ix, lx in IDX.items():
            c = f"{ix}_{sh}"
            uv = U[f"valid_w_{c}"].fillna(0).to_numpy()
            us = U[f"sum_wv_{c}"].fillna(0).to_numpy()
            rv = Rr.loc[pieces.ua_idx, f"valid_w_{c}"].fillna(0).to_numpy()
            rs = Rr.loc[pieces.ua_idx, f"sum_wv_{c}"].fillna(0).to_numpy()
            rt = Rr.loc[pieces.ua_idx, "total_w"].fillna(0).to_numpy()
            d[f"urban_{lx}"] = safe_div(us, uv)
            d[f"rural_{lx}"] = safe_div(rs, rv)
            d[f"delta_{lx}"] = d[f"urban_{lx}"] - d[f"rural_{lx}"]
            st[lx] = status(no_pixels, rv, uv)
            d[f"{lx}_urban_valid_coverage"] = safe_div(uv, utot)
            d[f"{lx}_rural_valid_coverage"] = safe_div(rv, rt)
            d[f"{lx}_urban_valid_weight_m2"] = uv
            d[f"{lx}_urban_total_weight_m2"] = utot
            d[f"{lx}_urban_sum_wv"] = us
        d["urban_distinct_cells"] = U["distinct_cells"].fillna(0).astype(np.int64).to_numpy()
        d["urban_clear_obs_mean"] = safe_div(U[f"sum_wclear_NDVI_{sh}"].fillna(0), U[f"valid_w_NDVI_{sh}"].fillna(0))
        d["overlap_share"] = pieces.overlap_share.to_numpy()
        d["piece_area_m2"] = pieces.piece_area_m2.to_numpy()
        d["tract_area_m2"] = pieces.tract_area_m2.to_numpy()
        d["sliver_flag"] = sliver
        d["in_vegetation_product"] = (pieces.piece_id <= nv_p).to_numpy()
        stat_differ += int((st["ndvi"] != st["evi"]).sum())
        d["quality_status_ndvi"] = st["ndvi"]
        d["quality_status_evi"] = st["evi"]
        if ua_complete is not None:
            d["test_ua_complete"] = pieces.GEOID20.map(ua_complete).to_numpy()
        rows.append(d)
    tp = pd.concat(rows, ignore_index=True)
    # nulls: a delta needs both sides; urban is kept when only the rural baseline is missing
    if split_status is None:
        split_status = stat_differ > 0
    if not split_status:
        tp.insert(tp.columns.get_loc("quality_status_ndvi"), "quality_status", tp["quality_status_ndvi"])
        tp = tp.drop(columns=["quality_status_ndvi", "quality_status_evi"])

    # ------------------------------------------------------------------ UA table
    urows = []
    for season in cfg["seasons"]:
        sh = SEASON_SHORT[season]
        d = pd.DataFrame({"ua_idx": uas.ua_idx.to_numpy(), "GEOID20": uas.GEOID20.to_numpy(),
                          "NAME20": uas.NAME20.to_numpy(), "year": year, "season": season})
        for zname, Z in (("rural", Rr), ("ua_all", A)):
            tot = Z["total_w"].fillna(0).to_numpy()
            for ix, lx in IDX.items():
                c = f"{ix}_{sh}"
                v = Z[f"valid_w_{c}"].fillna(0).to_numpy(); s = Z[f"sum_wv_{c}"].fillna(0).to_numpy()
                d[f"{zname}_{lx}"] = safe_div(s, v)
                d[f"{zname}_{lx}_valid_coverage"] = safe_div(v, tot)
                d[f"{zname}_{lx}_valid_weight_m2"] = v
                d[f"{zname}_{lx}_total_weight_m2"] = tot
                d[f"{zname}_{lx}_sum_wv"] = s
                d[f"{zname}_{lx}_clear_obs_mean"] = safe_div(Z[f"sum_wclear_{c}"].fillna(0), v)
            d[f"{zname}_distinct_cells"] = Z["distinct_cells"].fillna(0).astype(np.int64).to_numpy()
        for lx in IDX.values():
            d[f"delta_all_{lx}"] = d[f"ua_all_{lx}"] - d[f"rural_{lx}"]
        for c in ("pixel_m2", "water_m2", "built_m2", "nodata_m2", "snow_ice_m2"):
            d[f"ua_{c.replace('pixel_m2', 'pixel_area_m2')}"] = DU[c].fillna(0).to_numpy()
        d["in_vegetation_product"] = (uas.ua_idx <= nv_u).to_numpy()
        for lx in IDX.values():
            s = np.full(len(d), "ok", dtype=object)
            s[d[f"ua_all_{lx}_valid_weight_m2"] <= 0] = "ua_all_no_valid_value"
            s[d[f"rural_{lx}_valid_weight_m2"] <= 0] = "ua_no_valid_value"
            s[d["ua_pixel_area_m2"] <= 0] = "ua_no_pixels"
            d[f"quality_status_{lx}"] = s
        if ua_complete is not None:
            d["test_ua_complete"] = d.GEOID20.map(ua_complete)
        urows.append(d)
    tu = pd.concat(urows, ignore_index=True)
    if not split_status:
        tu.insert(tu.columns.get_loc("quality_status_ndvi"), "quality_status", tu["quality_status_ndvi"])
        tu = tu.drop(columns=["quality_status_ndvi", "quality_status_evi"])

    # ------------------------------------------------------------------ piece diagnostics
    dg = pd.DataFrame({"piece_id": pieces.piece_id.to_numpy(), "GEOID": pieces.GEOID.to_numpy(),
                       "GEOID20": pieces.GEOID20.to_numpy(), "ua_idx": pieces.ua_idx.to_numpy(),
                       "piece_area_m2": pieces.piece_area_m2.to_numpy(), "tract_area_m2": pieces.tract_area_m2.to_numpy(),
                       "overlap_share": pieces.overlap_share.to_numpy(), "sliver_flag": sliver,
                       "owns_pixels": ~no_pixels, "pixel_area_m2": pix})
    for c in ("valid_m2", "water_m2", "built_m2", "nodata_m2"):
        dg[c] = DP[c].fillna(0).to_numpy()
    dg["urban_total_weight_m2"] = U["total_w"].fillna(0).to_numpy()
    dg["urban_distinct_cells"] = U["distinct_cells"].fillna(0).astype(np.int64).to_numpy()
    dg["pixel_area_ratio"] = safe_div(pix, pieces.piece_area_m2)
    dg["in_vegetation_product"] = (pieces.piece_id <= nv_p).to_numpy()
    return tp, tu, dg, stat_differ


def write_table(df, out, name):
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, f"{name}.parquet.part"), index=False)
    os.replace(os.path.join(out, f"{name}.parquet.part"), os.path.join(out, f"{name}.parquet"))
    df.to_csv(os.path.join(out, f"{name}.csv.part"), index=False)
    os.replace(os.path.join(out, f"{name}.csv.part"), os.path.join(out, f"{name}.csv"))


def tile_box(t):
    lat = int(t[1:3]) * (1 if t[0] == "N" else -1)
    lon = int(t[4:7]) * (1 if t[3] == "E" else -1)
    return shapely.box(lon, lat, lon + 3, lat + 3)


def full_meta(cfg, tiles, pdir, tp, tu, dg, vd, split):
    """Provenance record written next to the tables (like urban_rural_veg.meta.json)."""
    from ndvi_common import sha256_file
    import pyarrow.parquet as pq
    man = pd.read_json(os.path.join(cfg["paths"]["composites"], "manifest.jsonl"), lines=True)
    man = man[man.C == cfg["grid"]["chunk_cells"]].drop_duplicates(["I", "J", "season"], keep="last")
    sig = json.loads(pq.read_schema(f"{pdir}/{tiles[0]}.parquet").metadata[b"ndvi"])
    st_col = ["quality_status"] if "quality_status" in tp else ["quality_status_ndvi", "quality_status_evi"]
    return {
        "utc": utc_now(), "product": "tract-piece urban-rural Landsat NDVI/EVI 2020, Method A",
        "year": cfg["year"], "seasons": {"JJA": "2020-06-01..2020-08-31", "DJF_same_year": "2020-01-01..2020-02-29 and 2020-12-01..2020-12-31"},
        "composites": {"source": "Landsat Collection 2 Level 2 (LT04/LT05/LE07<2017/LC08/LC09 T1_L2; 2020: LC08 only), Earth Engine",
                       "code": "scripts/ee_composite.py (GECC NDVI/EVI v1 code without its GLC water mask)",
                       "statistic": "per-pixel median of per-scene NDVI / EVI; clear_obs = count of clear observations",
                       "earth_engine_project": cfg["earth_engine"]["project"], "fetch": "ee.data.computePixels, high-volume endpoint",
                       "store": cfg["paths"]["composites"], "store_cfs": "/path/to/project/urbveg/ndvi/composites", "manifest_sha256": sha256_file(os.path.join(cfg["paths"]["composites"], "manifest.jsonl")),
                       "n_chunks_by_season": man.season.value_counts().to_dict(), "chunk_cells": cfg["grid"]["chunk_cells"],
                       "mean_nan_share_ndvi_by_season": man.groupby("season").nan_share_ndvi.mean().round(4).to_dict()},
        "grid": {"crs": "EPSG:4326", "cell_deg": cfg["grid"]["cell_deg"], "origin": [0.0, 0.0],
                 "cell_rule": "i = floor(lon/s), j = floor(-lat/s); each 10 m pixel takes the cell containing its centre",
                 "ua_buffer_cells": cfg["grid"]["ua_buffer_cells"]},
        "masks": {"worldcover": "ESA WorldCover 2020 v100, 10 m", "urban_excluded": cfg["classes"]["urban_excluded"],
                  "rural_excluded": cfg["classes"]["rural_excluded"], "ua_all_excluded": cfg["classes"]["ua_all_excluded"],
                  "snow_ice_70": "kept", "water": "WorldCover 80 only (no GLC water mask)"},
        "rasterization": {"window_px": sig["window_px"], "all_touched": sig["all_touched"],
                          "pixel_area": f"spherical, R = {sig['area_radius_m']} m, geodetic latitude"},
        "statistic": "value = sum(w*v)/sum(w) over pixels with a valid value; coverage = valid weight / total weight",
        "sliver_thresholds": {"overlap_share_lt": cfg["domains"]["sliver_overlap_share_lt"], "piece_area_m2_lt": None},
        "support_thresholds": None,
        "inputs_sha256": {"pieces_all": sig["pieces_all_sha256"], "uas_all": sig["uas_all_sha256"],
                          "pieces_veg": sig["pieces_veg_sha256"], "uas_veg": sig["uas_veg_sha256"]},
        "n_tiles": len(tiles), "worker_code_version": sig["code_version"],
        "n_rows": {"tract_piece_ndvi_evi_2020": len(tp), "ua_ndvi_evi_2020": len(tu), "piece_diagnostics_ndvi": len(dg)},
        "n_pieces": int(dg.piece_id.nunique()), "n_uas": int(tu.GEOID20.nunique()),
        "n_sliver_flagged": int(dg.sliver_flag.sum()), "n_pieces_without_pixels": int((~dg.owns_pixels).sum()),
        "ndvi_evi_validity_differ_zones": vd, "per_index_status": bool(split),
        "status_counts_piece_rows": {c: {s_: g[c].value_counts().to_dict() for s_, g in tp.groupby("season")} for c in st_col},
        "status_counts_ua_rows": {c: {s_: g[c].value_counts().to_dict() for s_, g in tu.groupby("season")}
                                  for c in (["quality_status"] if "quality_status" in tu else ["quality_status_ndvi", "quality_status_evi"])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=None)
    ap.add_argument("--partials-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="list partials in reverse tile order (gate G6)")
    ap.add_argument("--no-canonical", action="store_true", help="sum in listed order (gate G6, information only)")
    a = ap.parse_args()
    cfg = load_config()
    log = setup_logging("ndvi_tables")
    pdir = a.partials_dir or cfg["paths"]["partials_dir"]
    if a.tiles:
        tiles = a.tiles.split(",")
    elif a.test:
        tiles = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{pdir}/*.parquet"))
    else:
        tiles = pd.read_csv(cfg["worldcover"]["tiles_manifest_all"]).tile.tolist()
    order = tiles[::-1] if a.reverse else tiles
    red = reduce_partials([f"{pdir}/{t}.parquet" for t in order], canonical=not a.no_canonical)
    pieces = gpd.read_parquet(cfg["domains"]["pieces_all"])
    uas = gpd.read_parquet(cfg["domains"]["uas_all"])
    ua_complete = None
    if a.test:
        box = shapely.union_all([tile_box(t) for t in tiles])
        pieces = pieces[pieces.intersects(box)].reset_index(drop=True)
        uas_t = uas[uas.intersects(box)]
        tl = pd.read_csv(cfg["worldcover"]["tiles_manifest_all"])
        grid = gpd.GeoDataFrame({"tile": tl.tile}, geometry=[tile_box(t) for t in tl.tile], crs="EPSG:4326")
        j = gpd.sjoin(uas_t[["GEOID20", "geometry"]], grid, predicate="intersects")
        ua_complete = j.groupby("GEOID20").tile.apply(lambda s: set(s) <= set(tiles)).to_dict()
        uas = uas_t.reset_index(drop=True)
    vd = validity_differ(red, cfg["seasons"])
    split = sum(vd.values()) > 0
    tp, tu, dg, differ = build(cfg, red, pieces, uas, ua_complete=ua_complete, split_status=split)
    out = a.out or os.path.join(cfg["paths"]["output"], "census_tracts_ua_ndvi")
    write_table(tp, out, "tract_piece_ndvi_evi_2020")
    write_table(tu, out, "ua_ndvi_evi_2020")
    write_table(dg, out, "piece_diagnostics_ndvi")
    red.to_parquet(os.path.join(out, "reduced_sums.parquet"), index=False)
    if a.test:
        meta = {"utc": utc_now(), "tiles": tiles, "n_piece_rows": len(tp), "n_ua_rows": len(tu), "n_pieces": len(dg),
                "ndvi_evi_status_rows_differ": differ, "ndvi_evi_validity_differ_zones": vd, "test": True}
        write_json_atomic(os.path.join(out, "tables_meta.json"), meta)
    else:
        meta = full_meta(cfg, tiles, pdir, tp, tu, dg, vd, split)
        meta.update({"reduce_order": "reverse" if a.reverse else "manifest", "reduce_canonical": not a.no_canonical})
        write_json_atomic(os.path.join(out, "tract_piece_ndvi_evi_2020.meta.json"), meta)
    log.info("rows piece=%d ua=%d diag=%d validity_differ=%s split=%s out=%s", len(tp), len(tu), len(dg), vd, split, out)


if __name__ == "__main__":
    main()
