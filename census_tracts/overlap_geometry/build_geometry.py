"""Phase B geometry preparation (run once).

Products:
  ua.gpkg                       repaired Urban Area polygons, one per UACE20
  ua_rural_ring_10km.gpkg       10 km outward ring minus every Census UA
  ring_qa.parquet               ring geometry diagnostics and flags
  tract_ua_crosswalk.parquet    every positive-area tract-UA pair (QA)
  tract_primary_ua.parquet      one row per eligible tract
  excluded_tracts.parquet       tracts with no positive-area UA overlap
  geometry_qa.json              the required checks, including the proof that
                                no ring has positive-area overlap with any UA

Construction rules (locked):
  * ring = buffer(UA, 10 km) - union(all 2020 Census UAs), a fixed outward
    buffer, never normalized or equal-area;
  * the buffer is built in a per-UA azimuthal equidistant projection centred on
    that UA, so 10,000 m is a true distance for every UA including Alaska,
    Hawaii and the territories, rather than accepting Albers scale error;
  * the subtraction is done in the boundary files' own geographic coordinates,
    using the original UA vertices, so the ring and the UA share exact edges
    and the overlap is identically zero rather than zero "within tolerance";
  * rings of different UAs may overlap each other; that is allowed.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from pyproj import CRS, Transformer

import common

log = common.setup_logging("build_geometry")

GEOGRAPHIC_CRS = "EPSG:4269"  # NAD83, the vintage of both boundary files

_UA_TREE = None
_UA_GEOMS = None
_UA_IDS = None


def _aeqd(lon0: float, lat0: float) -> str:
    return (f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} +x_0=0 +y_0=0 "
            f"+ellps=GRS80 +units=m +no_defs")


def _init_worker(geoms_wkb, ids):
    global _UA_TREE, _UA_GEOMS, _UA_IDS
    _UA_GEOMS = shapely.from_wkb(geoms_wkb)
    _UA_IDS = ids
    _UA_TREE = STRtree(_UA_GEOMS)


def _ring_for_ua(i: int) -> dict:
    """Build one rural ring and its diagnostics."""
    geom = _UA_GEOMS[i]
    uace = _UA_IDS[i]
    lon0, lat0 = shapely.get_x(shapely.centroid(geom)), shapely.get_y(shapely.centroid(geom))

    fwd = Transformer.from_crs(GEOGRAPHIC_CRS, _aeqd(lon0, lat0), always_xy=True)
    inv = Transformer.from_crs(_aeqd(lon0, lat0), GEOGRAPHIC_CRS, always_xy=True)

    # Densify in degrees first: a straight edge in lon/lat is a curve in the
    # metric projection, and the buffer must follow the true edge.
    dense = shapely.segmentize(geom, common.SEGMENTIZE_DEG)
    proj = shapely.transform(
        dense, lambda c: np.column_stack(fwd.transform(c[:, 0], c[:, 1])))

    buf = shapely.buffer(proj, common.BUFFER_DISTANCE_M,
                         quad_segs=common.BUFFER_QUAD_SEGS)
    # Check the realized buffer distance in the metric construction CRS.
    ext = shapely.get_exterior_ring(shapely.get_geometry(buf, 0)) if shapely.get_type_id(buf) == 6 \
        else shapely.get_exterior_ring(buf)
    verts = shapely.get_coordinates(ext)
    if len(verts) > 5000:
        verts = verts[:: len(verts) // 5000]
    dists = shapely.distance(shapely.points(verts), proj)
    max_radius_m = float(np.nanmax(np.hypot(verts[:, 0], verts[:, 1]))) if len(verts) else 0.0

    buf = shapely.segmentize(buf, common.SEGMENTIZE_M)
    buf_geo = shapely.transform(buf, lambda c: np.column_stack(inv.transform(c[:, 0], c[:, 1])))
    buf_geo = shapely.make_valid(buf_geo)

    # Subtract every Census UA that touches the buffer, focal UA included.
    cand = _UA_TREE.query(buf_geo, predicate="intersects")
    cand_geoms = [_UA_GEOMS[j] for j in cand]
    union = shapely.union_all(cand_geoms) if cand_geoms else None
    ring = shapely.difference(buf_geo, union) if union is not None else buf_geo
    ring = shapely.make_valid(ring)
    # Keep polygonal parts only; a difference can leave slivers of lower
    # dimension that are not area.
    ring = shapely.get_parts(ring)
    ring = shapely.union_all([g for g in ring if shapely.get_type_id(g) in (3, 6)]) \
        if len(ring) else shapely.Polygon()
    if ring is None:
        ring = shapely.Polygon()

    n_parts = int(shapely.get_num_geometries(ring)) if not shapely.is_empty(ring) else 0
    parts = shapely.get_parts(ring) if n_parts else np.array([])
    part_areas = shapely.area(parts) if n_parts else np.array([])

    return {
        "UACE20": uace,
        "ring_wkb": shapely.to_wkb(ring),
        "buffer_wkb": shapely.to_wkb(buf_geo),
        "n_parts": n_parts,
        "largest_part_deg2": float(part_areas.max()) if n_parts else 0.0,
        "total_deg2": float(part_areas.sum()) if n_parts else 0.0,
        "n_interior_rings": int(sum(shapely.get_num_interior_rings(p) for p in parts))
                            if n_parts else 0,
        "n_neighbour_uas": int(max(0, len(cand) - 1)),
        "buffer_vertex_distance_min_m": float(np.nanmin(dists)) if len(dists) else None,
        "buffer_vertex_distance_max_m": float(np.nanmax(dists)) if len(dists) else None,
        "max_radius_m": max_radius_m,
        # AEQD scale error grows as (rho/R)^2/6; record the bound actually used.
        "aeqd_scale_error_bound": float((max_radius_m / 6371000.0) ** 2 / 6.0),
        "aeqd_lon0": lon0, "aeqd_lat0": lat0,
    }


def build_rings(ua: gpd.GeoDataFrame, workers: int) -> tuple:
    geoms_wkb = shapely.to_wkb(ua.geometry.values)
    ids = ua["UACE20"].tolist()
    log.info("building %d rings on %d workers", len(ua), workers)
    results = [None] * len(ua)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(geoms_wkb, ids)) as ex:
        for n, rec in enumerate(ex.map(_ring_for_ua, range(len(ua)), chunksize=8), 1):
            results[ids.index(rec["UACE20"])] = rec
            if n % 250 == 0:
                log.info("  %d/%d rings", n, len(ua))
    rings = gpd.GeoDataFrame(
        {"UACE20": [r["UACE20"] for r in results],
         "NAME20": ua.set_index("UACE20").loc[[r["UACE20"] for r in results],
                                              "NAME20"].values},
        geometry=[shapely.from_wkb(r["ring_wkb"]) for r in results],
        crs=GEOGRAPHIC_CRS)
    qa = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("ring_wkb", "buffer_wkb")} for r in results])
    buffers = gpd.GeoDataFrame(
        {"UACE20": [r["UACE20"] for r in results]},
        geometry=[shapely.from_wkb(r["buffer_wkb"]) for r in results],
        crs=GEOGRAPHIC_CRS)
    return rings, qa, buffers


def prove_zero_overlap(rings: gpd.GeoDataFrame, ua: gpd.GeoDataFrame) -> dict:
    """Every final ring must have zero positive-area intersection with every
    Census UA.  Checked against all UAs, not only the ones subtracted."""
    tree = STRtree(ua.geometry.values)
    ua_ea = ua.to_crs(common.EQUAL_AREA_CRS).geometry.values
    rows = []
    worst = 0.0
    for i, ring in enumerate(rings.geometry.values):
        if shapely.is_empty(ring):
            continue
        hits = tree.query(ring, predicate="intersects")
        if not len(hits):
            continue
        inter = shapely.intersection(ring, shapely.union_all([ua.geometry.values[j]
                                                              for j in hits]))
        if shapely.is_empty(inter) or shapely.area(inter) == 0:
            continue
        # Measure any residue in metres, not degrees.
        inter_ea = gpd.GeoSeries([inter], crs=GEOGRAPHIC_CRS).to_crs(
            common.EQUAL_AREA_CRS).geometry.values[0]
        a = float(shapely.area(inter_ea))
        worst = max(worst, a)
        if a > common.RING_OVERLAP_TOL_M2:
            rows.append({"UACE20": rings["UACE20"].iloc[i], "overlap_m2": a})
    return {"max_overlap_m2": worst, "tolerance_m2": common.RING_OVERLAP_TOL_M2,
            "n_rings_over_tolerance": len(rows),
            "offenders": rows[:50],
            "passed": len(rows) == 0}


def build_crosswalk(tracts: gpd.GeoDataFrame, ua: gpd.GeoDataFrame,
                    id_field: str, min_fraction: float) -> tuple:
    """All positive-area tract-UA intersections, with the primary selection."""
    tree = STRtree(ua.geometry.values)
    pairs = tree.query(tracts.geometry.values, predicate="intersects")
    log.info("%d candidate tract-UA pairs from the spatial index", pairs.shape[1])

    ti, ui = pairs[0], pairs[1]
    inter = shapely.intersection(tracts.geometry.values[ti], ua.geometry.values[ui])
    # Area must be measured in an equal-area CRS, never in degrees.
    inter_ea = gpd.GeoSeries(inter, crs=GEOGRAPHIC_CRS).to_crs(common.EQUAL_AREA_CRS)
    area = shapely.area(inter_ea.values)
    tract_area = shapely.area(
        tracts.to_crs(common.EQUAL_AREA_CRS).geometry.values)

    df = pd.DataFrame({
        "GEOID": tracts[id_field].values[ti],
        "UACE20": ua["UACE20"].values[ui],
        "NAME20": ua["NAME20"].values[ui],
        "overlap_area_m2": area,
        "tract_area_m2": tract_area[ti],
    })
    df["overlap_fraction_of_tract"] = np.where(
        df["tract_area_m2"] > 0, df["overlap_area_m2"] / df["tract_area_m2"], 0.0)

    touching_only = df[df["overlap_area_m2"] <= common.POSITIVE_AREA_TOL_M2]
    df = df[df["overlap_area_m2"] > common.POSITIVE_AREA_TOL_M2].copy()
    log.info("%d positive-area pairs; %d zero-area boundary touches discarded",
             len(df), len(touching_only))

    # Every positive-area pair stays in the QA crosswalk.  Only pairs covering
    # at least min_fraction of the tract count as an association.
    df["meets_min_overlap"] = df["overlap_fraction_of_tract"] >= min_fraction
    n_below = int((~df["meets_min_overlap"]).sum())
    log.info("%d of %d positive-area pairs are below the %.3f%% minimum "
             "overlap and do not count as associations", n_below, len(df),
             min_fraction * 100)

    # Largest overlap wins; exact ties break by ascending UACE20.
    df = df.sort_values(["GEOID", "overlap_area_m2", "UACE20"],
                        ascending=[True, False, True], kind="mergesort")
    qual = df[df["meets_min_overlap"]].copy()
    qual["rank"] = qual.groupby("GEOID").cumcount() + 1
    qual["is_primary"] = qual["rank"] == 1
    qual["ua_count"] = qual.groupby("GEOID")["UACE20"].transform("size")
    df = df.merge(qual[["GEOID", "UACE20", "rank", "is_primary", "ua_count"]],
                  on=["GEOID", "UACE20"], how="left")
    df["is_primary"] = df["is_primary"].fillna(False).astype(bool)

    primary = qual[qual["is_primary"]].rename(columns={
        "UACE20": "primary_UACE20", "NAME20": "primary_ua_name",
        "overlap_area_m2": "primary_overlap_area_m2",
        "overlap_fraction_of_tract": "primary_overlap_fraction_of_tract",
    })[["GEOID", "primary_UACE20", "primary_ua_name", "primary_overlap_area_m2",
        "primary_overlap_fraction_of_tract", "ua_count", "tract_area_m2"]]

    eligible = set(primary["GEOID"])
    all_ids = pd.Series(tracts[id_field].values, dtype="string")
    excluded = pd.DataFrame({"GEOID": all_ids[~all_ids.isin(eligible)].values})

    # Distinguish a tract with no positive-area overlap at all from one whose
    # every overlap is below the minimum fraction.
    best = df.groupby("GEOID")[["overlap_area_m2",
                                "overlap_fraction_of_tract"]].max()
    excluded = excluded.merge(best, left_on="GEOID", right_index=True, how="left")
    excluded = excluded.rename(columns={
        "overlap_area_m2": "max_overlap_area_m2",
        "overlap_fraction_of_tract": "max_overlap_fraction_of_tract"})
    has_any = excluded["max_overlap_area_m2"].notna()
    excluded["reason"] = np.where(
        has_any, "max_overlap_below_min_fraction", "no_positive_area_ua_overlap")
    excluded["min_overlap_fraction_required"] = min_fraction
    excluded["touches_ua_with_zero_area"] = excluded["GEOID"].isin(
        set(touching_only["GEOID"]))
    log.info("excluded %d tracts: %s", len(excluded),
             excluded["reason"].value_counts().to_dict())
    return df, primary, excluded, touching_only


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_ON_NODE", 16)))
    ap.add_argument("--min-overlap-fraction", type=float,
                    default=common.MIN_OVERLAP_FRACTION_OF_TRACT,
                    help="minimum intersection area as a fraction of the tract "
                         "for a tract-UA pair to count as an association")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv = json.loads(Path(args.inventory).read_text())
    ua_src = Path(inv["urban_areas"]["dataset_path"])
    tract_src = Path(inv["tracts"]["dataset_path"])
    tract_id = inv["tracts"]["identifier_field"]

    params = {"buffer_m": common.BUFFER_DISTANCE_M,
              "quad_segs": common.BUFFER_QUAD_SEGS,
              "segmentize_deg": common.SEGMENTIZE_DEG,
              "segmentize_m": common.SEGMENTIZE_M,
              "equal_area_crs": common.EQUAL_AREA_CRS,
              "geographic_crs": GEOGRAPHIC_CRS,
              "positive_area_tol_m2": common.POSITIVE_AREA_TOL_M2,
              "min_overlap_fraction_of_tract": args.min_overlap_fraction,
              "buffer_crs": "per-UA azimuthal equidistant (GRS80)"}
    inputs = [common.fingerprint(ua_src), common.fingerprint(tract_src)]

    ring_path = out / "ua_rural_ring_10km.gpkg"
    ua_path = out / "ua.gpkg"
    if not args.force and common.stage_complete(ring_path, inputs, params) \
            and common.stage_complete(ua_path, inputs, params):
        log.info("geometry already complete; nothing to do")
        return

    # --- Urban Areas -------------------------------------------------------
    log.info("reading Urban Areas %s", ua_src)
    ua = gpd.read_file(ua_src)
    ua = ua.set_crs(GEOGRAPHIC_CRS, allow_override=False) if ua.crs is None else ua
    keep = [c for c in ("UACE20", "GEOID20", "NAME20", "NAMELSAD20", "LSAD20",
                        "MTFCC20", "FUNCSTAT20", "ALAND20", "AWATER20",
                        "INTPTLAT20", "INTPTLON20") if c in ua.columns]
    ua = ua[keep + ["geometry"]].copy()
    ua["UACE20"] = ua["UACE20"].astype("string")
    n_invalid = int((~shapely.is_valid(ua.geometry.values)).sum())
    if n_invalid:
        log.info("repairing %d invalid UA geometries (no dissolve across IDs)", n_invalid)
        ua["geometry"] = shapely.make_valid(ua.geometry.values)
    if ua["UACE20"].duplicated().any():
        raise SystemExit("duplicate UACE20 values; refusing to dissolve them")
    ua = ua.to_crs(GEOGRAPHIC_CRS)

    # --- tracts ------------------------------------------------------------
    log.info("reading tracts %s", tract_src)
    tracts = gpd.read_file(tract_src)
    tracts[tract_id] = tracts[tract_id].astype("string")
    eq = inv["tracts"]["crs_equivalence"]
    if eq["max_displacement_m_approx"] > 0.01:
        raise SystemExit(
            "tract CRS is not coordinate-identical to NAD83 "
            f"(max displacement {eq['max_displacement_m_approx']} m); "
            "resolve the datum before continuing")
    log.info("tract CRS is coordinate-identical to %s (max displacement %.3g m); "
             "labelling it as such", GEOGRAPHIC_CRS, eq["max_displacement_m_approx"])
    tracts = tracts.set_crs(GEOGRAPHIC_CRS, allow_override=True)
    n_invalid_t = int((~shapely.is_valid(tracts.geometry.values)).sum())
    if n_invalid_t:
        log.info("repairing %d invalid tract geometries", n_invalid_t)
        tracts["geometry"] = shapely.make_valid(tracts.geometry.values)

    # --- rings -------------------------------------------------------------
    with common.stage_timer(log, "rings", out / "stage_times.json"):
        rings, ring_qa, buffers = build_rings(ua, args.workers)

    # areas and flags in equal-area metres
    rings_ea = rings.to_crs(common.EQUAL_AREA_CRS)
    ring_qa["ring_area_km2"] = shapely.area(rings_ea.geometry.values) / 1e6
    ua_ea = ua.to_crs(common.EQUAL_AREA_CRS)
    ring_qa["ua_area_km2"] = shapely.area(ua_ea.geometry.values) / 1e6
    buf_ea = buffers.to_crs(common.EQUAL_AREA_CRS)
    ring_qa["buffer_area_km2"] = shapely.area(buf_ea.geometry.values) / 1e6
    potential = ring_qa["buffer_area_km2"] - ring_qa["ua_area_km2"]
    ring_qa["potential_ring_area_km2"] = potential
    ring_qa["retained_fraction"] = np.where(potential > 0,
                                            ring_qa["ring_area_km2"] / potential, np.nan)
    ring_qa["is_empty"] = ring_qa["ring_area_km2"] <= 0
    ring_qa["largest_part_fraction"] = np.where(
        ring_qa["total_deg2"] > 0, ring_qa["largest_part_deg2"] / ring_qa["total_deg2"],
        np.nan)
    ring_qa["fragmented_flag"] = (ring_qa["n_parts"] > 1) & \
                                 (ring_qa["largest_part_fraction"] < 0.5)
    ring_qa["ring_valid"] = shapely.is_valid(rings.geometry.values)
    tol = common.BUFFER_DISTANCE_TOL_M
    ring_qa["buffer_distance_ok"] = (
        (ring_qa["buffer_vertex_distance_min_m"] > common.BUFFER_DISTANCE_M - tol) &
        (ring_qa["buffer_vertex_distance_max_m"] < common.BUFFER_DISTANCE_M + tol))

    # Ring area lying outside the tract footprint marks coastal water beyond the
    # Census water boundary and land across a national border.
    log.info("dissolving the tract footprint for the outside-US ring check")
    try:
        footprint = shapely.coverage_union_all(tracts.geometry.values)
    except Exception:
        footprint = shapely.union_all(tracts.geometry.values)
    outside = shapely.difference(rings.geometry.values, footprint)
    outside_ea = gpd.GeoSeries(outside, crs=GEOGRAPHIC_CRS).to_crs(common.EQUAL_AREA_CRS)
    ring_qa["ring_outside_tract_footprint_km2"] = shapely.area(outside_ea.values) / 1e6
    ring_qa["ring_outside_tract_footprint_fraction"] = np.where(
        ring_qa["ring_area_km2"] > 0,
        ring_qa["ring_outside_tract_footprint_km2"] / ring_qa["ring_area_km2"], 0.0)
    ring_qa["extends_outside_us_tract_footprint"] = \
        ring_qa["ring_outside_tract_footprint_fraction"] > 0.01

    gpd.GeoDataFrame({"UACE20": rings["UACE20"]}, geometry=list(outside),
                     crs=GEOGRAPHIC_CRS).to_file(
        out / "ring_outside_tract_footprint.gpkg", driver="GPKG")

    # --- proof: rings do not overlap any Census UA -------------------------
    with common.stage_timer(log, "ring_overlap_proof", out / "stage_times.json"):
        proof = prove_zero_overlap(rings, ua)
    log.info("ring/UA overlap proof: max %.6g m2, %d over tolerance",
             proof["max_overlap_m2"], proof["n_rings_over_tolerance"])

    # --- crosswalk ---------------------------------------------------------
    with common.stage_timer(log, "crosswalk", out / "stage_times.json"):
        cross, primary, excluded, touching = build_crosswalk(
            tracts, ua, tract_id, args.min_overlap_fraction)

    # --- crosswalk QA ------------------------------------------------------
    # The primary must be the largest overlap among the pairs that qualify as
    # associations, for the tracts that are actually eligible.
    qualifying = cross[cross["meets_min_overlap"]]
    max_by_tract = qualifying.groupby("GEOID")["overlap_area_m2"].max()
    prim_idx = primary.set_index("GEOID")["primary_overlap_area_m2"]
    qa = {
        "n_tracts": int(len(tracts)),
        "n_uas": int(len(ua)),
        "n_positive_area_pairs": int(len(cross)),
        "n_zero_area_touches_discarded": int(len(touching)),
        "min_overlap_fraction_of_tract": args.min_overlap_fraction,
        "n_pairs_meeting_min_overlap": int(len(qualifying)),
        "n_pairs_below_min_overlap_retained_for_qa": int(
            (~cross["meets_min_overlap"]).sum()),
        "n_eligible_tracts": int(len(primary)),
        "n_excluded_tracts": int(len(excluded)),
        "excluded_reasons": excluded["reason"].value_counts().to_dict(),
        "exactly_one_primary_per_tract": bool(
            cross[cross["is_primary"]].groupby("GEOID").size().eq(1).all()),
        "primary_is_in_candidate_list": bool(
            primary.merge(cross, left_on=["GEOID", "primary_UACE20"],
                          right_on=["GEOID", "UACE20"]).shape[0] == len(primary)),
        "no_duplicate_pairs": bool(not cross.duplicated(["GEOID", "UACE20"]).any()),
        "primary_equals_max_overlap": bool(
            np.allclose(prim_idx.reindex(max_by_tract.index).values,
                        max_by_tract.values, rtol=0, atol=1e-6)),
        "leading_zeros_preserved": {
            "tract_ids_with_leading_zero": int(
                primary["GEOID"].str.startswith("0").sum()),
            "ua_ids_with_leading_zero": int(
                ua["UACE20"].str.startswith("0").sum()),
            "tract_id_widths": sorted(set(primary["GEOID"].str.len().tolist())),
            "ua_id_widths": sorted(set(ua["UACE20"].str.len().tolist())),
        },
        "small_overlap_warnings": {
            "note": ("Pairs below the minimum overlap fraction are retained "
                     "here for QA but are not associations: they do not make a "
                     "tract eligible and cannot be primary.  They are mostly "
                     "boundary-vintage slivers between the 2020 tract file and "
                     "the TIGER 2025 Urban Area release."),
            "n_tracts_excluded_by_the_threshold": int(
                (excluded["reason"] == "max_overlap_below_min_fraction").sum()),
            "primary_overlap_lt_2pct_of_tract": int(
                (primary["primary_overlap_fraction_of_tract"] < 0.02).sum()),
            "primary_overlap_lt_1000_m2": int(
                (primary["primary_overlap_area_m2"] < 1000).sum()),
            "min_primary_overlap_m2": float(primary["primary_overlap_area_m2"].min()),
            "min_primary_overlap_fraction": float(
                primary["primary_overlap_fraction_of_tract"].min()),
        },
        "ring_overlap_proof": proof,
        "rings": {
            "n_empty": int(ring_qa["is_empty"].sum()),
            "n_fragmented": int(ring_qa["fragmented_flag"].sum()),
            "n_extending_outside_tract_footprint": int(
                ring_qa["extends_outside_us_tract_footprint"].sum()),
            "n_invalid": int((~ring_qa["ring_valid"]).sum()),
            "n_buffer_distance_out_of_range": int((~ring_qa["buffer_distance_ok"]).sum()),
            "buffer_distance_tolerance_m": common.BUFFER_DISTANCE_TOL_M,
            "realized_buffer_distance_min_m": float(
                ring_qa["buffer_vertex_distance_min_m"].min()),
            "realized_buffer_distance_max_m": float(
                ring_qa["buffer_vertex_distance_max_m"].max()),
            "buffer_distance_note": (
                "GEOS simplifies a buffer's input by 0.01 * distance, so the "
                "realized outward distance sits a few metres inside 10,000 m at "
                "some vertices.  The inner ring boundary is the exact UA "
                "boundary, because the subtraction uses the original vertices."),
            "ring_area_km2_describe": ring_qa["ring_area_km2"].describe().to_dict(),
            "retained_fraction_describe": ring_qa["retained_fraction"].describe().to_dict(),
            "max_aeqd_scale_error_bound": float(ring_qa["aeqd_scale_error_bound"].max()),
        },
    }

    # --- write -------------------------------------------------------------
    ua.to_file(ua_path, driver="GPKG", layer="ua")
    rings.to_file(ring_path, driver="GPKG", layer="ua_rural_ring_10km")
    for obj, name in ((ring_qa, "ring_qa.parquet"),
                      (cross, "tract_ua_crosswalk.parquet"),
                      (primary, "tract_primary_ua.parquet"),
                      (excluded, "excluded_tracts.parquet")):
        p = out / name
        with common.atomic_path(p) as tmp:
            obj.to_parquet(tmp, index=False)
        log.info("wrote %s (%d rows)", p, len(obj))
    common.write_json_atomic(out / "geometry_qa.json", qa)

    hard_fail = []
    if not proof["passed"]:
        hard_fail.append(f"{proof['n_rings_over_tolerance']} rings overlap a Census UA")
    for k in ("exactly_one_primary_per_tract", "primary_is_in_candidate_list",
              "no_duplicate_pairs", "primary_equals_max_overlap"):
        if not qa[k]:
            hard_fail.append(f"crosswalk check failed: {k}")
    if qa["n_invalid"] if "n_invalid" in qa else False:
        hard_fail.append("invalid rings")
    log.info("geometry QA written to %s", out / "geometry_qa.json")
    if hard_fail:
        for f in hard_fail:
            log.error("FAILED: %s", f)
        raise SystemExit("geometry gates failed; see geometry_qa.json")
    # Sidecars only after the gates pass, so a failed run cannot be mistaken
    # for a complete one by the next invocation.
    for p in (out / "ring_qa.parquet", out / "tract_ua_crosswalk.parquet",
              out / "tract_primary_ua.parquet", out / "excluded_tracts.parquet",
              ua_path, ring_path):
        common.write_sidecar(p, inputs, params)
    log.info("geometry complete: %d eligible tracts, %d excluded, %d rings",
             qa["n_eligible_tracts"], qa["n_excluded_tracts"], len(rings))


if __name__ == "__main__":
    main()
