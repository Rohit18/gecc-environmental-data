#!/usr/bin/env python3
"""Validate and merge non-CONUS WorldCover exports when all tracts are ready."""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import submit_census_tract_worldcover_exports as prior


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/worldcover_2020_nonconus"
RAW = OUT / "raw_exports"
CONUS = ROOT / "GECC_CensusTracts_release_v2/worldcover_urban_rural"
AREAS = re.compile(r"WC2020_NONCONUS_tract_v(?:2_areas_s(?:15|60|66|69|72|78)_n25|[35]_areas_s02_n25_a20000)_b\d{3}\.csv$")
CROSS = re.compile(r"WC2020_NONCONUS_tract_v(?:1_crosswalk_s72_n25_b000|2_crosswalk_s\d{2}_n25(?:_a20000)?_b\d{3})\.csv$")
TILE = re.compile(r"WC2020_NONCONUS_tract_v4_areas_ak_aleutian_02016000100_tile_t(\d{2})\.csv$")
# Codes are text. Without ua_ids here, a batch whose tracts all have one UA
# (or none) parses as float and writes "2332.0" for UA 02332.
AREA_DTYPE = {"GEOID": str, "STATEFP": str, "primary_ua_id": str, "ua_ids": str}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_aleutian_tiles():
    plan_path = OUT / "ak_aleutian_tile_plan.json"
    if not plan_path.exists():
        return None, []
    expected = {item["index"] for item in json.loads(plan_path.read_text())["tiles"]}
    files = {int(match[1]): path for path in RAW.glob("*.csv")
             if (match := TILE.fullmatch(path.name))}
    missing = sorted(expected - set(files))
    if missing:
        return None, missing
    if set(files) != expected:
        raise RuntimeError(f"Unexpected Aleutian tile exports: {sorted(set(files)-expected)}")
    rows = []
    for index in sorted(expected):
        frame = pd.read_csv(files[index], dtype=AREA_DTYPE)
        if len(frame) != 1 or int(frame.repair_tile_index.iloc[0]) != index:
            raise ValueError(f"Invalid repair tile CSV: {files[index].name}")
        rows.append(frame.iloc[0])
    result = rows[0].drop("repair_tile_index").copy()
    for row in rows:
        if row.GEOID != "02016000100" or row.STATEFP != "02":
            raise ValueError("Repair tile has wrong tract")
        for field in prior.META[2:]:
            a, b = result[field], row[field]
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.01):
                    raise ValueError(f"Repair tile metadata differs: {field}")
            elif a != b:
                raise ValueError(f"Repair tile metadata differs: {field}")
    for zone in ("urban", "rural", "tract"):
        zone_area = float(result["tract_geometry_area_m2" if zone == "tract"
                                 else f"{zone}_zone_area_m2"])
        raster = sum(float(row[f"{zone}_raster_area_m2"] or 0) for row in rows)
        valid = sum(float(row[f"{zone}_valid_area_m2"] or 0) for row in rows)
        result[f"{zone}_raster_area_m2"] = raster
        result[f"{zone}_valid_area_m2"] = valid
        has_valid = zone_area > 0 and valid > 0
        result[f"{zone}_has_valid_worldcover"] = has_valid
        result[f"{zone}_fraction_denominator_m2"] = valid if has_valid else np.nan
        result[f"{zone}_coverage_fraction"] = valid / raster if zone_area > 0 and raster > 0 else np.nan
        class_areas = {}
        for code in prior.CODES:
            field = f"{zone}_wc{code}_area_m2"
            area = sum(float(row[field]) for row in rows if pd.notna(row[field]))
            class_areas[code] = area
            result[field] = area if has_valid or zone_area == 0 else np.nan
        for group, codes in prior.GROUPS.items():
            area = sum(class_areas[code] for code in codes)
            result[f"{zone}_{group}_area_m2"] = area if has_valid or zone_area == 0 else np.nan
            if group == "vegetation":
                result[f"{zone}_vegetation_fraction"] = area / valid if has_valid else np.nan
    path = OUT / "ak_aleutian_02016000100_repaired_area.csv"
    pd.DataFrame([result])[prior.selectors()].to_csv(path, index=False)
    return path, []


def validate_areas(data):
    assert list(data.columns) == prior.selectors()
    assert data.GEOID.str.len().eq(11).all()
    assert data.GEOID.is_unique
    assert data.GEOID.str[:2].eq(data.STATEFP).all()
    assert data.tract_boundary_year.eq(2020).all()
    assert data.worldcover_year.eq(2020).all()
    assert data.tract_geometry_area_m2.gt(0).all()
    assert np.allclose(data.urban_zone_area_m2 + data.rural_zone_area_m2,
                       data.tract_geometry_area_m2, rtol=1e-8, atol=0.01)
    for zone in ("urban", "rural", "tract"):
        code_fields = [f"{zone}_wc{code}_area_m2" for code in prior.CODES]
        valid = data[f"{zone}_valid_area_m2"]
        assert valid.ge(0).all()
        assert np.allclose(data[code_fields].fillna(0).sum(axis=1), valid,
                           rtol=1e-8, atol=0.05)
        for group, codes in prior.GROUPS.items():
            group_area = data[f"{zone}_{group}_area_m2"]
            expected = data[[f"{zone}_wc{code}_area_m2" for code in codes]].fillna(0).sum(axis=1)
            assert np.allclose(group_area.fillna(0), expected, rtol=1e-8, atol=0.05)
        positive = valid.gt(0)
        assert np.allclose(data.loc[positive, f"{zone}_vegetation_area_m2"] / valid[positive],
                           data.loc[positive, f"{zone}_vegetation_fraction"],
                           rtol=1e-8, atol=1e-8)
    for code in prior.CODES:
        urban, rural, tract = (data[f"{zone}_wc{code}_area_m2"].fillna(0)
                               for zone in ("urban", "rural", "tract"))
        assert np.allclose(urban + rural, tract, rtol=1e-8, atol=0.05)


def validate_ua_ids(data, cross):
    """ua_ids must be each tract's sorted crosswalk UA codes, pipe-delimited."""
    expected = cross.sort_values("UACE20").groupby("GEOID").UACE20.agg("|".join)
    actual = data.set_index("GEOID").ua_ids.fillna("")
    mismatched = actual.ne(expected.reindex(actual.index).fillna(""))
    if mismatched.any():
        raise ValueError(f"ua_ids differ from the crosswalk for {int(mismatched.sum())} "
                         f"tracts, e.g. {actual[mismatched].head(3).to_dict()}")
    codes = actual[actual.ne("")].str.split("|").explode()
    if not codes.str.fullmatch(r"\d{5}").all():
        raise ValueError("ua_ids contains codes that are not five-digit text")


def build_national(nonconus_areas, nonconus_crosswalk):
    conus_areas_path = CONUS / "CensusTract_WorldCover2020_urban_rural.csv"
    conus_crosswalk_path = CONUS / "CensusTract_UA2020_crosswalk_all.csv"
    conus = pd.read_csv(conus_areas_path, dtype=AREA_DTYPE)
    if len(conus) != 83776 or set(conus.GEOID) & set(nonconus_areas.GEOID):
        raise ValueError("CONUS source count or partition differs from expected")
    national = pd.concat([conus, nonconus_areas], ignore_index=True)
    if len(national) != 85528 or national.GEOID.nunique() != 85528:
        raise ValueError("National WorldCover tract count differs from 2020 universe")
    validate_areas(national)
    conus_cross = pd.read_csv(conus_crosswalk_path,
                              dtype={"GEOID": str, "UACE20": str})
    national_cross = pd.concat([conus_cross, nonconus_crosswalk],
                                ignore_index=True)
    if national_cross.duplicated(["GEOID", "UACE20"]).any():
        raise ValueError("Duplicate national tract–urban area pair")
    counts = national_cross.groupby("GEOID").size()
    if not national.ua_count.eq(national.GEOID.map(counts).fillna(0).astype(int)).all():
        raise ValueError("National crosswalk counts differ from area table")
    validate_ua_ids(national, national_cross)
    areas_path = OUT / "CensusTract_WorldCover2020_allUS_areas.csv"
    cross_path = OUT / "CensusTract_WorldCover2020_allUS_UA_crosswalk.csv"
    national.sort_values("GEOID").to_csv(areas_path, index=False)
    national_cross.sort_values(["GEOID", "UACE20"]).to_csv(cross_path, index=False)
    return {"tracts": len(national), "crosswalk_pairs": len(national_cross),
            "conus_source": str(conus_areas_path.relative_to(ROOT)),
            "outputs": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                         "sha256": sha256(path)} for path in (areas_path, cross_path)]}


def main():
    expected = set(json.loads((OUT / "expected_geoids.json").read_text())["geoids"])
    area_files = sorted(path for path in RAW.glob("*.csv") if AREAS.fullmatch(path.name))
    frames = []
    for path in area_files:
        frame = pd.read_csv(path, dtype=AREA_DTYPE)
        if list(frame.columns) != prior.selectors():
            raise ValueError(f"Unexpected WorldCover columns: {path.name}")
        frames.append(frame)
    repaired, tile_missing = merge_aleutian_tiles()
    if repaired:
        frames.append(pd.read_csv(repaired, dtype=AREA_DTYPE))
    data = pd.concat(frames, ignore_index=True)
    validate_areas(data)
    if not set(data.GEOID) <= expected:
        raise ValueError("Export contains GEOIDs outside non-CONUS target")
    missing = sorted(expected - set(data.GEOID))
    coverage = {state: {"expected": sum(x.startswith(state) for x in expected),
                        "exported": int(data.STATEFP.eq(state).sum()),
                        "missing": sum(x.startswith(state) for x in missing)}
                for state in ("02", "15", "60", "66", "69", "72", "78")}
    progress = {"area_files": len(area_files), "area_rows": len(data),
                "tile_repair_missing_indices": tile_missing,
                "coverage": coverage, "missing_geoids": missing}
    (OUT / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    print(json.dumps({key: value for key, value in progress.items()
                      if key != "missing_geoids"}, indent=2))
    if missing:
        return

    cross_files = sorted(path for path in RAW.glob("*.csv") if CROSS.fullmatch(path.name))
    cross = pd.concat([pd.read_csv(path, dtype={"GEOID": str, "UACE20": str})
                       for path in cross_files], ignore_index=True)
    if list(cross.columns) != prior.PAIR_FIELDS:
        raise ValueError("Unexpected crosswalk columns")
    if cross.duplicated(["GEOID", "UACE20"]).any():
        raise ValueError("Duplicate tract–UA pair")
    if not set(cross.GEOID) <= expected:
        raise ValueError("Crosswalk contains GEOIDs outside non-CONUS target")
    if not cross.is_primary.isin([0, 1]).all():
        raise ValueError("Crosswalk has invalid primary flags")
    for field in ("overlap_area_m2", "overlap_fraction_of_tract",
                  "overlap_fraction_of_ua"):
        if cross[field].isna().any() or cross[field].lt(0).any():
            raise ValueError(f"Crosswalk has invalid {field}")
    for field in ("overlap_fraction_of_tract", "overlap_fraction_of_ua"):
        if cross[field].gt(1 + 1e-9).any():
            raise ValueError(f"Crosswalk has {field} above 1")
    counts = cross.groupby("GEOID").size()
    if not data.ua_count.eq(data.GEOID.map(counts).fillna(0).astype(int)).all():
        raise ValueError("Crosswalk count differs from area table")
    validate_ua_ids(data, cross)
    primary = cross[cross.is_primary.eq(1)].set_index("GEOID")
    if primary.index.has_duplicates or set(primary.index) != set(data.loc[data.ua_count.gt(0), "GEOID"]):
        raise ValueError("Primary urban area is missing or duplicated")
    area_primary = data.set_index("GEOID").loc[primary.index]
    if not area_primary.primary_ua_id.eq(primary.UACE20).all():
        raise ValueError("Primary urban area ID differs from area table")
    if not np.allclose(area_primary.primary_ua_overlap_area_m2,
                       primary.overlap_area_m2, rtol=1e-8, atol=0.01):
        raise ValueError("Primary urban area overlap differs from area table")
    areas_path = OUT / "CensusTract_WorldCover2020_nonCONUS_areas.csv"
    cross_path = OUT / "CensusTract_WorldCover2020_nonCONUS_UA_crosswalk.csv"
    data.sort_values("GEOID").to_csv(areas_path, index=False)
    cross.sort_values(["GEOID", "UACE20"]).to_csv(cross_path, index=False)
    report = {"status": "PASS", "tracts": len(data), "crosswalk_pairs": len(cross),
              "coverage": coverage, "area_files": len(area_files),
              "crosswalk_files": len(cross_files),
              "tracts_with_valid_pixels": int(data.tract_valid_area_m2.gt(0).sum()),
              "tracts_without_valid_pixels": data.loc[
                  data.tract_valid_area_m2.eq(0), "GEOID"].tolist(),
              "minimum_tract_coverage_fraction": float(
                  data.tract_coverage_fraction.min()),
              "outputs": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                           "sha256": sha256(path)} for path in (areas_path, cross_path)]}
    report["national"] = build_national(data, cross)
    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
