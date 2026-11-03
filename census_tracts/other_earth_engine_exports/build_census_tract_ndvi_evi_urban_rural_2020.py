#!/usr/bin/env python3
"""Validate and package the 2020 Census Tract urban/rural NDVI and EVI exports.

Reads the 44 v2 Earth Engine CSVs (Drive zips, CSVs, or folders of either),
checks them, joins Urban Area metadata, and writes one row per tract and season
with urban - rural differences. Exits 1 without writing the table if any check
fails.

Usage:
    python build_census_tract_ndvi_evi_urban_rural_2020.py ~/Downloads/<zip> [...]
        [--v1 <v1 2020 NDVI/EVI exports: zips, CSVs or folders>]

Checks: all 44 files present once; schema and constants; 85,528 unique GEOIDs
per season matching the WorldCover tract universe; urban + rural pixels =
pixels centred in the tract; zone counts within tract counts; index ranges and
null patterns; tract pixel totals equal v1's pixel-total exports (repo zip;
Alaska's tiered totals are not in it and are skipped); and, with --v1, whole-
tract stats equal the v1 2020 release. Derived columns are described in
census_tract_ndvi_evi_2020_urban_rural_plan.md.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import submit_census_tract_ndvi_evi_urban_rural_2020 as ur

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CensusTracts/ndvi_evi_urban_rural_2020"
TABLE = OUT / "CensusTract_NDVI_EVI_2020_urban_rural.csv"
REPORT = OUT / "CensusTract_NDVI_EVI_2020_urban_rural_validation.json"
WORLDCOVER = ROOT / "GECC_CensusTracts_release_allUS_20260930/worldcover_urban_rural"
V1_PIXEL_TOTALS_ZIP = ROOT / "GECC_CensusTract_NDVI_EVI_v1-20261009T211714Z-1-001.zip"
SEASONS = {"JJA": "JJA", "DJFsameYear": "DJF_same_year"}
EXPORT_RE = re.compile(
    rf"^{ur.TASK_PREFIX}(JJA|DJFsameYear)_2020_b(\d{{3}})(?:_(\d+m_g\d{{2}}))?$"
)
V1_DATA_RE = re.compile(
    r"^CensusTract_NDVI_EVI_v1_(JJA|DJFsameYear)_2020_b(\d{3})(?:_(\d+m_g\d{2}))?$"
)
ALASKA_GROUPS = ("30m_g01", "100m_g01", "100m_g02", "100m_g03")
# A difference resting on fewer valid pixels than this in either zone is flagged.
MIN_ZONE_VALID_PIXELS = 9
# The per-tract reduction can count one more or one fewer straddling edge
# pixel than v1's batch reduceRegions (4 of 85,351 tracts in the 2020 run:
# 29149480100, 48267950100, 53027000300, 72013300101). Such tracts are listed
# in the report; their stats are then compared with no tolerance assumption.
EDGE_PIXEL_TOLERANCE = 1
MEAN_TOLERANCE = 1e-9
MEDIAN_TOLERANCE = 2e-3
DIFF_STATS = ("mean", "median")


def expected_exports() -> set[str]:
    names = set()
    for season in SEASONS:
        for chunk in ur.vi.REGION_STATEFPS:
            for group in (ALASKA_GROUPS if chunk in ur.vi.GROUPED_CHUNKS else (None,)):
                suffix = f"_{group}" if group else ""
                names.add(f"{ur.TASK_PREFIX}{season}_2020_b{chunk:03d}{suffix}")
    return names


def csv_sources(paths: list[str]):
    """(stem, label, bytes) for every CSV in the given CSVs, zips, or folders."""
    for raw in paths:
        path = Path(raw).expanduser()
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for f in files:
            if f.suffix == ".csv":
                yield f.stem, str(f), f.read_bytes()
            elif f.suffix == ".zip":
                with zipfile.ZipFile(f) as z:
                    for name in z.namelist():
                        if name.endswith(".csv"):
                            yield Path(name).stem, f"{f.name}:{name}", z.read(name)


def load_exports(paths: list[str], problems: list[str]) -> pd.DataFrame:
    found: dict[str, tuple[str, bytes]] = {}
    for stem, label, content in csv_sources(paths):
        if not EXPORT_RE.match(stem):
            continue
        if stem in found and found[stem][1] != content:
            problems.append(f"{stem}: two different copies ({found[stem][0]}, {label})")
        found.setdefault(stem, (label, content))
    missing = sorted(expected_exports() - set(found))
    if missing:
        problems.append(f"{len(missing)} of 44 exports missing, e.g. {missing[:3]}")
    frames = []
    for stem, (label, content) in sorted(found.items()):
        frame = pd.read_csv(io.BytesIO(content), dtype={"GEOID": str, "StFIPS": str,
                                                        "tract_group": str})
        if list(frame.columns) != ur.output_selectors():
            problems.append(f"{stem}: unexpected columns")
            continue
        season, chunk, group = EXPORT_RE.match(stem).groups()
        bad = {
            "season": not frame.season.eq(SEASONS[season]).all(),
            "batch": not frame.batch.eq(int(chunk)).all(),
            "tract_group": not frame.tract_group.fillna("").eq(group or "").all(),
            "year": not frame.year.eq(2020).all(),
            "glc_water_mask_year": not frame.glc_water_mask_year.eq(2020).all(),
            "reduction_crs": not frame.reduction_crs.eq(ur.reduction_crs(int(chunk))).all(),
        }
        problems += [f"{stem}: {field} differs from its file name" for field, b in bad.items() if b]
        frames.append(frame)
    if not frames:
        raise SystemExit("No v2 urban/rural exports found in the given paths")
    return pd.concat(frames, ignore_index=True)


def check_exports(df: pd.DataFrame, universe: set[str], problems: list[str]) -> dict:
    """Structural and numeric checks on the merged GEE exports."""
    summary = {}
    for season, part in df.groupby("season"):
        ok = part.GEOID.is_unique and set(part.GEOID) == universe
        if not ok:
            problems.append(f"{season}: {len(part)} rows, {part.GEOID.nunique()} unique; "
                            f"{len(universe - set(part.GEOID))} universe GEOIDs missing")
        summary[season] = len(part)
    if not df.GEOID.str.fullmatch(r"\d{11}").all() or not df.StFIPS.eq(df.GEOID.str[:2]).all():
        problems.append("GEOID format or StFIPS prefix")

    count = lambda col: df[col].fillna(0)  # noqa: E731 - zero where a zone has no pixels
    if (count("urban_pixel_count") + count("rural_pixel_count") != count("tract_centre_pixel_count")).any():
        problems.append("urban + rural pixels != tract_centre_pixel_count")
    if (count("tract_centre_pixel_count") > count("tract_pixel_count")).any():
        problems.append("tract_centre_pixel_count > tract_pixel_count")
    for band in ("NDVI_count", "EVI_count", "pixel_nonwater_count"):
        if (count(f"urban_{band}") + count(f"rural_{band}") > count(f"tract_{band}")).any():
            problems.append(f"urban + rural {band} > tract {band}")
    for zone in ur.ZONES:
        if (count(f"{zone}_NDVI_count") > count(f"{zone}_pixel_nonwater_count")).any():
            problems.append(f"{zone}: NDVI_count > pixel_nonwater_count")
        if (count(f"{zone}_pixel_nonwater_count") > count(f"{zone}_pixel_count")).any():
            problems.append(f"{zone}: pixel_nonwater_count > pixel_count")
        for index in ur.INDICES:
            n = count(f"{zone}_{index}_count")
            for stat in ("mean", "median", "stdDev"):
                col = f"{zone}_{index}_{stat}"
                if (df[col].isna() != n.eq(0)).any():
                    problems.append(f"{col}: null where count > 0, or value where count = 0")
            values = df[[f"{zone}_{index}_mean", f"{zone}_{index}_median"]]
            if (values.lt(-1) | values.gt(1)).any().any():
                problems.append(f"{zone}_{index}: mean/median outside [-1, 1]")
            if df[f"{zone}_{index}_stdDev"].lt(0).any():
                problems.append(f"{zone}_{index}_stdDev negative")
        observed = count(f"{zone}_NDVI_count") > 0
        if (df.loc[observed, f"{zone}_clear_obs_mean"] < 1).any():
            problems.append(f"{zone}_clear_obs_mean < 1 where observed")
    summary["ndvi_count_ne_evi_count_rows"] = int((count("tract_NDVI_count") != count("tract_EVI_count")).sum())
    return summary


def check_v1_pixel_totals(df: pd.DataFrame, problems: list[str]) -> dict:
    """Tract pixel totals against v1's per-batch pixel-total exports."""
    if not V1_PIXEL_TOTALS_ZIP.exists():
        return {"status": "skipped: v1 zip not found"}
    totals = []
    with zipfile.ZipFile(V1_PIXEL_TOTALS_ZIP) as z:
        for name in z.namelist():
            if re.search(r"pixel_totals_30m_b\d{3}\.csv$", name):
                totals.append(pd.read_csv(z.open(name), dtype={"GEOID": str},
                                          usecols=["GEOID", "pixel_total_count",
                                                   "pixel_nonwater_count_2020"]))
    v1 = pd.concat(totals).set_index("GEOID")
    jja = df[df.season.eq("JJA") & df.GEOID.isin(v1.index)].set_index("GEOID")
    gaps = pd.concat([
        (jja.tract_pixel_count.fillna(0) - v1.loc[jja.index, "pixel_total_count"].fillna(0)),
        (jja.tract_pixel_nonwater_count.fillna(0)
         - v1.loc[jja.index, "pixel_nonwater_count_2020"].fillna(0)),
    ]).abs()
    over = gaps[gaps > EDGE_PIXEL_TOLERANCE]
    if len(over):
        problems.append(f"{len(over)} tract pixel totals differ from v1 by more than "
                        f"{EDGE_PIXEL_TOLERANCE} pixel, e.g. {over.head(3).to_dict()}")
    edge = sorted(set(gaps[(gaps > 0) & (gaps <= EDGE_PIXEL_TOLERANCE)].index))
    return {"tracts_compared": len(jja), "exact_matches": len(jja) - len(set(gaps[gaps > 0].index)),
            "one_edge_pixel_differences": edge, "larger_differences": len(over),
            "not_compared": "Alaska (v1 tiered EPSG:3338 totals not in the repo zip)"}


def check_v1_stats(df: pd.DataFrame, v1_paths: list[str] | None, problems: list[str]) -> dict:
    """Whole-tract stats against the released v1 2020 NDVI/EVI exports."""
    if not v1_paths:
        return {"status": "skipped: pass --v1 with the v1 2020 exports to compare"}
    frames = []
    for stem, _, content in csv_sources(v1_paths):
        if match := V1_DATA_RE.match(stem):
            frame = pd.read_csv(io.BytesIO(content), dtype={"GEOID": str})
            frame["season"] = SEASONS[match.group(1)]
            frames.append(frame)
    if not frames:
        problems.append("--v1 given but no v1 2020 NDVI/EVI exports found")
        return {"status": "no v1 files found"}
    v1 = pd.concat(frames).drop_duplicates(["GEOID", "season"]).set_index(["GEOID", "season"])
    ours = df.set_index(["GEOID", "season"])
    common = ours.index.intersection(v1.index)
    # Tract-seasons whose pixel count differs by one edge pixel cannot match
    # v1 exactly; list them and compare the rest strictly.
    count_gap = (ours.loc[common, "tract_NDVI_count"].fillna(0)
                 - v1.loc[common, "NDVI_count"].fillna(0)).abs()
    if (count_gap > EDGE_PIXEL_TOLERANCE).any():
        problems.append(f"{int((count_gap > EDGE_PIXEL_TOLERANCE).sum())} tract-seasons' "
                        "NDVI_count differs from v1 by more than one edge pixel")
    edge = count_gap[count_gap > 0].index
    result = {"tract_seasons_compared": len(common),
              "one_edge_pixel_exceptions": [f"{g} {s}" for g, s in edge]}
    common = common.difference(edge)
    for index in ur.INDICES:
        for stat, tolerance in (("mean", MEAN_TOLERANCE), ("stdDev", MEAN_TOLERANCE),
                                ("median", MEDIAN_TOLERANCE), ("count", 0)):
            a = ours.loc[common, f"tract_{index}_{stat}"]
            b = v1.loc[common, f"{index}_{stat}"]
            bad = int(((a - b).abs() > tolerance).sum() + (a.isna() != b.isna()).sum())
            result[f"{index}_{stat}_mismatches"] = bad
            if bad:
                problems.append(f"tract_{index}_{stat} differs from v1 in {bad} tract-seasons")
    return result


def ua_metadata(universe: set[str]) -> pd.DataFrame:
    """UA lists from the crosswalk; vector zone areas from the WorldCover table."""
    cross = pd.read_csv(WORLDCOVER / "CensusTract_UA2020_crosswalk_all.csv",
                        dtype={"GEOID": str, "UACE20": str})
    by_tract = cross.sort_values("UACE20").groupby("GEOID")
    meta = pd.DataFrame({
        "ua_ids": by_tract.UACE20.agg("|".join),
        "ua_count": by_tract.size(),
        "primary_ua_id": cross[cross.is_primary.eq(1)].set_index("GEOID").UACE20,
        "primary_ua_name": cross[cross.is_primary.eq(1)].set_index("GEOID").NAME20,
    })
    wc = pd.read_csv(WORLDCOVER / "CensusTract_WorldCover2020_urban_rural.csv",
                     usecols=["GEOID", "urban_zone_area_m2", "rural_zone_area_m2"],
                     dtype={"GEOID": str}).set_index("GEOID")
    meta = wc.join(meta)
    meta["ua_count"] = meta.ua_count.fillna(0).astype(int)
    meta["ua_ids"] = meta.ua_ids.fillna("")
    meta["ua_match_status"] = np.where(meta.ua_count > 0, "matched", "no_overlap")
    if set(meta.index) != universe:
        raise SystemExit("WorldCover tract universe differs from the expected 85,528")
    return meta


def zone_split(urban_valid: pd.Series, rural_valid: pd.Series) -> np.ndarray:
    return np.select([urban_valid & rural_valid, urban_valid, rural_valid],
                     ["both", "urban_only", "rural_only"], "no_valid_pixels")


def derive(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Final table: metadata, differences and flags, then the zone statistics."""
    out = df.rename(columns={"StFIPS": "STATEFP"}).join(meta, on="GEOID")
    urban_n, rural_n = out.urban_NDVI_count.fillna(0), out.rural_NDVI_count.fillna(0)
    out["zone_split"] = zone_split(urban_n > 0, rural_n > 0)
    out["small_zone_flag"] = out.zone_split.eq("both") & (
        np.minimum(urban_n, rural_n) < MIN_ZONE_VALID_PIXELS)
    for index in ur.INDICES:
        for stat in DIFF_STATS:
            # Null unless both zones have valid pixels (pandas propagates NaN).
            out[f"{index}_{stat}_diff"] = out[f"urban_{index}_{stat}"] - out[f"rural_{index}_{stat}"]
    for zone in ur.ZONES:
        nonwater = out[f"{zone}_pixel_nonwater_count"].fillna(0)
        out[f"{zone}_observed_fraction"] = (
            out[f"{zone}_NDVI_count"].fillna(0) / nonwater.where(nonwater > 0))
    leading = [
        "GEOID", "STATEFP", "year", "season", "ua_match_status", "ua_count", "ua_ids",
        "primary_ua_id", "primary_ua_name", "urban_zone_area_m2", "rural_zone_area_m2",
        "zone_split", "small_zone_flag",
        *[f"{index}_{stat}_diff" for index in ur.INDICES for stat in DIFF_STATS],
    ]
    zone_cols = []
    for zone in ur.ZONES:
        zone_cols += [f"{zone}_{index}_{stat}" for index in ur.INDICES for stat in ur.INDEX_STATS]
        zone_cols += [f"{zone}_clear_obs_mean", f"{zone}_observed_fraction",
                      f"{zone}_pixel_count", f"{zone}_pixel_nonwater_count"]
    trailing = ["tract_centre_pixel_count", "batch", "batch_label", "tract_group",
                "reduction_scale_m", "reduction_crs", "regional_scene_count",
                "glc_water_mask_year"]
    return out[leading + zone_cols + trailing].sort_values(["GEOID", "season"],
                                                         ascending=[True, False])


def summarize(out: pd.DataFrame, meta: pd.DataFrame) -> dict:
    summary = {}
    for season, part in out.groupby("season"):
        both = part[part.zone_split.eq("both")]
        vector = zone_split(part.urban_zone_area_m2 > 0, part.rural_zone_area_m2 > 0)
        summary[season] = {
            "zone_split": part.zone_split.value_counts().to_dict(),
            "small_zone_flag": int(part.small_zone_flag.sum()),
            "zone_split_vs_worldcover_vector": pd.crosstab(vector, part.zone_split.values)
            .to_dict(),
            **{f"{index}_mean_diff": {
                "median": float(both[f"{index}_mean_diff"].median()),
                "iqr": [float(both[f"{index}_mean_diff"].quantile(q)) for q in (.25, .75)],
                "share_negative": float((both[f"{index}_mean_diff"] < 0).mean()),
            } for index in ur.INDICES},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="v2 exports: Drive zips, CSVs, or folders")
    parser.add_argument("--v1", nargs="+", help="v1 2020 NDVI/EVI exports to compare tract stats")
    args = parser.parse_args()

    problems: list[str] = []
    df = load_exports(args.paths, problems)
    universe = set(pd.read_csv(WORLDCOVER / "CensusTract_WorldCover2020_urban_rural.csv",
                               usecols=["GEOID"], dtype={"GEOID": str}).GEOID)
    report = {
        "rows_by_season": check_exports(df, universe, problems),
        "v1_pixel_totals": check_v1_pixel_totals(df, problems),
        "v1_tract_stats": check_v1_stats(df, args.v1, problems),
    }
    report["problems"] = problems
    report["status"] = "FAIL" if problems else "PASS"
    OUT.mkdir(parents=True, exist_ok=True)
    if not problems:
        meta = ua_metadata(universe)
        out = derive(df, meta)
        out.to_csv(TABLE, index=False)
        report["table"] = {"path": str(TABLE.relative_to(ROOT)), "rows": len(out),
                           "columns": len(out.columns)}
        report["summary"] = summarize(out, meta)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "summary"}, indent=2, default=str))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
