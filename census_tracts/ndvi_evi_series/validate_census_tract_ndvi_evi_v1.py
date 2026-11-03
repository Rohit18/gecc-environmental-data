#!/usr/bin/env python3
"""Validate the version-1 Census Tract NDVI/EVI exports downloaded from Drive.

Usage:
    python validate_census_tract_ndvi_evi_v1.py ~/Downloads/ndvi_evi_zips/ [more paths]

Paths may be CSVs, Drive zips, or folders containing either. The per-batch
pixel-total exports define each batch's tract list (85,528 tracts in all). Alaska
(batch 14) is exported in size-tier tract groups on EPSG:3338 (_30m_pNN for tracts
up to 5,000 km2, _100m_pNN above): its group files are combined per season-year,
and only its tiered EPSG:3338 group pixel totals are used.

Checks per NDVI/EVI file: presence of all 41 years x 2 seasons x 19 batches,
duplicate copies, column schema, year/season/batch/GLC-year labels, exact tract
list, value ranges, EVI_count <= NDVI_count, clear_obs_mean >= 1 where valid, and
observed fraction (NDVI_count / pixel_nonwater_count_<glc year>) <= 1.

Writes outputs/census_tract_ndvi_evi_v1/validation/{report.md, issues.csv,
season_year_summary.csv} and exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

import submit_census_tract_ndvi_evi_exports as wf

OUT = Path(__file__).resolve().parent / "outputs/census_tract_ndvi_evi_v1/validation"
DATA_RE = re.compile(rf"^{wf.TASK_PREFIX}(JJA|DJFsameYear)_(\d{{4}})_b(\d{{3}})(?:_(\d+m_p\d{{2}}))?$")
TOTALS_RE = re.compile(rf"^{wf.TASK_PREFIX}pixel_totals_(?:30m_)?b(\d{{3}})(?:_(\d+m_p\d{{2}}))?(_epsg\d+)?$")
SEASON_LABEL = {"JJA": "JJA", "DJFsameYear": "DJF_same_year"}
FRACTION_TOLERANCE = 1e-9


def iter_csvs(paths: list[str]):
    """Yield (name stem, source label, bytes) for every CSV in the inputs."""
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


def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), dtype={wf.ID_FIELD: str, wf.STATE_FIELD: str})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    issues: list[dict] = []

    def issue(file: str, check: str, detail: str) -> None:
        issues.append({"file": file, "check": check, "detail": detail})

    data_files: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    group_totals: dict[int, dict[str, pd.DataFrame]] = defaultdict(dict)
    totals: dict[int, tuple[str, pd.DataFrame]] = {}
    for stem, source, raw in iter_csvs(args.paths):
        stem = re.sub(r" \(\d+\)$", "", stem)  # Drive duplicate copies "name (1)"
        if DATA_RE.match(stem):
            data_files[stem].append((source, raw))
        elif m := TOTALS_RE.match(stem):
            chunk, group, grid = int(m.group(1)), m.group(2), m.group(3)
            if chunk in wf.GROUPED_CHUNKS:
                if group and grid:
                    group_totals[chunk][group] = read_csv(raw)
                continue  # ungrouped/default-grid totals are superseded
            totals[chunk] = (stem, read_csv(raw))
    for chunk, parts in group_totals.items():
        combined = pd.concat(parts.values(), ignore_index=True)
        if combined[wf.ID_FIELD].duplicated().any():
            issue(f"pixel_totals b{chunk:03d}", "rows", "a tract appears in more than one group")
        totals[chunk] = (f"pixel_totals b{chunk:03d}_epsg groups {sorted(parts)}", combined)

    # --- tract universe from pixel totals
    for chunk in wf.REGION_STATEFPS:
        if chunk not in totals:
            issue(f"pixel_totals b{chunk:03d}", "missing", "no pixel-totals file")
        elif chunk in wf.CRS_BY_CHUNK and "_epsg" not in totals[chunk][0]:
            issue(totals[chunk][0], "grid", f"default-grid totals used; expected {wf.CRS_BY_CHUNK[chunk]} file")
    universe = sum(len(df) for _, df in totals.values())
    if len(totals) == len(wf.REGION_STATEFPS) and universe != wf.EXPECTED_TRACT_COUNT:
        issue("pixel_totals", "universe", f"{universe} tracts, expected {wf.EXPECTED_TRACT_COUNT}")

    expected_cols = wf.output_selectors()
    summary = []
    for year in wf.EXPECTED_YEARS:
        for season in ("JJA", "DJFsameYear"):
            for chunk in wf.REGION_STATEFPS:
                stem = f"{wf.TASK_PREFIX}{season}_{year}_b{chunk:03d}"
                if chunk in wf.GROUPED_CHUNKS:
                    stems = [f"{stem}_{key}" for key in sorted(group_totals.get(chunk, {}))]
                else:
                    stems = [stem]
                parts, missing = [], []
                for part in stems:
                    copies = data_files.get(part, [])
                    if not copies:
                        missing.append(part)
                        continue
                    if len({hashlib.sha256(raw).hexdigest() for _, raw in copies}) > 1:
                        issue(part, "duplicate", f"{len(copies)} differing copies: {[s for s, _ in copies]}")
                    parts.append(read_csv(copies[0][1]))
                if not stems or missing:
                    issue(stem, "missing", "file not found" if not stems or len(stems) == 1 else f"missing groups: {[m[len(stem) + 1:] for m in missing]}")
                    if not parts:
                        continue
                df = pd.concat(parts, ignore_index=True)
                if list(df.columns) != expected_cols:
                    issue(stem, "schema", f"columns {list(df.columns)}")
                    continue
                labels = {
                    "year": {year},
                    "season": {SEASON_LABEL[season]},
                    "batch": {chunk},
                    "batch_label": {wf.REGION_LABELS[chunk]},
                    "glc_water_mask_year": {wf.glc_map_year(year)},
                }
                for col, want in labels.items():
                    got = set(df[col].dropna().unique())
                    if got != want:
                        issue(stem, "label", f"{col}={sorted(got)}, expected {sorted(want)}")
                if df[wf.ID_FIELD].duplicated().any():
                    issue(stem, "rows", f"{df[wf.ID_FIELD].duplicated().sum()} duplicate GEOIDs")
                bad_id = ~df[wf.ID_FIELD].str.fullmatch(r"\d{11}") | (df[wf.ID_FIELD].str[:2] != df[wf.STATE_FIELD])
                if bad_id.any():
                    issue(stem, "geoid", f"{bad_id.sum()} malformed GEOID/StFIPS")
                if chunk in totals:
                    ref = set(totals[chunk][1][wf.ID_FIELD])
                    got = set(df[wf.ID_FIELD])
                    if got != ref:
                        issue(stem, "rows", f"{len(ref - got)} tracts missing, {len(got - ref)} unexpected")
                valid = df["NDVI_count"].fillna(0) > 0
                # EVI can be undefined where NDVI is not (EVI's denominator reaches
                # zero on some bright, blue-dominated winter pixels), so EVI_count
                # may be lower than NDVI_count but never higher.
                ndvi_n, evi_n = df["NDVI_count"].fillna(0), df["EVI_count"].fillna(0)
                if (evi_n > ndvi_n).any():
                    issue(stem, "values", "EVI_count > NDVI_count")
                evi_short = int((evi_n < ndvi_n).sum())
                for col, has in (("NDVI_mean", valid), ("NDVI_median", valid),
                                 ("EVI_mean", evi_n > 0), ("EVI_median", evi_n > 0)):
                    if df.loc[has, col].isna().any() or df.loc[~has, col].notna().any():
                        issue(stem, "values", f"{col} missing where its count>0 or present where count=0")
                    if ((df[col] < -1) | (df[col] > 1)).any():
                        issue(stem, "values", f"{col} outside [-1, 1]")
                if (df.loc[valid, "clear_obs_mean"] < 1 - 1e-9).any():
                    issue(stem, "values", "clear_obs_mean < 1 where count > 0")
                frac = pd.Series(dtype=float)
                if chunk in totals:
                    denom_col = f"pixel_nonwater_count_{wf.glc_map_year(year)}"
                    t = totals[chunk][1][[wf.ID_FIELD, denom_col]]
                    m = df.merge(t, on=wf.ID_FIELD, how="left")
                    frac = (m["NDVI_count"].fillna(0) / m[denom_col]).where(m[denom_col] > 0)
                    over = frac > 1 + FRACTION_TOLERANCE
                    if over.any():
                        issue(stem, "fraction", f"{over.sum()} tracts with observed fraction > 1 (max {frac.max():.4f})")
                summary.append({
                    "year": year, "season": SEASON_LABEL[season], "batch": chunk,
                    "tracts": len(df), "with_values": int(valid.sum()),
                    "ndvi_mean_median": df.loc[valid, "NDVI_mean"].median(),
                    "evi_mean_median": df.loc[valid, "EVI_mean"].median(),
                    "clear_obs_median": df.loc[valid, "clear_obs_mean"].median(),
                    "single_scene_tracts": int((df["clear_obs_mean"] < 1.5).sum()),
                    "evi_count_below_ndvi": evi_short,
                    "fraction_ge_0_5": int((frac >= 0.5).sum()),
                })

    summary_df = pd.DataFrame(summary)
    issues_df = pd.DataFrame(issues, columns=["file", "check", "detail"])
    issues_df.to_csv(OUT / "issues.csv", index=False)
    by_sy = pd.DataFrame()
    if not summary_df.empty:
        by_sy = summary_df.groupby(["year", "season"]).agg(
            batches=("batch", "nunique"), tracts=("tracts", "sum"),
            with_values=("with_values", "sum"), fraction_ge_0_5=("fraction_ge_0_5", "sum"),
            single_scene_tracts=("single_scene_tracts", "sum"),
            evi_count_below_ndvi=("evi_count_below_ndvi", "sum"),
            ndvi_mean_median=("ndvi_mean_median", "median"),
            evi_mean_median=("evi_mean_median", "median"),
            clear_obs_median=("clear_obs_median", "median"),
        ).reset_index()
        by_sy.to_csv(OUT / "season_year_summary.csv", index=False)

    found = len(summary_df)
    expected = len(wf.EXPECTED_YEARS) * 2 * len(wf.REGION_STATEFPS)
    counts = issues_df["check"].value_counts().to_dict() if not issues_df.empty else {}
    lines = [
        "# Census Tract NDVI/EVI v1: validation report", "",
        f"- NDVI/EVI files checked: {found} of {expected} expected",
        f"- Pixel-totals batches found: {len(totals)} of {len(wf.REGION_STATEFPS)} ({universe:,} tracts)",
        f"- Issues: {len(issues_df)} " + (str(counts) if counts else "(none)"),
        "", "Per season-year summary: `season_year_summary.csv`; issue list: `issues.csv`.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not issues_df.empty:
        print("\nFirst issues:")
        print(issues_df.groupby("check").head(3).to_string(index=False))
    return 1 if len(issues_df) else 0


if __name__ == "__main__":
    sys.exit(main())
