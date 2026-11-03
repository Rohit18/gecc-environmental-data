#!/usr/bin/env python3
"""Assemble the validated NDVI/EVI v1 exports into a standalone release module.

No Earth Engine calls are made. Refuses missing, conflicting, or unexpected
production exports and never overwrites a completed package. A failed build
retains its .tmp directory for inspection. Use the source validator first.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from contextlib import ExitStack
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import submit_census_tract_ndvi_evi_exports as wf

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs/census_tract_ndvi_evi_v1"
PREFIX = "CensusTract_ndvi_evi_1985_2025"
GEO_FIELDS = ["GEOID", "StFIPS", "CoFIPS", "TrFIPS", "LAT_GeoCen", "LON_GeoCen", "Area_Land", "Area_Water", "Pop"]
STR_FIELDS = {"GEOID", "StFIPS", "CoFIPS", "TrFIPS", "season", "batch_label", "export_group", "analysis_crs", "sensor_era", "eligible_satellites"}
BOOL_FIELDS = {"water_only_tract", "low_coverage", "single_scene", "l7_slc_off_era", "oli_era"}
INT_FIELDS = {"year", "batch", "regional_scene_count", "glc_water_mask_year", "NDVI_count", "EVI_count", "Pop", "boundary_year", "analysis_scale_m", "alaska_scale_m", "pixel_total_count", "pixel_glc_gap_count", "pixel_nonwater_count"}
DATA_RE = re.compile(r"^CensusTract_NDVI_EVI_v1_(JJA|DJFsameYear)_(\d{4})_b(\d{3})(?:_(\d+m_p\d{2}))?$")
TOTAL_RE = re.compile(r"^CensusTract_NDVI_EVI_v1_pixel_totals_(?:30m_)?b(\d{3})(?:_(\d+m_p\d{2})_epsg3338)?$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(raw):
    return pd.read_csv(io.BytesIO(raw), dtype={"GEOID": str, "StFIPS": str}, float_precision="round_trip")


def geography(path):
    """Read the release's one-feature-per-line GeoJSON without loading geometry."""
    rows = []
    with path.open() as f:
        first = f.readline()
        require('"FeatureCollection"' in first and first.rstrip().endswith("["), "Expected release line-delimited FeatureCollection")
        for line in f:
            line = line.strip()
            if line in {"", "]}"}:
                continue
            feature = json.loads(line.removesuffix(","))
            require(feature.get("type") == "Feature", "Expected one GeoJSON Feature per line")
            rows.append({k: feature["properties"][k] for k in GEO_FIELDS})
    df = pd.DataFrame(rows)
    require(len(df) == 85528 and df.GEOID.nunique() == 85528, "Wrong shared-geography universe")
    require(df.GEOID.str.fullmatch(r"\d{11}").all(), "Malformed shared geography GEOID")
    require((df.GEOID == df.StFIPS + df.CoFIPS + df.TrFIPS).all(), "Malformed geography FIPS components")
    require(df.Area_Land.notna().all() and (df.Area_Land >= 0).all(), "Missing or negative land area")
    require((df.Area_Land == 0).sum() == 368, "Expected 368 water-only tracts")
    return df.sort_values("GEOID").reset_index(drop=True)


def expected_exports(tiers):
    groups = sorted(tiers.group.unique())
    require(groups == [f"100m_p{i:02d}" for i in range(1, 9)] + [f"30m_p{i:02d}" for i in range(1, 7)], "Wrong Alaska group list")
    expected = {}
    for batch in range(19):
        for group in (groups if batch == 14 else [""]):
            suffix = f"_{group}" if group else ""
            total = f"{wf.TASK_PREFIX}pixel_totals_b014_{group}_epsg3338" if group else f"{wf.TASK_PREFIX}pixel_totals_30m_b{batch:03d}"
            expected[total] = ("totals", None, None, batch, group)
            for year in range(1985, 2026):
                for season in ["DJFsameYear", "JJA"]:
                    expected[f"{wf.TASK_PREFIX}{season}_{year}_b{batch:03d}{suffix}"] = ("data", year, season, batch, group)
    return expected


def index_exports(archives, expected):
    found, ignored = {}, []
    for archive in archives:
        for member in archive.namelist():
            if not member.endswith(".csv"):
                continue
            stem = re.sub(r" \(\d+\)$", "", Path(member).stem)
            if stem not in expected:
                # A new pNN production group/year must not silently disappear.
                data_match = DATA_RE.fullmatch(stem)
                total_match = TOTAL_RE.fullmatch(stem)
                if (data_match and (int(data_match.group(3)) != 14 or data_match.group(4))) or (total_match and total_match.group(2)):
                    raise ValueError(f"Unexpected production export: {member}")
                ignored.append({"archive": Path(archive.filename).name, "member": member})
                continue
            raw = archive.read(member)  # zipfile verifies each member's CRC.
            digest = hashlib.sha256(raw).hexdigest()
            if stem in found:
                require(digest == found[stem][2], f"Conflicting duplicate export: {stem}")
                found[stem][3].append(f"{Path(archive.filename).name}:{member}")
            else:
                found[stem] = [archive, member, digest, [f"{Path(archive.filename).name}:{member}"]]
    missing = sorted(set(expected) - set(found))
    require(not missing, f"Missing {len(missing)} production exports: {missing[:5]}")
    return found, ignored


def load_export(found, stem):
    archive, member, _, _ = found[stem]
    return read_csv(archive.read(member))


def sensor_fields(year, season):
    """Catalogue eligibility, never an assertion of actual tract contribution."""
    ranges = [(date(year, 6, 1), date(year, 9, 1))] if season == "JJA" else [(date(year, 1, 1), date(year, 3, 1)), (date(year, 12, 1), date(year + 1, 1, 1))]
    periods = [("L4", date(1982, 8, 22), date(1993, 6, 25)), ("L5", date(1984, 3, 16), date(2012, 5, 6)), ("L7", date(1999, 5, 28), date(2017, 1, 1)), ("L8", date(2013, 5, 1), date(2026, 1, 1)), ("L9", date(2021, 10, 31), date(2026, 1, 1))]
    eligible = ";".join(sensor for sensor, start, stop in periods if any(a < stop and start < b for a, b in ranges))
    era = "pre_OLI" if year < 2013 else "ETM_OLI_overlap" if year <= 2016 else "OLI_only"
    return {"eligible_satellites": eligible, "sensor_era": era, "l7_slc_off_era": 2003 <= year <= 2016, "oli_era": year >= 2013}


def arrow_table(df):
    fields = []
    for col in df:
        dtype = pa.string() if col in STR_FIELDS else pa.bool_() if col in BOOL_FIELDS else pa.int64() if col in INT_FIELDS or col.startswith("pixel_nonwater_count_") else pa.float64()
        fields.append(pa.field(col, dtype))
    return pa.Table.from_pandas(df, schema=pa.schema(fields), preserve_index=False)


def write_pair(df, folder, stem):
    table = arrow_table(df)
    df.to_csv(folder / f"{stem}.csv", index=False)
    pq.write_table(table, folder / f"{stem}.parquet", compression="zstd")
    require(pq.read_table(folder / f"{stem}.parquet").equals(table), f"Parquet round-trip failed: {stem}")


def prepare_frame(raw, meta, year, season, batch, group):
    require(list(raw) == wf.output_selectors(), "Source schema differs from production selectors")
    require(raw.GEOID.is_unique, "Duplicate GEOID in export")
    expected = meta.loc[(meta.batch == batch) & (meta.export_group == group)]
    require(set(raw.GEOID) == set(expected.GEOID), f"Wrong tract set for {year}/{season}/b{batch}/{group}")
    for col, value in {"year": year, "season": season, "batch": batch, "batch_label": wf.REGION_LABELS[batch], "glc_water_mask_year": wf.glc_map_year(year)}.items():
        require(raw[col].eq(value).all(), f"Wrong source label: {col}")
    numeric = raw.select_dtypes(include="number")
    require(not np.isinf(numeric.to_numpy(dtype=float)).any(), "Infinite source value")
    for index in ["NDVI", "EVI"]:
        count = raw[f"{index}_count"].fillna(0)
        require((count >= 0).all() and (count == count.round()).all(), f"Invalid {index} counts")
        for stat in ["mean", "median"]:
            values = raw[f"{index}_{stat}"]
            require(values.notna().equals(count.gt(0)), f"{index} null/count mismatch")
            require(values.dropna().between(-1, 1).all(), f"{index} outside [-1,1]")
        require(raw[f"{index}_stdDev"].dropna().ge(0).all(), "Negative standard deviation")
    require(raw.EVI_count.fillna(0).le(raw.NDVI_count.fillna(0)).all(), "EVI_count > NDVI_count")
    require(raw.loc[raw.NDVI_count.fillna(0).gt(0), "clear_obs_mean"].ge(1 - 1e-9).all(), "Missing or low observation depth")
    denom = f"pixel_nonwater_count_{wf.glc_map_year(year)}"
    extra = [k for k in meta if k not in raw and (not k.startswith("pixel_nonwater_count_") or k == denom)]
    df = raw.merge(meta[["GEOID", *extra]], on="GEOID", validate="one_to_one").rename(columns={denom: "pixel_nonwater_count"})
    numerator = df.NDVI_count.fillna(0)
    require(not ((df.pixel_nonwater_count == 0) & (numerator > 0)).any(), "Data where nonwater support is zero")
    df["observed_fraction"] = (numerator / df.pixel_nonwater_count).where(df.pixel_nonwater_count > 0)
    require(df.observed_fraction.dropna().between(0, 1 + 1e-9).all(), "Observed fraction outside [0,1]")
    df["low_coverage"] = df.observed_fraction.lt(0.5).astype("boolean").mask(df.observed_fraction.isna())
    df["single_scene"] = df.clear_obs_mean.lt(1.5).astype("boolean").mask(df.clear_obs_mean.isna())
    for col, value in sensor_fields(year, season).items():
        df[col] = value
    df["boundary_year"] = 2020
    return df.sort_values("GEOID").reset_index(drop=True)


def coverage(df, scope):
    valid = df.NDVI_count.fillna(0).gt(0)
    return {"scope": scope, "year": int(df.year.iloc[0]), "season": df.season.iloc[0], "records": len(df), "with_ndvi": int(valid.sum()), "without_ndvi": int((~valid).sum()), "with_evi": int(df.EVI_count.fillna(0).gt(0).sum()), "evi_count_below_ndvi": int(df.EVI_count.fillna(0).lt(df.NDVI_count.fillna(0)).sum()), "nonwater_support_zero": int(df.pixel_nonwater_count.eq(0).sum()), "observed_fraction_defined": int(df.observed_fraction.notna().sum()), "observed_fraction_ge_0_5": int(df.observed_fraction.ge(0.5).sum()), "low_coverage": int(df.low_coverage.sum()), "single_scene": int(df.single_scene.sum()), "water_only_tracts": int(df.water_only_tract.sum()), "ndvi_mean_median": df.NDVI_mean.median(), "evi_mean_median": df.EVI_mean.median(), "clear_obs_mean_median": df.clear_obs_mean.median(), "observed_fraction_median": df.observed_fraction.median()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--geography", type=Path, default=ROOT / "GECC_CensusTracts_release_allUS_20260930/geography/CensusTract_2020.geojson")
    parser.add_argument("--tiers", type=Path, default=RUN / "alaska_tract_tiers.csv")
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--guide", type=Path, default=ROOT / "dataset_documentation/census_tract_ndvi_evi_v1_DATA_GUIDE.md")
    parser.add_argument("--out", type=Path, default=RUN / "release/ndvi_evi")
    args = parser.parse_args()
    staging = args.out.with_name(args.out.name + ".tmp")
    require(not args.out.exists() and not staging.exists(), f"Output already exists: {args.out} or {staging}")
    # Evidence and documentation are required before beginning the build.
    require((args.run_dir / "validation/issues.csv").is_file(), "Run source validation first")
    require(pd.read_csv(args.run_dir / "validation/issues.csv").empty, "Source validation has issues")
    require(args.guide.is_file(), "Missing data guide")
    tiers = pd.read_csv(args.tiers, dtype={"GEOID": str})
    require(len(tiers) == 177 and tiers.GEOID.is_unique and tiers.scale_m.value_counts().to_dict() == {30: 137, 100: 40}, "Wrong Alaska tier inventory")
    require(tiers.grid.eq("EPSG:3338").all(), "Wrong Alaska projection")
    require((tiers.scale_m.eq(30) == tiers.tract_km2.le(5000)).all(), "Wrong Alaska size tiers")
    geo = geography(args.geography)
    require(set(tiers.GEOID) == set(geo.loc[geo.StFIPS.eq("02"), "GEOID"]), "Alaska tiers differ from shared geography")
    expected = expected_exports(tiers)
    staging.mkdir(parents=True)
    for sub in ["qa", "provenance", "code"]:
        (staging / sub).mkdir()
    with ExitStack() as stack:
        archives = [stack.enter_context(zipfile.ZipFile(p.expanduser())) for p in args.archives]
        print("Indexing and checking source exports...", flush=True)
        found, ignored = index_exports(archives, expected)
        total_frames = []
        inventory = []
        for stem, spec in expected.items():
            kind, year, season, batch, group = spec
            archive, member, digest, copies = found[stem]
            inventory.append({"export_name": stem + ".csv", "kind": kind, "year": year, "season": season, "batch": batch, "export_group": group, "sha256": digest, "selected_archive": Path(archive.filename).name, "selected_member": member, "copies": json.dumps(copies)})
            if kind != "totals":
                continue
            df = load_export(found, stem)
            required = ["GEOID", "StFIPS", "pixel_total_count", "pixel_glc_gap_count", *[f"pixel_nonwater_count_{y}" for y in wf.GLC_MAP_YEARS]]
            require(list(df) == required, f"Wrong totals schema: {stem}")
            require(df.GEOID.is_unique and df.GEOID.str[:2].eq(df.StFIPS).all(), "Malformed totals identifiers")
            if batch == 14:
                require(set(df.GEOID) == set(tiers.loc[tiers.group.eq(group), "GEOID"]), f"Wrong total group membership: {stem}")
            else:
                require(df.StFIPS.isin(wf.REGION_STATEFPS[batch]).all(), "Wrong totals batch")
            for col in required[2:]:
                # A null count from an empty reduction means no supported cells.
                values = df[col].fillna(0)
                require(values.ge(0).all() and values.eq(values.round()).all(), "Invalid pixel total")
                df[col] = values.astype("int64")
                if col != "pixel_total_count":
                    require(df[col].le(df.pixel_total_count).all(), "Support exceeds total cells")
            df["batch"], df["export_group"] = batch, group
            df["analysis_scale_m"] = wf.group_scale(group or None)
            df["analysis_crs"] = "EPSG:3338" if batch == 14 else "EPSG:4326"
            df["alaska_scale_m"] = pd.Series(wf.group_scale(group) if batch == 14 else pd.NA, index=df.index, dtype="Int64")
            total_frames.append(df)
        totals = pd.concat(total_frames, ignore_index=True)
        require(len(totals) == 85528 and totals.GEOID.is_unique and set(totals.GEOID) == set(geo.GEOID), "Pixel totals differ from shared geography")
        meta = geo.merge(totals, on=["GEOID", "StFIPS"], validate="one_to_one")
        meta["water_only_tract"] = meta.Area_Land.eq(0)
        meta = meta.sort_values("GEOID").reset_index(drop=True)
        write_pair(meta, staging, "CensusTract_ndvi_evi_tract_support")
        shutil.copy2(args.tiers, staging / "alaska_tract_tiers.csv")
        pd.DataFrame(inventory).to_csv(staging / "provenance/EXPORTS.csv", index=False)
        pd.DataFrame(ignored).to_csv(staging / "provenance/IGNORED_EXPORTS.csv", index=False)

        summaries, row_group_hashes = [], []
        ever_valid = pd.Series(False, index=geo.GEOID)
        writer = None
        rows = 0
        try:
            with (staging / f"{PREFIX}.csv").open("w", newline="") as csv_out:
                for year in range(1985, 2026):
                    for source_season in ["DJFsameYear", "JJA"]:
                        season = "DJF_same_year" if source_season == "DJFsameYear" else "JJA"
                        frames = []
                        for stem, (kind, y, s, batch, group) in expected.items():
                            if (kind, y, s) == ("data", year, source_season):
                                frames.append(prepare_frame(load_export(found, stem), meta, year, season, batch, group))
                        period = pd.concat(frames, ignore_index=True).sort_values("GEOID").reset_index(drop=True)
                        require(len(period) == 85528 and period.GEOID.is_unique and set(period.GEOID) == set(geo.GEOID), "Incomplete season-year")
                        table = arrow_table(period)
                        if writer is None:
                            writer = pq.ParquetWriter(staging / f"{PREFIX}.parquet", table.schema, compression="zstd")
                        writer.write_table(table, row_group_size=85528)
                        # Value hashes verify every serialized Parquet row group below.
                        row_group_hashes.append(table_hash(table))
                        period.to_csv(csv_out, header=(rows == 0), index=False)
                        rows += len(period)
                        ever_valid.loc[period.GEOID] |= period.NDVI_count.fillna(0).gt(0).to_numpy()
                        summaries.append(coverage(period, "ALL_US"))
                        summaries.extend(coverage(part, state) for state, part in period.groupby("StFIPS"))
                    print(f"Packaged {year}: {rows:,} rows", flush=True)
        finally:
            if writer:
                writer.close()
        require(rows == 7013296, "Wrong long-table record count")
        parquet = pq.ParquetFile(staging / f"{PREFIX}.parquet")
        require(parquet.metadata.num_rows == rows and parquet.num_row_groups == 82, "Wrong Parquet dimensions")
        for i, digest in enumerate(row_group_hashes):
            require(table_hash(parquet.read_row_group(i)) == digest, f"Parquet round-trip mismatch, period {i}")
            if (i + 1) % 20 == 0 or i + 1 == len(row_group_hashes):
                print(f"Verified Parquet serialization: {i + 1}/82 periods", flush=True)

    summary = pd.DataFrame(summaries)
    summary.to_csv(staging / f"{PREFIX}_coverage.csv", index=False)
    never = geo.loc[~geo.GEOID.map(ever_valid)].copy()
    never.to_csv(staging / "qa/tracts_without_ndvi_in_any_period.csv", index=False)
    (staging / "provenance/validation").mkdir()
    for p in (args.run_dir / "validation").iterdir():
        if p.is_file():
            shutil.copy2(p, staging / "provenance/validation" / p.name)
    for p in ["eecu_ledger.json", "eecu_ledger_alaska.json", "runner.log", "runner_alaska.log"]:
        shutil.copy2(args.run_dir / p, staging / "provenance" / p)
    for p in ["submit_census_tract_ndvi_evi_exports.py", "test_census_tract_ndvi_evi_exports.py", "run_census_tract_ndvi_evi_queue.py", "validate_census_tract_ndvi_evi_v1.py", "verify_census_tract_ndvi_evi_package.py", "test_build_census_tract_ndvi_evi_v1.py", Path(__file__).name]:
        shutil.copy2(ROOT / p, staging / "code" / p)
    shutil.copy2(args.guide, staging / "DATA_GUIDE.md")
    # Retain the evidence underlying the stated continuity caveat, not raw samples.
    harmony = staging / "provenance/harmonization_test"
    harmony.mkdir()
    harmony_source = ROOT / "harmonization_test"
    if not harmony_source.is_dir():
        harmony_source = args.run_dir / "harmonization_test"
    for name in ["README.md", "tract_level_offsets.csv", "index_level_coefficients.json", "index_level_cv.csv"]:
        shutil.copy2(harmony_source / name, harmony / name)
    archive_records = [{"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in args.archives]
    all_us = summary.loc[summary.scope.eq("ALL_US")]
    report = {"status": "PASS", "built_at_utc": datetime.now(timezone.utc).isoformat(), "rows": rows, "tracts": 85528, "years": [1985, 2025], "seasons": ["JJA", "DJF_same_year"], "production_data_exports": 2624, "production_pixel_total_exports": 32, "logical_season_year_batches": 1558, "water_only_tracts": 368, "alaska_30m_tracts": 137, "alaska_100m_tracts": 40, "records_with_ndvi": int(all_us.with_ndvi.sum()), "records_without_ndvi": int(all_us.without_ndvi.sum()), "evi_count_below_ndvi_records": int(all_us.evi_count_below_ndvi.sum()), "tracts_without_ndvi_in_any_period": len(never), "archives": archive_records, "shared_geography_sha256": sha256(args.geography), "alaska_tiers_sha256": sha256(args.tiers), "checks": ["all expected exports present; all duplicate copies identical", "source schema, labels, finite values, count/stat consistency and nonnegative standard deviations", "exact GEOID equality with shared release geography in every period", "exact Alaska group membership and size tier contract", "positive nonwater support wherever indices observed; observed fraction <= 1", "all 82 Parquet row groups equal assembled source and derived values", "CSV versus Parquet verification required separately; see csv_parquet_validation.json"], "software": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pa.__version__}, "compute": {"projects": ["your-ee-project", "your-ee-project"], "approximate_total_eecu_hours_including_tests_and_failed_attempts": 38400, "note": "Handoff total; individual queue ledgers omit some pilot, retry and your-ee-project Alaska activity. Do not sum ledgers as a complete billing total."}}
    (staging / "qa/assembly_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (staging / "code/requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in {"pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pa.__version__, "earthengine-api": __import__('importlib.metadata', fromlist=['version']).version('earthengine-api')}.items()) + "\n")
    (staging / "README.md").write_text(f"# Census Tract NDVI/EVI v1\n\n{rows:,} rows cover 85,528 2020 Census tracts, 1985–2025, JJA and same-calendar-year DJF. Values are unadjusted across Landsat sensors. Read DATA_GUIDE.md before trend analysis.\n\n- `{PREFIX}.csv` and `.parquet`: canonical long table, one tract/year/season per row.\n- `CensusTract_ndvi_evi_tract_support.csv` and `.parquet`: fixed tract attributes, grid, and all year-specific nonwater denominators.\n- `{PREFIX}_coverage.csv`: national and state/territory coverage for every period.\n- `alaska_tract_tiers.csv`: all 177 Alaska tracts and their 30/100 m tiers.\n- `DATA_GUIDE.md`: method, field definitions, missing values and limitations.\n- `qa/`: source and assembly checks, CSV/Parquet parity, coverage report, and never-observed tract list.\n- `provenance/`: source-export manifest, archive hashes in assembly validation, queue records and continuity-study summaries.\n- `code/`: production workflow, validator and assembly recipe.\n\nGEOID and FIPS are text; specify string dtypes when reading CSV. Parquet preserves identifier types and nullable booleans. Blank index values are missing. Counts in the long table retain source nulls; support-table null counts are normalized to zero. All ancillary flags retain the source values. No wide table is supplied, following the sibling seasonal air-temperature product.\n\nThis is a standalone module prepared for the all-US collection. Join the collection's `geography/CensusTract_2020.geojson` by text GEOID; geometry is not duplicated here. Existing collection-wide documentation and manifests must be refreshed when integrating this module.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(args.out)
    print(json.dumps({"output": str(args.out), "rows": rows, "status": "PASS"}, indent=2), flush=True)


def table_hash(table):
    # Strip pandas metadata and normalize chunk boundaries and null backing bytes.
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema.remove_metadata()) as out:
        out.write_table(pa.Table.from_pylist(table.to_pylist(), schema=table.schema.remove_metadata()))
    return hashlib.sha256(sink.getvalue()).hexdigest()


if __name__ == "__main__":
    main()
