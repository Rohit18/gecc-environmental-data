#!/usr/bin/env python3
"""Validate and package the complete 1 km CONUS Census Tract GRIDMET/PRISM series.

The raw Earth Engine CSVs are retained in their existing download folders.
This builder resolves byte-identical duplicate downloads, verifies every
source-year table, and produces one typed Parquet long table plus a gzip CSV,
coverage report, source manifest, validation record, and README.  No raster
statistics are recalculated and no missing values are imputed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "GRIDMET_PRISM"
DEFAULT_OUT = ROOT / "CensusTracts" / "air_temperature_gridmet_prism"
WORLD_COVER = ROOT / "CensusTracts" / "worldcover_urban_rural" / "CensusTract_WorldCover2020_urban_rural.csv"
FILENAME = re.compile(r"GECC_(GRIDMET|PRISM)_CensusTracts_AirTemp_(\d{4})_sameyear_1km\.csv$")
EXPECTED_YEARS = {"GRIDMET": range(1980, 2026), "PRISM": range(1981, 2026)}
EXPECTED_ROWS = 83_776
FIELDS = [
    "GEOID", "source_area_m2", "dataset", "geography_type", "year", "djf_convention",
    "annual_day_count", "JJA_day_count", "DJF_day_count", "source_scale_m", "reduction_scale_m",
    "missing_mean_count", "all_six_means_valid", "fallback_applied",
    "annual_tmax_mean", "annual_tmax_median", "annual_tmax_stdDev", "annual_tmax_count",
    "annual_tmin_mean", "annual_tmin_median", "annual_tmin_stdDev", "annual_tmin_count",
    "JJA_tmax_mean", "JJA_tmax_median", "JJA_tmax_stdDev", "JJA_tmax_count",
    "JJA_tmin_mean", "JJA_tmin_median", "JJA_tmin_stdDev", "JJA_tmin_count",
    "DJF_tmax_mean", "DJF_tmax_median", "DJF_tmax_stdDev", "DJF_tmax_count",
    "DJF_tmin_mean", "DJF_tmin_median", "DJF_tmin_stdDev", "DJF_tmin_count",
]
MEAN_FIELDS = [f"{period}_{variable}_mean" for period in ("annual", "JJA", "DJF") for variable in ("tmax", "tmin")]
COUNT_FIELDS = [f"{period}_{variable}_count" for period in ("annual", "JJA", "DJF") for variable in ("tmax", "tmin")]
NUMERIC_FIELDS = [field for field in FIELDS if field not in {"GEOID", "dataset", "geography_type", "djf_convention", "reduction_grid_alignment", "fallback_applied"}]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_geoids() -> set[str]:
    if not WORLD_COVER.is_file():
        raise RuntimeError(f"Missing CONUS reference GEOID table: {WORLD_COVER}")
    with WORLD_COVER.open(newline="", encoding="utf-8") as handle:
        ids = {row["GEOID"] for row in csv.DictReader(handle)}
    if len(ids) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} WorldCover CONUS GEOIDs, found {len(ids):,}")
    return ids


def find_sources(root: Path) -> dict[tuple[str, int], list[Path]]:
    found: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for path in root.rglob("*.csv"):
        match = FILENAME.fullmatch(path.name)
        if match:
            found[(match.group(1), int(match.group(2)))].append(path)
    expected = {(dataset, year) for dataset, years in EXPECTED_YEARS.items() for year in years}
    missing = sorted(expected - set(found))
    unexpected = sorted(set(found) - expected)
    if missing or unexpected:
        raise RuntimeError(f"Incomplete/unexpected 1 km source-year set; missing={missing}, unexpected={unexpected}")
    return found


def select_sources(found: dict[tuple[str, int], list[Path]]) -> tuple[dict[tuple[str, int], Path], list[dict[str, Any]]]:
    selected: dict[tuple[str, int], Path] = {}
    manifest: list[dict[str, Any]] = []
    for key, paths in sorted(found.items()):
        checksums = {path: sha256_file(path) for path in paths}
        if len(set(checksums.values())) != 1:
            raise RuntimeError(f"Conflicting duplicate downloads for {key}: {[str(path) for path in paths]}")
        path = min(paths, key=lambda item: (len(str(item)), str(item)))
        selected[key] = path
        manifest.append({
            "dataset": key[0], "year": key[1], "source_path": str(path.relative_to(ROOT)),
            "sha256": checksums[path], "bytes": path.stat().st_size, "duplicate_file_count": len(paths),
        })
    return selected, manifest


def _missing_count(row: pd.Series) -> int:
    return int(sum(pd.isna(row[field]) for field in MEAN_FIELDS))


def validate_frame(frame: pd.DataFrame, dataset: str, year: int, wanted_ids: set[str]) -> tuple[Counter[str], set[str]]:
    if frame.columns.tolist() != FIELDS:
        raise RuntimeError(f"{dataset} {year}: schema differs from the 1 km release contract")
    if len(frame) != EXPECTED_ROWS or frame.GEOID.nunique() != EXPECTED_ROWS:
        raise RuntimeError(f"{dataset} {year}: expected {EXPECTED_ROWS:,} unique rows, found {len(frame):,}")
    geoids = set(frame.GEOID.astype(str))
    if geoids != wanted_ids:
        raise RuntimeError(f"{dataset} {year}: GEOID universe differs from the CONUS reference")
    if set(frame.dataset) != {dataset} or set(frame.geography_type) != {"CensusTracts"} or set(frame.year.astype(int)) != {year}:
        raise RuntimeError(f"{dataset} {year}: dataset/geography/year metadata differs from filename")
    if set(frame.djf_convention) != {"same-year"} or set(frame.reduction_scale_m.astype(float)) != {1000.0}:
        raise RuntimeError(f"{dataset} {year}: DJF or 1 km reduction metadata is invalid")
    fallback = frame.fallback_applied.astype(str).str.lower()
    if not fallback.isin({"false", "0"}).all():
        raise RuntimeError(f"{dataset} {year}: fallback_applied is not uniformly false")
    reported_missing = frame.missing_mean_count.astype(float).astype(int)
    computed_missing = frame[MEAN_FIELDS].isna().sum(axis=1)
    if not (reported_missing == computed_missing).all():
        raise RuntimeError(f"{dataset} {year}: missing_mean_count disagrees with the six mean fields")
    complete = frame.all_six_means_valid.astype(float).astype(int)
    if not (complete == (computed_missing == 0).astype(int)).all():
        raise RuntimeError(f"{dataset} {year}: all_six_means_valid disagrees with null values")
    for field in NUMERIC_FIELDS:
        if field in {"year", "missing_mean_count", "all_six_means_valid"}:
            continue
        values = pd.to_numeric(frame[field], errors="coerce").dropna()
        if not values.map(math.isfinite).all():
            raise RuntimeError(f"{dataset} {year}: non-finite numeric value in {field}")
    coverage: Counter[str] = Counter()
    coverage["tract_records"] = len(frame)
    coverage["complete_six_means"] = int((computed_missing == 0).sum())
    coverage["some_means_missing"] = int(((computed_missing > 0) & (computed_missing < 6)).sum())
    coverage["all_six_means_missing"] = int((computed_missing == 6).sum())
    for field in MEAN_FIELDS:
        coverage[f"non_null_{field}"] = int(frame[field].notna().sum())
    return coverage, geoids


def arrow_schema() -> pa.Schema:
    fields: list[pa.Field] = []
    for name in FIELDS:
        if name in {"GEOID", "dataset", "geography_type", "djf_convention"}:
            typ = pa.string()
        elif name == "fallback_applied":
            typ = pa.bool_()
        elif name in {"year", "annual_day_count", "JJA_day_count", "DJF_day_count", "missing_mean_count", "all_six_means_valid", *COUNT_FIELDS}:
            typ = pa.int64()
        else:
            typ = pa.float64()
        fields.append(pa.field(name, typ))
    return pa.schema(fields)


def normalise_for_output(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["GEOID"] = out["GEOID"].astype("string")
    out["fallback_applied"] = out["fallback_applied"].astype(str).str.lower().isin({"true", "1"})
    for field in {"year", "annual_day_count", "JJA_day_count", "DJF_day_count", "missing_mean_count", "all_six_means_valid", *COUNT_FIELDS}:
        out[field] = pd.to_numeric(out[field], errors="raise").astype("int64")
    for field in set(FIELDS) - {"GEOID", "dataset", "geography_type", "djf_convention", "fallback_applied", "year", "annual_day_count", "JJA_day_count", "DJF_day_count", "missing_mean_count", "all_six_means_valid", *COUNT_FIELDS}:
        out[field] = pd.to_numeric(out[field], errors="coerce")
    return out[FIELDS]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(folder: Path) -> None:
    (folder / "README.md").write_text(
        "# Census Tract GRIDMET and PRISM air temperature\n\n"
        "## Package contents\n\n"
        "- `CensusTract_air_temperature_gridmet_prism_1980_2025.parquet` — canonical typed long table.\n"
        "- `CensusTract_air_temperature_gridmet_prism_1980_2025.csv.gz` — the same records in compressed CSV form.\n"
        "- `CensusTract_air_temperature_gridmet_prism_coverage.csv` — availability by source and year.\n"
        "- `CensusTract_air_temperature_gridmet_prism_source_manifest.csv` — selected raw file path, SHA-256, size, and duplicate-download count.\n"
        "- `CensusTract_air_temperature_gridmet_prism_validation.json` — source and table checks.\n"
        "- `../geography/CensusTract_2020.geojson` — shared geometry, joined by text `GEOID`.\n\n"
        "## Scope\n\n"
        "The table covers **83,776 2020 Census Tracts in CONUS** (the 48 contiguous states and Washington, DC). "
        "GRIDMET covers 1980–2025 and PRISM covers 1981–2025. Each source-year has one row for every CONUS tract; "
        "an unavailable raster statistic remains null rather than being filled or substituted. The GEOID universe exactly "
        "matches the CONUS LAI and WorldCover urban/rural modules.\n\n"
        "## Sources and processing\n\n"
        "GRIDMET is `IDAHO_EPSCOR/GRIDMET` and PRISM is `projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_DAILY`, "
        "as exported through Google Earth Engine. Both use the established 1,000 m reduction variant. This is the "
        "analysis scale, not a claim that the sources have the same native resolution: `source_scale_m` records the "
        "source grid scale. These are separate alternative temperature sources and must not be merged into or treated "
        "as replacements for the national SVCM-SP `air_temperature` module.\n\n"
        "Each record contains annual, JJA, and same-year DJF summaries of daily Tmax and Tmin. `*_mean`, `*_median`, "
        "and `*_stdDev` fields are degrees Celsius; `*_count` fields are raster-sample counts. DJF means January, "
        "February, and December of the labeled calendar year. `missing_mean_count`, `all_six_means_valid`, and the "
        "coverage report describe missingness. `fallback_applied` is always false: no fallback, imputation, water mask, "
        "or alternate raster source was applied during packaging.\n\n"
        "## Provenance\n\n"
        "The builder validates all 91 source-year exports, rejects conflicting duplicate downloads, and records the selected "
        "raw paths and SHA-256 hashes in the source manifest. See `provenance/` for scientific-source and code lineage.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out_dir.exists() or args.out_dir.with_name(args.out_dir.name + ".tmp").exists():
        raise RuntimeError(f"Refusing to overwrite an existing package directory: {args.out_dir}")
    wanted_ids = expected_geoids()
    selected, source_manifest = select_sources(find_sources(args.source_root))

    staging = args.out_dir.with_name(args.out_dir.name + ".tmp")
    staging.mkdir(parents=True)
    parquet_path = staging / "CensusTract_air_temperature_gridmet_prism_1980_2025.parquet"
    csv_path = staging / "CensusTract_air_temperature_gridmet_prism_1980_2025.csv.gz"
    coverage_rows: list[dict[str, Any]] = []
    validation_files: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    try:
        with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as csv_handle:
            csv_writer = csv.DictWriter(csv_handle, fieldnames=FIELDS)
            csv_writer.writeheader()
            for (dataset, year), path in sorted(selected.items()):
                frame = pd.read_csv(path, dtype={"GEOID": "string"}, keep_default_na=True)
                coverage, _ = validate_frame(frame, dataset, year, wanted_ids)
                output = normalise_for_output(frame)
                csv_writer.writerows(output.where(pd.notna(output), None).to_dict("records"))
                table = pa.Table.from_pandas(output, schema=arrow_schema(), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
                writer.write_table(table)
                coverage_rows.append({"dataset": dataset, "year": year, **coverage})
                validation_files.append({"dataset": dataset, "year": year, "source_path": str(path.relative_to(ROOT)),
                                         "rows": len(output), "unique_geoids": int(output.GEOID.nunique()), "passed": True})
        if writer is None:
            raise RuntimeError("No GRIDMET/PRISM source tables were selected")
        writer.close()
        writer = None
        coverage_fields = ["dataset", "year", "tract_records", "complete_six_means", "some_means_missing", "all_six_means_missing", *[f"non_null_{field}" for field in MEAN_FIELDS]]
        write_csv(staging / "CensusTract_air_temperature_gridmet_prism_coverage.csv", coverage_fields, coverage_rows)
        write_csv(staging / "CensusTract_air_temperature_gridmet_prism_source_manifest.csv",
                  ["dataset", "year", "source_path", "sha256", "bytes", "duplicate_file_count"], source_manifest)
        validation = {
            "package": "CensusTracts/air_temperature_gridmet_prism",
            "assembly": "dataset_documentation/build_census_tract_gridmet_prism.py",
            "scope": "CONUS 2020 Census Tracts (48 contiguous states plus DC)",
            "expected_geoids": EXPECTED_ROWS,
            "source_year_files": validation_files,
            "source_years": {dataset: [min(years), max(years), len(years)] for dataset, years in EXPECTED_YEARS.items()},
            "source_manifest_sha256": sha256_file(staging / "CensusTract_air_temperature_gridmet_prism_source_manifest.csv"),
            "output_sha256": {"parquet": sha256_file(parquet_path), "csv_gz": sha256_file(csv_path)},
            "passed": True,
        }
        (staging / "CensusTract_air_temperature_gridmet_prism_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_readme(staging)
        os.replace(staging, args.out_dir)
    except Exception:
        if writer is not None:
            writer.close()
        raise
    print(f"Packaged verified GRIDMET/PRISM release at {args.out_dir}")


if __name__ == "__main__":
    main()
