#!/usr/bin/env python3
"""Verify all CSV/Parquet values and all original seasonal-export fields.

Usage: python verify_census_tract_ndvi_evi_package.py MODULE_DIR SOURCE_ZIP [...]
Writes qa/csv_parquet_validation.json only after every check succeeds.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def check_pair(csv, parquet):
    file = pq.ParquetFile(parquet)
    schema = file.schema_arrow.remove_metadata()
    dtypes = {}
    for field in schema:
        dtypes[field.name] = str if pa.types.is_string(field.type) else "boolean" if pa.types.is_boolean(field.type) else "Int64" if pa.types.is_integer(field.type) else "float64"
    rows = 0
    with pd.read_csv(csv, dtype=dtypes, float_precision="round_trip", chunksize=85528) as chunks:
        for i, df in enumerate(chunks):
            if "export_group" in df:
                df["export_group"] = df.export_group.fillna("")
            actual = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            expected = file.read_row_group(i)
            if not actual.equals(expected):
                bad = [name for name in schema.names if not actual[name].equals(expected[name])]
                raise ValueError(f"CSV/Parquet mismatch, group {i}: {bad}")
            rows += len(df)
    if rows != file.metadata.num_rows:
        raise ValueError("CSV and Parquet row counts differ")
    return {"csv": csv.name, "parquet": parquet.name, "rows": rows, "columns": len(schema), "all_values_equal": True, "identifier_type": "string"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=Path)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    folder = args.module
    pairs = []
    for parquet in sorted(folder.glob("*.parquet")):
        print(f"Checking every CSV/Parquet value: {parquet.name}", flush=True)
        pairs.append(check_pair(parquet.with_suffix(".csv"), parquet))
    if len(pairs) != 2:
        raise ValueError("Expected the long and support table pairs")

    manifest = pd.read_csv(folder / "provenance/EXPORTS.csv", dtype={"export_group": str})
    manifest = manifest.loc[manifest.kind.eq("data")]
    source_rows = source_exports = 0
    long = pq.ParquetFile(folder / "CensusTract_ndvi_evi_1985_2025.parquet")
    with ExitStack() as stack:
        archives = {p.name: stack.enter_context(zipfile.ZipFile(p)) for p in args.archives}
        if len(archives) != len(args.archives):
            raise ValueError("Duplicate source archive basename")
        for i in range(long.num_row_groups):
            period = long.read_row_group(i)
            year, season = period["year"][0].as_py(), period["season"][0].as_py()
            selection = manifest.loc[manifest.year.eq(year) & manifest.season.eq("DJFsameYear" if season == "DJF_same_year" else season)]
            if len(selection) != 32:
                raise ValueError(f"Expected 18 non-Alaska + 14 Alaska exports for {year}/{season}")
            frames = []
            for entry in selection.itertuples():
                raw = archives[entry.selected_archive].read(entry.selected_member)
                frames.append(pd.read_csv(io.BytesIO(raw), dtype={"GEOID": str, "StFIPS": str}, float_precision="round_trip"))
            source = pd.concat(frames, ignore_index=True).sort_values("GEOID").reset_index(drop=True)
            fields = list(source.columns)
            schema = pa.schema([period.schema.field(name) for name in fields])
            expected = pa.Table.from_pandas(source, schema=schema, preserve_index=False)
            if not expected.equals(period.select(fields)):
                raise ValueError(f"Original export values changed: {year}/{season}")
            source_rows += len(source)
            source_exports += len(selection)
            if season == "JJA":
                print(f"Original source values preserved through {year}", flush=True)
    if source_rows != 7013296 or source_exports != 2624:
        raise ValueError("Incomplete source fidelity verification")
    result = {"status": "PASS", "verified_at_utc": datetime.now(timezone.utc).isoformat(), "csv_parquet_pairs": pairs, "source_exports_compared": source_exports, "source_rows_compared": source_rows, "original_export_fields_unchanged": True, "comparison": "Every field of both CSV/Parquet pairs, and every original field from all 2,624 seasonal exports; exact binary float equality after round-trip decimal parsing, exact text identifiers and nulls."}
    (folder / "qa/csv_parquet_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
