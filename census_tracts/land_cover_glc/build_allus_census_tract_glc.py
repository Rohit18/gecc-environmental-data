#!/usr/bin/env python3
"""Join validated PR/VI GLC results to the national table on all 85,528 tracts.

Unavailable GLC source pixels remain null; sample flags are zero. Processing is
streamed because the source table exceeds 1 GB.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "GECC_CensusTracts_release_v2/land_cover/CensusTract_land_cover_1985_2022.csv"
ADDON = ROOT / "outputs/glc_nonconus/CensusTract_GLC_nonCONUS_1985_2022_wide.csv"
NONCONUS = ROOT / "outputs/worldcover_2020_nonconus/CensusTract_WorldCover2020_nonCONUS_areas.csv"
OUT = ROOT / "outputs/glc_nonconus/CensusTract_GLC_allUS_1985_2022.csv"
VALIDATION = ROOT / "outputs/glc_nonconus/allUS_validation.json"
YEARS = [1985, 1990, 1995] + list(range(2000, 2023))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    with NONCONUS.open(newline="") as stream:
        nonconus = {}
        for row in csv.DictReader(stream):
            geoid = row["GEOID"]
            if geoid in nonconus:
                raise ValueError(f"Duplicate non-CONUS GEOID: {geoid}")
            nonconus[geoid] = row
    if len(nonconus) != 1752:
        raise ValueError(f"Expected 1752 add-on tracts, got {len(nonconus)}")

    with ORIGINAL.open(newline="") as source, ADDON.open(newline="") as extra, OUT.open("w", newline="") as target:
        original_reader = csv.reader(source)
        original_fields = next(original_reader)
        addon_reader = csv.DictReader(extra)
        addon_fields = addon_reader.fieldnames
        if not set(addon_fields) - {"STATEFP"} <= set(original_fields):
            raise ValueError("Add-on GLC schema is not a subset of the national schema")
        if set(original_fields) - set(addon_fields) != {
            "StFIPS", "CoFIPS", "TrFIPS", "Area_Land", "Area_Water",
            "LAT_GeoCen", "LON_GeoCen", "Pop"}:
            raise ValueError("Unexpected national-only GLC columns")
        output_fields = original_fields + ["glc_source_status"]
        writer = csv.writer(target)
        writer.writerow(output_fields)
        index = {name: i for i, name in enumerate(original_fields)}
        flag_indexes = [index[f"has_glc_sampled_area_{year}"] for year in YEARS]
        geoids = set()
        statuses = Counter()
        by_state = Counter()

        def add(geoid, row):
            if geoid in geoids:
                raise ValueError(f"Duplicate all-US GEOID: {geoid}")
            if len(geoid) != 11 or len(row) != len(original_fields):
                raise ValueError(f"Malformed GLC row: {geoid}")
            flags = [row[i] for i in flag_indexes]
            if all(v == "1" for v in flags):
                status = "sampled_all_years"
            elif all(v == "0" for v in flags):
                status = "no_valid_source_pixels"
            else:
                status = "partial_year_coverage"
            writer.writerow(row + [status])
            geoids.add(geoid)
            statuses[status] += 1
            by_state[(geoid[:2], status)] += 1

        for row in original_reader:
            add(row[index["GEOID"]], row)
        if len(geoids) != 84414:
            raise ValueError(f"Unexpected national input count: {len(geoids)}")
        overlap = geoids & nonconus.keys()
        if len(overlap) != 638 or {g[:2] for g in overlap} != {"02", "15"}:
            raise ValueError("Expected exactly Alaska and Hawaii in the original 84,414 GEOIDs")

        addon_geoids = set()
        for rec in addon_reader:
            geoid = rec["GEOID"]
            if geoid in addon_geoids or geoid not in nonconus or geoid in geoids:
                raise ValueError(f"Unexpected or duplicate GLC add-on: {geoid}")
            addon_geoids.add(geoid)
            worldcover = nonconus[geoid]
            row = [rec.get(name, "") for name in original_fields]
            row[index["StFIPS"]] = geoid[:2]
            row[index["CoFIPS"]] = geoid[2:5]
            row[index["TrFIPS"]] = geoid[5:]
            row[index["Area_Land"]] = worldcover["Area_Land"]
            row[index["Area_Water"]] = worldcover["Area_Water"]
            add(geoid, row)
        if len(addon_geoids) != 1013 or {g[:2] for g in addon_geoids} != {"72", "78"}:
            raise ValueError(f"Unexpected PR/VI add-on count: {len(addon_geoids)}")

        for geoid in sorted(nonconus.keys() - overlap - addon_geoids):
            wc = nonconus[geoid]
            row = [""] * len(original_fields)
            row[index["GEOID"]] = geoid
            row[index["StFIPS"]] = geoid[:2]
            row[index["CoFIPS"]] = geoid[2:5]
            row[index["TrFIPS"]] = geoid[5:]
            row[index["Area_Land"]] = wc["Area_Land"]
            row[index["Area_Water"]] = wc["Area_Water"]
            for i in flag_indexes:
                row[i] = "0"
            add(geoid, row)

    if len(geoids) != 85528:
        raise ValueError(f"All-US output is incomplete: {len(geoids)}")
    validation = {
        "status": "PASS", "rows": len(geoids), "unique_geoids": len(geoids),
        "source_rows": 84414, "pr_vi_added_rows": len(addon_geoids),
        "other_geography_rows_without_glc_pixels": len(nonconus) - len(overlap) - len(addon_geoids),
        "columns": len(output_fields), "source_status_counts": dict(statuses),
        "source_status_by_state": [
            {"state": state, "status": status, "tracts": n}
            for (state, status), n in sorted(by_state.items())],
        "output": str(OUT.relative_to(ROOT)), "bytes": OUT.stat().st_size,
        "sha256": sha256(OUT),
    }
    VALIDATION.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({k: validation[k] for k in ("status", "rows", "columns", "source_status_counts", "bytes")}, indent=2))


if __name__ == "__main__":
    main()
