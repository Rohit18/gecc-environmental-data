#!/usr/bin/env python3
"""Validate the Earth Engine tract export and prepare NERSC-ready polygons."""

import csv
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parent
DIR = ROOT / "outputs/nonconus_tract_boundaries"
SOURCE = DIR / "GECC_nonCONUS_2020_tract_boundaries_v2.geojson"
REFERENCE = ROOT / "outputs/worldcover_2020_nonconus/CensusTract_WorldCover2020_nonCONUS_areas.csv"
OUT = DIR / "CensusTract_2020_nonCONUS_NERSC.gpkg"
REPORT = DIR / "validation.json"
COUNTS = {"02": 177, "15": 461, "60": 18, "66": 57, "69": 26, "72": 981, "78": 32}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    tract = gpd.read_file(SOURCE)
    with REFERENCE.open(newline="") as stream:
        expected = {row["GEOID"] for row in csv.DictReader(stream)}
    counts = tract.STATEFP.value_counts().sort_index().to_dict()
    if (len(tract) != 1752 or tract.GEOID.nunique() != 1752
            or set(tract.GEOID) != expected or counts != COUNTS
            or tract.geometry.isna().any() or tract.geometry.is_empty.any()
            or not tract.geometry.is_valid.all() or tract.crs.to_epsg() != 4326):
        raise ValueError("Source tract GeoJSON failed universe or geometry validation")
    tract = tract[["GEOID", "STATEFP", "COUNTYFP", "TRACTCE", "ALAND", "AWATER", "geometry"]].copy()
    tract = tract.rename(columns={"STATEFP": "StFIPS", "COUNTYFP": "CoFIPS",
                                  "TRACTCE": "TrFIPS", "ALAND": "Area_Land",
                                  "AWATER": "Area_Water"})
    if not (tract.GEOID == tract.StFIPS + tract.CoFIPS + tract.TrFIPS).all():
        raise ValueError("Boundary GEOID is inconsistent with component FIPS fields")
    tract["Pop"] = pd.Series(pd.NA, index=tract.index, dtype="Int64")
    tract = tract.sort_values("GEOID").reset_index(drop=True)
    tract.to_file(OUT, layer="tracts", driver="GPKG")
    check = gpd.read_file(OUT, layer="tracts")
    if len(check) != 1752 or set(check.GEOID) != expected or not check.geometry.is_valid.all():
        raise ValueError("Round-trip GeoPackage validation failed")
    rec = {"status": "PASS", "source": str(SOURCE.relative_to(ROOT)),
           "source_sha256": sha256(SOURCE), "output": str(OUT.relative_to(ROOT)),
           "output_sha256": sha256(OUT), "rows": 1752, "unique_geoids": 1752,
           "state_counts": counts, "missing_population": 1752,
           "crs": "EPSG:4326", "invalid_geometries": 0,
           "matched_worldcover_nonconus_geoids": True}
    REPORT.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({"status": rec["status"], "rows": rec["rows"],
                      "state_counts": counts, "bytes": OUT.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
