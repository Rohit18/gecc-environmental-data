#!/usr/bin/env python3
"""Merge and validate the PR/VI GLC-FCS30D Earth Engine CSV exports."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from submit_census_tract_glc_exports import EXPECTED_YEARS, GLC_CLASSES


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/glc_nonconus"
RAW = OUT / "raw_exports"
PATTERN = re.compile(r"GLC_NONCONUS_CensusTract_v1_(\d{4})_s(72|78)_n100_b(\d{3})\.csv$")
GROUPS = ("vegetation", "tree", "shrub", "grass", "crop")
FIELDS = ["GEOID", "year", "batch", "source_area_m2", "total_area_m2",
          "glc_coverage_fraction", "vegetation_area_m2", "vegetation_fraction",
          "tree_area_m2", "tree_fraction", "shrub_area_m2", "shrub_fraction",
          "grass_area_m2", "grass_fraction", "crop_area_m2", "crop_fraction"]
FIELDS += [f"area_class_{code}" for code in GLC_CLASSES]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ids = json.loads((OUT / "expected_geoids.json").read_text())["geoids"]
    assert len(ids) == 1013 and len(set(ids)) == 1013
    by_state = {state: sorted(geoid for geoid in ids if geoid.startswith(state))
                for state in ("72", "78")}
    assert len(by_state["72"]) == 981 and len(by_state["78"]) == 32
    expected = {(year, state, batch): set(geoids[start:start + 100])
                for year in EXPECTED_YEARS
                for state, geoids in by_state.items()
                for batch, start in enumerate(range(0, len(geoids), 100))}
    files = {}
    for path in RAW.glob("*.csv"):
        match = PATTERN.fullmatch(path.name)
        if match:
            key = (int(match[1]), match[2], int(match[3]))
            if key in files:
                raise ValueError(f"Duplicate export for {key}")
            files[key] = path
    if set(files) != set(expected):
        missing, extra = sorted(set(expected) - set(files)), sorted(set(files) - set(expected))
        raise RuntimeError(f"Export inventory incomplete: missing={missing[:10]} extra={extra[:10]}")

    frames = []
    for key in sorted(expected):
        path = files[key]
        frame = pd.read_csv(path, dtype={"GEOID": str}, float_precision="round_trip")
        if list(frame.columns) != FIELDS:
            raise ValueError(f"Unexpected columns: {path.name}")
        year, state, batch = key
        if (len(frame) != len(expected[key]) or frame.GEOID.duplicated().any()
                or set(frame.GEOID) != expected[key]
                or not frame.year.eq(year).all()
                or not frame.batch.eq(batch).all()):
            raise ValueError(f"Rows or batch keys do not match TIGER: {path.name}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    assert len(data) == 1013 * len(EXPECTED_YEARS)
    assert not data.duplicated(["GEOID", "year"]).any()

    raw_class = [f"area_class_{code}" for code in GLC_CLASSES]
    assert data[raw_class].ge(0).all().all()
    assert np.allclose(data[raw_class].sum(axis=1), data.total_area_m2,
                       rtol=1e-9, atol=0.01)
    assert data.source_area_m2.gt(0).all()
    assert np.allclose(data.total_area_m2 / data.source_area_m2,
                       data.glc_coverage_fraction, rtol=1e-9, atol=1e-8)
    for group in GROUPS:
        area = data[f"{group}_area_m2"]
        fraction = data[f"{group}_fraction"]
        assert area.ge(0).all() and area.le(data.total_area_m2 + 0.01).all()
        supported = data.total_area_m2.gt(0)
        assert np.allclose(area[supported] / data.total_area_m2[supported],
                           fraction[supported], rtol=1e-9, atol=1e-8)
        assert fraction[supported].between(0, 1 + 1e-9).all()
        data.loc[~supported, f"{group}_fraction"] = np.nan
    data["has_glc_sampled_area"] = data.total_area_m2.gt(0).astype(int)

    # Independent duplicate export: first 25 PR tracts, 2020.
    pilot = RAW / "GLC_NONCONUS_CensusTract_v1_2020_s72_n25_b000.csv"
    pilot_checked = 0
    if pilot.exists():
        sample = pd.read_csv(pilot, dtype={"GEOID": str}, float_precision="round_trip")
        production = data[data.year.eq(2020)].set_index("GEOID").loc[sample.GEOID]
        for field in FIELDS[3:]:
            if not np.allclose(sample[field].to_numpy(float),
                               production[field].to_numpy(float),
                               rtol=1e-8, atol=0.01, equal_nan=True):
                raise ValueError(f"Pilot and production differ: {field}")
        pilot_checked = len(sample)

    long_path = OUT / "CensusTract_GLC_nonCONUS_1985_2022_long.csv"
    coverage_path = OUT / "CensusTract_GLC_nonCONUS_year_coverage.csv"
    wide_path = OUT / "CensusTract_GLC_nonCONUS_1985_2022_wide.csv"
    data = data.drop(columns="batch").sort_values(["GEOID", "year"])
    data.to_csv(long_path, index=False)
    coverage = data.assign(state=data.GEOID.str[:2]).groupby(["year", "state"]).agg(
        rows=("GEOID", "size"),
        sampled_tracts=("has_glc_sampled_area", "sum"),
        sampled_area_m2=("total_area_m2", "sum"),
    ).reset_index()
    coverage.to_csv(coverage_path, index=False)

    metrics = [field for field in data if field not in ("GEOID", "year")]
    wide_parts = []
    for year in EXPECTED_YEARS:
        part = data[data.year.eq(year)].set_index("GEOID")[metrics].reindex(ids)
        part.columns = [f"{field}_{year}" for field in part]
        wide_parts.append(part)
    wide = pd.concat(wide_parts, axis=1)
    wide.insert(0, "STATEFP", [geoid[:2] for geoid in ids])
    wide.insert(0, "GEOID", ids)
    wide["land_cover_source"] = "GLC-FCS30D"
    wide["area_units"] = "m2"
    wide["fraction_denominator"] = "total sampled GLC raster area; zero-area fractions null"
    wide.to_csv(wide_path, index=False)

    # Read back the deliverables, including the wide key and zero-support rule.
    back = pd.read_csv(long_path, dtype={"GEOID": str})
    wide_back = pd.read_csv(wide_path, dtype={"GEOID": str})
    assert len(back) == len(data) and not back.duplicated(["GEOID", "year"]).any()
    assert len(wide_back) == 1013 and wide_back.GEOID.is_unique
    assert set(wide_back.GEOID) == set(ids)
    assert back.has_glc_sampled_area.eq(back.total_area_m2.gt(0).astype(int)).all()
    for group in GROUPS:
        assert back.loc[back.has_glc_sampled_area.eq(0), f"{group}_fraction"].isna().all()

    report = {
        "status": "PASS", "tracts": 1013, "years": list(EXPECTED_YEARS),
        "production_files": len(files), "long_rows": len(back),
        "wide_rows": len(wide_back), "wide_columns": len(wide_back.columns),
        "pilot_rows_cross_checked": pilot_checked,
        "sampled_tracts_by_state": {state: int(coverage[coverage.state.eq(state)].sampled_tracts.min())
                                    for state in ("72", "78")},
        "outputs": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                     "sha256": sha256(path)}
                    for path in (long_path, wide_path, coverage_path)],
    }
    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
