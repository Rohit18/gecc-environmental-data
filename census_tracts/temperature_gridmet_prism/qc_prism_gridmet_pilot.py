#!/usr/bin/env python3
"""Quality-check GECC PRISM/GRIDMET pilot CSV exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PERIODS = ("annual", "JJA", "DJF")
VARIABLES = ("tmax", "tmin")
MEAN_FIELDS = tuple(f"{p}_{v}_mean" for p in PERIODS for v in VARIABLES)
COUNT_FIELDS = tuple(f"{p}_{v}_count" for p in PERIODS for v in VARIABLES)
STD_FIELDS = tuple(f"{p}_{v}_stdDev" for p in PERIODS for v in VARIABLES)

EXPECTED_FILES = {
    "Pilot_GRIDMET_CensusTracts_AirTemp_1980_2020_sameyear_native.csv": {
        "dataset": "GRIDMET",
        "geography": "CensusTracts",
        "id": "GEOID",
        "years": [1980, 1981, 2000, 2020],
    },
    "Pilot_GRIDMET_ZCTAs_AirTemp_1980_2020_sameyear_native.csv": {
        "dataset": "GRIDMET",
        "geography": "ZCTAs",
        "id": "GEOID20",
        "years": [1980, 1981, 2000, 2020],
    },
    "Pilot_PRISM_CensusTracts_AirTemp_1981_2020_sameyear_native.csv": {
        "dataset": "PRISM",
        "geography": "CensusTracts",
        "id": "GEOID",
        "years": [1981, 2000, 2020],
    },
    "Pilot_PRISM_ZCTAs_AirTemp_1981_2020_sameyear_native.csv": {
        "dataset": "PRISM",
        "geography": "ZCTAs",
        "id": "GEOID20",
        "years": [1981, 2000, 2020],
    },
}


def load_files(folder: Path) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for filename, config in EXPECTED_FILES.items():
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={config["id"]: "string"})
        loaded[filename] = frame
    return loaded


def check_file(filename: str, frame: pd.DataFrame) -> dict[str, object]:
    config = EXPECTED_FILES[filename]
    id_field = config["id"]
    years = config["years"]
    expected_columns = {
        id_field,
        "dataset",
        "geography_type",
        "year",
        "djf_convention",
        "missing_mean_count",
        "all_six_means_valid",
        *MEAN_FIELDS,
        *COUNT_FIELDS,
        *STD_FIELDS,
    }
    missing_columns = sorted(expected_columns - set(frame.columns))

    identifiers = frame[id_field]
    unique_ids = int(identifiers.nunique(dropna=False))
    expected_rows = unique_ids * len(years)
    duplicate_id_years = int(frame.duplicated([id_field, "year"]).sum())
    year_values = sorted(frame["year"].dropna().astype(int).unique().tolist())

    mean_values = frame[list(MEAN_FIELDS)].apply(pd.to_numeric, errors="coerce")
    count_values = frame[list(COUNT_FIELDS)].apply(pd.to_numeric, errors="coerce")
    std_values = frame[list(STD_FIELDS)].apply(pd.to_numeric, errors="coerce")

    order_violations = 0
    for period in PERIODS:
        order_violations += int(
            (frame[f"{period}_tmin_mean"] > frame[f"{period}_tmax_mean"]).sum()
        )

    expected_day_counts = frame["year"].astype(int).map(
        lambda year: 366 if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0) else 365
    )
    day_count_mismatches = int(
        (frame["annual_day_count"].astype(int) != expected_day_counts).sum()
        + (frame["JJA_day_count"].astype(int) != 92).sum()
        + (
            frame["DJF_day_count"].astype(int)
            != expected_day_counts.map(lambda days: 91 if days == 366 else 90)
        ).sum()
    )

    checks = {
        "dataset": config["dataset"],
        "geography": config["geography"],
        "rows": int(len(frame)),
        "unique_ids": unique_ids,
        "expected_rows": expected_rows,
        "years": year_values,
        "missing_columns": missing_columns,
        "duplicate_id_years": duplicate_id_years,
        "null_ids": int(identifiers.isna().sum()),
        "null_means": int(mean_values.isna().sum().sum()),
        "reported_missing_means": int(frame["missing_mean_count"].sum()),
        "invalid_all_six_flags": int((frame["all_six_means_valid"] != 1).sum()),
        "nonpositive_counts": int((count_values <= 0).sum().sum()),
        "negative_stddev": int((std_values < 0).sum().sum()),
        "temperature_order_violations": order_violations,
        "implausible_means": int(((mean_values < -60) | (mean_values > 55)).sum().sum()),
        "day_count_mismatches": day_count_mismatches,
        "dataset_label_mismatches": int((frame["dataset"] != config["dataset"]).sum()),
        "geography_label_mismatches": int(
            (frame["geography_type"] != config["geography"]).sum()
        ),
        "djf_label_mismatches": int((frame["djf_convention"] != "same-year").sum()),
        "fallback_rows": int(frame["fallback_applied"].astype(bool).sum()),
        "minimum_pixel_count": float(count_values.min().min()),
        "maximum_pixel_count": float(count_values.max().max()),
        "minimum_mean_c": float(mean_values.min().min()),
        "maximum_mean_c": float(mean_values.max().max()),
    }
    checks["passed"] = bool(
        len(frame) == expected_rows
        and year_values == years
        and not missing_columns
        and all(
            checks[key] == 0
            for key in (
                "duplicate_id_years",
                "null_ids",
                "null_means",
                "reported_missing_means",
                "invalid_all_six_flags",
                "nonpositive_counts",
                "negative_stddev",
                "temperature_order_violations",
                "implausible_means",
                "day_count_mismatches",
                "dataset_label_mismatches",
                "geography_label_mismatches",
                "djf_label_mismatches",
            )
        )
    )
    return checks


def cross_source_checks(frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for geography, id_field in (("CensusTracts", "GEOID"), ("ZCTAs", "GEOID20")):
        gridmet = next(
            frame
            for name, frame in frames.items()
            if "GRIDMET" in name and geography in name
        )
        prism = next(
            frame
            for name, frame in frames.items()
            if "PRISM" in name and geography in name
        )
        common_years = sorted(set(gridmet["year"]) & set(prism["year"]))
        gridmet = gridmet[gridmet["year"].isin(common_years)]
        prism = prism[prism["year"].isin(common_years)]
        merged = gridmet.merge(
            prism,
            on=[id_field, "year"],
            how="outer",
            suffixes=("_gridmet", "_prism"),
            indicator=True,
        )

        for field in MEAN_FIELDS:
            paired = merged.loc[
                merged["_merge"] == "both",
                [f"{field}_gridmet", f"{field}_prism"],
            ].dropna()
            gridmet_values = paired[f"{field}_gridmet"].astype(float)
            prism_values = paired[f"{field}_prism"].astype(float)
            difference = prism_values - gridmet_values
            results.append(
                {
                    "geography": geography,
                    "metric": field,
                    "n": int(len(paired)),
                    "mean_bias_prism_minus_gridmet_c": float(difference.mean()),
                    "mae_c": float(difference.abs().mean()),
                    "rmse_c": float(np.sqrt(np.mean(np.square(difference)))),
                    "max_abs_difference_c": float(difference.abs().max()),
                    "correlation": float(gridmet_values.corr(prism_values)),
                    "unmatched_rows": int((merged["_merge"] != "both").sum()),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    frames = load_files(args.folder.resolve())
    file_checks = {
        filename: check_file(filename, frame)
        for filename, frame in frames.items()
    }
    cross_checks = cross_source_checks(frames)
    summary = {
        "all_files_passed": all(item["passed"] for item in file_checks.values()),
        "file_checks": file_checks,
        "cross_source_checks": cross_checks,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
