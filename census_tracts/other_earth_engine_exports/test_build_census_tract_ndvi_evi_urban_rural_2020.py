#!/usr/bin/env python3

import math
import unittest

import pandas as pd

import build_census_tract_ndvi_evi_urban_rural_2020 as build
import submit_census_tract_ndvi_evi_urban_rural_2020 as ur


def export_row(geoid: str, urban_n: int, rural_n: int, **overrides) -> dict:
    """One GEE export row: zone counts as given, 4 straddling edge pixels."""
    row = {col: None for col in ur.output_selectors()}
    row.update({"GEOID": geoid, "StFIPS": geoid[:2], "year": 2020, "season": "JJA",
                "batch": 0, "batch_label": "new_england", "tract_group": None,
                "reduction_scale_m": 30, "reduction_crs": "EPSG:4326",
                "regional_scene_count": 100, "glc_water_mask_year": 2020})
    for zone, n, mean in (("urban", urban_n, 0.3), ("rural", rural_n, 0.5),
                          ("tract", urban_n + rural_n + 4, 0.42)):
        for index in ur.INDICES:
            row[f"{zone}_{index}_count"] = n
            for stat, value in (("mean", mean), ("median", mean), ("stdDev", 0.1)):
                row[f"{zone}_{index}_{stat}"] = value if n else None
        row[f"{zone}_clear_obs_mean"] = 5.0 if n else None
        row[f"{zone}_pixel_count"] = n
        row[f"{zone}_pixel_nonwater_count"] = n
    row["tract_centre_pixel_count"] = urban_n + rural_n
    row.update(overrides)
    return row


def meta_for(geoids) -> pd.DataFrame:
    return pd.DataFrame({
        "urban_zone_area_m2": 1.0, "rural_zone_area_m2": 1.0, "ua_ids": "00001",
        "ua_count": 1, "primary_ua_id": "00001", "primary_ua_name": "A",
        "ua_match_status": "matched",
    }, index=pd.Index(list(geoids), name="GEOID"))


class PackagingTests(unittest.TestCase):
    def test_expected_exports(self) -> None:
        names = build.expected_exports()
        self.assertEqual(len(names), 44)
        self.assertIn("CensusTract_NDVI_EVI_UR2020_v2_DJFsameYear_2020_b014_100m_g03", names)
        self.assertTrue(all(build.EXPORT_RE.match(n) for n in names))

    def test_derive_differences_and_flags(self) -> None:
        df = pd.DataFrame([export_row("09001000100", 100, 50),
                           export_row("09001000200", 100, 3),
                           export_row("09001000300", 80, 0)])
        out = build.derive(df, meta_for(df.GEOID)).set_index("GEOID")
        self.assertAlmostEqual(out.loc["09001000100", "NDVI_mean_diff"], -0.2)
        self.assertEqual(out.loc["09001000100", "zone_split"], "both")
        self.assertFalse(out.loc["09001000100", "small_zone_flag"])
        self.assertTrue(out.loc["09001000200", "small_zone_flag"])
        self.assertEqual(out.loc["09001000300", "zone_split"], "urban_only")
        self.assertTrue(math.isnan(out.loc["09001000300", "EVI_median_diff"]))
        self.assertFalse(out.loc["09001000300", "small_zone_flag"])
        self.assertEqual(out.loc["09001000300", "urban_observed_fraction"], 1.0)
        self.assertTrue(math.isnan(out.loc["09001000300", "rural_observed_fraction"]))
        self.assertEqual(list(out.columns[:3]), ["STATEFP", "year", "season"])

    def test_checks_pass_clean_rows(self) -> None:
        df = pd.DataFrame([export_row("09001000100", 100, 50), export_row("09001000300", 80, 0)])
        problems: list[str] = []
        build.check_exports(df, set(df.GEOID), problems)
        self.assertEqual(problems, [])

    def test_checks_catch_partition_and_nulls(self) -> None:
        df = pd.DataFrame([export_row("09001000100", 100, 50, tract_centre_pixel_count=151),
                           export_row("09001000300", 80, 0, rural_NDVI_mean=0.1)])
        problems: list[str] = []
        build.check_exports(df, set(df.GEOID) | {"09001000400"}, problems)
        text = " ".join(problems)
        self.assertIn("tract_centre_pixel_count", text)
        self.assertIn("rural_NDVI_mean: null", text)
        self.assertIn("universe GEOIDs missing", text)


if __name__ == "__main__":
    unittest.main()
