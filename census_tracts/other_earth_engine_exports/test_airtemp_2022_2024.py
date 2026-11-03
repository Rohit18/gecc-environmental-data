#!/usr/bin/env python3
"""Local unit tests for the 2022-2024 air-temperature workflow."""

from __future__ import annotations

import unittest

import airtemp_2022_2024_common as common


class AirTemperatureWorkflowTests(unittest.TestCase):
    def test_parse_years(self) -> None:
        self.assertEqual(common.parse_years("2024,2022-2023,2022"), [2022, 2023, 2024])

    def test_expected_day_counts(self) -> None:
        self.assertEqual(common.expected_day_count(2022, "jja"), 92)
        self.assertEqual(common.expected_day_count(2022, "djf"), 90)
        self.assertEqual(common.expected_day_count(2024, "djf"), 91)

    def test_expected_2024_djf_dates(self) -> None:
        dates = common.expected_dates(2024, "djf")
        self.assertEqual(len(dates), 91)
        self.assertIn("2024-02-29", dates)
        self.assertIn("2024-12-31", dates)

    def test_established_task_names(self) -> None:
        self.assertEqual(
            common.task_description("tracts", "jja", 2022),
            "CensusTract_AirTemp_JJA_2022",
        )
        self.assertEqual(
            common.task_description("tracts", "djf", 2022),
            "CensusTract_AirTemp_DJF_2022",
        )
        self.assertEqual(
            common.task_description("zctas", "jja", 2022),
            "ZCTA_AirTemp_JJA_2022",
        )
        self.assertEqual(
            common.task_description("zctas", "djf", 2022),
            "ZCTA_AirTemp_DJF_2022",
        )
        self.assertEqual(
            common.task_description("fuas", "jja", 2022),
            "FUA_AirTemp_JJA_2022",
        )
        self.assertEqual(
            common.task_description("fuas", "djf", 2022),
            "FUA_AirTemp_DJFsameYear_2022",
        )


if __name__ == "__main__":
    unittest.main()
