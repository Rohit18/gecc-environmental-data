#!/usr/bin/env python3

import tempfile
import unittest
import zipfile
from pathlib import Path

import submit_census_tract_ndvi_evi_exports as workflow


class CensusTractVegetationIndexWorkflowTests(unittest.TestCase):
    def test_completed_years_exclude_partial_2026(self) -> None:
        self.assertEqual(workflow.EXPECTED_YEARS[0], 1985)
        self.assertEqual(workflow.EXPECTED_YEARS[-1], 2025)
        self.assertEqual(len(workflow.EXPECTED_YEARS), 41)
        self.assertNotIn(2026, workflow.EXPECTED_YEARS)

    def test_every_state_has_one_spatial_batch(self) -> None:
        mapping = workflow.state_to_chunk_mapping()
        self.assertEqual(len(mapping), 56)
        self.assertEqual(mapping["06"], 12)
        self.assertEqual(mapping["48"], 9)
        self.assertEqual(mapping["41"], 13)
        self.assertEqual(mapping["02"], 14)
        self.assertEqual(mapping["15"], 15)
        self.assertEqual(mapping["72"], 16)
        self.assertEqual(mapping["78"], 16)
        self.assertEqual(mapping["66"], 17)
        self.assertEqual(mapping["69"], 17)
        self.assertEqual(mapping["60"], 18)
        territory_chunks = {mapping[c] for c in workflow.TERRITORY_STATEFPS}
        for chunk in territory_chunks:
            self.assertTrue(
                set(workflow.REGION_STATEFPS[chunk]) <= set(workflow.TERRITORY_STATEFPS)
            )
        self.assertEqual(workflow.EXPECTED_TRACT_COUNT, 85528)
        self.assertEqual(set(workflow.REGION_LABELS), set(workflow.REGION_STATEFPS))

    def test_season_date_ranges(self) -> None:
        self.assertEqual(
            workflow.season_date_ranges(2022, "jja"),
            [("2022-06-01", "2022-09-01")],
        )
        self.assertEqual(
            workflow.season_date_ranges(2022, "djf"),
            [("2022-01-01", "2022-03-01"), ("2022-12-01", "2023-01-01")],
        )
        with self.assertRaises(ValueError):
            workflow.season_date_ranges(2022, "mam")

    def test_parse_helpers(self) -> None:
        self.assertEqual(
            workflow.parse_int_ranges("1985,2000-2002,2001"),
            [1985, 2000, 2001, 2002],
        )
        self.assertEqual(workflow.parse_csv("JJA, djf"), ["jja", "djf"])
        with self.assertRaises(ValueError):
            workflow.parse_int_ranges("2022-2020")

    def test_task_names(self) -> None:
        self.assertEqual(
            workflow.task_description(2022, "jja", 0),
            "CensusTract_NDVI_EVI_v1_JJA_2022_b000",
        )
        self.assertEqual(
            workflow.task_description(2022, "djf", 15),
            "CensusTract_NDVI_EVI_v1_DJFsameYear_2022_b015",
        )

    def test_combined_output_schema(self) -> None:
        selectors = workflow.output_selectors()
        self.assertIn("NDVI_mean", selectors)
        self.assertIn("NDVI_count", selectors)
        self.assertIn("EVI_mean", selectors)
        self.assertIn("EVI_count", selectors)
        self.assertIn("clear_obs_mean", selectors)
        self.assertEqual(len(selectors), len(set(selectors)))

    def test_glc_water_mask_years(self) -> None:
        self.assertEqual(workflow.GLC_EXCLUDED_CLASSES, (210, 187))
        self.assertEqual(len(workflow.GLC_MAP_YEARS), 26)
        cases = {1985: 1985, 1989: 1985, 1990: 1990, 1999: 1995,
                 2000: 2000, 2013: 2013, 2022: 2022, 2025: 2022}
        for year, map_year in cases.items():
            self.assertEqual(workflow.glc_map_year(year), map_year)
        for year in workflow.EXPECTED_YEARS:
            self.assertIn(workflow.glc_map_year(year), workflow.GLC_MAP_YEARS)
        with self.assertRaises(ValueError):
            workflow.glc_map_year(1984)
        self.assertIn("glc_water_mask_year", workflow.output_selectors())
        totals = workflow.pixel_totals_selectors()
        self.assertIn("pixel_nonwater_count_1985", totals)
        self.assertIn("pixel_nonwater_count_2022", totals)
        self.assertIn("pixel_glc_gap_count", totals)
        self.assertEqual(len(totals), 4 + 26)
        self.assertEqual(workflow.WORLDCOVER_WATER_CLASS, 80)

    def test_plan_tasks_skips_and_limits_retries(self) -> None:
        states = {
            "a": ["COMPLETED"],
            "b": ["FAILED"],
            "c": ["FAILED", "FAILED"],
            "d": ["FAILED", "RUNNING"],
        }
        done, available, retries, given_up = workflow.plan_tasks(
            ["a", "b", "c", "d", "e", "f"], states, {"f"}
        )
        self.assertEqual(done, ["a", "d", "f"])
        self.assertEqual(available, ["b", "e"])
        self.assertEqual(retries, {"b"})
        self.assertEqual(given_up, ["c"])
        _, available, retries, given_up = workflow.plan_tasks(
            ["c"], states, set(), retry_given_up=True
        )
        self.assertEqual((available, retries, given_up), (["c"], {"c"}, []))

    def test_completed_descriptions_reads_zips_and_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "loose.csv").write_text("x")
            for i, name in enumerate(["one", "two"]):
                with zipfile.ZipFile(root / f"drive-1-00{i}.zip", "w") as z:
                    z.writestr(f"folder/{name}.csv", "x")
            self.assertEqual(
                workflow.completed_descriptions([tmp]), {"loose", "one", "two"}
            )
            self.assertEqual(
                workflow.completed_descriptions([str(root / "drive-1-001.zip")]),
                {"two"},
            )
        self.assertEqual(workflow.completed_descriptions(None), set())

    def test_pixel_totals_descriptions(self) -> None:
        self.assertEqual(
            workflow.pixel_totals_description(16),
            "CensusTract_NDVI_EVI_v1_pixel_totals_30m_b016",
        )
        self.assertTrue(
            workflow.pixel_totals_description(0).startswith(workflow.TASK_PREFIX)
        )

    def test_alaska_grid(self) -> None:
        self.assertEqual(workflow.reduce_grid_kwargs(14), {"crs": "EPSG:3338"})
        self.assertEqual(workflow.reduce_grid_kwargs(0), {})
        self.assertEqual(
            workflow.pixel_totals_description(14),
            "CensusTract_NDVI_EVI_v1_pixel_totals_30m_b014_epsg3338",
        )

    def test_alaska_groups(self) -> None:
        rows = [("02001", 200_000.0), ("02002", 100_000.0), ("02003", 1.0)] + [
            (f"021{i:02d}", 1.0) for i in range(30)
        ]
        groups = workflow.group_tracts(rows)
        self.assertEqual(groups[0], ["02001"])
        self.assertEqual(len(groups[1]), 25)
        self.assertEqual(sum(len(g) for g in groups), len(rows))
        self.assertEqual(workflow.group_tracts(list(reversed(rows))), groups)

    def test_alaska_size_tiers(self) -> None:
        rows = [("02001", 200_000.0), ("02002", 6_000.0), ("02003", 5_000.0)] + [
            (f"021{i:02d}", 1.0) for i in range(30)
        ]
        keyed = workflow.tiered_groups(14, rows)
        self.assertEqual(list(keyed), ["30m_p01", "30m_p02", "100m_p01"])  # 31 small tracts -> 25 + 6
        # 200,000 + 6,000 km2 fit under the 250,000 km2 group limit together
        self.assertEqual(keyed["100m_p01"], ["02001", "02002"])
        self.assertIn("02003", keyed["30m_p01"])  # exactly 5,000 km2 stays at 30 m
        self.assertEqual(sum(len(g) for g in keyed.values()), len(rows))
        self.assertEqual(workflow.group_scale("100m_p02"), 100)
        self.assertEqual(workflow.group_scale("30m_p01"), 30)
        self.assertEqual(workflow.group_scale(None), 30)
        self.assertEqual(list(workflow.tiered_groups(3, rows))[0], "30m_p01")
        jobs = workflow.build_jobs([1985], ["jja"], [13, 14], {14: ["30m_p01", "100m_p01"]})
        self.assertEqual(
            [j.description for j in jobs],
            [
                "CensusTract_NDVI_EVI_v1_JJA_1985_b013",
                "CensusTract_NDVI_EVI_v1_JJA_1985_b014_30m_p01",
                "CensusTract_NDVI_EVI_v1_JJA_1985_b014_100m_p01",
            ],
        )
        self.assertEqual(
            workflow.pixel_totals_description(14, "100m_p02"),
            "CensusTract_NDVI_EVI_v1_pixel_totals_b014_100m_p02_epsg3338",
        )
        self.assertEqual(
            workflow.pixel_totals_description(3),
            "CensusTract_NDVI_EVI_v1_pixel_totals_30m_b003",
        )

    def test_alaska_deferred_by_default(self) -> None:
        self.assertEqual(workflow.DEFERRED_CHUNKS, {14})
        self.assertIn(14, workflow.REGION_STATEFPS)

    def test_build_jobs(self) -> None:
        jobs = workflow.build_jobs([2022], ["jja", "djf"], [0, 1])
        self.assertEqual(len(jobs), 4)
        self.assertEqual((jobs[0].season, jobs[0].chunk), ("jja", 0))
        self.assertEqual((jobs[-1].season, jobs[-1].chunk), ("djf", 1))


if __name__ == "__main__":
    unittest.main()
