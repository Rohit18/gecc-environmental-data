#!/usr/bin/env python3

import unittest

import submit_census_tract_glc_exports as workflow


class CensusTractGlcWorkflowTests(unittest.TestCase):
    def test_expected_years(self) -> None:
        self.assertEqual(len(workflow.EXPECTED_YEARS), 26)
        self.assertEqual(workflow.EXPECTED_YEARS[:3], (1985, 1990, 1995))
        self.assertEqual(workflow.EXPECTED_YEARS[-1], 2022)

    def test_parse_int_ranges(self) -> None:
        self.assertEqual(
            workflow.parse_int_ranges("1985,1990,2000-2002,2001"),
            [1985, 1990, 2000, 2001, 2002],
        )
        with self.assertRaises(ValueError):
            workflow.parse_int_ranges("2022-2020")

    def test_task_description_is_stable(self) -> None:
        self.assertEqual(
            workflow.task_description(2022, 0, 10),
            "GLC_CensusTract_class_areas_2022_b000of010",
        )

    def test_output_schema_contains_raw_and_derived_metrics(self) -> None:
        selectors = workflow.output_selectors()
        self.assertIn("GEOID", selectors)
        self.assertIn("vegetation_fraction", selectors)
        self.assertIn("tree_fraction", selectors)
        self.assertIn("area_class_190", selectors)
        self.assertEqual(len(selectors), len(set(selectors)))

    def test_build_jobs(self) -> None:
        jobs = workflow.build_jobs([2021, 2022], [0, 3], 10)
        self.assertEqual(len(jobs), 4)
        self.assertEqual((jobs[0].year, jobs[0].chunk), (2021, 0))
        self.assertEqual((jobs[-1].year, jobs[-1].chunk), (2022, 3))


if __name__ == "__main__":
    unittest.main()
