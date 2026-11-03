#!/usr/bin/env python3

import unittest

import submit_census_tract_ndvi_evi_exports as vi
import submit_census_tract_ndvi_evi_urban_rural_2020 as workflow


def preview_row(**overrides):
    """A consistent two-zone preview row and its v1 counterpart."""
    row = {}
    for index in workflow.INDICES:
        row.update({
            f"urban_{index}_count": 40, f"rural_{index}_count": 60,
            f"tract_{index}_count": 100, f"urban_{index}_mean": 0.2,
            f"rural_{index}_mean": 0.5, f"tract_{index}_mean": 0.38,
            f"tract_{index}_median": 0.4, f"tract_{index}_stdDev": 0.1,
        })
    # 110 pixels centred in the tract plus 8 straddling edge pixels.
    for band in ("pixel_count", "pixel_nonwater_count"):
        row.update({f"urban_{band}": 45, f"rural_{band}": 65, f"tract_{band}": 118})
    row["tract_centre_pixel_count"] = 110
    row["tract_clear_obs_mean"] = 7.5
    v1 = {f"{index}_{stat}": row[f"tract_{index}_{stat}"]
          for index in workflow.INDICES for stat in workflow.INDEX_STATS}
    v1["clear_obs_mean"] = 7.5
    row.update(overrides)
    return row, v1


class UrbanRuralWorkflowTests(unittest.TestCase):
    def test_task_names(self) -> None:
        self.assertEqual(workflow.task_description("jja", 0),
                         "CensusTract_NDVI_EVI_UR2020_v2_JJA_2020_b000")
        self.assertEqual(workflow.task_description("djf", 14, "100m_g02"),
                         "CensusTract_NDVI_EVI_UR2020_v2_DJFsameYear_2020_b014_100m_g02")
        # Distinct from v1 so v1's task bookkeeping never sees these tasks.
        self.assertFalse(workflow.TASK_PREFIX.startswith(vi.TASK_PREFIX))

    def test_jobs_expand_grouped_batches(self) -> None:
        jobs = workflow.build_jobs(["jja", "djf"], [0, 14],
                                   {14: ["30m_g01", "100m_g01"]})
        self.assertEqual(len(jobs), 6)
        self.assertIn(("djf", 14, "100m_g01"), jobs)
        self.assertIn(("jja", 0, None), jobs)
        all_chunks = list(vi.REGION_STATEFPS)
        alaska_groups = {14: ["30m_g01", "100m_g01", "100m_g02", "100m_g03"]}
        self.assertEqual(len(workflow.build_jobs(list(vi.SEASONS), all_chunks, alaska_groups)), 44)

    def test_selectors(self) -> None:
        fields = workflow.output_selectors()
        self.assertEqual(len(fields), len(set(fields)))
        self.assertEqual(fields[0], "GEOID")
        for zone in workflow.ZONES:
            for name in ("NDVI_mean", "EVI_median", "EVI_count", "clear_obs_mean",
                         "pixel_count", "pixel_nonwater_count"):
                self.assertIn(f"{zone}_{name}", fields)
        self.assertEqual(fields[-1], "tract_centre_pixel_count")
        self.assertEqual(len(fields), 11 + 3 * 11 + 1)

    def test_preview_scale_follows_alaska_tiers(self) -> None:
        self.assertEqual(workflow.preview_scale(14, 5_000), 30)
        self.assertEqual(workflow.preview_scale(14, 5_001), 100)
        self.assertEqual(workflow.preview_scale(12, 50_000), 30)
        self.assertEqual(workflow.reduction_crs(14), "EPSG:3338")
        self.assertEqual(workflow.reduction_crs(0), "EPSG:4326")

    def test_preview_check_passes_consistent_row(self) -> None:
        self.assertEqual(workflow.check_preview_row(*preview_row()), [])

    def test_preview_check_catches_partition_and_v1_mismatch(self) -> None:
        row, v1 = preview_row(rural_NDVI_count=61)
        self.assertTrue(any("NDVI_count" in p for p in workflow.check_preview_row(row, v1)))
        row, v1 = preview_row(rural_pixel_count=64)
        self.assertTrue(any("centre-in-tract" in p for p in workflow.check_preview_row(row, v1)))
        row, v1 = preview_row(tract_centre_pixel_count=119, rural_pixel_count=74)
        self.assertTrue(any("exceed" in p for p in workflow.check_preview_row(row, v1)))
        row, v1 = preview_row()
        v1["EVI_mean"] += 1e-6
        self.assertTrue(any("tract_EVI_mean" in p for p in workflow.check_preview_row(row, v1)))
        row, v1 = preview_row()
        v1["NDVI_median"] += 1e-3  # inside the histogram-median tolerance
        self.assertEqual(workflow.check_preview_row(row, v1), [])

    def test_preview_check_allows_empty_zone(self) -> None:
        empty = {f"urban_{index}_{stat}": None for index in workflow.INDICES
                 for stat in ("mean", "median", "stdDev")}
        empty.update({f"urban_{index}_count": 0 for index in workflow.INDICES})
        empty.update({"urban_pixel_count": 0, "urban_pixel_nonwater_count": 0})
        empty.update({f"rural_{index}_count": 100 for index in workflow.INDICES})
        empty.update({"rural_pixel_count": 110, "rural_pixel_nonwater_count": 110})
        self.assertEqual(workflow.check_preview_row(*preview_row(**empty)), [])
        row, v1 = preview_row(urban_NDVI_mean=None)
        self.assertTrue(any("null-ness" in p for p in workflow.check_preview_row(row, v1)))


if __name__ == "__main__":
    unittest.main()
