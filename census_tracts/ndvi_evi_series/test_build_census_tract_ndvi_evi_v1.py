"""Scientific packaging boundaries: denominators, nulls, tiers and sensor dates."""
import pandas as pd
import pytest

import build_census_tract_ndvi_evi_v1 as package


def example():
    raw = pd.DataFrame([
        {"GEOID": "01001020100", "StFIPS": "01", "year": 2020, "season": "JJA", "batch": 8, "batch_label": "east_south_central", "regional_scene_count": 100, "glc_water_mask_year": 2020,
         "NDVI_mean": 0.6, "NDVI_median": 0.7, "NDVI_stdDev": 0.1, "NDVI_count": 4, "EVI_mean": 0.4, "EVI_median": 0.5, "EVI_stdDev": 0.1, "EVI_count": 3, "clear_obs_mean": 1.25},
        {"GEOID": "01001020200", "StFIPS": "01", "year": 2020, "season": "JJA", "batch": 8, "batch_label": "east_south_central", "regional_scene_count": 100, "glc_water_mask_year": 2020,
         "NDVI_mean": None, "NDVI_median": None, "NDVI_stdDev": None, "NDVI_count": 0, "EVI_mean": None, "EVI_median": None, "EVI_stdDev": None, "EVI_count": None, "clear_obs_mean": None},
    ])[package.wf.output_selectors()]
    # Reverse metadata to ensure joins use GEOID rather than input row order.
    meta = pd.DataFrame({"GEOID": ["01001020200", "01001020100"], "batch": [8, 8], "export_group": ["", ""], "pixel_nonwater_count_2020": [0, 10], "pixel_nonwater_count_2019": [999, 999]})
    return raw, meta


def test_year_matched_support_and_nullable_flags():
    raw, meta = example()
    df = package.prepare_frame(raw, meta, 2020, "JJA", 8, "")
    assert df.loc[0, "observed_fraction"] == 0.4
    assert df.loc[0, "low_coverage"] and df.loc[0, "single_scene"]
    assert pd.isna(df.loc[1, "observed_fraction"])
    assert pd.isna(df.loc[1, "low_coverage"]) and pd.isna(df.loc[1, "single_scene"])
    assert "pixel_nonwater_count_2019" not in df
    pd.testing.assert_frame_equal(df[raw.columns], raw, check_dtype=False)


def test_unobserved_with_positive_support_has_zero_fraction():
    raw, meta = example()
    meta.loc[meta.GEOID.eq("01001020200"), "pixel_nonwater_count_2020"] = 10
    df = package.prepare_frame(raw, meta, 2020, "JJA", 8, "")
    assert df.loc[1, "observed_fraction"] == 0
    assert df.loc[1, "low_coverage"]
    assert pd.isna(df.loc[1, "single_scene"])


def test_no_support_cannot_have_measurements():
    raw, meta = example()
    meta.loc[meta.GEOID.eq("01001020100"), "pixel_nonwater_count_2020"] = 0
    with pytest.raises(ValueError, match="support is zero"):
        package.prepare_frame(raw, meta, 2020, "JJA", 8, "")


def test_coverage_cannot_exceed_nonwater_support():
    raw, meta = example()
    meta.loc[meta.GEOID.eq("01001020100"), "pixel_nonwater_count_2020"] = 3
    with pytest.raises(ValueError, match="fraction outside"):
        package.prepare_frame(raw, meta, 2020, "JJA", 8, "")


def test_partial_mission_years_use_the_selected_season():
    assert package.sensor_fields(2012, "JJA")["eligible_satellites"] == "L7"
    assert package.sensor_fields(2012, "DJF_same_year")["eligible_satellites"] == "L5;L7"
    assert package.sensor_fields(2021, "JJA")["eligible_satellites"] == "L8"
    assert package.sensor_fields(2021, "DJF_same_year")["eligible_satellites"] == "L8;L9"
    assert package.sensor_fields(2016, "JJA")["l7_slc_off_era"]
    assert not package.sensor_fields(2017, "JJA")["l7_slc_off_era"]
