#!/usr/bin/env python3
"""Submit two Alaska NDVI/EVI test exports (JJA 1985) to choose an Alaska approach.

Alaska batch-14 tasks ran out of memory at tileScale 4 and 16 on the default
EPSG:4326 grid. These tests use the production masking, compositing, and water
mask from submit_census_tract_ndvi_evi_exports.py and differ only in which
tracts are reduced and on which grid:

A: the largest tract (02185000200, ~234,600 km2) alone, EPSG:3338 at 30 m.
B: 25 Anchorage tracts (02020...), default grid (as production), tileScale 16.

Each row also has nonwater_count from the same reduction, so NDVI_count /
nonwater_count is the observed fraction on that grid. Task names use their own
prefix so the production queue runner ignores them.
"""

import argparse

import ee

import submit_census_tract_ndvi_evi_exports as wf

PREFIX = "CensusTract_NDVI_EVI_AKtest_"
YEAR, SEASON = 1985, "jja"
TESTS = {
    "A_largest_tract_EPSG3338": {"crs": "EPSG:3338"},
    "B_anchorage_25_default_grid": {"crs": None},
}


def test_tracts(name: str) -> ee.FeatureCollection:
    alaska = wf.chunk_source(14)
    if name.startswith("A_"):
        return alaska.filter(ee.Filter.eq(wf.ID_FIELD, "02185000200"))
    return (
        alaska.filter(ee.Filter.stringStartsWith(wf.ID_FIELD, "02020"))
        .sort(wf.ID_FIELD)
        .limit(25)
    )


def test_table(name: str, crs: str | None) -> ee.FeatureCollection:
    tracts = test_tracts(name)
    region = tracts.geometry(maxError=1000)
    composite, scene_count = wf.seasonal_composite(region, YEAR, SEASON)
    keep = wf.not_water_mask(wf.glc_map_year(YEAR))
    image = composite.updateMask(keep).select(["NDVI", "EVI", "clear_obs"]).addBands(
        ee.Image.constant(1).updateMask(keep).rename("nonwater")
    )
    kwargs = dict(
        collection=tracts,
        reducer=wf.stats_reducer(),
        scale=wf.SCALE_M,
        tileScale=16,
        maxPixelsPerRegion=wf.DEFAULT_MAX_PIXELS_PER_REGION,
    )
    if crs:
        kwargs["crs"] = crs
    stats = image.reduceRegions(**kwargs)
    return stats.map(
        lambda f: ee.Feature(
            None,
            {
                wf.ID_FIELD: f.get(wf.ID_FIELD),
                "test": name,
                "crs": crs or "default (EPSG:4326 grid)",
                "regional_scene_count": scene_count,
                "NDVI_mean": f.get("NDVI_mean"),
                "NDVI_median": f.get("NDVI_median"),
                "NDVI_count": f.get("NDVI_count"),
                "EVI_mean": f.get("EVI_mean"),
                "clear_obs_mean": f.get("clear_obs_mean"),
                "nonwater_count": f.get("nonwater_count"),
            },
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    ee.Initialize(project=wf.DEFAULT_PROJECT)
    for name, opts in TESTS.items():
        n = test_tracts(name).size().getInfo()
        description = f"{PREFIX}{name}_JJA_{YEAR}"
        print(f"{description}: {n} tract(s), crs={opts['crs'] or 'default'}")
        if not args.submit:
            continue
        task = ee.batch.Export.table.toDrive(
            collection=test_table(name, opts["crs"]),
            description=description,
            folder=wf.DRIVE_FOLDER,
            fileNamePrefix=description,
            fileFormat="CSV",
        )
        task.start()
        print(f"STARTED: {description} ({task.id})")
    if not args.submit:
        print("Dry run only. Add --submit to start the tests.")


if __name__ == "__main__":
    main()
