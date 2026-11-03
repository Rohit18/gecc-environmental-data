#!/usr/bin/env python3
"""Submit the 11 Alaska groups for JJA 1985 at 100 m, to compare with 30 m.

Production Alaska (batch 14) runs in 11 tract groups on EPSG:3338 at 30 m. The
30 m JJA 1985 groups cost 361.5 EECU-hours in total, about 11x the rest of the
country per km2. This test reruns the same groups, year and season with
identical masking, compositing and water mask, changing only the reduction
scale to 100 m, to measure (1) the compute saving and (2) how far each tract's
values move from the 30 m results. Output columns match production.

Task names use their own prefix so the production queue runner ignores them.
"""

import argparse

import ee

import submit_census_tract_ndvi_evi_exports as wf

PREFIX = "CensusTract_NDVI_EVI_AKtest100m_"
YEAR, SEASON, CHUNK, SCALE_M, TILE_SCALE = 1985, "jja", 14, 100, 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    ee.Initialize(project=wf.DEFAULT_PROJECT)
    groups = wf.chunk_groups(CHUNK)
    print(f"Alaska groups: {len(groups)} ({', '.join(str(len(g)) for g in groups)} tracts)")
    for group in range(1, len(groups) + 1):
        description = f"{PREFIX}JJA_{YEAR}_b{CHUNK:03d}_g{group:02d}"
        if not args.submit:
            print(f"  would start {description}")
            continue
        table = wf.seasonal_stats_table(
            wf.tract_source(CHUNK, groups, group),
            YEAR, SEASON, CHUNK, TILE_SCALE,
            wf.DEFAULT_MAX_PIXELS_PER_REGION, scale_m=SCALE_M,
        )
        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=description,
            folder=wf.DRIVE_FOLDER,
            fileNamePrefix=description,
            fileFormat="CSV",
            selectors=wf.output_selectors(),
        )
        task.start()
        print(f"STARTED: {description}")
    if not args.submit:
        print("Dry run only. Add --submit to start the tests.")


if __name__ == "__main__":
    main()
