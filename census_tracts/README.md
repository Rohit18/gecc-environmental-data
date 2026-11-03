# Census tracts

Code for GECC Census Tract Environmental Data, release 1.0: 12 environmental metric tables for 2020 Census tracts and tract-Urban Area overlaps in the United States and territories. The dataset's codebook describes each table's method, definitions and limitations.

| Released table | Folder | Platform |
| --- | --- | --- |
| temperature_svcm | [temperature_svcm/](temperature_svcm/) | Python, NERSC |
| temperature_gridmet, temperature_prism | [temperature_gridmet_prism/](temperature_gridmet_prism/) | Earth Engine, Python |
| temperature_daymet | [temperature_daymet/](temperature_daymet/) | Python, NERSC |
| extreme_days | [extreme_days/](extreme_days/) | Python, NERSC |
| land_cover_glc | [land_cover_glc/](land_cover_glc/) | Earth Engine, Python |
| ndvi_evi_series | [ndvi_evi_series/](ndvi_evi_series/) | Earth Engine, Python |
| ndvi_evi_ua_pieces | [ndvi_evi_ua_pieces/](ndvi_evi_ua_pieces/) | Earth Engine composites, Python on NERSC |
| uhi_ua_pieces | [uhi_ua_pieces/](uhi_ua_pieces/) with [overlap_geometry/](overlap_geometry/) | Python, NERSC |
| canopy_water_excluded, lai_water_excluded | [canopy_lai_water_excluded/](canopy_lai_water_excluded/) | Python, NERSC |
| vegetation_ua_pieces | [vegetation_ua_pieces/](vegetation_ua_pieces/) (also builds the FUA grid variant) | Python, NERSC |

Other folders:

- [overlap_geometry/](overlap_geometry/): tract-Urban Area overlap geometry, rural references and ESA WorldCover fractions used by the overlap tables.
- [other_earth_engine_exports/](other_earth_engine_exports/): supporting Earth Engine exports, including WorldCover summaries, territory tract boundaries, Earth Engine LAI statistics, the SVCM-SP 2022-2024 source extension and an earlier urban-rural NDVI/EVI version.
- [release_packaging/](release_packaging/): scripts that assembled the release, including the GRIDMET/PRISM split, the column dictionary, the codebook and collection guide PDFs, wording corrections, redundant-column removal and the text documentation.

Inputs are the source products cited in the codebook and the 2020 TIGER/Line tract and 2025 TIGER/Line 2020 Urban Area boundaries. Paths are placeholders (see the repository README). Several NERSC workflows were run in stages with validation checks between them; each folder's scripts follow that order.
