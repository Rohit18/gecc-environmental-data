# GECC Environmental Data

Processing code for environmental datasets produced for the Gateway Exposome Coordinating Center (GECC) (https://gatewayexposome.org/). Code is organised by geography:

| Folder | Geography | Status |
| --- | --- | --- |
| [census_tracts/](census_tracts/) | 2020 U.S. Census tracts and tract-Urban Area overlaps | GECC Census Tract Environmental Data, release 1.0 |

Code for Functional Urban Areas (FUAs), ZIP Code Tabulation Areas (ZCTAs) and City Health Dashboard cities will be added in their own folders as those datasets are released.

The repository contains code only. Source data and outputs are not included. Each folder's README describes its scripts, inputs and outputs.

## Running the code

The Python and JavaScript scripts were run on Google Earth Engine (Python API and Code Editor) and on the NERSC Perlmutter system. They are provided to document the methods. Paths and project names that were specific to the original machines are replaced by placeholders such as `/path/to/GECC`, `/path/to/project`, `/path/to/scratch` and `your-ee-project`. Set these to your own locations, Earth Engine project and NERSC allocation before running a script. Source datasets must be obtained from their providers; the dataset documentation lists them.

## Citation

Cite the dataset you use by its DOI and version, and cite the source products listed in its documentation. To cite this code, cite this repository (https://github.com/Rohit18/gecc-environmental-data).

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The datasets are licensed separately (GECC Census Tract Environmental Data: CC BY 4.0).

## Acknowledgment

This work was funded as a pilot project under the Gateway Exposome Coordinating Center (GECC). This work used analytical resources developed and disseminated by the Gateway Exposome Coordinating Center (GECC) (https://gatewayexposome.org/). The GECC is funded by the National Institute on Aging (NIA), under award U24AG088894. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a Department of Energy User Facility. Several datasets were processed in Google Earth Engine (Gorelick, N. et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sensing of Environment 202, 18-27. DOI 10.1016/j.rse.2017.06.031).

Authors: TC Chakraborty; Rohit Mukherjee. Contact: TC Chakraborty (tirthankar.chakraborty@pnnl.gov).
