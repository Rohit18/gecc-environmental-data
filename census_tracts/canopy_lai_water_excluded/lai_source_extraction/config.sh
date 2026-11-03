#!/bin/bash
# Shared paths for the 2020 Census Tract LAI workflow (CONUS). Source this file, don't run it.
# Every path can be overridden from the environment before sourcing.

export LAI_ACCOUNT="${LAI_ACCOUNT:-your_nersc_project}"
export LAI_CODE="${LAI_CODE:-$HOME/GECC/census_tract_lai}"
export LAI_ROOT="${LAI_ROOT:-$PSCRATCH/gecc_census_tract_lai}"
export ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/gecc-lai}"

# Inputs: the downloaded native-grid CONUS LAI tiles, their VRT, and the 2020 tracts.
export LAI_DOWNLOAD="${LAI_DOWNLOAD:-$PSCRATCH/lai_global_mosaic_2020}"
export LAI_TILES_DIR="${LAI_TILES_DIR:-$LAI_DOWNLOAD/tiles}"
export LAI_TILE_LIST="${LAI_TILE_LIST:-$LAI_DOWNLOAD/conus_tiles_2048.txt}"
export LAI_VRT="${LAI_VRT:-$LAI_DOWNLOAD/conus.vrt}"
export TRACTS="${TRACTS:-$HOME/GECC/Census_2020_US_Tract_Polygons/Census_2020_US_Tract_Polygons.shp}"
export EE_EXPORTS="${EE_EXPORTS:-$LAI_ROOT/inputs/ee_exports}"

# Optional Earth Engine reference sample (05_ee_reference_sample.py).
export EE_PROJECT="${EE_PROJECT:-your-ee-project}"
export LAI_LEDGER="${LAI_LEDGER:-$LAI_DOWNLOAD/eecu_ledger_your-ee-project.jsonl}"

export WORK="$LAI_ROOT/work"
export MANIFEST_DIR="$WORK/manifest"
export SHARD_DIR="$WORK/shards"
export QA_DIR="$WORK/qa"
export OUT_DIR="$LAI_ROOT/outputs"
export LOG_DIR="$LAI_ROOT/logs"
export N_SHARDS="${N_SHARDS:-128}"

export GDAL_CACHEMAX="${GDAL_CACHEMAX:-1024}"
export GDAL_MAX_DATASET_POOL_SIZE="${GDAL_MAX_DATASET_POOL_SIZE:-2000}"
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

activate_env() {
    module load conda
    conda activate "$ENV_PREFIX"
}
