#!/bin/bash
export DAYMET_ROOT="${DAYMET_ROOT:-/path/to/scratch/daymet_tract_temperature}"
export DAYMET_CODE="$DAYMET_ROOT/code"
export DAYMET_CFS=/path/to/project/gecc/daymet
export ENV_PREFIX="$DAYMET_ROOT/env/daymet"
export POLYGONS=/path/to/project/gecc/era5_land/inputs/CensusTract_2020_US_85528.parquet
export DAILY_DIR="$DAYMET_CFS/daily"
export STAGING_DIR="$DAYMET_ROOT/work/staging"
export COMPOSITE_DIR="$DAYMET_CFS/composites"
export MANIFEST="$DAYMET_CFS/manifests/daymet_manifest.json"
export OUT_DIR="$DAYMET_CFS/outputs"
export AUDIT_DIR="$DAYMET_CFS/audits"
export LOG_DIR="$DAYMET_ROOT/logs"
export DAYMET_YEARS="${DAYMET_YEARS:-1980-2025}"
export DAYMET_WORKERS="${DAYMET_WORKERS:-4}"
export DAYMET_SCRATCH_BUDGET_GB="${DAYMET_SCRATCH_BUDGET_GB:-150}"
export GDAL_CACHEMAX=256
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
activate_env() {
    export PATH="$ENV_PREFIX/bin:$PATH"
    export PROJ_DATA="$ENV_PREFIX/share/proj"
    export GDAL_DATA="$ENV_PREFIX/share/gdal"
}
