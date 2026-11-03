#!/bin/bash
set -euo pipefail
export PATH=/path/to/home/.conda/envs/gecc-lai/bin:$PATH
export PROJ_DATA=/path/to/home/.conda/envs/gecc-lai/share/proj
export GDAL_CACHEMAX=128
export GDAL_MAX_DATASET_POOL_SIZE=128
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
