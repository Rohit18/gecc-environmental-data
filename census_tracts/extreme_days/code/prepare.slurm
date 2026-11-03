#!/bin/sh
#SBATCH --account=your_nersc_project
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --time-min=00:30:00
#SBATCH --job-name=tract_stage3_prepare
#SBATCH --output=/path/to/project/gecc/extreme_temperature_tract2020/stage3_full/logs/prepare_%j.out
#SBATCH --error=/path/to/project/gecc/extreme_temperature_tract2020/stage3_full/logs/prepare_%j.err
set -eu
export PROJ_DATA=/path/to/home/.conda/envs/gecc-lai/share/proj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GDAL_CACHEMAX=64
/bin/date -u
/bin/hostname
/usr/bin/time -v /path/to/home/.conda/envs/gecc-lai/bin/python -I /path/to/project/gecc/extreme_temperature_tract2020/stage3_full/code/prepare_production.py
