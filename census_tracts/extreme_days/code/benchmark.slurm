#!/bin/sh
#SBATCH --account=your_nersc_project
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=tract_stage3_benchmark
#SBATCH --output=/path/to/project/gecc/extreme_temperature_tract2020/stage3_full/logs/benchmark_%j.out
#SBATCH --error=/path/to/project/gecc/extreme_temperature_tract2020/stage3_full/logs/benchmark_%j.err
set -eu
export PROJ_DATA=/path/to/home/.conda/envs/gecc-lai/share/proj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GDAL_CACHEMAX=64
/bin/date -u
/bin/hostname
PYTHON=/path/to/home/.conda/envs/gecc-lai/bin/python
CODE=/path/to/project/gecc/extreme_temperature_tract2020/stage3_full/code/production.py
/usr/bin/time -v "$PYTHON" -I "$CODE" benchmark
