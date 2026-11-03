#!/bin/sh
#SBATCH --account=your_nersc_project
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --time-min=00:30:00
#SBATCH --job-name=tract_v2_finalize
#SBATCH --output=/path/to/project/gecc/extreme_temperature_tract2020/stage3_full/runs/storage_v2/logs/finalize_%j.out
#SBATCH --error=/path/to/project/gecc/extreme_temperature_tract2020/stage3_full/runs/storage_v2/logs/finalize_%j.err
set -eu
export PROJ_DATA=/path/to/home/.conda/envs/gecc-lai/share/proj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GDAL_CACHEMAX=64
/usr/bin/time -v /path/to/home/.conda/envs/gecc-lai/bin/python -I /path/to/project/gecc/extreme_temperature_tract2020/stage3_full/runs/storage_v2/qa/storage_compatibility_v2.py finalize
/usr/bin/time -v /path/to/home/.conda/envs/gecc-lai/bin/python -I /path/to/project/gecc/extreme_temperature_tract2020/stage3_full/runs/storage_v2/qa/finish_report.py
