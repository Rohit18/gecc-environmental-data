#!/bin/bash
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J gecc-water-pilot
#SBATCH -o /path/to/scratch/gecc_tract_watermask_20261013/logs/pilot-%j.out
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
test -f /path/to/scratch/gecc_tract_watermask_20261013/inputs/WORLDCOVER_READY.json
python /path/to/scratch/gecc_tract_watermask_20261013/code/test_water_source.py
python /path/to/scratch/gecc_tract_watermask_20261013/code/pilot.py
