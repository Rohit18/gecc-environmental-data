#!/bin/bash
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J gecc-water-chm-pilot
#SBATCH -o /path/to/scratch/gecc_tract_watermask_20261013/logs/chm-benchmark-%j.out
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
test -f /path/to/scratch/gecc_tract_watermask_20261013/validation/PILOT_PASSED.json
python /path/to/scratch/gecc_tract_watermask_20261013/code/extract.py chm --workers 1 --limit 1
