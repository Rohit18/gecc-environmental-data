#!/bin/bash
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J gecc-water-lai
#SBATCH -o /path/to/scratch/gecc_tract_watermask_20261013/logs/lai-%j.out
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
test -f /path/to/scratch/gecc_tract_watermask_20261013/validation/PILOT_PASSED.json
python /path/to/scratch/gecc_tract_watermask_20261013/code/extract.py lai --workers 32
python /path/to/scratch/gecc_tract_watermask_20261013/code/merge_validate.py lai
