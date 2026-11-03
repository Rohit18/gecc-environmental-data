#!/bin/bash
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J gecc-water-chm-merge
#SBATCH -o /path/to/scratch/gecc_tract_watermask_20261013/logs/chm-merge-%j.out
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
python /path/to/scratch/gecc_tract_watermask_20261013/code/merge_validate.py chm
