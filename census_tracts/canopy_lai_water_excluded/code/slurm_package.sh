#!/bin/bash
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J gecc-water-package
#SBATCH -o /path/to/scratch/gecc_tract_watermask_20261013/logs/package-%j.out
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
python /path/to/scratch/gecc_tract_watermask_20261013/code/merge_validate.py package
