#!/bin/bash
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J gecc-water-finish
#SBATCH -o /path/to/scratch/gecc_tract_watermask_20261013/logs/finish-%j.out
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
python /path/to/scratch/gecc_tract_watermask_20261013/code/merge_validate.py chm
python /path/to/scratch/gecc_tract_watermask_20261013/code/merge_validate.py package
