#!/bin/bash
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J gecc-water-fetch
#SBATCH -o /path/to/scratch/gecc_tract_watermask_20261013/logs/fetch-%j.out
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
python /path/to/scratch/gecc_tract_watermask_20261013/code/fetch_missing.py
