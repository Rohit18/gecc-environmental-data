#!/bin/bash
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --array=0-7%4
#SBATCH -J gecc-water-chm1m
#SBATCH -o /path/to/scratch/gecc_tract_watermask_20261013/logs/chm-%A_%a.out
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
test -f /path/to/scratch/gecc_tract_watermask_20261013/validation/PILOT_PASSED.json
python /path/to/scratch/gecc_tract_watermask_20261013/code/extract.py chm --workers 24 --shards 8 --shard "$SLURM_ARRAY_TASK_ID"
