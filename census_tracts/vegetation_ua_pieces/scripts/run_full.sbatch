#!/bin/bash
# Stage 5 full run — DO NOT SUBMIT before user go-ahead.
# Sizing (manifests/run_cost_estimate.json): 4,327 windows with UA over 135 tiles, 0.27-0.37 s/window
# -> ~0.3-0.45 core-hours of work; peak ~4.8 GB per worker process. 32 worker processes on half a node in the
# shared QOS: expected wall ~5-10 min incl. reduce; charge well under 1 node-hour.
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 64
#SBATCH --mem=200G
#SBATCH -t 00:45:00
#SBATCH -J urbveg_full
#SBATCH -o /path/to/scratch/urbveg/logs/slurm-%j.out
cd /path/to/scratch/urbveg/scripts
source ./env.sh
python run_all.py --procs 32
