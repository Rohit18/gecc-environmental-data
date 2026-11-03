#!/bin/bash
# Stage 3 full weights run .
# Sizing (qa/stage2/stage2_columbus.json): 4,327 UA windows over 135 tiles at ~0.48 s/window + ~3 s start per tile
# -> ~0.7 core-hours; peak 3.2 GB per worker at Columbus. 32 worker processes on half a node, shared QOS:
# expected wall ~5 min incl. reduce; charge well under 1 node-hour.
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 64
#SBATCH --mem=200G
#SBATCH -t 00:45:00
#SBATCH -J uhi_weights
#SBATCH -o /path/to/scratch/urbveg/uhi/logs/slurm-%j.out
cd /path/to/scratch/urbveg/uhi/scripts
source ../../scripts/env.sh
python uhi_run_all.py --procs 32
