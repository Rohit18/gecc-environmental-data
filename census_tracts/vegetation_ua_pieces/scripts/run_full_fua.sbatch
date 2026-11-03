#!/bin/bash
# Full run, profile fua_fishnet. Sizing: 20,562 windows with an FUA over 1,151 tiles; 0.45-0.53 s/window
# -> ~2.6-3.0 core-hours; ~4.5 GB per worker -> 96 workers on one node (~430 GB). Expected wall < 15 min.
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J urbveg_fua
#SBATCH -o /path/to/scratch/urbveg/logs/slurm-fua-%j.out
cd /path/to/scratch/urbveg/scripts
source ./env.sh
export URBVEG_RUN=fua_fishnet
python run_all.py --procs 96 && python finalize.py
