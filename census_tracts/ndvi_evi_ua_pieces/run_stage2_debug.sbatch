#!/bin/bash
# Stage 2 full run: NDVI worker on all 142 WorldCover tiles (process pool, one subprocess per tile, single retry),
# then tables, gates G1-G6 (stop at the first failure) and summary statistics.
# Sizing (qa/stage1_benchmark.json): 4,360 windows x ~1 s + ~3 s/tile -> ~1.3 core-hours; peak ~4 GB per worker.
# Run on the debug QOS (whole node, 32 workers): the shared-QOS job 59748158 waited 15 min for resources.
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J ndvi_stage2
#SBATCH -o /path/to/scratch/urbveg/ndvi/logs/slurm-%j.out
set -o pipefail
cd /path/to/scratch/urbveg/ndvi/scripts
source ../../scripts/env.sh
T() { echo "=== $* $(date -u +%T)"; /usr/bin/time -v "$@" 2>&1 | grep -v -E '^\s+(Average|Exit|File|Major|Minor|Page|Percent|Signals|Socket|Swaps|System|User|Voluntary|Involuntary|Command)'; }
T python -W ignore ndvi_run_all.py --procs 32 || exit 1
T python -W ignore ndvi_tables.py || exit 1
T python -W ignore stage2_gates.py || exit 2
T python -W ignore summary_stats.py
echo "=== done $(date -u +%T)"
