#!/bin/bash
# Stage 1 test runs: NDVI worker on the six test tiles (one process per tile), then gates and draft tables.
#SBATCH -A your_nersc_project
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J ndvi_stage1
#SBATCH -o /path/to/scratch/urbveg/ndvi/logs/slurm-%j.out
cd /path/to/scratch/urbveg/ndvi/scripts
source ../../scripts/env.sh
TILES="N39W084 N21W159 N12E144 N15E144 S15W171 N15W066"
for t in $TILES; do /usr/bin/time -v python -W ignore ndvi_worker.py $t > ../logs/worker_$t.out 2>&1 & done
wait
grep -h -E "OK|FAIL|Elapsed|Maximum resident" ../logs/worker_*.out
python stage1_gates.py N39W084 N15W066 > ../logs/stage1_gates.out 2>&1; echo "gates exit $?"
python -W ignore ndvi_tables.py --test --tiles ${TILES// /,} --out ../data/output/stage1_test; echo "tables exit $?"
