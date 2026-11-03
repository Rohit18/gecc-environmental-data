#!/bin/bash
# Stage 4c on a compute node: Columbus QA rasters, window-size benchmark, worker run for the Columbus tile.
cd /path/to/scratch/urbveg/scripts; source ./env.sh
T() { echo "=== $* $(date -u +%T)"; /usr/bin/time -v "$@" 2> >(grep -E 'Elapsed|Maximum resident|Traceback|Error' >&2); echo "exit=$?"; }
T python qa_city.py --tile N39W084 --bbox -83.30 39.70 -82.50 40.40 --name columbus_oh
for w in 3000 4500 6000 9000; do T python bench_worker.py N39W084 $w; done
