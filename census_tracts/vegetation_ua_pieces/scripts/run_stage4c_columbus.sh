#!/bin/bash
# Stage 4c (Columbus): official partial (config settings), window-invariance check, clamp-mode seam QA, test reduce.
cd /path/to/scratch/urbveg/scripts; source ./env.sh
T() { echo "=== $* $(date -u +%T)"; /usr/bin/time -v "$@" 2>&1 | grep -E 'Elapsed|Maximum resident|Traceback|Error|^\{|OK|FAIL|SKIP|exit' ; }
T python urbveg_worker.py N39W084
T python bench_worker.py N39W084 6000 --save
T python qa_city.py --tile N39W084 --bbox -83.30 39.70 -82.50 40.40 --name columbus_oh_clamp --overhang clamp
T python reduce.py --tiles N39W084 --name columbus
