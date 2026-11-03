#!/bin/bash
cd /path/to/scratch/urbveg/scripts; source ./env.sh
T() { echo "=== $* $(date -u +%T)"; /usr/bin/time -v "$@" 2>&1 | grep -E 'Elapsed|Maximum resident|Traceback|Error|^ "n_|"failures|total_windows|s_per_window|est_total|n_tiles' ; }
T python verify_glc_tiles.py 16 --manifest ../manifests/glc_tiles_8239305_fua.csv --tag _fua
URBVEG_RUN=fua_fishnet T python extrapolate_cost.py ../logs/worker/N39W084.stats.json ../manifests/bench/N39W075_w3000.json
