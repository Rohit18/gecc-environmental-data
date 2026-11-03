#!/bin/bash
# GLC test-tile checks on a compute node: version comparison, alignment + Columbus clips, benchmark.
set -uo pipefail
cd /path/to/scratch/urbveg; source scripts/env.sh
VC=data/raw/glc_fcs30d/version_compare
BB="-83.30 39.70 -82.50 40.40"
run() { echo "=== $* $(date -u +%T)"; /usr/bin/time -v "$@" 2> >(grep -E 'Elapsed|Maximum resident|Traceback|Error' >&2) || echo "FAILED $*"; }
run python scripts/glc_version_compare.py
for rec in 8239305 15063683; do
  for t in W85N40 W85N45; do
    run python scripts/alignment_check.py --wc N39W084 --glc $(ls $VC/$rec/*_${t}_Annual*.tif) --year 2020 \
        --city-bbox $BB --city-name columbus_oh_${rec}_${t} --ua-geoid 19234
  done
  run python scripts/benchmark.py --wc N39W084 --glc $(ls $VC/$rec/*_W85N40_Annual*.tif) --year 2020
  cp manifests/benchmark.json manifests/benchmark_${rec}.json
done
