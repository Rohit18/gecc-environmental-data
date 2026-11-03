#!/bin/bash
# Phase 1 prep steps on a compute node. Logs to logs/.
set -uo pipefail
cd /path/to/scratch/urbveg
source scripts/env.sh
run() { echo "=== $* $(date -u +%T)"; /usr/bin/time -v "$@" 2> >(grep -E 'Elapsed|Maximum resident|Traceback|Error|error' >&2) || echo "FAILED $*"; }
for s in "$@"; do
  case $s in
    boundaries) run python scripts/inspect_boundaries.py ;;
    wc_select)  run python scripts/select_worldcover_tiles.py ;;
    glc_select) run python scripts/select_glc_tiles.py 8239305; run python scripts/select_glc_tiles.py 15063683 ;;
    wc_fetch)   run bash scripts/fetch_worldcover.sh ;;
    wc_verify)  run python scripts/verify_worldcover.py N39W084 N36W078 ;;
  esac
done
