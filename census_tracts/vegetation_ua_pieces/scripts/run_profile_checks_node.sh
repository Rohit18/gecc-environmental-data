#!/bin/bash
# (1) census regression after the run-profile refactor; (2) FUA run on the Columbus WorldCover tile.
cd /path/to/scratch/urbveg/scripts; source ./env.sh
T() { echo "=== $* $(date -u +%T)"; /usr/bin/time -v "$@" 2>&1 | grep -E 'Elapsed|Maximum resident|Traceback|Error|^\{|OK|FAIL|SKIP|^ "n_|exit' ; }
T python verify_glc_tiles.py 64 --manifest ../manifests/glc_tiles_8239305_fua.csv --tag _fua
export URBVEG_RUN=census_tracts_ua
T python bench_worker.py N39W084 3000 --save
T python reduce.py --tiles $(python -c "import pandas as pd;print(' '.join(pd.read_csv('../manifests/worldcover_tiles.csv').tile))") --name census_regression
export URBVEG_RUN=fua_fishnet
T python urbveg_worker.py N39W084
T python qa_city.py --tile N39W084 --bbox -83.30 39.70 -82.50 40.40 --name columbus_oh
T python bench_worker.py N39W075 3000
