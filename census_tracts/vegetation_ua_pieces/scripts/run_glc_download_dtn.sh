#!/bin/bash
# Run on a DTN (dtn01). Downloads the selected GLC_FCS30D annual tiles in N parallel shards.
# Idempotent: fetch_glc_tiles.py skips tiles present with the expected size.
R=/global/path/to/scratch/urbveg
REC=${1:-8239305}; N=${2:-4}; MAN=${3:-}
for k in $(seq 0 $((N-1))); do
  nohup $R/env/bin/python $R/scripts/fetch_glc_tiles.py $REC --root $R --shard $k/$N ${MAN:+--manifest $MAN} \
    > $R/logs/fetch_glc_${REC}_shard${k}of${N}${MAN:+_fua}.out 2>&1 &
done
wait
echo "ALL SHARDS DONE $(date -u +%FT%TZ)" >> $R/logs/fetch_glc_${REC}${MAN:+_fua}_done.txt
