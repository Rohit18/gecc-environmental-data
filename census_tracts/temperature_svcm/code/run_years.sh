#!/usr/bin/env bash
# Select, extract and composite a batch of years, then delete that batch's daily
# rasters so disk use stays bounded. Composites and audits are kept.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${AIRTEMP_PYTHON:?set AIRTEMP_PYTHON}
YEARS="$1"; JOBID="$2"
for y in $YEARS; do
  srun --jobid="$JOBID" --overlap -N1 -n1 -c 8 "$PY" "$ROOT/code/select_members.py" "$y" "$AIRTEMP_SOURCE_DIR" "$ROOT/work/lists" &
done; wait
for y in $YEARS; do
  mkdir -p "$ROOT/work/daily/$y"
  srun --jobid="$JOBID" --overlap -N1 -n1 -c 6 tar -xf "$AIRTEMP_SOURCE_DIR/NorthAmerica$y.tar" -C "$ROOT/work/daily/$y" -T "$ROOT/work/lists/members_$y.txt" &
done; wait
echo "extracted: $YEARS"
Y=$(echo $YEARS | tr ' ' ',')
srun --jobid="$JOBID" --overlap -N1 -n1 -c 128 "$PY" "$ROOT/code/composites.py" "$Y" "$ROOT/work/lists" "$ROOT/work/daily" "$ROOT/work/composites_$(echo $YEARS | cut -d' ' -f1)" 24
for y in $YEARS; do rm -rf "$ROOT/work/daily/$y"; done
echo "composited and cleaned: $YEARS"
