#!/bin/bash
# Populate data/raw/worldcover_2020/tiles from manifests/worldcover_tiles.csv.
#  - action=symlink_existing: symlink to the existing size-matching copy on CFS (no copy, no download)
#  - action=download: aws s3 cp --no-sign-request (only *_Map.tif); intended to run on a DTN for large sets
# Idempotent: skips files present with the S3 size. Never overwrites. Logs to logs/.
# Usage: fetch_worldcover.sh [--dry-run|""] [manifest.csv]
set -uo pipefail
ROOT=${URBVEG_ROOT:-/path/to/scratch/urbveg}
ENV=${URBVEG_ENV:-/path/to/scratch/urbveg/env}
DRY=${1:-}
MAN=${2:-$ROOT/manifests/worldcover_tiles.csv}
OUT=$ROOT/data/raw/worldcover_2020/tiles
mkdir -p "$OUT"
LOG=$ROOT/logs/fetch_worldcover_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
n_ok=0; n_skip=0; n_err=0
while IFS=, read -r tile fname s3_uri s3_bytes existing_path action; do
  dest=$OUT/$fname
  if [[ -e "$dest" ]]; then
    sz=$(stat -L -c %s "$dest")
    if [[ "$sz" == "$s3_bytes" ]]; then echo "SKIP $fname"; n_skip=$((n_skip+1)); continue; fi
    echo "ERROR $fname present with size $sz != $s3_bytes (not overwriting)"; n_err=$((n_err+1)); continue
  fi
  if [[ -n "$DRY" ]]; then echo "DRY $action $fname"; continue; fi
  case "$action" in
    symlink_existing) ln -s "$existing_path" "$dest" && echo "LINK $fname -> $existing_path" && n_ok=$((n_ok+1)) ;;
    download)
      $ENV/bin/aws s3 cp --no-sign-request --only-show-errors "$s3_uri" "$dest.part" \
        && [[ $(stat -c %s "$dest.part") == "$s3_bytes" ]] && mv "$dest.part" "$dest" \
        && echo "GET $fname" && n_ok=$((n_ok+1)) || { echo "ERROR download $fname"; n_err=$((n_err+1)); } ;;
    *) echo "ERROR unknown action $action for $fname"; n_err=$((n_err+1)) ;;
  esac
done < <($ENV/bin/python - <<EOF
import pandas as pd
d = pd.read_csv("$MAN")
for r in d.itertuples():
    print(",".join(str(x) for x in [r.tile, r.fname, r.s3_uri, int(r.s3_bytes), r.existing_path if isinstance(r.existing_path, str) else "", r.action]))
EOF
)
echo "done ok=$n_ok skip=$n_skip err=$n_err"
