#!/bin/bash
# Idempotent download of the 2020 TIGER/Line tract files for AS (60), GU (66), MP (69) and VI (78), in the same
# layout as the vegetation run's 52 files (zip/ + one unzipped folder per file), under the ndvi subproject.
# Same fetch logic as ../../scripts/download_boundaries.sh (skip when size == server Content-Length).
set -uo pipefail
ROOT=/path/to/scratch/urbveg/ndvi
D=$ROOT/data/raw/boundaries/tracts_2020
mkdir -p $D/zip $ROOT/logs
LOG=$ROOT/logs/download_tracts_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
fetch() {  # url destdir
  local url=$1 dest=$2/$(basename "$1")
  local remote=$(curl -sfIL "$url" | tr -d '\r' | awk -F': ' 'tolower($1)=="content-length"{v=$2} END{print v}')
  if [[ -z "$remote" ]]; then echo "ERROR no content-length: $url"; return 1; fi
  if [[ -f "$dest" && $(stat -c %s "$dest") == "$remote" ]]; then echo "SKIP $dest ($remote B)"; return 0; fi
  curl -sfL --retry 5 --retry-delay 5 -o "$dest.part" "$url" || { echo "ERROR download $url"; return 1; }
  if [[ $(stat -c %s "$dest.part") != "$remote" ]]; then echo "ERROR size mismatch $url"; return 1; fi
  mv "$dest.part" "$dest"; echo "OK $dest ($remote B)"
}
T=https://www2.census.gov/geo/tiger/TIGER2020/TRACT
for f in 60 66 69 78; do fetch $T/tl_2020_${f}_tract.zip $D/zip; done
for z in $D/zip/*.zip; do
  d=$D/$(basename "$z" .zip); mkdir -p "$d"; unzip -n -q "$z" -d "$d" && echo "UNZIP $z"
done
sha256sum $D/zip/*.zip
