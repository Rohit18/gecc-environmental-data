#!/bin/bash
# Idempotent download of 2020 TIGER/Line tracts (50 states + DC + PR) and 2020 Urban Areas.
# Skips files already present whose size equals the server Content-Length. Logs to logs/.
set -uo pipefail
ROOT=/path/to/scratch/urbveg
LOG=$ROOT/logs/download_boundaries_$(date -u +%Y%m%dT%H%M%SZ).log
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
# 50 states + DC (11) + PR (72); excludes AS 60, GU 66, MP 69, VI 78
FIPS="01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56 72"
for f in $FIPS; do fetch $T/tl_2020_${f}_tract.zip $ROOT/data/raw/boundaries/tracts_2020/zip; done
U=https://www2.census.gov/geo/tiger/TIGER2020/UAC
for z in tl_2020_us_uac20.zip tl_2020_us_uac20_corrected.zip 2020_UA_ATLANTA_GAINESVILLE_JUNE_2023_UPDATE.zip 2020_UA_NEW_ORLEANS_JUNE_2023_UPDATE.zip; do
  fetch $U/$z $ROOT/data/raw/boundaries/urban_areas_2020/zip
done
# unzip each archive into its own folder (no overwrite)
for z in $ROOT/data/raw/boundaries/tracts_2020/zip/*.zip $ROOT/data/raw/boundaries/urban_areas_2020/zip/*.zip; do
  d=$(dirname $(dirname "$z"))/$(basename "$z" .zip); mkdir -p "$d"; unzip -n -q "$z" -d "$d" && echo "UNZIP $z"
done
echo "N tract zips: $(ls $ROOT/data/raw/boundaries/tracts_2020/zip/*.zip | wc -l)"
