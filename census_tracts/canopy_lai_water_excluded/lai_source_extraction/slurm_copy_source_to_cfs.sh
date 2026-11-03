#!/bin/bash
#SBATCH --job-name=lai-src-cfs
#SBATCH --qos=xfer
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/project/gecc/lai_2020_source/logs/%x-%j.out
# Copy the native-grid LAI 2020 source tiles (CONUS) from scratch to CFS and verify
# every tile against the sha256 recorded by 01_raster_manifest.py. Runs on the xfer
# QOS (no compute allocation). Resumable: rsync skips files already copied, and the
# verification always re-hashes everything on CFS.
#   mkdir -p /path/to/project/gecc/lai_2020_source/logs
#   sbatch -A your_nersc_project slurm_copy_source_to_cfs.sh
set -euo pipefail
source "${LAI_CODE:-$HOME/GECC/census_tract_lai}/config.sh"
DEST="${LAI_SOURCE_CFS:-/path/to/project/gecc/lai_2020_source}"
SRC="$LAI_DOWNLOAD"
PAR=16
RS="rsync -rlt --chmod=Dug+rwx,Dg+s,Fug+rw,o-rwx"
mkdir -p "$DEST/tiles" "$DEST/download" "$DEST/logs"

echo "$(date '+%F %T') copying tiles $SRC/tiles -> $DEST/tiles"
find "$SRC/tiles" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort \
    | xargs -P "$PAR" -I{} $RS "$SRC/tiles/{}/" "$DEST/tiles/{}/"
$RS "$SRC/conus.vrt" "$SRC/conus_tiles_2048.txt" "$SRC/conus_tiles_2048_nodata.txt" \
    "$SRC/conus_tile_files.txt" "$SRC/coverage_2048.npy" "$DEST/"
$RS "$SRC/download_tiles.py" "$SRC/conus_download.log" "$SRC/eecu_ledger_your-ee-project.jsonl" "$DEST/download/"
$RS "$MANIFEST_DIR/tile_manifest.csv" "$MANIFEST_DIR/raster_validation.json" "$DEST/"
echo "$(date '+%F %T') copy done"

# Verify: expected sha256 per tile from the manifest vs a fresh hash of every tile on CFS.
cd "$DEST"
awk -F, 'NR > 1 {print $5 "  " $1}' tile_manifest.csv | sort -k2 > tiles.sha256
find tiles -name '*.tif' | sort | xargs -P "$PAR" -n 256 sha256sum | sort -k2 > logs/tiles.sha256.cfs
n_expected=$(wc -l < tiles.sha256)
n_found=$(wc -l < logs/tiles.sha256.cfs)
if diff -q tiles.sha256 logs/tiles.sha256.cfs > /dev/null; then
    tiles_ok=true
else
    tiles_ok=false
    diff tiles.sha256 logs/tiles.sha256.cfs | head -20
fi
vrt_expected=$(python3 -c "import json; print(json.load(open('raster_validation.json'))['vrt']['sha256'])")
vrt_found=$(sha256sum conus.vrt | cut -d' ' -f1)
echo "$(date '+%F %T') tiles expected=$n_expected found=$n_found identical=$tiles_ok vrt_identical=$([ "$vrt_expected" = "$vrt_found" ] && echo true || echo false)"
[ "$tiles_ok" = true ] && [ "$vrt_expected" = "$vrt_found" ] && [ "$n_expected" = "$n_found" ]
echo "VERIFIED"
