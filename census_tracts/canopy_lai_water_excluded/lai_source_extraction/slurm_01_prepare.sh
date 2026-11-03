#!/bin/bash
#SBATCH --job-name=lai-prepare
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
# Raster validation + tile manifest (01), then CONUS tract selection + shards (02).
# Submit through submit_all.sh, or:  sbatch -A your_nersc_project -o $LOG_DIR/%x-%j.out slurm_01_prepare.sh
set -eo pipefail
source "${LAI_CODE:-$HOME/GECC/census_tract_lai}/config.sh"
activate_env
cd "$LAI_CODE"
mkdir -p "$MANIFEST_DIR"

python 01_raster_manifest.py --tiles-dir "$LAI_TILES_DIR" --tile-list "$LAI_TILE_LIST" --vrt "$LAI_VRT" \
    --out-dir "$MANIFEST_DIR" --workers 64
python 02_prepare_tracts.py --tracts "$TRACTS" --tile-manifest "$MANIFEST_DIR/tile_manifest.csv" \
    --out-dir "$MANIFEST_DIR" --n-shards "$N_SHARDS"
