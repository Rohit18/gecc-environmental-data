#!/bin/bash
#SBATCH --job-name=lai-merge
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# Earth Engine semantics QA (07, if the reference sample exists), then merge, validate,
# compare with the Earth Engine exports and write the deliverables (06).
# Submit through submit_all.sh, or:  sbatch -A your_nersc_project -o $LOG_DIR/%x-%j.out slurm_03_merge.sh
set -eo pipefail
source "${LAI_CODE:-$HOME/GECC/census_tract_lai}/config.sh"
activate_env
cd "$LAI_CODE"
mkdir -p "$OUT_DIR"

# Earth Engine semantics QA (07) needs the optional reference sample from 05.
if [ -f "$QA_DIR/ee_reference_sample.parquet" ]; then
    python 07_ee_semantics.py --manifest-dir "$MANIFEST_DIR" --shard-dir "$SHARD_DIR" --qa-dir "$QA_DIR" \
        --vrt "$LAI_VRT" --ee-exports "$EE_EXPORTS"
fi
python 06_merge_validate.py --manifest-dir "$MANIFEST_DIR" --shard-dir "$SHARD_DIR" --qa-dir "$QA_DIR" \
    --ee-exports "$EE_EXPORTS" --tile-list "$LAI_TILE_LIST" --tiles-dir "$LAI_TILES_DIR" --out-dir "$OUT_DIR"
