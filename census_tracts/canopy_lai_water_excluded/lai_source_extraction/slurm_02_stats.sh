#!/bin/bash
#SBATCH --job-name=lai-stats
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
# Per-shard exactextract statistics (03), then the independent recomputation sample (04).
# Measured: one 1.4e9-bbox-pixel shard = 99 s and 3.5 GB RSS on one worker; 128 shards on 64 workers ~5 min.
# Complete shards are skipped on resubmission (content hashes, not timestamps).
# Submit through submit_all.sh, or:  sbatch -A your_nersc_project -o $LOG_DIR/%x-%j.out slurm_02_stats.sh
set -eo pipefail
source "${LAI_CODE:-$HOME/GECC/census_tract_lai}/config.sh"
activate_env
cd "$LAI_CODE"
mkdir -p "$SHARD_DIR" "$QA_DIR"

python 03_tract_stats.py --manifest-dir "$MANIFEST_DIR" --vrt "$LAI_VRT" --out-dir "$SHARD_DIR" --workers 64
python 04_independent_check.py --manifest-dir "$MANIFEST_DIR" --vrt "$LAI_VRT" \
    --out "$QA_DIR/independent_check.parquet" --workers 64 \
    --include-geoids-from "$QA_DIR/ee_reference_sample.parquet"
