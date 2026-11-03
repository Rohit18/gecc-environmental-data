#!/bin/bash
# Submit prepare -> stats -> merge as an afterok chain. Safe to rerun: completed
# shards are skipped, and each stage re-validates its inputs.
#   bash submit_all.sh            # all three stages
#   bash submit_all.sh stats      # start from the stats stage (manifests already built)
#   bash submit_all.sh merge      # merge/validate only
# The optional Earth Engine reference sample (05) runs on a login node, between
# prepare and stats:  source config.sh && activate_env && python 05_ee_reference_sample.py ...
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/config.sh"
mkdir -p "$LOG_DIR"
start="${1:-prepare}"
dep=()
submit() {  # name script
    local id
    id=$(sbatch --parsable -A "$LAI_ACCOUNT" -o "$LOG_DIR/%x-%j.out" "${dep[@]}" "$LAI_CODE/$2")
    echo "$1: job $id"
    dep=(--dependency=afterok:"$id")
}
case "$start" in
    prepare) submit prepare slurm_01_prepare.sh; submit stats slurm_02_stats.sh; submit merge slurm_03_merge.sh ;;
    stats)   submit stats slurm_02_stats.sh; submit merge slurm_03_merge.sh ;;
    merge)   submit merge slurm_03_merge.sh ;;
    *) echo "unknown stage $start" >&2; exit 2 ;;
esac
