#!/bin/bash
#SBATCH --job-name=daymet-stats
#SBATCH --account=your_nersc_project
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --licenses=cfs,scratch
#SBATCH --output=/path/to/scratch/daymet_tract_temperature/logs/%x-%j.out
set -euo pipefail
source /path/to/scratch/daymet_tract_temperature/code/config.sh
activate_env
cd "$DAYMET_CODE"
python zonal_stats.py --years 2020 --workers 1
python spot_checks.py --year 2020
python merge_validate.py --years 2020
# Capture fresh quotas and run pilot_report.py on a login node: compute nodes
# lack NERSC's GPFS quota helper. Purge is an I/O-only xfer/login stage after
# the report and composite audits pass; see README.md.
# No post-pilot jobs or downloads are submitted here.
