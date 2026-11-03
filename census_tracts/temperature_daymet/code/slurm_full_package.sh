#!/bin/bash
#SBATCH --job-name=daymet-full-package
#SBATCH --account=your_nersc_project
#SBATCH --qos=xfer
#SBATCH --time=04:00:00
#SBATCH --licenses=cfs,scratch
#SBATCH --output=/path/to/scratch/daymet_tract_temperature/logs/%x-%j.out
set -euo pipefail
source /path/to/scratch/daymet_tract_temperature/code/config.sh
activate_env
cd "$DAYMET_CODE"
python full_run_report.py
python package_release.py --years 1980-2025
python full_run_report.py
