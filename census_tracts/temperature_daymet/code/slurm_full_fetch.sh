#!/bin/bash
#SBATCH --job-name=daymet-full-fetch
#SBATCH --account=your_nersc_project
#SBATCH --qos=xfer
#SBATCH --time=12:00:00
#SBATCH --licenses=cfs,scratch
#SBATCH --output=/path/to/scratch/daymet_tract_temperature/logs/%x-%j.out
set -euo pipefail
source /path/to/scratch/daymet_tract_temperature/code/config.sh
activate_env
cd "$DAYMET_CODE"
python daymet_fetch.py fetch-cfs --years "${1:?year list required}" --workers 4 --approved-post-pilot
