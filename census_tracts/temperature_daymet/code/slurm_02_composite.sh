#!/bin/bash
#SBATCH --job-name=daymet-composite
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
python daymet_composite.py --years 2020 --workers 1
