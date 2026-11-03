#!/bin/sh
set -eu
export PROJ_DATA=/path/to/home/.conda/envs/gecc-lai/share/proj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GDAL_CACHEMAX=64
ROOT=/path/to/project/gecc/extreme_temperature_tract2020/stage3_full/runs/constant_means_v3
PYTHON=/path/to/home/.conda/envs/gecc-lai/bin/python
exec > "$ROOT/logs/interactive_${SLURM_JOB_ID}.out" 2> "$ROOT/logs/interactive_${SLURM_JOB_ID}.err"
/bin/date -u
/bin/hostname
for phase in production finalize; do
 /usr/bin/time -v "$PYTHON" -I "$ROOT/qa/constant_reducer_v3.py" "$phase"
done
/usr/bin/time -v "$PYTHON" -I "$ROOT/qa/finish_report.py"
/bin/date -u
