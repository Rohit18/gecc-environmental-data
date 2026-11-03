#!/bin/bash
# Run inside one allocated CPU node. Keep source extraction code unchanged so
# existing checksum-verified parts remain reusable.
source /path/to/scratch/gecc_tract_watermask_20261013/code/job_env.sh
ROOT=/path/to/scratch/gecc_tract_watermask_20261013
SHARD=${1:?specify canopy shard 0 or 1}
if [[ "$SHARD" == 1 ]]; then
  if python - <<'PY'
import json,sys
sys.path.insert(0,'/path/to/scratch/gecc_tract_watermask_20261013/code')
from extract import ROOT,signature
p=ROOT/'validation/lai_validation.json'
if not p.exists():raise SystemExit(1)
r=json.loads(p.read_text())
raise SystemExit(0 if r.get('passed') and r.get('signature')==signature() else 1)
PY
  then
    echo 'LAI already complete and validated; resuming canopy only'
  else
    python "$ROOT/code/extract.py" lai --workers 64
    python "$ROOT/code/merge_validate.py" lai
  fi
fi
python "$ROOT/code/extract.py" chm --workers 64 --shards 2 --shard "$SHARD"
