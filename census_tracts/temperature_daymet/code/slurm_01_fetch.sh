#!/bin/bash
#SBATCH --job-name=daymet-fetch-2020
#SBATCH --account=your_nersc_project
#SBATCH --qos=xfer
#SBATCH --time=02:00:00
#SBATCH --licenses=cfs,scratch
#SBATCH --output=/path/to/scratch/daymet_tract_temperature/logs/%x-%j.out
set -euo pipefail
source /path/to/scratch/daymet_tract_temperature/code/config.sh
activate_env
cd "$DAYMET_CODE"
python - <<'PY'
import json
from common import CFS
gate=json.loads((CFS/'audits/geometry_preflight.json').read_text())
tests=json.loads((CFS/'audits/unit_tests.json').read_text())
assert gate['passed'] and gate['scale_corrected_threshold']==.02, 'Geometry gate must pass'
assert tests['passed'], 'Unit tests must pass'
PY
python daymet_fetch.py fetch --years 2020 --workers "$DAYMET_WORKERS" --scratch-budget-gb "$DAYMET_SCRATCH_BUDGET_GB"
python daymet_fetch.py archive --years 2020
python daymet_fetch.py verify --years 2020
python - <<'PY'
from common import *
import json
manifest=json.loads((CFS/'manifests/daymet_manifest.json').read_text())
files=[r for r in manifest['files'] if r['year']==2020]
assert len(files)==6 and all(r['status']=='verified_cfs' for r in files)
peak=sum(p.stat().st_size for p in (ROOT/'work/staging').rglob('*') if p.is_file())
atomic_json(CFS/'audits/download_2020.json',dict(passed=True,created_utc=utc_now(),files=files,source_staging_peak_bytes=peak,archive_bytes=sum(r['bytes'] for r in files)))
PY
