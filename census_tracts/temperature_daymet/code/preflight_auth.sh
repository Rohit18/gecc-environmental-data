#!/bin/bash
# Small, 2020-only Earthdata authentication and durable-archive preflight.
# Run inside tmux. A failed fetch stops before any archive or later download.
set -euo pipefail
CODE=/path/to/scratch/daymet_tract_temperature/code
PYTHON=/path/to/scratch/daymet_tract_temperature/env/daymet/bin/python
cd "$CODE"
"$PYTHON" daymet_fetch.py fetch --years 2020 --domains hi --measures tmax --workers 1
"$PYTHON" daymet_fetch.py archive --years 2020 --domains hi --measures tmax
"$PYTHON" daymet_fetch.py verify --years 2020 --domains hi --measures tmax
"$PYTHON" - <<'PY'
import datetime,json
from pathlib import Path
manifest=json.loads(Path('/path/to/project/gecc/daymet/manifests/daymet_manifest.json').read_text())
record=next(r for r in manifest['files'] if r['filename']=='daymet_v4_daily_hi_tmax_2020.nc')
assert record['status']=='verified_cfs'
report={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'file':record}
Path('/path/to/project/gecc/daymet/audits/authentication_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('Small Hawaii staging -> CFS archive -> checksum verification PASSED',flush=True)
PY
