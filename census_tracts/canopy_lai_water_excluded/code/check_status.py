"""Read-only progress summary of this run's jobs, parts and validated outputs."""
import json
import subprocess
from pathlib import Path
ROOT=Path('/path/to/scratch/gecc_tract_watermask_20261013')
jobs=json.loads((ROOT/'jobs.json').read_text())
ids=','.join(str(v) for k,v in jobs.items() if isinstance(v,int))
subprocess.run(['sacct','-j',ids,'-X','--format=JobID,JobName%25,State,Elapsed,ExitCode'],check=True)
for kind in ['lai','chm']:
    parts=list((ROOT/f'work/{kind}').glob('*.done.json'))
    print(kind,'completed parts:',len(parts),'of',496 if kind=='lai' else 4195)
    p=ROOT/f'validation/{kind}_validation.json'
    if p.exists():print(p.read_text())
print('outputs:',[p.name for p in (ROOT/'outputs').iterdir()])
