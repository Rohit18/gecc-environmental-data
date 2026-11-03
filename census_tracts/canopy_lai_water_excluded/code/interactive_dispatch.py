"""Replace queued GECC jobs only after interactive nodes have been granted."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/path/to/scratch/gecc_tract_watermask_20261013')
job=os.environ['SLURM_JOB_ID']
nodes=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
assert len(nodes)==2,(job,nodes)
assert json.loads((ROOT/'validation/PILOT_PASSED.json').read_text())['passed']
jobs=json.loads((ROOT/'jobs.json').read_text())
old=[str(jobs[k]) for k in ['lai','chm_array','chm_merge','package']]
# Do not start competing writers; these are solely this run's batch jobs.
active=set(subprocess.check_output(['squeue','-h','-u','your_user','-o','%A'],text=True).split())
previous=jobs.get('interactive_allocation')
if previous is not None and str(previous) in active and str(previous)!=job:
    raise SystemExit(f'previous GECC interactive allocation {previous} is still active; avoid duplicate processing')
queued=[j for j in old if j in active]
if queued:subprocess.run(['scancel',*queued],check=True)
if previous is not None and int(previous)!=int(job):
    jobs.setdefault('previous_interactive_allocations',[]).append({
        'job_id':previous,'nodes':jobs.get('interactive_nodes'),
        'started_utc':jobs.get('interactive_started_utc')})
jobs['superseded_batch_jobs']={k:jobs[k] for k in ['lai','chm_array','chm_merge','package']}
jobs['interactive_allocation']=int(job)
jobs['interactive_nodes']=nodes
jobs['interactive_started_utc']=time.strftime('%FT%TZ',time.gmtime())
(ROOT/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
print('INTERACTIVE_GRANTED',job,nodes,flush=True)
processes=[]
for shard,node in enumerate(nodes):
    cmd=['srun','--jobid',job,'--nodes=1','--ntasks=1','--cpus-per-task=128',
         '--exclusive','--mem=0','--nodelist',node,
         '--output',str(ROOT/f'logs/interactive-{job}-node{shard}.out'),
         'bash',str(ROOT/'code/interactive_node.sh'),str(shard)]
    processes.append(subprocess.Popen(cmd))
codes=[p.wait() for p in processes]
if any(codes):
    raise SystemExit(f'interactive extraction steps failed: {codes}; completed parts remain resumable')
subprocess.run(['srun','--jobid',job,'--nodes=1','--ntasks=1','--cpus-per-task=8',
                '--nodelist',nodes[0],'python',str(ROOT/'code/merge_validate.py'),'chm'],check=True)
subprocess.run(['srun','--jobid',job,'--nodes=1','--ntasks=1','--cpus-per-task=4',
                '--nodelist',nodes[0],'python',str(ROOT/'code/merge_validate.py'),'package'],check=True)
print('INTERACTIVE_PIPELINE_COMPLETE',job,flush=True)
