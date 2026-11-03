#!/usr/bin/env python3
"""Persistently orchestrate the series; stop on any gate or job failure."""
import argparse, fcntl, json, re, subprocess, sys, time
from common import *
from merge_validate import SEASONAL_GATE_POLICY
STATE=CFS/'audits/full_run_state.json';CHECKS=[1980,2000,2024,2025]
SCRIPT=ROOT/'code/run_cpu_batch.sh'

def cmd(args):return subprocess.check_output(args,text=True).strip()
def save():
    state['updated_utc']=utc_now();atomic_json(STATE,state)
def slurm(job):
    rows=cmd(['sacct','-X','-n','-P','-j',str(job),'--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes']).splitlines()
    for row in rows:
        fields=row.split('|')
        if fields[0]==str(job):return dict(jobid=int(job),state=fields[1],exitcode=fields[2],seconds=int(fields[3]),nodes=int(fields[4]))
    return None

def used_hours():
    total=0
    for job in state['cpu_jobs']:
        row=slurm(job['jobid']) if job.get('jobid') else None
        if row:job.update(accounting=row);total+=row['seconds']*row['nodes']/3600
    state['cpu_node_hours_used']=total;save();return total

def wait_download(years,job):
    while True:
        row=slurm(job)
        records=[r for r in json.loads((CFS/'manifests/daymet_manifest.json').read_text())['files'] if r['year'] in years]
        if len(records)==6*len(years) and all(r['status']=='verified_cfs' for r in records):return
        if row and row['state'] not in ['PENDING','RUNNING','COMPLETING','CONFIGURING']:
            raise RuntimeError(f'Download job {job} ended before required archives verified: {row}')
        time.sleep(30)

def cpu(years,workers=4,check=False,final=False,validate_only=False):
    remaining=30-used_hours()
    if remaining<.5:raise RuntimeError('Insufficient remaining CPU node-hour budget; stopped before exceeding 30')
    minutes=min(240,int((remaining-.05)*60))
    args=['--years',','.join(map(str,years)),'--workers',str(workers)]
    if check:args+=['--check-years']
    if final:args+=['--final']
    if validate_only:args+=['--validate-only']
    log=ROOT/'logs'/f'full_cpu_{years[0]}_{years[-1]}_{"final" if final else "checks" if check else "batch"}_{len(state["cpu_jobs"])}.log'
    job=dict(years=years,workers=workers,check_years=check,final=final,validate_only=validate_only,log=str(log),reserved_max_node_hours=minutes/60)
    with log.open('w') as out:
        proc=subprocess.Popen(['salloc','--account=your_nersc_project','--constraint=cpu','--qos=interactive','--nodes=1','--ntasks=1','--cpus-per-task=128',f'--time={minutes}','--licenses=cfs,scratch','--job-name=daymet-full-cpu','srun','bash',str(SCRIPT),*args],stdout=out,stderr=subprocess.STDOUT)
        state['cpu_jobs'].append(job);save()
        while proc.poll() is None:
            if not job.get('jobid'):
                match=re.search(r'Granted job allocation (\d+)',log.read_text())
                if match:job['jobid']=int(match[1]);save()
            time.sleep(15)
    if not job.get('jobid'):
        match=re.search(r'Granted job allocation (\d+)',log.read_text())
        if match:job['jobid']=int(match[1])
    used_hours()
    if proc.returncode:raise RuntimeError(f'CPU batch failed (exit {proc.returncode}); inspect {log}')
    if not job.get('jobid') or not (CFS/'audits'/f'cpu_batch_{job["jobid"]}.json').exists():raise RuntimeError('CPU batch has no success audit')
    for year in years:
        if final:break
        audit=json.loads((CFS/'outputs'/f'CensusTract_Daymet_{year}_validation.json').read_text())
        if not audit['passed']:raise RuntimeError(f'{year} failed a gate')
        state['gate_results'][str(year)]={k:g['passed'] for k,g in audit['gates'].items()}
    save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-download-job',type=int,required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--revalidate-checks',action='store_true');a=parser.parse_args()
    controller_lock=(CFS/'audits/full_controller.lock').open('a')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if STATE.exists():
        if not a.resume:raise SystemExit('Existing full-run state: use --resume after inspecting the prior failure')
        state=json.loads(STATE.read_text())
        for job in state['cpu_jobs']:
            row=slurm(job['jobid']) if job.get('jobid') else None
            if row and row['state'] in ['RUNNING','PENDING','CONFIGURING','COMPLETING']:
                raise SystemExit(f'Existing CPU allocation {job["jobid"]} is active; wait for it before resuming')
        for year in state['approved_years']:
            audit=CFS/'outputs'/f'CensusTract_Daymet_{year}_validation.json'
            if audit.exists() and not json.loads(audit.read_text())['passed']:
                prior=json.loads(audit.read_text());failed=[k for k,g in prior['gates'].items() if not g['passed']]
                revision=state.get('seasonal_gate_change_authorized',{})
                if not (a.revalidate_checks and year in CHECKS and failed==['ordering'] and revision.get('policy')==SEASONAL_GATE_POLICY):
                    raise SystemExit(f'{year} has a failed scientific gate: continuation requires review')
        state.pop('error',None);state['status']='running_checks'
    else:
        state=dict(status='running_checks',approved_years=list(range(1980,2026)),check_years=CHECKS,cpu_node_hour_limit=30,cpu_jobs=[dict(jobid=59746093,phase='2020 pilot compositing'),dict(jobid=59748776,phase='2020 pilot statistics')],download_jobs=[a.check_download_job],gate_results={},created_utc=utc_now(),scratch_staging_limit_bytes=150000000000)
    save()
    try:
        wait_download(CHECKS,a.check_download_job)
        if a.revalidate_checks:
            cpu(sorted(CHECKS+[2020]),4,validate_only=True)
            fingerprints=json.loads((CFS/'audits/data_fingerprints_before_CA_revalidation.json').read_text())
            changed=[name for name,info in fingerprints['files'].items() if Path(name).stat().st_size!=info['bytes'] or sha256_of(name)!=info['sha256']]
            atomic_json(CFS/'audits/data_unchanged_after_CA_revalidation.json',dict(passed=not changed,changed=changed,files_verified=len(fingerprints['files']),created_utc=utc_now(),policy=SEASONAL_GATE_POLICY))
            if changed:raise RuntimeError(f'Revalidation changed existing scientific data files: {changed}')
        elif not all(str(y) in state['gate_results'] for y in CHECKS):cpu(CHECKS,4,check=True)
        for year in CHECKS:
            audit=json.loads((CFS/'outputs'/f'CensusTract_Daymet_{year}_validation.json').read_text())
            assert audit['passed'] and audit['gates']['ordering'].get('seasonal_gate_policy')==SEASONAL_GATE_POLICY, f'{year}: current gate policy must pass'
        bench=json.loads((CFS/'audits/concurrency_benchmark.json').read_text());assert bench['complete']
        workers=bench['chosen_workers'];state.pop('failed_gate',None);state.update(status='running_full_series',chosen_workers=workers,check_years_all_gates_passed=True,seasonal_gate_policy=SEASONAL_GATE_POLICY);save()
        remaining=[y for y in range(1980,2026) if y not in CHECKS+[2020]]
        download=state['download_jobs'][-1] if len(state['download_jobs'])>1 else None
        row=slurm(download) if download else None
        if not row or row['state'] not in ['PENDING','RUNNING','COMPLETING','CONFIGURING','COMPLETED']:
            download=int(cmd(['sbatch','--parsable',str(ROOT/'code/slurm_full_fetch.sh'),','.join(map(str,remaining))]).split(';')[0]);state['download_jobs'].append(download);save()
        for start in range(0,len(remaining),workers):
            batch=remaining[start:start+workers];wait_download(batch,download)
            if not all(str(y) in state['gate_results'] for y in batch):cpu(batch,workers)
        pilot=json.loads((CFS/'outputs/CensusTract_Daymet_2020_validation.json').read_text());assert pilot['passed']
        state['gate_results']['2020']={k:g['passed'] for k,g in pilot['gates'].items()};save()
        cpu(list(range(1980,2026)),workers,final=True)
        state['status']='validated';used_hours();save()
        # I/O-only package verifies daily archive hashes without spending CPU allocations.
        pkg=int(cmd(['sbatch','--parsable',str(ROOT/'code/slurm_full_package.sh')]));state['package_job']=pkg;save()
        while True:
            row=slurm(pkg)
            if row and row['state']=='COMPLETED' and row['exitcode']=='0:0':break
            if row and row['state'] not in ['PENDING','RUNNING','COMPLETING','CONFIGURING']:raise RuntimeError(f'Package job failed: {row}')
            time.sleep(30)
        state.update(status='complete',release_packaged=True,release_path=json.loads((CFS/'audits/release_location.json').read_text())['path']);save()
    except BaseException as error:
        state.update(status='stopped',error=str(error));save()
        # No more downloads proceed after a scientific gate failure.
        for job in state['download_jobs']:
            row=slurm(job)
            if row and row['state'] in ['PENDING','RUNNING','CONFIGURING']:subprocess.run(['scancel',str(job)],check=False)
        raise
