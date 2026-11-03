#!/usr/bin/env python3
"""Compute a specified batch, run independent checks, and enforce identical gates."""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from common import *
p=argparse.ArgumentParser();p.add_argument('--years',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--check-years',action='store_true');p.add_argument('--final',action='store_true');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
started=time.monotonic();years=parse_years(a.years)
def run(script,*args):subprocess.run([sys.executable,script,*args],check=True)
if a.check_years:
    assert years==[1980,2000,2024,2025]
    run('benchmark_concurrency.py')
if not a.final:
    if not a.validate_only:
        run('daymet_composite.py','--years',a.years,'--workers',str(a.workers))
        run('zonal_stats.py','--years',a.years,'--workers',str(min(a.workers,8)))
        with ThreadPoolExecutor(max_workers=min(a.workers,8)) as pool:
            list(pool.map(lambda year:run('spot_checks.py','--year',str(year)),years))
    # Separate per-year validation avoids a tag collision for noncontiguous checks.
    # Their destinations are independent, so validate in parallel within the node.
    with ThreadPoolExecutor(max_workers=min(a.workers,8)) as pool:
        list(pool.map(lambda year:run('merge_validate.py','--years',str(year)),years))
else:run('merge_validate.py','--years',a.years)
atomic_json(CFS/'audits'/f'cpu_batch_{os.environ["SLURM_JOB_ID"]}.json',dict(years=years,workers=a.workers,check_years=a.check_years,final=a.final,validate_only=a.validate_only,passed=True,elapsed_seconds=time.monotonic()-started,created_utc=utc_now()))
