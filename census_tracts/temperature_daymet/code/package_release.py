#!/usr/bin/env python3
"""Package verified Daymet tables/code/audits; reference, never copy daily data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo
from common import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years',default='1980-2025')
    parser.add_argument('--date',default=dt.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d'))
    args=parser.parse_args();years=parse_years(args.years)
    tag=str(years[0]) if len(years)==1 else f'{years[0]}_{years[-1]}'
    validation=CFS/'outputs'/f'CensusTract_Daymet_{tag}_validation.json'
    audit=json.loads(validation.read_text())
    if not audit.get('passed') or audit['years']!=years:raise SystemExit('Packaging blocked: missing/passing validation or wrong year set')
    manifest=json.loads((CFS/'manifests/daymet_manifest.json').read_text())
    wanted=[r for r in manifest['files'] if r['year'] in years]
    if len(wanted)!=6*len(years) or any(r['status']!='verified_cfs' for r in wanted):raise SystemExit('Packaging blocked: missing verified source archives')
    # Packaging verifies archive bytes but never copies any daily source file.
    from daymet_fetch import verify_file
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda record: verify_file(immutable_read_path(record['cfs_path']), record, record['sha256']), wanted))
    print(f'Archive checksums verified: {len(wanted)} files', flush=True)
    code=Path(__file__).resolve().parent
    plan={}
    for suffix in ['parquet','csv.gz','validation.json']:
        name=f'CensusTract_Daymet_{tag}.{suffix}' if suffix!='validation.json' else f'CensusTract_Daymet_{tag}_validation.json'
        plan['data/'+name]=CFS/'outputs'/name
    for label in ['coverage_by_StFIPS_year','tracts_with_missing']:
        name=f'CensusTract_Daymet_{label}_{tag}.csv';plan['data/'+name]=CFS/'outputs'/name
    for year in years:
        for suffix in ['csv','parquet']:
            name=f'CensusTract_Daymet_{year}.{suffix}';plan['data/per_year/'+name]=CFS/'outputs/per_year'/name
        plan[f'validation/spot_checks_{year}.json']=CFS/'audits'/f'spot_checks_{year}.json'
        plan[f'validation/per_year_validation_{year}.json']=CFS/'outputs'/f'CensusTract_Daymet_{year}_validation.json'
        plan[f'validation/California_seasonal_inversions/{year}.csv']=CFS/'outputs'/f'CensusTract_Daymet_California_seasonal_inversions_{year}.csv'
        for domain in DOMAINS:
            source=composite_path(CFS/'composites',year,domain).with_suffix('.json')
            plan['validation/composite_sidecars/'+source.name]=source
    for name in ['unit_tests.json','geometry_preflight.json','input_storage_preflight.json','authentication_preflight.json']:
        plan['validation/'+name]=CFS/'audits'/name
    for name in ['full_run_state.json','full_run_report.json','full_run_report.md','gate_results_1980_2025.csv','concurrency_benchmark.json','nonterritory_missing_reasons.json','geometry_area_checks.parquet','planar_area_distortion_by_StFIPS.csv','quota_before_full.txt','quota_after_full_download.txt','runtime_versions.json']:
        plan['validation/'+name]=CFS/'audits'/name
    plan['validation/seasonal_gate_change_20261013.json']=CFS/'audits/seasonal_gate_change_20261013.json'
    name=f'CensusTract_Daymet_California_seasonal_inversions_{tag}.csv'
    plan['validation/'+name]=CFS/'outputs'/name
    plan['validation/data_unchanged_after_CA_revalidation.json']=CFS/'audits/data_unchanged_after_CA_revalidation.json'
    plan['validation/archive_read_path_performance_20261013.json']=CFS/'audits/archive_read_path_performance_20261013.json'
    plan['provenance/daymet_manifest.json']=CFS/'manifests/daymet_manifest.json'
    plan['provenance/daily_archive_README.md']=CFS/'daily/README.md'
    plan['provenance/CensusTract_2020_US_85528.provenance.json']=POLYGONS.with_suffix('.provenance.json')
    for source in sorted((CFS/'audits/source_docs').glob('*')):
        if source.is_file():plan['provenance/source_docs/'+source.name]=source
    for pattern in ['*.py','*.sh','*.md','environment.yml','tests/*.py']:
        for path in sorted(code.glob(pattern)):plan['code/'+str(path.relative_to(code))]=path
    for name in ['README.md','METHODS.md']:plan[name]=code/name
    missing=[str(p) for p in plan.values() if not p.is_file()]
    if missing:raise SystemExit(f'Packaging blocked by missing inputs/outputs: {missing}')
    destination=CFS/'release'/f'GECC_CensusTract_Daymet_{years[0]}_{years[-1]}_{args.date}'
    if destination.exists():
        for line in (destination/'SHA256SUMS').read_text().splitlines():
            expected,name=line.split(maxsplit=1)
            if sha256_of(destination/name)!=expected:raise SystemExit('Existing package checksum mismatch')
        atomic_json(CFS/'audits/release_location.json',dict(path=str(destination),checksum_listing_sha256=sha256_of(destination/'SHA256SUMS'),created_utc=utc_now()))
        print(f'Existing release verifies: {destination}');return
    temporary=destination.with_name(destination.name+'.part')
    if temporary.exists():raise SystemExit(f'Interrupted package exists at {temporary}; inspect before replacing')
    temporary.mkdir(parents=True)
    for relative,source in plan.items():
        target=temporary/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha256_of(target)!=sha256_of(source):raise SystemExit(f'Package copy checksum mismatch: {relative}')
    # This report becomes public only after the verified package is renamed.
    # Record its final location rather than a pre-package existence diagnostic.
    report_path=temporary/'validation/full_run_report.json'
    report=json.loads(report_path.read_text());old_path=report['release_path']
    report.update(release_path=str(destination),release_exists=True)
    report['cfs_allocated_bytes']=int(subprocess.check_output(['du','-s','--block-size=1',str(CFS)],text=True).split()[0])
    report['cfs_apparent_bytes']=int(subprocess.check_output(['du','-s','--apparent-size','--block-size=1',str(CFS)],text=True).split()[0])
    atomic_json(report_path,report)
    packaged_state_path=temporary/'validation/full_run_state.json'
    packaged_state=json.loads(packaged_state_path.read_text())
    packaged_state.update(status='complete',release_packaged=True,release_path=str(destination))
    atomic_json(packaged_state_path,packaged_state)
    report_md=temporary/'validation/full_run_report.md'
    report_lines=report_md.read_text().replace(old_path,str(destination)).splitlines()
    report_lines=[f'CFS dataset tree allocated: {report["cfs_allocated_bytes"]:,} bytes; apparent: {report["cfs_apparent_bytes"]:,} bytes.' if line.startswith('CFS dataset tree allocated:') else line for line in report_lines]
    report_md.write_text('\n'.join(report_lines)+'\n')
    code_files=sorted(p for p in (temporary/'code').rglob('*') if p.is_file())
    (temporary/'code/CODE_SHA256SUMS').write_text(''.join(f'{sha256_of(p)}  {p.relative_to(temporary/"code")}\n' for p in code_files))
    atomic_json(temporary/'RELEASE_INFO.json',dict(created_utc=utc_now(),dataset='DAYMET',version='4.1',
        doi='10.3334/ORNLDAAC/2129',years=years,rows=85528*len(years),validation_passed=True,
        daily_archive=str(CFS/'daily'),daily_archive_copied_into_release=False,
        calendar_rule='365 days; leap years include February 29 and omit December 31',
        cpu_node_hours_used=json.loads((CFS/'audits/full_run_state.json').read_text())['cpu_node_hours_used'],
        chosen_years_per_node=json.loads((CFS/'audits/concurrency_benchmark.json').read_text())['chosen_workers'],
        code_snapshot_sha256=sha256_of(temporary/'code/CODE_SHA256SUMS')))
    files=sorted(p for p in temporary.rglob('*') if p.is_file())
    (temporary/'SHA256SUMS').write_text(''.join(f'{sha256_of(p)}  {p.relative_to(temporary)}\n' for p in files))
    temporary.rename(destination)
    atomic_json(CFS/'audits/release_location.json',dict(path=str(destination),checksum_listing_sha256=sha256_of(destination/'SHA256SUMS'),created_utc=utc_now()))
    print(f'Release verified: {destination}')


if __name__=='__main__':main()
