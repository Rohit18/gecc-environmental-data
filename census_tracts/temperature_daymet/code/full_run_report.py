#!/usr/bin/env python3
"""Summarize verified gates, coverage, Slurm node-hours and permanent storage."""
import datetime as dt, json, subprocess
from zoneinfo import ZoneInfo
from common import *
state=json.loads((CFS/'audits/full_run_state.json').read_text())
validation=json.loads((CFS/'outputs/CensusTract_Daymet_1980_2025_validation.json').read_text())
assert validation['passed'] and validation['years']==list(range(1980,2026))
inversions=validation['reported']['California_seasonal_inversions']
inversion_csv=Path(inversions['audit_csv'])
inversion_table=pd.read_csv(inversion_csv,dtype={'GEOID':str})
assert len(inversion_table)==inversions['total']
summary=['<!-- BEGIN CALIFORNIA INVERSION SUMMARY -->',f'Across 1980–2025, {inversions["total"]} California tract-years have JJA Tmax <= same-year DJF Tmax. These are reported climate observations. The seasonal hard gate applies to CONUS and Alaska outside California; Tmax >= Tmin remains a hard gate everywhere.',f'Audit: `validation/{inversion_csv.name}` in the release; `{inversion_csv}` on CFS. Margin is JJA minus DJF Tmax (degrees C).','', '| Year | California inverted tract-years |','|---|---:|']
summary += [f'| {year} | {count} |' for year,count in inversions['counts_by_year'].items()]
summary += ['<!-- END CALIFORNIA INVERSION SUMMARY -->']
readme=ROOT/'code/README.md';text=readme.read_text();start_marker='<!-- BEGIN CALIFORNIA INVERSION SUMMARY -->';end_marker='<!-- END CALIFORNIA INVERSION SUMMARY -->'
if start_marker in text:
    start=text.index(start_marker);end=text.index(end_marker,start)+len(end_marker);text=text[:start]+'\n'.join(summary)+text[end:]
else:text+='\n'+'\n'.join(summary)+'\n'
text=text.replace('scientific data files are unchanged; the full run has resumed.', 'scientific data files are unchanged. All 46 years passed every hard gate; the completed release contains 3,934,288 rows and 38 columns.')
readme.write_text(text)
manifest=json.loads((CFS/'manifests/daymet_manifest.json').read_text());assert len(manifest['files'])==276 and all(r['status']=='verified_cfs' for r in manifest['files'])
rows=[]
for year in range(1980,2026):
    audit=json.loads((CFS/'outputs'/f'CensusTract_Daymet_{year}_validation.json').read_text());assert audit['passed']
    rows.append(dict(year=year,**{k:v['passed'] for k,v in audit['gates'].items()},all_gates_passed=audit['passed']))
pd.DataFrame(rows).to_csv(CFS/'audits/gate_results_1980_2025.csv',index=False)
bench=json.loads((CFS/'audits/concurrency_benchmark.json').read_text())
missing=pd.read_csv(CFS/'outputs/CensusTract_Daymet_tracts_with_missing_1980_2025.csv',dtype={'GEOID':str,'StFIPS':str})
reasons=json.loads((CFS/'audits/nonterritory_missing_reasons.json').read_text())
nonterritory=missing[~missing.StFIPS.isin(UNSUPPORTED_FIPS)]
expected={r['GEOID'] for r in reasons['tracts']}
assert set(nonterritory.GEOID)==expected, 'Unexpected non-territory missing tract: investigate before release'
coverage=pd.read_csv(CFS/'outputs/CensusTract_Daymet_coverage_by_StFIPS_year_1980_2025.csv',dtype={'StFIPS':str})
allocated=int(subprocess.check_output(['du','-s','--block-size=1',str(CFS)],text=True).split()[0])
apparent=int(subprocess.check_output(['du','-s','--apparent-size','--block-size=1',str(CFS)],text=True).split()[0])
scratch=int(subprocess.check_output(['du','-s','--block-size=1',str(ROOT)],text=True).split()[0])
pilot=json.loads((CFS/'audits/pilot_report_2020.json').read_text())
workspace_peak=max(scratch,pilot['workspace_peak_allocated_bytes'])
staging_peak=pilot['source_staging_peak_bytes']
node_hours=state['cpu_node_hours_used'];assert node_hours<=30
release=CFS/'release'/f'GECC_CensusTract_Daymet_1980_2025_{dt.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d")}'
location=CFS/'audits/release_location.json'
if location.exists():release=Path(json.loads(location.read_text())['path'])
report=dict(created_utc=utc_now(),years=list(range(1980,2026)),rows=85528*46,all_gates_passed=True,gate_results=rows,missingness_by_StFIPS_year=coverage.to_dict('records'),nonterritory_missing_reasons=reasons['tracts'],cpu_node_hours_used=node_hours,cpu_node_hour_budget=30,cpu_jobs=state['cpu_jobs'],concurrency=bench,source_archive_bytes=sum(r['bytes'] for r in manifest['files']),cfs_allocated_bytes=allocated,cfs_apparent_bytes=apparent,scratch_allocated_bytes=scratch,workspace_peak_allocated_bytes_at_least=workspace_peak,source_staging_peak_bytes=staging_peak,source_scratch_staging_bytes=sum(p.stat().st_size for p in (ROOT/'work/staging').rglob('*') if p.is_file()),release_path=str(release),release_exists=release.exists(),GRIDMET_PRISM_comparison='Skipped by user instruction')
report['California_seasonal_inversions']=inversions
atomic_json(CFS/'audits/full_run_report.json',report)
lines=['# Daymet 1980–2025 run report','',f'All 46 years pass every gate; {85528*46:,} rows and 38 columns.',f'CPU node-hours used, including the 2020 pilot and concurrency benchmark: **{node_hours:.4f} / 30**.',f'Chosen concurrency: **{bench["chosen_workers"]} years per CPU node**.',f'Daily archive: **{report["source_archive_bytes"]:,} bytes** ({report["source_archive_bytes"]/1e12:.6f} TB).',f'CFS dataset tree allocated: {allocated:,} bytes; apparent: {apparent:,} bytes.',f'Release: `{release}`','', 'Detailed gate results: `gate_results_1980_2025.csv`.', 'Missingness: `CensusTract_Daymet_coverage_by_StFIPS_year_1980_2025.csv`.', '', '| GEOID | StFIPS | Reason |','|---|---|---|']
lines += [f'| {r["GEOID"]} | {r["StFIPS"]} | {r["reason"]} |' for r in reasons['tracts']]
lines += ['', '| Concurrent processes | Native cells/second | Seconds | Peak GB/process |','|---:|---:|---:|---:|']
lines += [f'| {r["workers"]} | {r["native_cells_per_second"]:.1f} | {r["wall_seconds"]:.1f} | {r["peak_per_process_bytes"]/1e9:.3f} |' for r in bench['results']]
lines += ['',bench['method'],'','## Missingness by StFIPS and year','',coverage.to_csv(index=False),'','## CPU job accounting','']
for job in state['cpu_jobs']:
    accounting=job.get('accounting',{});lines.append(f'- Job {job.get("jobid")}: {accounting.get("seconds")} seconds × {accounting.get("nodes")} nodes; years {job.get("years",job.get("phase"))}.')
lines += ['', '## California seasonal inversions','']+summary
(CFS/'audits/full_run_report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:report[k] for k in ['all_gates_passed','cpu_node_hours_used','source_archive_bytes','cfs_allocated_bytes','release_path']},indent=2))
