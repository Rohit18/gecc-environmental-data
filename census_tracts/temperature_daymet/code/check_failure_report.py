#!/usr/bin/env python3
"""Report the stopped check-year run without continuing or relaxing any gate."""
import json,subprocess
from common import *
state=json.loads((CFS/'audits/full_run_state.json').read_text())
assert state['status']=='stopped', 'Failure report is only for a stopped run'
checks=[1980,2000,2024,2025];audits={};status=[]
for year in [2020]+checks:
    path=CFS/'outputs'/f'CensusTract_Daymet_{year}_validation.json'
    if path.exists():
        audit=json.loads(path.read_text());audits[str(year)]=audit
        status.append(dict(year=year,validation='PASS' if audit['passed'] else 'FAIL',gates={k:v['passed'] for k,v in audit['gates'].items()}))
    else:status.append(dict(year=year,validation='NOT RUN AFTER STOP',gates=None))
roster=pd.read_parquet(POLYGONS,columns=['GEOID','StFIPS','Area_Land'])
frames=[]
for year in [2020]+checks:
    path=CFS/'outputs/working'/f'daymet_tracts_{year}.parquet'
    if path.exists():
        frame=pd.read_parquet(path,columns=['GEOID','year','missing_mean_count'])
        frame=frame.merge(roster,on='GEOID',validate='one_to_one')
        frames.append(frame)
df=pd.concat(frames,ignore_index=True)
coverage=(df.assign(all6=df.missing_mean_count==0,some=(df.missing_mean_count>0)&(df.missing_mean_count<6),none=df.missing_mean_count==6)
 .groupby(['StFIPS','year']).agg(tracts_total=('GEOID','size'),with_all_six_means=('all6','sum'),with_some_missing=('some','sum'),with_all_missing=('none','sum')).reset_index())
coverage_path=CFS/'audits/check_years_coverage_by_StFIPS_year.csv';coverage.to_csv(coverage_path,index=False)
manifest=json.loads((CFS/'manifests/daymet_manifest.json').read_text());verified=[r for r in manifest['files'] if r['status']=='verified_cfs']
assert {r['year'] for r in verified}=={2020,*checks}
bench=json.loads((CFS/'audits/concurrency_benchmark.json').read_text());source=json.loads((CFS/'audits/failed_seasonal_gate_source_recheck.json').read_text());geometry=json.loads((CFS/'audits/geometry_preflight.json').read_text())
used=int(subprocess.check_output(['du','-s','--block-size=1',str(CFS)],text=True).split()[0])
apparent=int(subprocess.check_output(['du','-s','--apparent-size','--block-size=1',str(CFS)],text=True).split()[0])
report=dict(created_utc=utc_now(),status='stopped_on_check_year_gate_failure',first_failed_year=2024,failed_gate='ordering:JJA_tmax_mean > DJF_tmax_mean for CONUS and Alaska',failed_GEOID='06075980401',year_status=status,source_daily_independent_recheck=source,concurrency=bench,cpu_node_hours_used=state['cpu_node_hours_used'],cpu_node_hour_limit=30,cpu_jobs=state['cpu_jobs'],archive_files_verified=len(verified),archive_years=sorted({r['year'] for r in verified}),source_archive_bytes=sum(r['bytes'] for r in verified),cfs_allocated_bytes=used,cfs_apparent_bytes=apparent,missingness_by_StFIPS_year=coverage.to_dict('records'),missingness_csv=str(coverage_path),release_packaged=False,remaining_years_download_started=False,geometry_reference_comparison=geometry,GRIDMET_PRISM_comparison='Skipped by user instruction')
atomic_json(CFS/'audits/check_year_failure_report.json',report)
lines=['# Daymet check-year run: stopped on a scientific gate','', 'The 2024 seasonal-ordering gate failed for GEOID **06075980401**. The requirement is JJA Tmax > same-year DJF Tmax for every CONUS/Alaska tract with data. No gate or source value was changed. The remaining-year download and release packaging did not start.', '', '| Year | Full validation |','|---|---|']
lines += [f'| {r["year"]} | {r["validation"]} |' for r in status]
lines += ['', '2024 passes the other ten gates. 2025 composites, statistics and eight independent spots completed; its full table validation was halted after the 2024 failure.', '', '| Year | JJA Tmax, °C | DJF Tmax, °C | JJA − DJF, °C |','|---|---:|---:|---:|']
lines += [f'| {r["year"]} | {r["source_daily_weighted_means_C"]["JJA"]:.9f} | {r["source_daily_weighted_means_C"]["DJF"]:.9f} | {r["JJA_minus_DJF_C"]:.9f} |' for r in source['results']]
lines += ['', 'Independent extraction directly from the original daily Tmax files reproduces both seasonal inversions exactly. The three native cells each have weight 1.0 and all 365 daily values valid; JJA has 92 days and DJF has 90. The tract has positive Census land area (419,323 m²).', '', f'CPU node-hours: **{state["cpu_node_hours_used"]:.6f} / 30**, including the pilot and canceled benchmark allocations.', f'CFS dataset tree allocated: **{used:,} bytes** ({used/1e9:.3f} GB).', f'Permanent daily archive: **{report["source_archive_bytes"]:,} bytes**, 30 verified files for 1980, 2000, 2020, 2024 and 2025.', 'All daily originals remain on CFS.', '', '## Concurrency benchmark','', '| Processes | Native cells/second | Peak GB/process |','|---:|---:|---:|']
lines += [f'| {r["workers"]} | {r["native_cells_per_second"]:.1f} | {r["peak_per_process_bytes"]/1e9:.3f} |' for r in bench['results']]
lines += ['',f'Fastest tested setting: {bench["chosen_workers"]} years per node. Four-year production ran at four-way concurrency; the selected setting was not used for remaining years because the check gate failed.', '', bench['method'], '', '## Missingness','',f'By StFIPS × year for all five computed years: `{coverage_path}`.', '', '| StFIPS | Fully missing per computed year |','|---|---:|']
for fips,g in coverage.groupby('StFIPS'):
    values=sorted(g.with_all_missing.unique())
    if max(values)>0:lines.append(f'| {fips} | {", ".join(map(str,values))} |')
lines += ['', 'All computed years have 105 completely missing tracts, no partially missing tracts, and 85,423 tracts with all six means. All other StFIPS groups have zero missing tracts. USVI has data for all 32 tracts.', '', '## Geometry comparison','', '| Check | NERSC | Off-NERSC reference |','|---|---:|---:|', f'| Invalid geometries | {geometry["projected_invalid"]} | 0 |', f'| Maximum scale-corrected deviation | {100*geometry["maximum_deviation"]["absolute_scale_corrected_deviation"]:.6f}% (15003981200) | about 1.7% (15003981200) |', f'| 02016000100 corrected ratio | {geometry["aleutian"]["scale_corrected_ratio"]:.9f} | 1.003 |', '', 'Raw planar distortion by StFIPS remains reported separately in `planar_area_distortion_by_StFIPS.csv`; it is not a gate. Projected geometry repairs: zero.', '', 'GRIDMET/PRISM comparison skipped as requested. No release path exists for this stopped run.']
(CFS/'audits/check_year_failure_report.md').write_text('\n'.join(lines)+'\n')
state['gate_results']={str(r['year']):r['gates'] for r in status if r['gates'] is not None}
state.update(failed_gate=dict(year=2024,GEOID='06075980401',gate='seasonal_ordering',JJA_tmax_mean=14.26956526265628,DJF_tmax_mean=14.461296307599104),check_years_all_gates_passed=False,release_packaged=False,failure_report=str(CFS/'audits/check_year_failure_report.md'),updated_utc=utc_now())
atomic_json(CFS/'audits/full_run_state.json',state)
print(json.dumps({k:report[k] for k in ['status','cpu_node_hours_used','source_archive_bytes','cfs_allocated_bytes','release_packaged']},indent=2))
