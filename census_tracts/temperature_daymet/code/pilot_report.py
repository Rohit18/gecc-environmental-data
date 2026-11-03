#!/usr/bin/env python3
"""Record pilot results and resource estimates, then stop for review."""
import json
import argparse
from pathlib import Path
import subprocess
from common import *


def quota_bytes(value):
    import re
    number,unit=re.fullmatch(r'([0-9.]+)([A-Za-z]+)',value.strip()).groups()
    units={'TiB':1024**4,'GiB':1024**3,'MiB':1024**2,'TB':10**12,'GB':10**9}
    return float(number)*units[unit]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quota-file',default=str(CFS/'audits/quota_after_pilot.txt'),help='showquota output captured on a login node')
    args=parser.parse_args()
    validation=json.loads((CFS/'outputs/CensusTract_Daymet_2020_validation.json').read_text())
    if not validation['passed']:raise SystemExit('Pilot report refuses failed validation; inspect its gate JSON')
    manifest=json.loads((CFS/'manifests/daymet_manifest.json').read_text())
    files=[r for r in manifest['files'] if r['year']==2020]
    sides={d:json.loads(composite_path(CFS/'composites',2020,d).with_suffix('.json').read_text()) for d in DOMAINS}
    stats=json.loads((CFS/'audits/zonal_stats_2020.json').read_text())
    spot=json.loads((CFS/'audits/spot_checks_2020.json').read_text())
    total_source=sum(r['cmr_bytes'] for r in manifest['files'])
    per_year_composite=sum(s['composite_bytes'] for s in sides.values())
    table_bytes=sum(p.stat().st_size for p in (CFS/'outputs/per_year').glob('*2020*'))
    combined_bytes=sum((CFS/'outputs'/name).stat().st_size for name in ['CensusTract_Daymet_2020.parquet','CensusTract_Daymet_2020.csv.gz'])
    projected_cfs=total_source+46*(per_year_composite+2*table_bytes+2*combined_bytes)+(CFS/'outputs/working/daymet_tracts_2020.parquet').stat().st_size*46
    current_cfs_bytes=sum(p.stat().st_size for p in CFS.rglob('*') if p.is_file())
    workspace_peak_bytes=int(subprocess.check_output(['du','-s','-B1',str(ROOT)],text=True).split()[0])
    previous_report=CFS/'audits/pilot_report_2020.json'
    if previous_report.exists():
        workspace_peak_bytes=max(workspace_peak_bytes,json.loads(previous_report.read_text()).get('workspace_peak_allocated_bytes',0))
    quota_path=Path(args.quota_file)
    if quota_path.exists():quota_text=quota_path.read_text()
    else:quota_text=subprocess.check_output(['bash','-lc','showquota your_nersc_project'],text=True)
    (CFS/'audits/quota_after_pilot.txt').write_text(quota_text)
    quotas={}
    for line in quota_text.splitlines():
        if '|' not in line:continue
        parts=[p.strip() for p in line.split('|')]
        if len(parts)>4 and parts[1] in ['pscratch','your_nersc_project']:
            used=quota_bytes(parts[2]);limit=quota_bytes(parts[3])
            quotas[parts[1]]=dict(used_bytes=used,quota_bytes=limit,free_bytes=limit-used)
    composite_seconds=sum(s['elapsed_seconds'] for s in sides.values())
    # This is an extrapolation from one pilot, not a measured full-series job.
    cpu_year_seconds=composite_seconds+stats['elapsed_seconds']+spot['elapsed_seconds']+validation['elapsed_seconds']
    total_cpu_node_hours=46*cpu_year_seconds/3600.
    aggregate_MB_s=sum(r.get('throughput_MB_s',0) for r in files if r['domain']=='na')
    download_estimate=total_source/1e6/aggregate_MB_s if aggregate_MB_s>0 else None
    source_staging=json.loads((CFS/'audits/download_2020.json').read_text())['source_staging_peak_bytes']
    jobs=json.loads((CFS/'audits/pilot_jobs.json').read_text())
    fetch_elapsed=None
    records=subprocess.check_output(['sacct','-j',str(jobs['fetch_job_id']),'--format=JobID,ElapsedRaw','-n','-P'],text=True)
    for line in records.splitlines():
        fields=line.split('|')
        if fields[0]==str(jobs['fetch_job_id']):fetch_elapsed=int(fields[1])
    report=dict(pilot_year=2020,passed=True,created_utc=utc_now(),stopped_for_user_review=True,post_pilot_approved=False,
        gates=validation['gates'],reported=validation['reported'],geometry=json.loads((CFS/'audits/geometry_preflight.json').read_text()),
        downloads=[{k:r.get(k) for k in ['filename','bytes','download_seconds','throughput_MB_s','sha256','cfs_path']} for r in files],
        composites={d:{k:s[k] for k in ['elapsed_seconds','peak_rss_bytes','composite_bytes','composite_sha256','source_metadata']} for d,s in sides.items()},
        statistics_seconds=stats['elapsed_seconds'],statistics_peak_rss_bytes=stats['peak_rss_bytes'],
        spot_check_seconds=spot['elapsed_seconds'],merge_validate_seconds=validation['elapsed_seconds'],
        fetch_and_archive_job_seconds=fetch_elapsed,
        workspace_peak_allocated_bytes=workspace_peak_bytes,
        source_staging_peak_bytes=source_staging,archive_2020_bytes=sum(r['bytes'] for r in files),
        full_series_source_bytes=total_source,projected_full_series_cfs_bytes=projected_cfs,projected_additional_cfs_bytes=max(0,projected_cfs-current_cfs_bytes),quotas_after_pilot=quotas,
        estimates=dict(cpu_serial_node_hours_and_wall_hours_46years=total_cpu_node_hours,
            ideal_cpu_node_hours_and_wall_hours_with_4_year_workers=total_cpu_node_hours/4,
            download_wall_hours_with_2_large_streams=download_estimate/3600 if download_estimate else None,
            fetch_archive_verify_wall_hours_full_series=fetch_elapsed*total_source/sum(r['bytes'] for r in files)/3600 if fetch_elapsed else None,
            caveats='Extrapolated from 2020, including composites, statistics, spot checks and validation. Four-worker estimate assumes ideal scaling within one CPU node; actual I/O contention will reduce it. Queue waits and packaging are additional. Source staging limited to one year, <=150 GB.'))
    atomic_json(CFS/'audits/pilot_report_2020.json',report)
    text='# Daymet 2020 pilot — stopped for review\n\nAll validation gates passed. No post-pilot download or job has been submitted.\n\n'
    text+='| Gate | Passed |\n|---|---|\n'+''.join(f'| {name} | {gate["passed"]} |\n' for name,gate in validation['gates'].items())
    text+='\nDetailed missingness, USVI coverage, reference comparisons, per-file throughput, per-domain time/memory, quotas and extrapolations are in `pilot_report_2020.json`.\n'
    text+='\nSource archive: '+str(CFS/'daily/2020')+'\n\nTables: '+str(CFS/'outputs/per_year')+'\n'
    text+='\n## Geometry reference comparison\n\n| Check | NERSC | Off-NERSC reference |\n|---|---:|---:|\n'
    geometry=report['geometry']
    text+=f'| Invalid projected geometries | {geometry["projected_invalid"]} | 0 |\n'
    text+=f'| Empty projected geometries | {geometry["projected_empty"]} | — |\n'
    text+=f'| make_valid changes | {len(geometry["make_valid_changed"])} | — |\n'
    text+=f'| Maximum corrected deviation | {100*geometry["maximum_deviation"]["absolute_scale_corrected_deviation"]:.6f}% ({geometry["maximum_deviation"]["GEOID"]}) | about 1.7% (15003981200) |\n'
    text+=f'| 02016000100 corrected ratio | {geometry["aleutian"]["scale_corrected_ratio"]:.9f} | 1.003 |\n'
    text+='\nRing traversal is normalized only for signed geodesic-area measurement. The audit retains all unoriented-area results; no source vertices or topology changed. Raw planar distortion is reported separately and projection round-trip is not a gate.\n'
    text+='\n### Raw planar/geodesic distortion by StFIPS (reported, not gated)\n\n| StFIPS | Tracts | Minimum ratio | Median ratio | Maximum ratio |\n|---|---:|---:|---:|---:|\n'
    for row in geometry['raw_ratio_by_StFIPS']:
        text+=f'| {row["StFIPS"]} | {row["count"]} | {row["min"]:.6f} | {row["median"]:.6f} | {row["max"]:.6f} |\n'
    text+='\n## Missingness by state and territory, 2020\n\n| StFIPS | Tracts | All six means | Some missing | All missing |\n|---|---:|---:|---:|---:|\n'
    for row in validation['reported']['missingness_by_StFIPS_year']:
        text+=f'| {row["StFIPS"]} | {row["tracts_total"]} | {row["with_all_six_means"]} | {row["with_some_missing"]} | {row["with_all_missing"]} |\n'
    text+='\nThe hard missingness gate uses CONUS tracts with Area_Land > 0 only. Territory and water-only counts are reported without that gate.\n'
    text+='\n### Water-only tracts\n\n| StFIPS | Water-only tracts | With all six means |\n|---|---:|---:|\n'
    for row in validation['reported']['water_only_by_StFIPS_year']:
        text+=f'| {row["StFIPS"]} | {row["tracts"]} | {row["with_all_six_means"]} |\n'
    text+=f'\nUSVI coverage: {stats["usvi_covered"]}; {stats["usvi_tracts_with_valid_cells"]} of {stats["usvi_total"]} tracts intersect valid cells. See zonal_stats_2020.json for individual tract coverage.\n'
    text+='\n## Transfer, compute and storage\n\n| File | GB | Download seconds | MB/s |\n|---|---:|---:|---:|\n'
    for row in files:
        text+=f'| {row["filename"]} | {row["bytes"]/1e9:.6f} | {row.get("download_seconds",0):.2f} | {row.get("throughput_MB_s",0):.2f} |\n'
    text+=f'\nFetch/archive/verification job elapsed: {fetch_elapsed} seconds.\n'
    text+='\n| Domain | Composite seconds | Peak process RSS, GB | Composite GB |\n|---|---:|---:|---:|\n'
    for domain,side in sides.items():text+=f'| {domain} | {side["elapsed_seconds"]:.2f} | {side["peak_rss_bytes"]/1e9:.4f} | {side["composite_bytes"]/1e9:.4f} |\n'
    text+=f'\nZonal statistics: {stats["elapsed_seconds"]:.2f} seconds; peak RSS {stats["peak_rss_bytes"]/1e9:.4f} GB. Spot checks: {spot["elapsed_seconds"]:.2f} seconds. Merge/validation: {validation["elapsed_seconds"]:.2f} seconds.\n'
    text+=f'\nPeak source staging: {source_staging/1e9:.6f} GB. Measured whole-workspace allocated bytes before purge: {workspace_peak_bytes/1e9:.6f} GB. 2020 daily CFS archive: {report["archive_2020_bytes"]/1e9:.6f} GB.\n'
    text+=f'\nProjected 1980–2025 source archive: {total_source/1e12:.6f} TB. Projected CFS total including composites, working/per-year/combined tables and release table copies: {projected_cfs/1e12:.6f} TB.\n'
    for name,quota in quotas.items():text+=f'\n{name}: remaining quota {quota["free_bytes"]/1024**4:.4f} TiB after the pilot.\n'
    text+='\n### Full-series estimates\n\nThese are pilot extrapolations, not a submitted run.\n\n'
    for key,value in report['estimates'].items():
        if key!='caveats' and value is not None:text+=f'- {key}: {value:.3f} hours\n'
    text+='\n'+report['estimates']['caveats']+'\n'
    text+='\n## ERA5-Land comparison, 2020\n\nOnly tract-years with all six means in both sources are used. Differences are Daymet minus reference.\n\n| Metric | Shared tracts | Mean difference, C | RMSE, C | Pearson r |\n|---|---:|---:|---:|---:|\n'
    for name,metrics in validation['reported']['ERA5Land_comparison']['metrics'].items():
        text+=f'| {name} | {metrics["n"]} | {metrics["mean_difference_C"]:.6f} | {metrics["rmse_C"]:.6f} | {metrics["pearson_correlation"]:.6f} |\n'
    text+='\nGRIDMET/PRISM: '+validation['reported']['GRIDMET_PRISM_comparison']+'.\n'
    text+='\n## Partial validity and fractional counts\n\n| Domain | Band | Partial-validity cells set to NaN |\n|---|---|---:|\n'
    for domain,side in sides.items():
        for band,info in side['periods'].items():text+=f'| {domain} | {band} | {info["cells_some_days_missing_set_nan"]} |\n'
    text+=f'\nDistinct JJA_tmax_mean values in 2020: {validation["reported"]["distinct_JJA_tmax_mean_by_year"]["2020"]}.\n'
    text+='\n| Band | Fraction of all tracts with count < 1 | Fraction with data and count < 1 |\n|---|---:|---:|\n'
    for band,info in validation['reported']['count_below_1_by_band'].items():
        text+=f'| {band} | {info["fraction_all_tracts"]:.6f} | {info["fraction_with_data"]:.6f} |\n'
    (CFS/'audits/pilot_report_2020.md').write_text(text)
    preflight_path=CFS/'audits/input_storage_preflight.json'
    preflight=json.loads(preflight_path.read_text())
    if 'area_gate' in preflight:preflight['superseded_raw_area_gate']=preflight.pop('area_gate')
    preflight.update(status='pilot_completed_all_gates_passed',pilot_2020_run=True,pipeline_adaptation_complete=True,blocking_gate=None,pilot_report=str(CFS/'audits/pilot_report_2020.json'))
    atomic_json(preflight_path,preflight)
    (ROOT/'PREFLIGHT_STATUS.md').write_text('# Daymet workflow status\n\n2020 pilot completed and all gates passed. STOPPED for review. No later year is approved or scheduled.\n\nDurable report: '+str(CFS/'audits/pilot_report_2020.json')+'\n')
    print('2020 PILOT PASSED — STOPPED for review; no full-series job submitted',flush=True)


if __name__=='__main__':main()
