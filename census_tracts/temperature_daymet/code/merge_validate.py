#!/usr/bin/env python3
"""Write matching 38-column release tables and enforce every scientific gate."""
import argparse
import json
import os
import time
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import CRS
from common import *
from daymet_composite import verify_composite

ERA5_REFERENCE=Path('/path/to/project/gecc/era5_land/release/GECC_CensusTract_ERA5Land_1980_2025_20261012/data/CensusTract_ERA5Land_1980_2025.parquet')
MEAN_COLUMNS=[f'{b}_mean' for b in BAND_NAMES]
SEASONAL_GATE_POLICY='CONUS_AK_outside_CA_hard_CA_report_20261013'
INVERSION_COLUMNS=['GEOID','year','JJA_tmax_mean','DJF_tmax_mean','margin']


def temperature_ordering_violations(df):
    return {p:int(((df[f'{p}_tmax_mean']<df[f'{p}_tmin_mean']) & df[f'{p}_tmax_mean'].notna() & df[f'{p}_tmin_mean'].notna()).sum()) for p in PERIOD_MONTHS}


def seasonal_ordering_results(df):
    scope=(~df.StFIPS.isin(NON_CONUS_FIPS)|(df.StFIPS=='02'))&df.JJA_tmax_mean.notna()&df.DJF_tmax_mean.notna()
    inverted=df.loc[scope & (df.JJA_tmax_mean<=df.DJF_tmax_mean),['GEOID','StFIPS','year','JJA_tmax_mean','DJF_tmax_mean']].copy()
    inverted['margin']=inverted.JJA_tmax_mean-inverted.DJF_tmax_mean
    california=inverted.loc[inverted.StFIPS=='06',INVERSION_COLUMNS].sort_values(['year','GEOID']).reset_index(drop=True)
    hard=inverted.loc[inverted.StFIPS!='06',INVERSION_COLUMNS].sort_values(['year','GEOID']).reset_index(drop=True)
    return hard,california


def write_csv(df,path,compression=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.part')
    df.to_csv(temporary,index=False,lineterminator='\n',compression=compression)
    os.replace(temporary,path)


def write_parquet(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.part')
    df.to_parquet(temporary,index=False);os.replace(temporary,path)


def read_csv_typed(path):
    strings={c:str for c,t in OUTPUT_DTYPES.items() if t=='str'}
    df=pd.read_csv(path,dtype=strings,keep_default_na=False,float_precision='round_trip',
                   na_values={c:[''] for c,t in OUTPUT_DTYPES.items() if t=='float64'})
    if list(df.columns)!=OUTPUT_COLUMNS:raise ValueError('CSV schema/order mismatch')
    df['fallback_applied']=df.fallback_applied.map({'False':False,'True':True,False:False,True:True})
    return df.astype(OUTPUT_DTYPES)


def frames_identical(a,b):
    if list(a.columns)!=list(b.columns) or len(a)!=len(b):return False
    for column in a.columns:
        if a[column].dtype!=b[column].dtype:return False
        if a[column].dtype=='float64':
            if not np.array_equal(a[column].to_numpy(),b[column].to_numpy(),equal_nan=True):return False
        elif not (a[column].to_numpy()==b[column].to_numpy()).all():return False
    return True


def compare_reference(df,path):
    reference=pd.read_parquet(path,columns=['GEOID','year','dataset','all_six_means_valid']+MEAN_COLUMNS,filters=[('year','==',2020)])
    current=df.loc[(df.year==2020)&(df.all_six_means_valid==1)]
    output=dict(path=str(path),year=2020,selection='both sources have all six means',metrics={})
    for dataset,group in reference.groupby('dataset'):
        group=group.loc[group.all_six_means_valid==1]
        joined=current.merge(group,on=['GEOID','year'],suffixes=('_daymet','_ref'),validate='one_to_one')
        for column in MEAN_COLUMNS:
            left=joined[column+'_daymet'].to_numpy();right=joined[column+'_ref'].to_numpy();difference=left-right
            output['metrics'][f'{dataset}:{column}']=dict(n=len(joined),mean_difference_C=float(difference.mean()) if len(joined) else None,
                median_difference_C=float(np.median(difference)) if len(joined) else None,
                sd_difference_C=float(difference.std()) if len(joined) else None,
                rmse_C=float(np.sqrt((difference**2).mean())) if len(joined) else None,
                pearson_correlation=float(np.corrcoef(left,right)[0,1]) if len(joined)>1 else None)
    return output


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years',default='2020')
    parser.add_argument('--in-dir',default=str(CFS/'outputs/working'))
    parser.add_argument('--out-dir',default=str(CFS/'outputs'))
    parser.add_argument('--reference',default='',help='optional verified GRIDMET/PRISM table')
    args=parser.parse_args();years=parse_years(args.years)
    tag=str(years[0]) if len(years)==1 else f'{years[0]}_{years[-1]}'
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    originals=load_roster();expected_ids=set(originals.GEOID)
    manifest={r['filename']:r for r in json.loads((CFS/'manifests/daymet_manifest.json').read_text())['files']}
    geometry=json.loads((CFS/'audits/geometry_preflight.json').read_text())
    tests=json.loads((CFS/'audits/unit_tests.json').read_text())
    gates={};reported={};frames=[];calendar_bad=[];units_bad=[];grid_bad=[];integrity_bad=[];mask_bad=[];csv_bad=[]
    sidecars={};reference_schema=pq.read_schema(ERA5_REFERENCE).remove_metadata()
    for year in years:
        source=Path(args.in_dir)/f'daymet_tracts_{year}.parquet'
        table=pd.read_parquet(source)
        if list(table.columns)!=OUTPUT_COLUMNS:raise ValueError(f'{year}: not the exact 38-column schema')
        if pq.read_schema(source).remove_metadata()!=reference_schema:integrity_bad.append(f'{year}: Arrow physical types differ from ERA5-Land')
        if len(table)!=EXPECTED_TRACTS or set(table.GEOID)!=expected_ids or table.duplicated(['GEOID','year']).any():integrity_bad.append(f'{year}: tract rows/keys mismatch')
        if not table.GEOID.map(lambda x:isinstance(x,str) and len(x)==11 and x.isdigit()).all():integrity_bad.append(f'{year}: GEOID type/length')
        expected_area=originals.set_index('GEOID').Area_Land.reindex(table.GEOID).to_numpy(dtype='float64')
        if not np.array_equal(table.source_area_m2.to_numpy(),expected_area):integrity_bad.append(f'{year}: source area differs from Area_Land')
        for column,value in [('dataset','DAYMET'),('geography_type','CensusTracts'),('djf_convention','same-year'),('year',year),('source_scale_m',1000.),('reduction_scale_m',1000.),('fallback_applied',False)]:
            if not (table[column]==value).all():integrity_bad.append(f'{year}: {column} wrong')
        computed_missing=table[MEAN_COLUMNS].isna().sum(axis=1)
        if not np.array_equal(computed_missing.to_numpy(),table.missing_mean_count.to_numpy()) or not np.array_equal((computed_missing==0).astype('int64').to_numpy(),table.all_six_means_valid.to_numpy()):integrity_bad.append(f'{year}: missingness indicators wrong')
        for band in BAND_NAMES:
            empty=table[f'{band}_count'].isna()
            if not table.loc[empty,[f'{band}_{s}' for s in STAT_SUFFIXES]].isna().all().all():mask_bad.append(f'{year} {band}: count-null stats inconsistent')
            if table.loc[~empty,f'{band}_count'].le(0).any() or not np.isfinite(table.loc[~empty,[f'{band}_{s}' for s in STAT_SUFFIXES]].to_numpy()).all():mask_bad.append(f'{year} {band}: invalid positive-count statistics')
        sides={}
        for domain in DOMAINS:
            side=verify_composite(composite_path(CFS/'composites',year,domain),manifest);sides[domain]=side
            for measure,limits in side['raw_C'].items():
                if limits[0]<RAW_C_RANGE[0] or limits[1]>RAW_C_RANGE[1]:units_bad.append(f'{year} {domain} {measure}: raw range')
            for band,info in side['periods'].items():
                period=band.split('_')[0]
                expected=[d.isoformat() for d in expected_dates(year,period)]
                if info['dates']!=expected or info['expected_days']!=len(expected):calendar_bad.append(f'{year} {domain} {band}: exact date set')
                if info['mean_degC_min'] is not None and (info['mean_degC_min']< -80 or info['mean_degC_max']>70):units_bad.append(f'{year} {domain} {band}: composite mean range')
            if not CRS(side['crs']).equals(CRS(geometry['crs'])) or side['grid']['spacing_m']!=1000:grid_bad.append(f'{year} {domain}: CRS/spacing mismatch')
            baseline=json.loads((CFS/'manifests'/f'grid_{domain}.json').read_text())
            if side['grid']!=baseline:grid_bad.append(f'{year} {domain}: grid changed')
        sidecars[year]=sides
        for key,days in expected_day_counts(year).items():
            if not (table[key]==days).all() or any(s[key]!=days for s in sides.values()):calendar_bad.append(f'{year}: {key}')
        release_parquet=out/'per_year'/f'CensusTract_Daymet_{year}.parquet'
        release_csv=out/'per_year'/f'CensusTract_Daymet_{year}.csv'
        # Revalidation preserves existing data bytes. Export only missing files
        # from the already computed table; mismatches fail the integrity gate.
        if not release_parquet.exists():write_parquet(table,release_parquet)
        if not release_csv.exists():write_csv(table,release_csv)
        if not frames_identical(table,pd.read_parquet(release_parquet)) or not frames_identical(table,read_csv_typed(release_csv)):csv_bad.append(year)
        frames.append(table)
    df=pd.concat(frames,ignore_index=True)
    df=df.merge(originals[['GEOID','StFIPS','Area_Land']],on='GEOID',validate='many_to_one')
    conus=~df.StFIPS.isin(NON_CONUS_FIPS)
    def gate(name,passed,**detail):gates[name]=dict(passed=bool(passed),**detail)
    gate('geometry',geometry['passed'] and geometry['scale_corrected_threshold']==.02,maximum_deviation=geometry['maximum_deviation'],make_valid_changed=len(geometry['make_valid_changed']),off_nersc_reference=geometry['off_nersc_reference'],aleutian=geometry['aleutian'])
    gate('calendar',not calendar_bad,problems=calendar_bad,counts_by_year={y:expected_day_counts(y) for y in years})
    means=df[MEAN_COLUMNS].to_numpy();out_of_range=int(((means< -80)|(means>70)).sum())
    gate('units',not units_bad and out_of_range==0,problems=units_bad,tract_means_out_of_range=out_of_range)
    ordering=temperature_ordering_violations(df)
    seasonal,california=seasonal_ordering_results(df)
    inversion_csv=out/f'CensusTract_Daymet_California_seasonal_inversions_{tag}.csv'
    write_csv(california,inversion_csv)
    reported['California_seasonal_inversions']=dict(audit_csv=str(inversion_csv),total=int(len(california)),counts_by_year={int(y):int((california.year==y).sum()) for y in years},criterion='JJA_tmax_mean <= DJF_tmax_mean',margin_definition='JJA_tmax_mean - DJF_tmax_mean',gated=False)
    gate('ordering',not any(ordering.values()) and seasonal.empty,tmax_below_tmin=ordering,seasonal_gate_policy=SEASONAL_GATE_POLICY,hard_fail_scope='CONUS or Alaska with StFIPS != 06',JJA_not_above_DJF_CONUS_AK_outside_CA=seasonal.to_dict('records'))
    gate('grid',not grid_bad,problems=grid_bad,native_grid=True,raster_reprojection=False)
    gate('band_missingness',not mask_bad,problems=mask_bad)
    gate('integrity',not integrity_bad and not csv_bad and not df.duplicated(['GEOID','year']).any(),problems=integrity_bad,csv_parquet_mismatch_years=csv_bad,columns=38,rows=len(df),rows_by_year={int(y):int(len(g)) for y,g in df.groupby('year')})
    coverage=(df.assign(all6=df.missing_mean_count==0,some=(df.missing_mean_count>0)&(df.missing_mean_count<6),none=df.missing_mean_count==6)
        .groupby(['StFIPS','year']).agg(tracts_total=('GEOID','size'),with_all_six_means=('all6','sum'),with_some_missing=('some','sum'),with_all_missing=('none','sum')).reset_index())
    write_csv(coverage,out/f'CensusTract_Daymet_coverage_by_StFIPS_year_{tag}.csv')
    positive_land=df.loc[conus&(df.Area_Land>0)]
    missing=positive_land.groupby('year').missing_mean_count.apply(lambda s:float((s>0).mean()))
    gate('missingness_CONUS_positive_land',bool((missing<=.005).all()),threshold=.005,fraction_by_year={int(y):float(v) for y,v in missing.items()})
    uncovered=df.loc[df.StFIPS.isin(UNSUPPORTED_FIPS)]
    gate('uncovered_territories',len(uncovered)==101*len(years) and uncovered[METRIC_COLUMNS].isna().all().all(),tracts_per_year=101)
    reported['missingness_by_StFIPS_year']=coverage.to_dict('records')
    water=df.loc[df.Area_Land==0]
    reported['water_only_by_StFIPS_year']=water.groupby(['StFIPS','year']).agg(tracts=('GEOID','size'),with_all_six_means=('all_six_means_valid','sum')).reset_index().to_dict('records')
    reported['distinct_JJA_tmax_mean_by_year']={int(y):int(g.JJA_tmax_mean.nunique()) for y,g in df.groupby('year')}
    reported['count_below_1_by_band']={b:dict(fraction_all_tracts=float((df[f'{b}_count']<1).mean()),fraction_with_data=float((df.loc[df[f'{b}_count'].notna(),f'{b}_count']<1).mean())) for b in BAND_NAMES}
    reported['partial_validity_cells']={y:{d:{b:s['periods'][b]['cells_some_days_missing_set_nan'] for b in BAND_NAMES} for d,s in sides.items()} for y,sides in sidecars.items()}
    reported['usvi_coverage']={y:json.loads((CFS/'audits'/f'zonal_stats_{y}.json').read_text())['usvi_covered'] for y in years}
    reported['raw_planar_geodesic_ratio_by_StFIPS']=geometry['raw_ratio_by_StFIPS']
    gate('unit_tests',tests['passed'],tests_run=tests['tests_run'])
    spot={y:json.loads((CFS/'audits'/f'spot_checks_{y}.json').read_text()) for y in years}
    gate('spot_checks',all(s['passed'] for s in spot.values()),results={y:dict(tracts=len(s['tracts']),max_abs_diff=s['max_abs_diff']) for y,s in spot.items()})
    if 2020 in years:
        reported['ERA5Land_comparison']=compare_reference(df,ERA5_REFERENCE)
        if args.reference:
            if sha256_of(args.reference)!='2524fa90e2bca9695888433fdf6b28ffdc338bc97cbedf4515ae1a2d37c8ba90':raise ValueError('GRIDMET/PRISM reference hash mismatch')
            reported['GRIDMET_PRISM_comparison']=compare_reference(df,Path(args.reference))
        else:reported['GRIDMET_PRISM_comparison']='Skipped by user instruction; comparison will run outside NERSC'
    missing_rows=df.loc[df.missing_mean_count>0,['GEOID','StFIPS','year','source_area_m2','missing_mean_count']+MEAN_COLUMNS]
    write_csv(missing_rows,out/f'CensusTract_Daymet_tracts_with_missing_{tag}.csv')
    combined=df[OUTPUT_COLUMNS].sort_values(['year','GEOID']).reset_index(drop=True)
    parquet=out/f'CensusTract_Daymet_{tag}.parquet';csv=out/f'CensusTract_Daymet_{tag}.csv.gz'
    if not parquet.exists():write_parquet(combined,parquet)
    if not csv.exists():write_csv(combined,csv,compression='gzip')
    same=frames_identical(combined,pd.read_parquet(parquet)) and frames_identical(combined,read_csv_typed(csv))
    gates['integrity']['combined_csv_parquet_identical']=same
    gates['integrity']['passed'] &= same
    audit=dict(dataset='DAYMET',version='4.1',years=years,crs=geometry['crs'],domains=list(DOMAINS),calendar_rule='365 days; leap years include February 29 and omit December 31',
                gates=gates,reported=reported,passed=all(g['passed'] for g in gates.values()),created_utc=utc_now(),
                elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    atomic_json(out/f'CensusTract_Daymet_{tag}_validation.json',audit)
    print(json.dumps({k:v['passed'] for k,v in gates.items()},indent=2),flush=True)
    if not audit['passed']:raise SystemExit('Validation FAILED; no specification changed')


if __name__=='__main__':main()
