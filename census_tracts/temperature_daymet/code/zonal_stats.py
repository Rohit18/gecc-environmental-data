#!/usr/bin/env python3
"""Native-grid coverage-weighted exactextract statistics, retaining every tract."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import resource

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from pyproj import CRS
from common import *
from daymet_composite import verify_composite

OPS=['mean','median','stdev','count']
OP_SUFFIX={'mean':'mean','median':'median','stdev':'stdDev','count':'count'}


def reduce_composite(composite,tracts):
    with rasterio.open(composite) as src:
        if list(src.descriptions)!=BAND_NAMES:raise ValueError('Composite band order mismatch')
        if not CRS(src.crs).equals(CRS(tracts.crs),ignore_axis_order=True):raise ValueError('Polygon/raster CRS mismatch')
    res=exact_extract(str(composite),tracts[['GEOID','geometry']],OPS,include_cols=['GEOID'],output='pandas')
    if len(res)!=len(tracts) or not np.array_equal(res.GEOID.to_numpy(),tracts.GEOID.to_numpy()):
        raise ValueError('exactextract changed the tract row set/order')
    out=pd.DataFrame(index=range(len(tracts)))
    for index,band in enumerate(BAND_NAMES,1):
        columns={}
        for op in OPS:
            name=next((n for n in [f'{band}_{op}',f'band_{index}_{op}'] if n in res.columns),None)
            if name is None:raise ValueError(f'exactextract missing {band} {op}')
            columns[op]=res[name].to_numpy(dtype='float64')
        missing=~(columns['count']>0)
        for op in OPS:
            values=columns[op].copy();values[missing]=np.nan
            if not np.isfinite(values[~missing]).all():raise ValueError(f'{band} {op}: non-finite with positive count')
            out[f'{band}_{OP_SUFFIX[op]}']=values
    return out[METRIC_COLUMNS]


def build_table(metrics,tracts,year,counts=None):
    counts=counts or expected_day_counts(year)
    missing=metrics[[f'{b}_mean' for b in BAND_NAMES]].isna().sum(axis=1).to_numpy(dtype='int64')
    table=pd.DataFrame(dict(GEOID=tracts.GEOID.to_numpy(),source_area_m2=tracts.Area_Land.to_numpy(dtype='float64'),
        dataset='DAYMET',geography_type='CensusTracts',year=year,djf_convention='same-year',**counts,
        source_scale_m=1000.,reduction_scale_m=1000.,missing_mean_count=missing,
        all_six_means_valid=(missing==0).astype('int64'),fallback_applied=False))
    for column in METRIC_COLUMNS:table[column]=metrics[column].to_numpy(dtype='float64')
    return table[OUTPUT_COLUMNS].astype(OUTPUT_DTYPES)


def run_year(job):
    year,args=job
    started=time.monotonic()
    tracts=load_roster(args.polygons)
    manifest={r['filename']:r for r in json.loads(Path(args.manifest).read_text())['files']}
    composites={d:composite_path(args.composite_dir,year,d) for d in DOMAINS}
    sidecars={d:verify_composite(p,manifest) for d,p in composites.items()}
    crs=CRS(sidecars['na']['crs'])
    if any(not CRS(s['crs']).equals(crs) for s in sidecars.values()):raise ValueError('CRS mismatch across domains')
    projected,repairs=project_tracts(tracts,crs)
    geometry_audit=json.loads((CFS/'audits/geometry_preflight.json').read_text())
    if not geometry_audit.get('passed') or not CRS(geometry_audit['crs']).equals(crs):raise ValueError('Geometry gate/CRS mismatch')
    vi=projected.loc[tracts.StFIPS=='78'].reset_index(drop=True)
    vi_metrics=reduce_composite(composites['pr'],vi)
    vi_has_data=vi_metrics[[f'{b}_count' for b in BAND_NAMES]].gt(0).any(axis=1)
    usvi_covered=bool(vi_has_data.any())
    domains=assign_domains(tracts.StFIPS,usvi_covered)
    metrics=pd.DataFrame(np.nan,index=tracts.index,columns=METRIC_COLUMNS,dtype='float64')
    for domain in DOMAINS:
        index=np.flatnonzero(domains.to_numpy()==domain)
        subset=projected.iloc[index].reset_index(drop=True)
        reduced=reduce_composite(composites[domain],subset)
        metrics.iloc[index]=reduced.to_numpy()
        print(f'{year} {domain}: {len(index)} tracts reduced',flush=True)
    table=build_table(metrics,tracts,year)
    output=Path(args.out_dir)/f'daymet_tracts_{year}.parquet'
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.part')
    table.to_parquet(temporary,index=False);os.replace(temporary,output)
    assignment=pd.DataFrame(dict(GEOID=tracts.GEOID,StFIPS=tracts.StFIPS,domain=domains))
    assignment.to_parquet(CFS/'audits'/f'domain_assignment_{year}.parquet',index=False)
    counts=assignment.groupby(['StFIPS','domain']).size().reset_index(name='tracts')
    counts.to_csv(CFS/'audits'/f'domain_assignment_counts_{year}.csv',index=False)
    audit=dict(year=year,passed=True,created_utc=utc_now(),usvi_covered=usvi_covered,
               usvi_tracts_with_valid_cells=int(vi_has_data.sum()),usvi_total=len(vi),
               usvi_by_tract=[dict(GEOID=g,has_valid_cells=bool(v)) for g,v in zip(vi.GEOID,vi_has_data)],
               make_valid_changed=repairs,domain_counts=counts.to_dict('records'),
               table_sha256=sha256_of(output),composite_sha256s={d:s['composite_sha256'] for d,s in sidecars.items()},
               elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    atomic_json(CFS/'audits'/f'zonal_stats_{year}.json',audit)
    print(f'{year}: {len(table)} rows; missing all means {int((table.missing_mean_count==6).sum())}; {audit["elapsed_seconds"]:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years',default='2020')
    parser.add_argument('--polygons',default=str(POLYGONS))
    parser.add_argument('--manifest',default=str(CFS/'manifests/daymet_manifest.json'))
    parser.add_argument('--composite-dir',default=str(CFS/'composites'))
    parser.add_argument('--out-dir',default=str(CFS/'outputs/working'))
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    years=parse_years(args.years)
    pending=[]
    for year in years:
        output=Path(args.out_dir)/f'daymet_tracts_{year}.parquet'
        audit_path=CFS/'audits'/f'zonal_stats_{year}.json'
        if output.exists() and audit_path.exists():
            audit=json.loads(audit_path.read_text())
            if not audit.get('passed') or sha256_of(output)!=audit['table_sha256']:raise ValueError('Existing table checksum mismatch')
            for domain in DOMAINS:
                side=json.loads(composite_path(args.composite_dir,year,domain).with_suffix('.json').read_text())
                if side['composite_sha256']!=audit['composite_sha256s'][domain]:raise ValueError('Composite changed since table reduction')
            print(f'{year}: verified tract table exists; skipped',flush=True)
        else:pending.append((year,args))
    if pending:
        with ProcessPoolExecutor(max_workers=min(max(1,args.workers),len(pending))) as pool:list(pool.map(run_year,pending))


if __name__=='__main__':main()
