#!/usr/bin/env python3
"""Stream native Daymet chunk-aligned row blocks to complete-day period means."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

import netCDF4
import numpy as np
import rasterio
from rasterio.windows import Window
from common import *


def complete_means(values, dates, year):
    """Small-array reference used by scientific mask tests, not production cubes."""
    check_date_set(dates,expected_dates(year),'complete means')
    result={}
    for period,months in PERIOD_MONTHS.items():
        selected=np.asarray([d.month in months for d in dates])
        block=values[selected]
        valid=np.isfinite(block)
        result[period]=np.where(valid.all(axis=0),np.where(valid,block,0.).sum(axis=0)/len(block),np.nan)
    return result


def verify_composite(path, manifest):
    path=Path(path)
    side=json.loads(path.with_suffix('.json').read_text())
    if not side.get('passed') or path.stat().st_size!=side['composite_bytes'] or sha256_of(path)!=side['composite_sha256']:
        raise ValueError(f'{path}: existing composite checksum/audit mismatch')
    for filename,info in side['inputs'].items():
        record=manifest[filename]
        if record['status']!='verified_cfs' or record['sha256']!=info['sha256']:
            raise ValueError('Composite input manifest mismatch')
    with rasterio.open(path) as src:
        if list(src.descriptions)!=BAND_NAMES or src.transform.a!=1000 or src.transform.e!=-1000:
            raise ValueError('Existing composite band/grid mismatch')
    return side


def source_attributes(var):
    return {key:np.asarray(var.getncattr(key)).tolist() for key in
            ('units','_FillValue','missing_value','scale_factor','add_offset','grid_mapping') if key in var.ncattrs()}


def build_domain(year,domain,args):
    started=time.monotonic()
    manifest={r['filename']:r for r in json.loads(Path(args.manifest).read_text())['files']}
    output=composite_path(args.out_dir,year,domain)
    if output.exists() and output.with_suffix('.json').exists():
        side=verify_composite(output,manifest)
        print(f'{year} {domain}: verified composite exists; skipped',flush=True)
        return side
    source_files=[source_path(year,domain,measure,args.daily_dir) for measure in ('tmax','tmin')]
    inputs={}
    for path in source_files:
        record=manifest[path.name]
        if record['status']!='verified_cfs' or record['cfs_path']!=str(path) or not path.is_file() or path.stat().st_size!=record['bytes']:
            raise ValueError(f'{path.name}: source is not verified at the manifest CFS path')
        # Actual hash, not just a matching filename or prior manifest assertion.
        read_path=immutable_read_path(path)
        if sha256_of(read_path)!=record['sha256']:
            raise ValueError(f'{path.name}: source checksum mismatch')
        inputs[path.name]={'sha256':record['sha256'],'path_used':str(read_path),'archive_path':str(path),'bytes':record['bytes']}
    with netCDF4.Dataset(immutable_read_path(source_files[0])) as ds:
        crs,transform,x,y=read_grid(ds,'tmax')
    grid=dict(crs=crs.to_wkt(),transform=list(transform)[:6],height=len(y),width=len(x),spacing_m=1000)
    grid_file=CFS/'manifests'/f'grid_{domain}.json'
    if grid_file.exists():
        if json.loads(grid_file.read_text())!=grid:
            raise ValueError(f'{domain}: source CRS/transform/shape changed across years')
    else: atomic_json(grid_file,grid)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.part.tif')
    profile=dict(driver='GTiff',height=len(y),width=len(x),count=6,dtype='float64',
                 crs=crs.to_wkt(),transform=transform,nodata=np.nan,tiled=True,
                 blockxsize=256,blockysize=256,compress='zstd',predictor=3,BIGTIFF='IF_SAFER')
    audit=dict(dataset='DAYMET',version='4.1',year=year,domain=domain,units='degrees C',
               bands=BAND_NAMES,djf_convention='same-year',calendar_rule='365 days; leap years include February 29 and omit December 31',
               grid=grid,crs=crs.to_wkt(),inputs=inputs,periods={},source_metadata={},raw_C={},
               **expected_day_counts(year))
    with rasterio.Env(GDAL_CACHEMAX=256*1024*1024):
        with rasterio.open(temporary,'w',**profile) as dst:
            for i,name in enumerate(BAND_NAMES,1):dst.set_band_description(i,name)
            for measure,path in zip(('tmax','tmin'),source_files):
                with netCDF4.Dataset(immutable_read_path(path)) as ds:
                    source_crs,source_transform,sx,sy=read_grid(ds,measure)
                    if source_crs!=crs or source_transform!=transform or not np.array_equal(x,sx) or not np.array_equal(y,sy):
                        raise ValueError(f'{path.name}: tmax/tmin grid mismatch')
                    dates=decode_dates(ds,year)
                    variable=ds.variables[measure]
                    attributes=source_attributes(variable)
                    if str(attributes.get('units','')).lower() not in ('degrees c','degree_celsius','degrees_celsius','degc','celsius'):
                        raise ValueError(f'{path.name}: source units are not Celsius')
                    if variable.dimensions!=('time','y','x'):
                        raise ValueError(f'{path.name}: unexpected dimensions {variable.dimensions}')
                    chunking=variable.chunking()
                    ct,cy,cx=chunking if isinstance(chunking,list) else [1,min(64,len(y)),len(x)]
                    # Align rows to source chunks. If an entire daily plane is
                    # one chunk, this reads ONE plane at a time, never 365.
                    row_block=cy
                    time_block=ct
                    estimated_bytes=time_block*row_block*len(x)*8
                    if estimated_bytes>2*1024**3:
                        raise ValueError(f'{path.name}: source chunk read exceeds 2 GiB streaming budget')
                    variable.set_auto_maskandscale(True)
                    variable.set_var_chunk_cache(size=min(512*1024**2,max(16*1024**2,ct*cy*cx*4)),nelems=1009,preemption=.75)
                    audit['source_metadata'][measure]=dict(attributes=attributes,chunking=chunking,row_block=row_block,time_block=time_block)
                    raw_min,raw_max=np.inf,-np.inf
                    selectors={p:np.asarray([d.month in months for d in dates]) for p,months in PERIOD_MONTHS.items()}
                    period_info={p:dict(expected_days=len(expected_dates(year,p)),dates=[d.isoformat() for d in expected_dates(year,p)],
                        cells_total=len(x)*len(y),cells_all_days_valid=0,cells_some_days_missing_set_nan=0,cells_no_valid_day=0,
                        mean_degC_min=None,mean_degC_max=None) for p in PERIOD_MONTHS}
                    for row in range(0,len(y),row_block):
                        height=min(row_block,len(y)-row)
                        sums={p:np.zeros((height,len(x)),dtype='float64') for p in PERIOD_MONTHS}
                        counts={p:np.zeros((height,len(x)),dtype='uint16') for p in PERIOD_MONTHS}
                        for t0 in range(0,len(dates),time_block):
                            t1=min(t0+time_block,len(dates))
                            # netCDF4 masks raw fill values before applying scale;
                            # do not apply scale_factor/add_offset ourselves.
                            masked=np.ma.asarray(variable[t0:t1,row:row+height,:],dtype='float64')
                            values=masked.filled(np.nan)
                            finite=np.isfinite(values)
                            if finite.any():
                                lo=float(values[finite].min());hi=float(values[finite].max())
                                raw_min=min(raw_min,lo);raw_max=max(raw_max,hi)
                                if lo<RAW_C_RANGE[0] or hi>RAW_C_RANGE[1]:
                                    raise ValueError(f'{path.name}: daily Celsius outside -70..60: {lo}..{hi}')
                            for period in PERIOD_MONTHS:
                                selected=selectors[period][t0:t1]
                                if selected.any():
                                    v=values[selected];ok=finite[selected]
                                    sums[period]+=np.where(ok,v,0.).sum(axis=0)
                                    counts[period]+=ok.sum(axis=0,dtype='uint16')
                            del masked,values,finite
                        for period in PERIOD_MONTHS:
                            expected=len(expected_dates(year,period));count=counts[period]
                            complete=count==expected
                            mean=np.full(count.shape,np.nan,dtype='float64')
                            mean[complete]=sums[period][complete]/expected
                            info=period_info[period]
                            info['cells_all_days_valid']+=int(complete.sum())
                            info['cells_some_days_missing_set_nan']+=int(((count>0)&(count<expected)).sum())
                            info['cells_no_valid_day']+=int((count==0).sum())
                            if complete.any():
                                lo=float(mean[complete].min());hi=float(mean[complete].max())
                                if lo< -80 or hi>70:raise ValueError('Period mean outside -80..70 C')
                                info['mean_degC_min']=min(lo,info['mean_degC_min']) if info['mean_degC_min'] is not None else lo
                                info['mean_degC_max']=max(hi,info['mean_degC_max']) if info['mean_degC_max'] is not None else hi
                            dst.write(mean,BAND_NAMES.index(f'{period}_{measure}')+1,window=Window(0,row,len(x),height))
                        del sums,counts
                        if domain=='na':
                            print(f'{year} {domain} {measure}: streamed rows {row+height}/{len(y)}; elapsed {time.monotonic()-started:.1f}s',flush=True)
                    if not np.isfinite(raw_min):raise ValueError('Source has no valid daily cells')
                    audit['raw_C'][measure]=[raw_min,raw_max]
                    for period,info in period_info.items():audit['periods'][f'{period}_{measure}']=info
                    print(f'{year} {domain} {measure}: raw {raw_min:.3f}..{raw_max:.3f} C',flush=True)
            dst.update_tags(dataset='DAYMET',version='4.1',year=str(year),domain=domain,units='degrees C',djf_convention='same-year')
    os.replace(temporary,output)
    audit.update(passed=True,composite_bytes=output.stat().st_size,composite_sha256=sha256_of(output),
                 elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 created_utc=utc_now())
    atomic_json(output.with_suffix('.json'),audit)
    print(f'{year} {domain}: composite audited in {audit["elapsed_seconds"]:.1f}s; peak RSS {audit["peak_rss_bytes"]/1e9:.3f} GB',flush=True)
    return audit


def run_year(job):
    year,args=job
    # A fresh process for each domain gives a true domain-specific RSS peak;
    # otherwise ru_maxrss would retain the previous North America peak in HI/PR.
    results=[]
    for domain in DOMAINS:
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            results.append(pool.submit(build_domain,year,domain,args).result())
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years',default='2020')
    parser.add_argument('--manifest',default=str(CFS/'manifests/daymet_manifest.json'))
    parser.add_argument('--daily-dir',default=str(CFS/'daily'))
    parser.add_argument('--out-dir',default=str(CFS/'composites'))
    parser.add_argument('--workers',type=int,default=1,help='parallel years within one CPU node')
    args=parser.parse_args()
    geometry=json.loads((CFS/'audits/geometry_preflight.json').read_text())
    if not geometry.get('passed') or geometry.get('scale_corrected_threshold')!=.02:
        raise SystemExit('Approved geometry gate must pass before compositing')
    years=parse_years(args.years)
    with ProcessPoolExecutor(max_workers=min(max(1,args.workers),len(years))) as pool:
        list(pool.map(run_year,[(year,args) for year in years]))


if __name__=='__main__':main()
