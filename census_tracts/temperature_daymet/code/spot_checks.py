#!/usr/bin/env python3
"""Independent Shapely coverage and NumPy statistics from native composites."""
import argparse
import json
import time
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import shapely
from shapely.geometry import box
from pyproj import CRS
from common import *

ABS_TOL=1e-5
COUNT_REL_TOL=1e-6


def weighted_median(values,weights):
    distinct,inverse=np.unique(values,return_inverse=True)
    w=np.bincount(inverse,weights=weights)
    n=len(distinct)
    if n==1:return float(distinct[0])
    previous=np.r_[0.,np.cumsum(w)[:-1]]
    positions=np.arange(n)*w+(n-1)*previous
    target=.5*positions[-1]
    lower=int(np.clip(np.searchsorted(positions,target,side='right')-1,0,n-2))
    if positions[lower+1]==positions[lower]:return float(distinct[lower])
    return float(distinct[lower]+(target-positions[lower])/(positions[lower+1]-positions[lower])*(distinct[lower+1]-distinct[lower]))


def independent_stats(geometry,composite):
    values=[[] for _ in BAND_NAMES];weights=[[] for _ in BAND_NAMES]
    cells_total=0;nan_cells=np.zeros(6,dtype='int64')
    with rasterio.open(composite) as src:
        transform=src.transform;res=1000.
        xmin,ymin,xmax,ymax=geometry.bounds
        c0=max(0,int(np.floor((xmin-transform.c)/res)));c1=min(src.width,int(np.ceil((xmax-transform.c)/res)))
        r0=max(0,int(np.floor((transform.f-ymax)/res)));r1=min(src.height,int(np.ceil((transform.f-ymin)/res)))
        for row in range(r0,r1,128):
            height=min(128,r1-row)
            for column in range(c0,c1,128):
                width=min(128,c1-column)
                left=transform.c+column*res;top=transform.f-row*res
                clipped=geometry.intersection(box(left,top-height*res,left+width*res,top))
                if clipped.is_empty:continue
                rows,cols=np.meshgrid(np.arange(row,row+height),np.arange(column,column+width),indexing='ij')
                cells=shapely.box(transform.c+cols.ravel()*res,transform.f-(rows.ravel()+1)*res,
                                  transform.c+(cols.ravel()+1)*res,transform.f-rows.ravel()*res)
                fraction=shapely.area(shapely.intersection(cells,clipped))/(res*res)
                keep=fraction>0
                if not keep.any():continue
                fraction=fraction[keep];cells_total+=int(keep.sum())
                data=src.read(window=Window(column,row,width,height)).reshape(6,-1)[:,keep]
                for band in range(6):
                    finite=np.isfinite(data[band]);nan_cells[band]+=int((~finite).sum())
                    if finite.any():
                        values[band].append(data[band,finite]);weights[band].append(fraction[finite])
    result={'cells_intersected':cells_total}
    for index,name in enumerate(BAND_NAMES):
        if not values[index]:result[name]={s:None for s in STAT_SUFFIXES}
        else:
            v=np.concatenate(values[index]);w=np.concatenate(weights[index]);mean=float(np.sum(v*w)/w.sum())
            result[name]=dict(mean=mean,median=weighted_median(v,w),stdDev=float(np.sqrt(np.sum(w*(v-mean)**2)/w.sum())),count=float(w.sum()))
        result[name+'_nan_cells_intersected']=int(nan_cells[index])
    return result


def choose_tracts(original,projected,table,sidecars,usvi_covered):
    t=projected.merge(table[['GEOID','annual_tmax_count','missing_mean_count']],on='GEOID')
    conus=t.loc[~t.StFIPS.isin(NON_CONUS_FIPS)]
    transform=rasterio.Affine(*sidecars['na']['grid']['transform'])
    bounds=conus.geometry.bounds
    single=((np.floor((bounds.minx-transform.c)/1000)==np.floor((bounds.maxx-transform.c)/1000)) &
            (np.floor((transform.f-bounds.maxy)/1000)==np.floor((transform.f-bounds.miny)/1000)))
    small=conus.loc[single & (conus.missing_mean_count==0) & (conus.Area_Land>0)]
    # DC provides an unambiguous urban choice; other urban states are a
    # selection fallback only, never a data/source fallback.
    city=small.loc[small.StFIPS=='11']
    if city.empty:city=small.loc[small.StFIPS.isin(['06','36','17','48'])]
    if not city.empty:small=city
    if small.empty:raise ValueError('No small tract wholly inside one valid Daymet cell')
    rural=conus.loc[(conus.missing_mean_count==0)&(conus.Area_Land>0)].sort_values('Area_Land')
    coastal=conus.loc[conus.StFIPS.isin(['06','12','23','53']) & (conus.missing_mean_count==0) &
                     (conus.annual_tmax_count < conus.geometry.area/1e6-.5)].sort_values('Area_Land')
    if coastal.empty:raise ValueError('No coastal tract with masked native cells selected')
    alaska=t.loc[(t.StFIPS=='02')&(t.missing_mean_count==0)].copy()
    geographic=original.set_index('GEOID').geometry.representative_point()
    alaska=alaska.loc[alaska.GEOID.map(lambda g: geographic[g].y>60 and -160<geographic[g].x< -140)]
    if alaska.empty:raise ValueError('No interior Alaska tract with data')
    picks=dict(small_urban_single_cell=small.sort_values('Area_Land').GEOID.iloc[0],
               large_rural=rural.GEOID.iloc[-1],coastal=coastal.GEOID.iloc[-1],
               interior_alaska=alaska.sort_values('Area_Land').GEOID.iloc[-1],aleutian_antimeridian='02016000100')
    for fips,label in [('15','hawaii'),('72','puerto_rico')]+([('78','usvi')] if usvi_covered else []):
        sub=t.loc[(t.StFIPS==fips)&(t.missing_mean_count==0)]
        if sub.empty:raise ValueError(f'{label}: no complete-data spot-check tract')
        picks[label]=sub.sort_values('Area_Land').GEOID.iloc[-1]
    return picks


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year',type=int,default=2020)
    parser.add_argument('--table',default='')
    parser.add_argument('--out',default='')
    args=parser.parse_args();year=args.year
    table_path=Path(args.table or CFS/'outputs/working'/f'daymet_tracts_{year}.parquet')
    table=pd.read_parquet(table_path)
    originals=load_roster()
    composites={d:composite_path(CFS/'composites',year,d) for d in DOMAINS}
    sides={d:json.loads(p.with_suffix('.json').read_text()) for d,p in composites.items()}
    tracts,_=project_tracts(originals,CRS(sides['na']['crs']))
    stats_audit=json.loads((CFS/'audits'/f'zonal_stats_{year}.json').read_text())
    picks=choose_tracts(originals,tracts,table,sides,stats_audit['usvi_covered'])
    geometries=tracts.set_index('GEOID').geometry;rows=table.set_index('GEOID')
    domains=assign_domains(originals.StFIPS,stats_audit['usvi_covered'])
    domain_by_id=dict(zip(originals.GEOID,domains))
    details={};maximum={s:0. for s in STAT_SUFFIXES};passed=True
    for label,geoid in picks.items():
        domain=domain_by_id[geoid]
        independent=independent_stats(geometries[geoid],composites[domain])
        band_results={}
        for band in BAND_NAMES:
            expected=independent[band]
            got={s:None if pd.isna(rows.at[geoid,f'{band}_{s}']) else float(rows.at[geoid,f'{band}_{s}']) for s in STAT_SUFFIXES}
            agree=True
            for stat in STAT_SUFFIXES:
                if (expected[stat] is None)!=(got[stat] is None):agree=False
                elif expected[stat] is not None:
                    difference=abs(expected[stat]-got[stat]);maximum[stat]=max(maximum[stat],difference)
                    agree &= difference<=ABS_TOL or (stat=='count' and difference/max(expected[stat],1e-12)<=COUNT_REL_TOL)
            passed &= agree
            band_results[band]=dict(independent=expected,table=got,agree=bool(agree),nan_cells=independent[band+'_nan_cells_intersected'])
        details[label]=dict(GEOID=geoid,domain=domain,cells_intersected=independent['cells_intersected'],bands=band_results)
        print(f'{label}: {geoid} {domain}, {independent["cells_intersected"]} intersected cells',flush=True)
    audit=dict(year=year,passed=bool(passed),tracts=details,max_abs_diff=maximum,abs_tol_degC=ABS_TOL,
               count_relative_tolerance=COUNT_REL_TOL,table_sha256=sha256_of(table_path),created_utc=utc_now(),
               elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    atomic_json(args.out or CFS/'audits'/f'spot_checks_{year}.json',audit)
    if not passed:raise SystemExit('Independent spot checks FAILED')


if __name__=='__main__':main()
