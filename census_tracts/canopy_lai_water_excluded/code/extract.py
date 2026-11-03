"""Resumable native-grid LAI and 1 m canopy histograms with WC water excluded."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import get_context
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from exactextract import exact_extract
from water_source import PairedReader,MaskedSource,operations,support_weights

ROOT=Path('/path/to/scratch/gecc_tract_watermask_20261013')
WC=ROOT/'inputs/worldcover.vrt'
MAX_CELLS=4_000_000
CHM_GEOMS=None

def signature():
    h=hashlib.sha256()
    for p in sorted((ROOT/'code').glob('*.py')):
        if p.name in {'water_source.py','extract.py','lai_common.py','chm_quantiles.py'}:
            h.update(p.name.encode());h.update(p.read_bytes())
    for p in [ROOT/'inputs/worldcover_inventory.csv',ROOT/'inputs/lai_jobs.csv',ROOT/'inputs/chm_jobs.parquet']:
        h.update(p.read_bytes())
    return h.hexdigest()

def extract(source,polys,kind,strategy):
    reader=PairedReader(source,WC)
    masked,support=MaskedSource(reader),MaskedSource(reader,True)
    ops=operations(masked,support,kind=='lai')
    try:
        r=exact_extract([masked,support],polys[['GEOID','geometry']],ops,include_cols=['GEOID'],
                        output='pandas',strategy=strategy,max_cells_in_memory=MAX_CELLS)
    finally:reader.close()
    rows=[]
    for _,v in r.iterrows():
        hist=np.zeros(256,dtype=np.float64)
        u=v['unique'];f=v['frac'];n=v['wcount']
        if u is not None and len(u) and n>0:
            u=np.asarray(u,dtype=int)
            if (u<0).any() or (u>255).any():raise ValueError('invalid source value')
            hist[u]=np.asarray(f,dtype=float)*float(n)
        sw=support_weights(v)
        if not np.isclose(hist.sum()+sw[1],sw.sum(),rtol=1e-10,atol=1e-6):
            raise AssertionError(f'{v.GEOID}: masked + water support != source support')
        rows.append({'GEOID':v.GEOID,'hist':hist.tolist(),
                     'source_wcount':float(sw.sum()),'water_wcount':float(sw[1]),
                     'wc_unknown_wcount':float(sw[2]),'retained_wcount':float(hist.sum()),
                     'touched':int(v['touched']) if kind=='lai' else 0})
    return pd.DataFrame(rows)

def complete(path,sig):
    side=path.with_suffix('.done.json')
    if not(path.exists() and side.exists()):return False
    rec=json.loads(side.read_text())
    if rec['signature']!=sig:return False
    return rec['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()

def save(df,path,sig,secs):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.parquet')
    df.to_parquet(tmp,index=False);os.replace(tmp,path)
    side={'signature':sig,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'rows':len(df),'seconds':secs,'job_id':os.environ.get('SLURM_JOB_ID'),
          'finished_utc':time.strftime('%FT%TZ',time.gmtime())}
    path.with_suffix('.done.json').write_text(json.dumps(side,indent=2)+'\n')

def worker_lai(job):
    sig=job['signature'];path=ROOT/f'work/lai/part_{job["job"]:05d}.parquet'
    if complete(path,sig):return f'LAI {job["job"]}: already complete'
    t=time.time();g=gpd.read_parquet(job['geoms'])
    rows=[]
    for _,v in g.iterrows():
        if v.geometry.bounds[2]-v.geometry.bounds[0]>180:
            for side in (box(-180,-90,0,90),box(0,-90,180,90)):
                p=v.geometry.intersection(side)
                if not p.is_empty:rows.append({'GEOID':v.GEOID,'geometry':p})
        else:rows.append({'GEOID':v.GEOID,'geometry':v.geometry})
    split=gpd.GeoDataFrame(rows,geometry='geometry',crs=g.crs)
    df=extract(job['source'],split,'lai','feature-sequential')
    if df.GEOID.duplicated().any():
        out=[]
        for geoid,part in df.groupby('GEOID'):
            r={'GEOID':geoid,'hist':np.asarray(part['hist'].tolist()).sum(axis=0).tolist()}
            for k in ['source_wcount','water_wcount','wc_unknown_wcount','retained_wcount','touched']:
                r[k]=part[k].sum()
            out.append(r)
        df=pd.DataFrame(out)
    assert len(df)==len(g) and df.GEOID.is_unique
    save(df,path,sig,time.time()-t)
    return f'LAI {job["job"]}: {len(df)} tracts, {time.time()-t:.1f}s'

def worker_chm(job):
    sig=job['signature'];path=ROOT/f'work/chm/tile_{job["tile"]}.parquet'
    if complete(path,sig):return f'CHM {job["tile"]}: already complete'
    t=time.time()
    ids=json.loads(job['geoids'])
    g=CHM_GEOMS.loc[ids].reset_index()
    df=extract(job['source'],g,'chm','raster-sequential')
    assert len(df)==len(ids) and df.GEOID.is_unique
    save(df,path,sig,time.time()-t)
    return f'CHM {job["tile"]}: {len(df)} tracts, {time.time()-t:.1f}s'

def main():
    global CHM_GEOMS
    ap=argparse.ArgumentParser()
    ap.add_argument('kind',choices=['lai','chm'])
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--limit',type=int)
    a=ap.parse_args()
    sig=signature()
    if a.kind=='lai':
        jobs=pd.read_csv(ROOT/'inputs/lai_jobs.csv').to_dict('records');worker=worker_lai
    else:
        jobs=pd.read_parquet(ROOT/'inputs/chm_jobs.parquet').to_dict('records');worker=worker_chm
        CHM_GEOMS=gpd.read_parquet(ROOT/'inputs/chm_tracts.parquet').to_crs(3857).set_index('GEOID')
    jobs=jobs[a.shard::a.shards]
    if a.limit:jobs=jobs[:a.limit]
    for job in jobs:job['signature']=sig
    t=time.time()
    print(f'START {a.kind} shard {a.shard}/{a.shards}, jobs {len(jobs)}, workers {a.workers}, signature {sig}',flush=True)
    with get_context('fork').Pool(a.workers,maxtasksperchild=4) as pool:
        for i,msg in enumerate(pool.imap_unordered(worker,jobs),1):
            print(f'{time.strftime("%FT%TZ",time.gmtime())} [{i}/{len(jobs)}] {msg}',flush=True)
    print(f'DONE {a.kind} {time.time()-t:.1f}s',flush=True)

if __name__=='__main__':main()
