"""Fetch only WC2020 census coverage gaps after existing-cache inventory."""
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import rasterio
from prepare import ROOT

def fetch(r):
    dest=ROOT/'inputs/worldcover_missing'/Path(r['url']).name
    dest.parent.mkdir(exist_ok=True)
    side=Path(str(dest)+'.verified.json')
    if dest.exists() and side.exists():
        rec=json.loads(side.read_text())
        if rec['bytes']==dest.stat().st_size:return r['tile'],str(dest),rec
    last=None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(r['url'],method='HEAD'),timeout=60) as response:
                expected=int(response.headers['Content-Length']);etag=response.headers.get('ETag')
            tmp=dest.with_suffix('.part')
            h=hashlib.sha256()
            with urllib.request.urlopen(r['url'],timeout=180) as response,open(tmp,'wb') as f:
                while chunk:=response.read(8<<20):f.write(chunk);h.update(chunk)
            assert tmp.stat().st_size==expected,(r['tile'],tmp.stat().st_size,expected)
            with rasterio.open(tmp) as d:
                assert d.crs.to_epsg()==4326 and d.count==1 and d.width==36000 and d.height==36000
                rec={'tile':r['tile'],'url':r['url'],'bytes':expected,'sha256':h.hexdigest(),
                     'etag':etag,'crs':str(d.crs),'transform':list(d.transform)[:6],
                     'bounds':list(d.bounds),'year':2020,'version':'v100','band':'Map'}
            os.replace(tmp,dest);side.write_text(json.dumps(rec,indent=2)+'\n')
            print(f'FETCHED {r["tile"]} {expected} bytes',flush=True)
            return r['tile'],str(dest),rec
        except Exception as e:last=e;print(f'RETRY {r["tile"]} {attempt}: {e}',flush=True)
    raise RuntimeError(f'{r["tile"]}: {last}')

def main():
    inv=pd.read_csv(ROOT/'inputs/worldcover_inventory.csv',keep_default_na=False)
    missing=inv[inv.path==''].to_dict('records')
    with ThreadPoolExecutor(8) as pool:results=list(pool.map(fetch,missing))
    for tile,path,rec in results:
        k=inv.tile==tile
        inv.loc[k,'path']=path;inv.loc[k,'bytes']=rec['bytes'];inv.loc[k,'recorded_sha256']=rec['sha256']
    assert (inv.path!='').all()
    # Verify every reused raster header and every available historical size record.
    headers=[]
    for r in inv.itertuples():
        with rasterio.open(r.path) as d:
            assert d.count==1 and d.width==36000 and d.height==36000 and d.crs.to_epsg()==4326,r.path
            assert d.dtypes[0]=='uint8' and d.nodata==0,r.path
            assert os.stat(r.path).st_size==r.bytes,r.path
            headers.append({'tile':r.tile,'path':r.path,'width':d.width,'height':d.height,
                            'nodata':d.nodata,'crs':str(d.crs),'transform':list(d.transform)[:6]})
    inv.to_csv(ROOT/'inputs/worldcover_inventory.csv',index=False)
    (ROOT/'validation/worldcover_headers.json').write_text(json.dumps(headers,indent=2)+'\n')
    lst=ROOT/'inputs/worldcover_sources.txt';lst.write_text(''.join(p+'\n' for p in inv.path))
    subprocess.run(['gdalbuildvrt','-q','-overwrite','-srcnodata','0','-vrtnodata','0',
                    '-input_file_list',str(lst),str(ROOT/'inputs/worldcover.vrt')],check=True)
    (ROOT/'inputs/WORLDCOVER_READY.json').write_text(json.dumps({'tiles':len(inv),'new_downloads':len(results),
                            'inventory_sha256':hashlib.sha256((ROOT/'inputs/worldcover_inventory.csv').read_bytes()).hexdigest(),
                            'finished_utc':time.strftime('%FT%TZ',time.gmtime())},indent=2)+'\n')
    print('WORLDCOVER_READY',len(inv),'tiles;',len(results),'new downloads',flush=True)

if __name__=='__main__':main()
