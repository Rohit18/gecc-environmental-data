"""Independent actual-pixel mask/grid/coverage checks before the full jobs."""
import json
import time
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from rasterio.windows import Window
from shapely.geometry import box
from exactextract import exact_extract
from exactextract.operation import Operation
from exactextract.raster import NumPyRasterSource
from water_source import PairedReader,MaskedSource,operations,VALID_CLASSES
from extract import ROOT,WC

def check(source,geoid,geom,crs,kind):
    with rasterio.open(source) as d:
        point=geom.representative_point()
        row,col=d.index(point.x,point.y)
        col=max(0,min(d.width-64,col-32));row=max(0,min(d.height-64,row-32))
        win=Window(col,row,64,64);t=d.window_transform(win)
        a=d.read(1,window=win,masked=True)
        extent=rasterio.windows.bounds(win,d.transform)
        poly=geom.intersection(box(*extent))
        xx,yy=np.meshgrid(t.c+(np.arange(64)+.5)*t.a,t.f+(np.arange(64)+.5)*t.e)
        tx=pyproj.Transformer.from_crs(d.crs,4326,always_xy=True)
        lon,lat=tx.transform(xx,yy)
        with rasterio.open(WC) as wc:
            wr=np.floor((np.asarray(lat)-wc.transform.f)/wc.transform.e).astype(int)
            wc_col=np.floor((np.asarray(lon)-wc.transform.c)/wc.transform.a).astype(int)
            r0,c0=int(wr.min()),int(wc_col.min())
            raw=wc.read(1,window=Window(c0,r0,int(wc_col.max()-c0+1),int(wr.max()-r0+1)),
                        boundless=True,fill_value=0)
            classes=raw[wr-r0,wc_col-c0]
    reader=PairedReader(source,WC)
    _,codes=reader.read(col,row,64,64)
    expected_codes=np.where(classes==80,1,np.where(np.isin(classes,VALID_CLASSES),0,2))
    np.testing.assert_array_equal(codes,expected_codes)
    # Independent GEOS intersections of every native cell with the clipped tract.
    cells=shapely.box(xx-.5*t.a,yy+.5*t.e,xx+.5*t.a,yy-.5*t.e)
    weights=shapely.area(shapely.intersection(cells,poly))/abs(t.a*t.e)
    valid=~np.ma.getmaskarray(a)
    keep=valid & (classes!=80)
    h=np.bincount(a.data[keep].astype(int),weights=weights[keep],minlength=256)
    sw=np.bincount(expected_codes[valid],weights=weights[valid],minlength=3)
    # Extract a crop with the same adapter-derived mask. This also tests the C++ reducer.
    masked=reader.read(col,row,64,64)[0].data.astype(np.int16)
    masked[~keep]=-1
    support=expected_codes.astype(np.int16);support[~valid]=-1
    m=NumPyRasterSource(masked,*extent,nodata=-1,srs_wkt=crs.to_wkt(),name='masked')
    s=NumPyRasterSource(support,*extent,nodata=-1,srs_wkt=crs.to_wkt(),name='support')
    g=gpd.GeoDataFrame({'GEOID':[geoid]},geometry=[poly],crs=crs)
    out=exact_extract([m,s],g,operations(m,s,kind=='lai'),output='pandas',max_cells_in_memory=4096).iloc[0]
    actual=np.zeros(256)
    if out['unique'] is not None and len(out['unique']):
        actual[np.asarray(out['unique'],dtype=int)]=np.asarray(out['frac'])*out.wcount
    np.testing.assert_allclose(actual,h,atol=2e-3,rtol=1e-7)
    from water_source import support_weights
    np.testing.assert_allclose(support_weights(out),sw,atol=2e-3,rtol=1e-7)
    reader.close()
    return {'kind':kind,'GEOID':geoid,'source':source,'native_cells':4096,
            'water_weight':float(sw[1]),'unknown_weight':float(sw[2]),
            'kept_weight':float(h.sum()),'max_histogram_error':float(abs(actual-h).max()),
            'classification_matches':True,'passed':True}

def main():
    ids=['36061000100','12086000107','06075060100','15003000106','02185000200','72001030100','66010952400']
    lai=gpd.read_parquet(ROOT/'inputs/lai_tracts.parquet').set_index('GEOID')
    chm=gpd.read_parquet(ROOT/'inputs/chm_tracts.parquet').to_crs(3857).set_index('GEOID')
    jobs=pd.read_parquet(ROOT/'inputs/chm_jobs.parquet')
    results=[]
    for geoid in ids:
        if geoid not in lai.index:continue
        comp='addon' if geoid[:2] in {'02','15','60','66','69','72','78'} else 'conus'
        source=f'/path/to/scratch/gecc_lai_reproducibility_audit_20261012/rerun_work_20261012/{comp}_rerun.vrt'
        results.append(check(source,geoid,lai.loc[geoid].geometry,lai.crs,'lai'))
        point=chm.loc[geoid].geometry.representative_point()
        found=False
        for r in jobs[jobs.geoids.str.contains(geoid,regex=False)].itertuples():
            with rasterio.open(r.source) as d:
                b=d.bounds
                if not(b.left<=point.x<b.right and b.bottom<point.y<=b.top):continue
            results.append(check(r.source,geoid,chm.loc[geoid].geometry,chm.crs,'chm'))
            found=True;break
        if not found:raise RuntimeError(f'no canopy tile at pilot point {geoid}')
    assert len(results)>=6
    assert any(r['water_weight']>0 for r in results)
    assert any(r['kept_weight']>0 for r in results)
    pd.DataFrame(results).to_csv(ROOT/'validation/pilot_pixel_checks.csv',index=False)
    (ROOT/'validation/PILOT_PASSED.json').write_text(json.dumps({'checks':len(results),'passed':True,
                 'finished_utc':time.strftime('%FT%TZ',time.gmtime()),'max_histogram_error':max(r['max_histogram_error'] for r in results)},indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
    print('PILOT_PASSED',len(results),flush=True)

if __name__=='__main__':main()
