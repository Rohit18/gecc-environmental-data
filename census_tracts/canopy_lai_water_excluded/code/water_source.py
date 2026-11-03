"""Exclude WC2020 class 80 on the unchanged native LAI/CHM grid.

WorldCover classes are sampled by nearest neighbour at native pixel centres.
Only class 80 is removed. Unclassified WorldCover pixels are retained, but their
coverage is recorded independently and flagged; they are never called land.
The source raster's own nodata stays masked. No source values are resampled.
"""
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from exactextract.raster import RasterSource
from exactextract.operation import Operation

VALID_CLASSES = np.array([10,20,30,40,50,60,70,80,90,95,100],dtype=np.uint8)

class PairedReader:
    def __init__(self, source, worldcover):
        self.ds = rasterio.open(source)
        self.wc_ds = rasterio.open(worldcover)
        self.wc = WarpedVRT(self.wc_ds, crs=self.ds.crs, transform=self.ds.transform,
                            width=self.ds.width, height=self.ds.height,
                            src_nodata=0, nodata=0, resampling=Resampling.nearest,
                            tolerance=1e-9, warp_mem_limit=64)
        self.key=None
        self.arr=None
        self.classes=None
    def read(self,x0,y0,nx,ny):
        key=(x0,y0,nx,ny)
        if key != self.key:
            w=Window(x0,y0,nx,ny)
            a=self.ds.read(1,window=w,masked=True)
            wc=self.wc.read(1,window=w,masked=False)
            self.classes=np.where(wc==80,1,np.where(np.isin(wc,VALID_CLASSES),0,2)).astype(np.uint8)
            self.arr=a
            self.key=key
        return self.arr,self.classes
    def close(self):
        self.wc.close();self.wc_ds.close();self.ds.close()

class MaskedSource(RasterSource):
    def __init__(self, reader, support=False):
        super().__init__()
        self.reader=reader
        self.support=support
        self.set_name('support' if support else 'masked')
    def res(self):
        t=self.reader.ds.transform
        return t.a,-t.e
    def extent(self):
        b=self.reader.ds.bounds
        return b.left,b.bottom,b.right,b.top
    def srs_wkt(self): return self.reader.ds.crs.wkt
    def nodata_value(self): return -1
    def read_window(self,x0,y0,nx,ny):
        a,c=self.reader.read(x0,y0,nx,ny)
        original=np.ma.getmaskarray(a)
        if self.support:
            out=c.astype(np.int16)
            out[original]=-1
            return out
        out=a.data.astype(np.int16)
        out[original | (c==1)]=-1
        return out

def operations(masked,support,touched=False):
    ops=[Operation('unique','unique',masked),Operation('frac','frac',masked),
         Operation('count','wcount',masked),
         Operation('unique','support_unique',support),Operation('frac','support_frac',support),
         Operation('count','source_wcount',support)]
    if touched:
        ops.append(Operation('count','touched',masked,options={'coverage_weight':'none'}))
    return ops

def support_weights(row):
    w=np.zeros(3,dtype=np.float64)
    u=row['support_unique'];f=row['support_frac'];count=row['source_wcount']
    if u is not None and len(u) and count>0:
        w[np.asarray(u,dtype=int)]=np.asarray(f,dtype=float)*float(count)
    return w

def status(kept,weights):
    if weights.sum()==0:return 'source_no_valid_pixels'
    if weights[2]>max(1e-6,weights.sum()*1e-10):return 'worldcover_incomplete'
    if kept==0:return 'water_only'
    return 'ok'
