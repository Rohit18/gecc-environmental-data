"""Meaningful mask/grid/support tests with fractional boundary coverage."""
import tempfile
import unittest
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from exactextract import exact_extract
from water_source import PairedReader,MaskedSource,operations,support_weights

class MaskTests(unittest.TestCase):
    def run_case(self,source,wc,geom,nodata=None,scale=1):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name,a in [('source',source),('wc',wc)]:
                step=scale if name=='wc' else 1
                with rasterio.open(p/f'{name}.tif','w',driver='GTiff',width=a.shape[1],height=a.shape[0],
                                   count=1,dtype=a.dtype,crs='EPSG:4326',
                                   transform=from_origin(0,4,step,step),
                                   nodata=nodata if name=='source' else 0) as d:
                    d.write(a,1)
            reader=PairedReader(p/'source.tif',p/'wc.tif')
            masked,support=MaskedSource(reader),MaskedSource(reader,True)
            polygons=gpd.GeoDataFrame({'id':[1]},geometry=[geom],crs=4326)
            try:
                return exact_extract([masked,support],polygons,operations(masked,support,True),
                                     output='pandas',max_cells_in_memory=4).iloc[0]
            finally:reader.close()
    def test_water_zero_land_zero_and_source_nodata(self):
        source=np.array([[0,8,0,5],[-1,4,6,9]],dtype=np.int16)
        wc=np.array([[10,80,80,50],[80,80,10,0]],dtype=np.uint8)
        r=self.run_case(source,wc,box(0,2,4,4),nodata=-1)
        self.assertEqual(r.touched,4)
        self.assertEqual(r.wcount,4)
        self.assertEqual(set(r['unique']),{0,5,6,9})
        np.testing.assert_allclose(support_weights(r),[3,3,1])
    def test_fractional_boundary(self):
        source=np.array([[2,10]],dtype=np.uint8)
        wc=np.array([[10,80]],dtype=np.uint8)
        r=self.run_case(source,wc,box(.25,3,1.5,4))
        self.assertAlmostEqual(r.wcount,.75)
        np.testing.assert_allclose(support_weights(r),[.75,.5,0])
        self.assertEqual(r.touched,1)
    def test_nearest_class_upsampling(self):
        source=np.arange(8,dtype=np.uint8).reshape(2,4)
        wc=np.array([[80,10]],dtype=np.uint8)
        r=self.run_case(source,wc,box(0,2,4,4),scale=2)
        self.assertEqual(set(r['unique']),{2,3,6,7})
        np.testing.assert_allclose(support_weights(r),[4,4,0])
    def test_all_water_preserves_no_data(self):
        r=self.run_case(np.array([[10]],dtype=np.uint8),np.array([[80]],dtype=np.uint8),box(0,3,1,4))
        self.assertEqual(r.wcount,0)
        self.assertEqual(r.touched,0)
        np.testing.assert_allclose(support_weights(r),[0,1,0])

if __name__=='__main__':unittest.main()
