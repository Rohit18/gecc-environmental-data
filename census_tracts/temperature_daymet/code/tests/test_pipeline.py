"""Independent scientific calendar, mask, native-grid and geometry tests."""
import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box,MultiPolygon,Polygon
from common import *
from daymet_composite import complete_means,build_domain
from zonal_stats import reduce_composite,build_table


class CalendarTests(unittest.TestCase):
    def test_leap_year_passes_without_december31(self):
        dates=expected_dates(2020)
        self.assertEqual(len(dates),365)
        self.assertIn(dt.date(2020,2,29),dates)
        self.assertNotIn(dt.date(2020,12,31),dates)
        self.assertEqual(expected_day_counts(2020),{'annual_day_count':365,'JJA_day_count':92,'DJF_day_count':90})
        check_date_set(dates,expected_dates(2020),'leap')
    def test_extra_december31_fails(self):
        with self.assertRaises(ValueError):check_date_set(expected_dates(2020)+[dt.date(2020,12,31)],expected_dates(2020),'extra')
    def test_missing_february29_fails(self):
        with self.assertRaises(ValueError):check_date_set([d for d in expected_dates(2020) if d!=dt.date(2020,2,29)],expected_dates(2020),'gap')
    def test_duplicate_fails(self):
        dates=expected_dates(2020)
        with self.assertRaises(ValueError):check_date_set(dates+[dates[0]],dates,'duplicate')
    def test_common_year_counts(self):
        self.assertIn(dt.date(2021,12,31),expected_dates(2021))
        self.assertEqual(expected_day_counts(2021),{'annual_day_count':365,'JJA_day_count':92,'DJF_day_count':90})
    def test_same_year_djf(self):
        self.assertTrue(all(d.year==2020 for d in expected_dates(2020,'DJF')))


class MaskTests(unittest.TestCase):
    def test_production_stream_masks_fill_before_scaling_exactly_once(self):
        dates=expected_dates(2020)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);daily=root/'daily';out=root/'composites'
            (root/'manifests').mkdir()
            records=[]
            for measure,raw_value in [('tmax',200),('tmin',100)]:
                path=source_path(2020,'hi',measure,daily);path.parent.mkdir(parents=True,exist_ok=True)
                with netCDF4.Dataset(path,'w') as ds:
                    for dimension,size in [('time',365),('y',2),('x',2)]:ds.createDimension(dimension,size)
                    tv=ds.createVariable('time','f8',('time',));tv.units='days since 2020-01-01'
                    tv[:]=netCDF4.date2num([dt.datetime.combine(d,dt.time()) for d in dates],tv.units)
                    for name,coordinates in [('x',[500.,1500.]),('y',[1500.,500.])]:
                        coordinate=ds.createVariable(name,'f8',(name,));coordinate.units='m';coordinate[:]=coordinates
                    mapping=ds.createVariable('lambert_conformal_conic','i4')
                    mapping.setncatts({k:v for k,v in EXPECTED_CRS.to_cf().items() if k!='crs_wkt'})
                    var=ds.createVariable(measure,'i2',('time','y','x'),fill_value=-9999,chunksizes=(1,2,2))
                    var.units='degrees C';var.missing_value=np.int16(-9999);var.scale_factor=.1;var.add_offset=10.
                    var.grid_mapping='lambert_conformal_conic';var.set_auto_maskandscale(False)
                    data=np.full((365,2,2),raw_value,dtype='int16')
                    if measure=='tmax':data[dates.index(dt.date(2020,7,4)),0,0]=-9999
                    var[:]=data
                records.append(dict(filename=path.name,status='verified_cfs',cfs_path=str(path),bytes=path.stat().st_size,sha256=sha256_of(path)))
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'files':records}))
            args=SimpleNamespace(manifest=str(manifest),out_dir=str(out),daily_dir=str(daily))
            with patch('daymet_composite.CFS',root):side=build_domain(2020,'hi',args)
            with rasterio.open(composite_path(out,2020,'hi')) as src:array=src.read()
            self.assertTrue(np.isnan(array[0,0,0]))
            self.assertTrue(np.isnan(array[2,0,0]))
            self.assertEqual(array[4,0,0],30.)
            self.assertTrue((array[1::2]==20.).all())
            self.assertEqual(side['periods']['annual_tmax']['cells_some_days_missing_set_nan'],1)
            self.assertEqual(side['periods']['JJA_tmax']['cells_some_days_missing_set_nan'],1)
    def test_one_nan_masks_period_and_preserves_rows_and_zeros(self):
        dates=expected_dates(2020)
        values=np.full((365,2,2),20.,dtype='float64')
        values[dates.index(dt.date(2020,7,4)),0,0]=np.nan
        means=complete_means(values,dates,2020)
        self.assertTrue(np.isnan(means['annual'][0,0]))
        self.assertTrue(np.isnan(means['JJA'][0,0]))
        self.assertEqual(means['DJF'][0,0],20.)
        transform=rasterio.transform.from_origin(0,2000,1000,1000)
        array=np.stack([means[p] if m=='tmax' else np.full((2,2),10.) for p,m in BANDS])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'composite.tif'
            with rasterio.open(path,'w',driver='GTiff',height=2,width=2,count=6,dtype='float64',crs=EXPECTED_CRS,transform=transform,nodata=np.nan) as dst:
                dst.write(array)
                for index,name in enumerate(BAND_NAMES,1):dst.set_band_description(index,name)
            tracts=gpd.GeoDataFrame({'GEOID':['01001000100','01001000200'],'StFIPS':['01','01'],'Area_Land':[640000.,640000.]},geometry=[box(100,1100,900,1900),box(1100,1100,1900,1900)],crs=EXPECTED_CRS)
            table=build_table(reduce_composite(path,tracts),tracts,2020)
            self.assertEqual(len(table),2)
            self.assertEqual(list(table.columns),OUTPUT_COLUMNS)
            for period in ['annual','JJA']:
                for stat in STAT_SUFFIXES:self.assertTrue(pd.isna(table.loc[0,f'{period}_tmax_{stat}']))
            self.assertEqual(table.loc[0,'missing_mean_count'],2)
            self.assertEqual(table.loc[0,'annual_tmin_stdDev'],0.)
            self.assertAlmostEqual(table.loc[0,'annual_tmin_count'],.64,places=6)


class GeometryTests(unittest.TestCase):
    def test_same_winding_holes_are_subtracted_without_shape_change(self):
        outside=list(box(-100.1,40.,-100.,40.1).exterior.coords)
        hole=list(box(-100.08,40.02,-100.02,40.08).exterior.coords)
        geometry=Polygon(outside,[hole])
        original=gpd.GeoDataFrame({'GEOID':['01001000100'],'StFIPS':['01'],'Area_Land':[1.]},geometry=[geometry],crs=4326)
        projected,_=project_tracts(original,EXPECTED_CRS)
        metrics=geometry_area_metrics(original,projected,EXPECTED_CRS)
        self.assertLess(abs(metrics.scale_corrected_ratio.iloc[0]-1),.02)
        self.assertGreater(metrics.unoriented_signed_geodesic_area_m2.iloc[0],metrics.geodesic_area_m2.iloc[0])
        self.assertTrue(geometry.equals(shapely.orient_polygons(geometry)))
    def test_antimeridian_multipart_projects_valid_with_scale_correction(self):
        geometry=MultiPolygon([box(179.5,51.,179.9,52.),box(-179.9,51.,-179.5,52.)])
        original=gpd.GeoDataFrame({'GEOID':['02016000100'],'StFIPS':['02'],'Area_Land':[1.]},geometry=[geometry],crs=4326)
        projected,changes=project_tracts(original,EXPECTED_CRS)
        self.assertTrue(projected.geometry.is_valid.all())
        self.assertFalse(projected.geometry.is_empty.any())
        self.assertEqual(changes,[])
        metrics=geometry_area_metrics(original,projected,EXPECTED_CRS)
        self.assertLess(abs(metrics.scale_corrected_ratio.iloc[0]-1),.02)
        self.assertGreater(abs(metrics.raw_planar_geodesic_ratio.iloc[0]-1),.02)
    def test_domain_assignment(self):
        fips=['01','02','15','72','78','60','66','69']
        self.assertEqual(assign_domains(fips,False).tolist(),['na','na','hi','pr','none','none','none','none'])
        self.assertEqual(assign_domains(fips,True).tolist(),['na','na','hi','pr','pr','none','none','none'])


if __name__=='__main__':unittest.main()
