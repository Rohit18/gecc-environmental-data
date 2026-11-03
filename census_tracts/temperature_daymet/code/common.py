"""Locked scientific definitions shared by the Daymet tract pipeline."""
from __future__ import annotations

import calendar
import datetime as dt
from pathlib import Path

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Proj, Transformer
import shapely
from rasterio.transform import Affine

from daymet_fetch import atomic_json, parse_years, sha256_of, utc_now, immutable_read_path

CFS = Path('/path/to/project/gecc/daymet')
ROOT = Path('/path/to/scratch/daymet_tract_temperature')
POLYGONS = Path('/path/to/project/gecc/era5_land/inputs/CensusTract_2020_US_85528.parquet')
ROSTER_SHA256 = '5d998203a5e6cb5a32a1b2ac51a7144f06a2cb4fd3a12251acbf83bd08a60680'
EXPECTED_TRACTS = 85528
EXPECTED_CONUS = 83776
NON_CONUS_FIPS = ('02', '15', '60', '66', '69', '72', '78')
UNSUPPORTED_FIPS = ('60', '66', '69')
DOMAINS = ('na', 'hi', 'pr')
PERIOD_MONTHS = {'annual': tuple(range(1, 13)), 'JJA': (6, 7, 8), 'DJF': (1, 2, 12)}
BANDS = [('annual', 'tmax'), ('annual', 'tmin'), ('JJA', 'tmax'), ('JJA', 'tmin'), ('DJF', 'tmax'), ('DJF', 'tmin')]
BAND_NAMES = [f'{p}_{m}' for p, m in BANDS]
SOURCE_SCALE_M = 1000.0
RAW_C_RANGE = (-70., 60.)
EXPECTED_CRS = CRS.from_proj4('+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m')
STAT_SUFFIXES = ['mean', 'median', 'stdDev', 'count']
METRIC_COLUMNS = [f'{b}_{s}' for b in BAND_NAMES for s in STAT_SUFFIXES]
ID_COLUMNS = ['GEOID', 'source_area_m2', 'dataset', 'geography_type', 'year', 'djf_convention', 'annual_day_count', 'JJA_day_count', 'DJF_day_count', 'source_scale_m', 'reduction_scale_m', 'missing_mean_count', 'all_six_means_valid', 'fallback_applied']
OUTPUT_COLUMNS = ID_COLUMNS + METRIC_COLUMNS
OUTPUT_DTYPES = {
    'GEOID': 'str', 'source_area_m2': 'float64', 'dataset': 'str', 'geography_type': 'str',
    'year': 'int64', 'djf_convention': 'str', 'annual_day_count': 'int64', 'JJA_day_count': 'int64',
    'DJF_day_count': 'int64', 'source_scale_m': 'float64', 'reduction_scale_m': 'float64',
    'missing_mean_count': 'int64', 'all_six_means_valid': 'int64', 'fallback_applied': 'bool',
    **{c: 'float64' for c in METRIC_COLUMNS},
}


def expected_dates(year, period='annual'):
    return [dt.date(year, m, d) for m in PERIOD_MONTHS[period]
            for d in range(1, calendar.monthrange(year, m)[1]+1)
            if not (calendar.isleap(year) and m == 12 and d == 31)]


def expected_day_counts(year):
    return {f'{p}_day_count': len(expected_dates(year, p)) for p in PERIOD_MONTHS}


def check_date_set(found, expected, label):
    if len(found) != len(set(found)) or set(found) != set(expected):
        raise ValueError(f'{label}: calendar mismatch: {len(found)-len(set(found))} duplicates; '
                         f'missing {sorted(set(expected)-set(found))[:6]}; extra {sorted(set(found)-set(expected))[:6]}')


def source_path(year, domain, measure, daily_dir=None):
    return Path(daily_dir or CFS/'daily')/str(year)/f'daymet_v4_daily_{domain}_{measure}_{year}.nc'


def composite_path(out_dir, year, domain):
    return Path(out_dir)/str(year)/f'Daymet_seasonal_{domain}_{year}.tif'


def read_grid(ds, measure):
    var = ds.variables[measure]
    mapping = ds.variables[var.getncattr('grid_mapping')]
    cf = {k: mapping.getncattr(k) for k in mapping.ncattrs()}
    crs = CRS.from_cf(cf)
    # Compare numerical defining parameters, including the ellipsoid, rather
    # than CRS names (CF labels WGS84 as "undefined" in some files).
    params = crs.to_dict()
    for key, value in EXPECTED_CRS.to_dict().items():
        if key in ('type', 'no_defs', 'ellps'):
            continue
        actual = params.get(key)
        if isinstance(value, (int, float)):
            if actual is None or not np.isclose(float(actual), value, rtol=0, atol=1e-9):
                raise ValueError(f'CRS parameter mismatch: {key}={actual}, expected {value}')
        elif actual != value:
            raise ValueError(f'CRS parameter mismatch: {key}={actual}, expected {value}')
    if not np.isclose(crs.ellipsoid.semi_major_metre, 6378137., rtol=0, atol=1e-6) or not np.isclose(crs.ellipsoid.inverse_flattening, 298.257223563, rtol=0, atol=1e-9):
        raise ValueError('Daymet CRS ellipsoid is not WGS84')
    coords = []
    for name in ('x', 'y'):
        coordinate = ds.variables[name]
        if getattr(coordinate, 'units', '').lower() not in ('m', 'meter', 'meters', 'metre', 'metres'):
            raise ValueError(f'{name}: expected meter coordinates')
        values = np.asarray(coordinate[:], dtype='float64')
        expected_step = 1000. if name == 'x' else -1000.
        if values.size < 2 or not np.all(np.diff(values) == expected_step):
            raise ValueError(f'{name}: spacing is not exactly {expected_step} m')
        coords.append(values)
    x, y = coords
    transform = Affine(1000., 0., x[0]-500., 0., -1000., y[0]+500.)
    return crs, transform, x, y


def decode_dates(ds, year):
    tv = ds.variables['time']
    values = netCDF4.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))
    dates = [dt.date(v.year, v.month, v.day) for v in values]
    check_date_set(dates, expected_dates(year), f'{year} annual')
    for period, months in PERIOD_MONTHS.items():
        check_date_set([d for d in dates if d.month in months], expected_dates(year, period), f'{year} {period}')
    return dates


def load_roster(path=POLYGONS):
    if sha256_of(path) != ROSTER_SHA256:
        raise ValueError('Roster SHA-256 mismatch; source must not be rebuilt')
    gdf = gpd.read_parquet(path)
    if len(gdf) != EXPECTED_TRACTS or gdf.GEOID.nunique() != EXPECTED_TRACTS or not gdf.GEOID.map(lambda x: isinstance(x,str) and len(x)==11 and x.isdigit()).all():
        raise ValueError('Roster rows or GEOIDs mismatch')
    if not CRS(gdf.crs).equals(CRS.from_epsg(4326)):
        raise ValueError('Roster CRS mismatch')
    return gdf


def assign_domains(stfips, usvi_covered=False):
    fips = pd.Series(stfips).astype(str).str.zfill(2)
    domains = pd.Series('na', index=fips.index, dtype='str')
    domains.loc[fips == '15'] = 'hi'
    domains.loc[fips == '72'] = 'pr'
    domains.loc[fips.isin(UNSUPPORTED_FIPS)] = 'none'
    domains.loc[fips == '78'] = 'pr' if usvi_covered else 'none'
    return domains


def project_tracts(tracts, crs):
    """Project polygon vertices only; repair and log invalid projected shapes.

    The native LCC central meridian normalizes longitudes across ±180
    continuously; the original multipart Aleutian geometry needs no grid shift.
    """
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    result = tracts.copy()
    geometries = shapely.transform(tracts.geometry.to_numpy(), transformer.transform, interleaved=False)
    changed = []
    for index in np.flatnonzero(~shapely.is_valid(geometries)):
        before = geometries[index]
        after = shapely.make_valid(before)
        import hashlib
        changed.append(dict(GEOID=tracts.GEOID.iloc[index],reason=shapely.is_valid_reason(before),
                            before_wkb_sha256=hashlib.sha256(before.wkb).hexdigest(),
                            after_wkb_sha256=hashlib.sha256(after.wkb).hexdigest(),
                            before_area_m2=before.area,after_area_m2=after.area))
        geometries[index] = after
    result.geometry = gpd.GeoSeries(geometries, index=tracts.index, crs=crs)
    result = result.set_crs(crs, allow_override=True)
    if shapely.is_empty(geometries).any() or (~shapely.is_valid(geometries)).any():
        raise ValueError('Projected tracts contain empty or invalid geometry after make_valid')
    return result, changed


def geometry_area_metrics(original, projected, crs):
    geod = Geod(ellps='WGS84')
    # Geod accumulates signed rings. Some territory roster polygons have holes
    # wound in the SAME direction as their exterior: passing those rings
    # directly would add holes instead of subtracting them. Orient only the
    # temporary measurement geometry; vertices, topology and the roster stay
    # unchanged. See pyproj Geod.geometry_area_perimeter documentation.
    unoriented = np.array([abs(geod.geometry_area_perimeter(g)[0]) for g in original.geometry], dtype='float64')
    geodesic = np.array([abs(geod.geometry_area_perimeter(shapely.orient_polygons(g))[0]) for g in original.geometry], dtype='float64')
    points = original.geometry.representative_point()
    scales = np.asarray(Proj(crs).get_factors(points.x.to_numpy(), points.y.to_numpy()).areal_scale)
    planar = projected.geometry.area.to_numpy()
    return pd.DataFrame(dict(GEOID=original.GEOID.to_numpy(),StFIPS=original.StFIPS.to_numpy(),
        geodesic_area_m2=geodesic,unoriented_signed_geodesic_area_m2=unoriented,planar_area_m2=planar,areal_scale=scales,
        raw_planar_geodesic_ratio=planar/geodesic,scale_corrected_ratio=planar/(geodesic*scales)))
