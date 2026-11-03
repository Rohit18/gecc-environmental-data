#!/usr/bin/env python3
"""CONUS 2020 WorldCover tract areas and many-to-many UA crosswalk exports.

Dry run by default; --preview validates a real tract; --submit starts resumable
Drive exports. Existing READY/RUNNING/COMPLETED descriptions are skipped.
Run with /opt/anaconda3/envs/wsgee/bin/python in this workspace.
"""
import argparse
import json
import math
from pathlib import Path

import ee

TRACT_ASSET = 'projects/your-ee-project/assets/GECC/censusTracts2020US'
UA_ASSET = 'projects/your-ee-project/assets/GECC/tl_2025_us_uac20'
WC_ASSET = 'ESA/WorldCover/v100'
STATES = '01 04 05 06 08 09 10 11 12 13 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56'.split()
CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
GROUPS = {'tree': [10, 95], 'non_tree_vegetation': [20, 30, 40, 90, 100],
          'vegetation': [10, 20, 30, 40, 90, 95, 100]}
PREFIX = 'WC2020_CONUS_tract_v2'
FOLDER = 'GECC_CONUS_CensusTract_WorldCover2020'
OUT = Path(__file__).resolve().parent / 'outputs/worldcover_2020_tracts'
MANIFEST = OUT / f'submitted_tasks_{PREFIX}.jsonl'
META = ['GEOID', 'STATEFP', 'Area_Land', 'Area_Water', 'tract_geometry_area_m2',
        'urban_zone_area_m2', 'rural_zone_area_m2', 'primary_ua_id',
        'primary_ua_name', 'primary_ua_overlap_area_m2', 'ua_ids', 'ua_count',
        'ua_match_status', 'tract_boundary_asset', 'tract_boundary_year',
        'ua_boundary_asset', 'ua_boundary_year', 'ua_tiger_release_year',
        'worldcover_asset', 'worldcover_year']
PAIR_FIELDS = ['GEOID', 'UACE20', 'NAME20', 'overlap_area_m2',
               'overlap_fraction_of_tract', 'overlap_fraction_of_ua', 'is_primary']


def selectors():
    fields = list(META)
    for zone in ['urban', 'rural', 'tract']:
        fields += [f'{zone}_wc{code}_area_m2' for code in CODES]
        fields += [f'{zone}_{group}_area_m2' for group in GROUPS]
        fields += [f'{zone}_raster_area_m2', f'{zone}_valid_area_m2',
                   f'{zone}_fraction_denominator_m2',
                   f'{zone}_coverage_fraction', f'{zone}_has_valid_worldcover',
                   f'{zone}_vegetation_fraction']
    return fields


def inputs():
    tracts = ee.FeatureCollection(TRACT_ASSET).filter(ee.Filter.inList('StFIPS', STATES))
    uas = ee.FeatureCollection(UA_ASSET)
    wc = ee.ImageCollection(WC_ASSET).first().select('Map')
    return tracts, uas, wc


def overlaps(tract, uas):
    geom = tract.geometry()
    def intersect(ua):
        return ua.set('overlap_area_m2', ua.geometry().intersection(geom, 1).area(1))
    return uas.filterBounds(geom).map(intersect).filter(ee.Filter.gt('overlap_area_m2', 0))


def primary(matches):
    maximum = matches.aggregate_max('overlap_area_m2')
    return ee.Feature(ee.Algorithms.If(matches.size().gt(0),
        matches.filter(ee.Filter.eq('overlap_area_m2', maximum)).sort('UACE20').first(),
        ee.Feature(None, {'UACE20': '', 'NAME20': '', 'overlap_area_m2': 0})))


def area_image(wc, uas=None):
    """WorldCover class-area image; zones are supplied as geometries per tract."""
    valid = wc.remap(CODES, [1] * len(CODES), 0).unmask(0)
    area = ee.Image.pixelArea()
    bands = []
    for code in CODES:
        bands.append(area.multiply(wc.eq(code).unmask(0))
                     .rename(f'wc{code}_area_m2'))
    bands += [area.rename('raster_area_m2'),
              area.multiply(valid).rename('valid_area_m2')]
    return ee.Image.cat(bands)


def reduce_zone(image, geometry, projection):
    """Reduce a non-empty vector zone, returning an empty dictionary otherwise."""
    return ee.Dictionary(ee.Algorithms.If(
        geometry.area(1).gt(0),
        image.reduceRegion(ee.Reducer.sum(), geometry,
                           crs=projection['crs'],
                           crsTransform=projection['transform'],
                           maxPixels=2e10, tileScale=8),
        ee.Dictionary({})))


def painted_mask(geometry):
    """Rasterize a vector geometry on demand; target projection is set by reduction."""
    return ee.Image.constant(0).byte().paint(geometry, 1)


def tract_row(tract, uas, image, projection, single_pass=False,
              raster_tile_bounds=None):
    geom = tract.geometry()
    matches = overlaps(tract, uas)
    main = primary(matches)
    total = geom.area(1)
    # Build the zones from this tract's positive-area UA matches. The vector
    # areas are retained separately, while complementary masks on the tract
    # provide one shared WorldCover grid for urban/rural class totals.
    urban_geom = matches.geometry(1).intersection(geom, 1)
    rural_geom = geom.difference(urban_geom, 1)
    urban_area = urban_geom.area(1)
    rural_area = rural_geom.area(1)
    has_urban = urban_area.gt(0)
    has_rural = rural_area.gt(0)
    tract_mask = painted_mask(geom)
    urban_raw_mask = painted_mask(urban_geom).And(tract_mask)
    rural_raw_mask = tract_mask.And(urban_raw_mask.Not())
    zero_mask = ee.Image.constant(0).byte()
    # Explicitly force genuinely empty zones to zero. When both zones have
    # positive vector area, use complementary raster masks so class areas
    # partition the whole-tract raster totals without double-counting edges.
    urban_mask = ee.Image(ee.Algorithms.If(
        has_urban.And(has_rural), urban_raw_mask,
        ee.Algorithms.If(has_urban, tract_mask, zero_mask)))
    rural_mask = ee.Image(ee.Algorithms.If(
        has_urban.And(has_rural), rural_raw_mask,
        ee.Algorithms.If(has_rural, tract_mask, zero_mask)))
    if single_pass:
        # One scan of the WorldCover grid for all three complementary zones.
        # Prefix the bands so the summed dictionary still identifies each zone.
        bands = image.bandNames()
        stacked = ee.Image.cat([
            image.multiply(mask).rename(bands.map(
                lambda name: ee.String(zone).cat('_').cat(ee.String(name))))
            for zone, mask in [('urban', urban_mask), ('rural', rural_mask),
                               ('tract', tract_mask)]
        ])
        reduction_region = geom
        if raster_tile_bounds is not None:
            west, south, east, north = raster_tile_bounds
            # Pixel-centre membership makes adjacent repair tiles disjoint.
            # Normalize negative longitude because Aleutian bounds extend
            # east of 180 degrees in Earth Engine's unwrapped geometry.
            lonlat = ee.Image.pixelLonLat()
            lon = lonlat.select('longitude')
            if east > 180:
                lon = lon.where(lon.lt(0), lon.add(360))
            lat = lonlat.select('latitude')
            tile_mask = lon.gte(west).And(lon.lt(east)).And(
                lat.gte(south)).And(lat.lt(north))
            stacked = stacked.multiply(tile_mask)
            tile = ee.Geometry.Rectangle([west, south, east, north], None, False)
            reduction_region = geom.intersection(tile, 1)
        all_stats = stacked.reduceRegion(
            ee.Reducer.sum(), reduction_region, crs=projection['crs'],
            crsTransform=projection['transform'], maxPixels=1e12, tileScale=8)

        def stat(zone, field, default=0):
            return all_stats.get(f'{zone}_{field}', default)
    else:
        zone_stats = {
            'urban': image.multiply(urban_mask).reduceRegion(
                ee.Reducer.sum(), geom, crs=projection['crs'],
                crsTransform=projection['transform'], maxPixels=2e10, tileScale=8),
            'rural': image.multiply(rural_mask).reduceRegion(
                ee.Reducer.sum(), geom, crs=projection['crs'],
                crsTransform=projection['transform'], maxPixels=2e10, tileScale=8),
            'tract': image.multiply(tract_mask).reduceRegion(
                ee.Reducer.sum(), geom, crs=projection['crs'],
                crsTransform=projection['transform'], maxPixels=2e10, tileScale=8),
        }

        def stat(zone, field, default=0):
            return zone_stats[zone].get(field, default)
    props = {'GEOID': tract.get('GEOID'), 'STATEFP': tract.get('StFIPS'),
        'Area_Land': tract.get('Area_Land'), 'Area_Water': tract.get('Area_Water'),
        'tract_geometry_area_m2': total, 'urban_zone_area_m2': urban_area,
        'rural_zone_area_m2': rural_area,
        'primary_ua_id': main.get('UACE20'), 'primary_ua_name': main.get('NAME20'),
        'primary_ua_overlap_area_m2': main.get('overlap_area_m2'),
        'ua_ids': matches.aggregate_array('UACE20').distinct().sort().join('|'),
        'ua_count': matches.size(),
        'ua_match_status': ee.Algorithms.If(matches.size().gt(0), 'matched', 'no_overlap'),
        'tract_boundary_asset': TRACT_ASSET, 'tract_boundary_year': 2020,
        'ua_boundary_asset': UA_ASSET, 'ua_boundary_year': 2020,
        'ua_tiger_release_year': 2025, 'worldcover_asset': WC_ASSET, 'worldcover_year': 2020}
    for zone, zone_area in [('urban', urban_area), ('rural', rural_area),
                            ('tract', total)]:
        valid = ee.Number(stat(zone, 'valid_area_m2'))
        raster = ee.Number(stat(zone, 'raster_area_m2'))
        has_zone = zone_area.gt(0)
        has_pixels = raster.gt(0)
        has_valid = valid.gt(0)
        props[f'{zone}_valid_area_m2'] = valid
        props[f'{zone}_raster_area_m2'] = raster
        props[f'{zone}_fraction_denominator_m2'] = ee.Algorithms.If(
            has_zone.And(has_valid), valid, None)
        props[f'{zone}_has_valid_worldcover'] = has_zone.And(has_valid)
        props[f'{zone}_coverage_fraction'] = ee.Algorithms.If(
            has_zone.And(has_pixels), valid.divide(raster), None)
        for code in CODES:
            field = f'{zone}_wc{code}_area_m2'
            props[field] = ee.Algorithms.If(
                has_zone.And(has_valid), stat(zone, f'wc{code}_area_m2'),
                ee.Algorithms.If(has_zone, None, 0))
        for group, codes in GROUPS.items():
            value = ee.Number(0)
            for code in codes:
                value = value.add(ee.Number(stat(zone, f'wc{code}_area_m2')))
            props[f'{zone}_{group}_area_m2'] = ee.Algorithms.If(
                has_zone.And(has_valid), value,
                ee.Algorithms.If(has_zone, None, 0))
            if group == 'vegetation':
                props[f'{zone}_vegetation_fraction'] = ee.Algorithms.If(
                    has_zone.And(has_valid), value.divide(valid), None)
    return ee.Feature(None, props)


def pair_rows(tract, uas):
    tract = ee.Feature(tract)
    matches = overlaps(tract, uas)
    main = primary(matches)
    area = tract.geometry().area(1)
    def row(ua):
        overlap = ee.Number(ua.get('overlap_area_m2'))
        return ee.Feature(None, {'GEOID': tract.get('GEOID'), 'UACE20': ua.get('UACE20'),
            'NAME20': ua.get('NAME20'), 'overlap_area_m2': overlap,
            'overlap_fraction_of_tract': overlap.divide(area),
            'overlap_fraction_of_ua': overlap.divide(ua.geometry().area(1)),
            'is_primary': ee.String(ua.get('UACE20')).compareTo(main.get('UACE20')).eq(0)})
    return matches.map(row)


def check_row(row, pairs):
    ids = sorted(p['properties']['UACE20'] for p in pairs)
    assert row['ua_ids'] == '|'.join(ids)
    assert row['ua_count'] == len(ids) == len(set(ids))
    assert len(row['GEOID']) == 11
    assert all(len(code) == 5 for code in ids)
    assert sum(p['properties']['is_primary'] for p in pairs) == bool(ids)
    if ids:
        assert row['primary_ua_id'] in ids
    for code in CODES:
        u, r, t = (row.get(f'{z}_wc{code}_area_m2') for z in ['urban', 'rural', 'tract'])
        if None not in (u, r, t):
            assert math.isclose(u + r, t, abs_tol=0.01, rel_tol=1e-8)
    for zone in ['urban', 'rural', 'tract']:
        t, n, v = (row.get(f'{zone}_{g}_area_m2') for g in GROUPS)
        if None not in (t, n, v):
            assert math.isclose(t + n, v, abs_tol=0.01, rel_tol=1e-8)
        fraction = row.get(f'{zone}_vegetation_fraction')
        if fraction is not None:
            assert 0 <= fraction <= 1
        zone_area = row.get('tract_geometry_area_m2' if zone == 'tract'
                            else f'{zone}_zone_area_m2')
        if zone_area == 0:
            assert row.get(f'{zone}_raster_area_m2') == 0
            assert row.get(f'{zone}_vegetation_fraction') is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='your-ee-project')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--preview', nargs='*', default=[])
    parser.add_argument('--states', nargs='*', default=STATES)
    parser.add_argument('--batch-size', type=int, default=750)
    parser.add_argument('--max-new-tasks', type=int, default=1000)
    args = parser.parse_args()
    if not set(args.states) <= set(STATES) or args.batch_size < 1:
        raise ValueError('Invalid CONUS states or batch size')
    ee.Initialize(project=args.project)
    tracts, uas, wc = inputs()
    info = ee.Dictionary({'tract_count': tracts.size(),
        'tract_unique': tracts.aggregate_count_distinct('GEOID'),
        'state_counts': tracts.aggregate_histogram('StFIPS'),
        'ua_count': uas.size(), 'ua_unique': uas.aggregate_count_distinct('UACE20'),
        'tract_fields': tracts.first().propertyNames(), 'ua_fields': uas.first().propertyNames(),
        'projection': wc.projection()}).getInfo()
    assert info['tract_count'] == info['tract_unique'] > 0
    assert info['ua_count'] == info['ua_unique'] > 0
    assert {'UACE20', 'NAME20'} <= set(info['ua_fields'])
    assert {'GEOID', 'StFIPS', 'Area_Land', 'Area_Water'} <= set(info['tract_fields'])
    print(json.dumps(info), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'input_validation.json').write_text(json.dumps(info, indent=2))
    projection = info['projection']
    image = area_image(wc, uas)
    for geoid in args.preview:
        selected = tracts.filter(ee.Filter.eq('GEOID', geoid))
        assert selected.size().getInfo() == 1, geoid
        tract = ee.Feature(selected.first())
        row = tract_row(tract, uas, image, projection).toDictionary().getInfo()
        pairs = pair_rows(tract, uas).getInfo()['features']
        check_row(row, pairs)
        (OUT / f'preview_{PREFIX}_{geoid}.json').write_text(
            json.dumps({'row': row, 'pairs': pairs}, indent=2))
        print(f'Preview passed: {geoid}, UAs={row["ua_ids"]}, urban vegetation={row["urban_vegetation_area_m2"]}', flush=True)
    existing = {t.config.get('description'): t.status()['state'] for t in ee.batch.Task.list()
                if t.config.get('description', '').startswith(PREFIX)}
    started = 0
    for state in args.states:
        count = int(info['state_counts'][state])
        state_tracts = tracts.filter(ee.Filter.eq('StFIPS', state)).sort('GEOID')
        for offset in range(0, count, args.batch_size):
            chunk = state_tracts.toList(min(args.batch_size, count - offset), offset)
            for kind in ['areas', 'crosswalk']:
                name = f'{PREFIX}_{kind}_s{state}_n{args.batch_size}_b{offset // args.batch_size:03d}'
                if existing.get(name) in {'READY', 'RUNNING', 'COMPLETED'}:
                    print(f'SKIP {name} {existing[name]}', flush=True)
                    continue
                if not args.submit:
                    print(f'PLAN {name}', flush=True)
                    continue
                if started >= args.max_new_tasks:
                    return
                if kind == 'areas':
                    table = ee.FeatureCollection(chunk).map(lambda f: tract_row(f, uas, image, projection))
                    fields = selectors()
                else:
                    table = ee.FeatureCollection(chunk.map(lambda f: pair_rows(f, uas))).flatten()
                    fields = PAIR_FIELDS
                task = ee.batch.Export.table.toDrive(collection=table, description=name,
                    folder=FOLDER, fileNamePrefix=name, fileFormat='CSV', selectors=fields)
                task.start()
                record = {'description': name, 'task_id': task.id, 'state_fips': state,
                          'tract_count': min(args.batch_size, count - offset), 'kind': kind,
                          'drive_folder': FOLDER, 'project': args.project}
                with MANIFEST.open('a') as log:
                    log.write(json.dumps(record) + '\n')
                started += 1
                print(f'STARTED {name} {task.id}', flush=True)
    print(f'New tasks started: {started}', flush=True)


if __name__ == '__main__':
    main()
