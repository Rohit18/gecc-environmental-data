"""Rerun failed census tract LAI batch 072 of 120 in two exports.

The original task GECC_CensusTract2020_LAI_2020_uint8_source_b072of120 failed
with "Too many pixels" because tract 02185000200 (North Slope Borough, AK)
spans about 1.09e10 native LAI grid pixels in its bounding box.

1. Standard export: batch 072 without 02185000200, same computation as
   gee_census_tract_2020_lai_statistics.js v1.1.0.
2. Tiled export: 02185000200 split into cells aligned with the native LAI
   grid. The raw uint8 band is reduced to a 256-bin weighted histogram plus a
   count in each cell. Histograms and counts are summed across cells, then the
   statistics are computed in native LAI units. Tested against the standard
   reducer on 02020980200 split into 40 cells: min, max, median, P25 and P75
   match exactly. Mean, stdDev and count differ by <1.3e-5 relative because of
   pixels at tile boundaries (outputs/census_tract_lai_b072_rerun/validation.json).

Both CSVs share the original column schema. script_version marks the tiled
row.

  python submit_census_tract_lai_b072_rerun.py validate
  python submit_census_tract_lai_b072_rerun.py submit
"""
import argparse
import datetime
import json
import math
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/census_tract_lai_b072_rerun'
PROJECT = 'your-ee-project'

N_BATCHES = 120
BATCH = 72
BATCH_SEED = 20260914
LARGE_GEOID = '02185000200'
TRACT_ASSET = 'projects/your-ee-project/assets/GECC/censusTracts2020US'
LAI_ASSET = 'projects/your-ee-project/assets/LAI_2020_Global_Mosaic_uint8'
LAI_BAND = 'LAI_uint8'
LAI_INPUT_DIVISOR = 10
DRIVE_FOLDER = 'GECC_CensusTract2020_LAI_2020_uint8_source'
ANALYSIS_SCALE_M = 10
ANALYSIS_CRS = 'EPSG:4326'
TILE_SCALE = 8
MAX_PIXELS_PER_REGION = 1e10
SCRIPT_VERSION = '1.1.0'
TILED_VERSION = '1.1.0-tiled-histogram'

# Native LAI grid (EPSG:4326), from the asset's projection transform.
PIXEL_DEG = 8.983152841195215e-05
ORIGIN_X = -180.00001488697754
ORIGIN_Y = 90.00005235925298
CELL_W_PX = 22264  # about 2.0 degrees
CELL_H_PX = 11132  # about 1.0 degree; about 2.5e8 bbox pixels per cell

TAG = f'b{BATCH:03d}of{N_BATCHES:03d}'
BASE_NAME = f'GECC_CensusTract2020_LAI_2020_uint8_source_{TAG}'
STANDARD_DESC = f'{BASE_NAME}_excl{LARGE_GEOID}'
TILED_DESC = f'{BASE_NAME}_{LARGE_GEOID}_tiled'

ID_FIELDS = ['GEOID', 'StFIPS', 'CoFIPS', 'TrFIPS', 'Area_Land', 'Area_Water', 'Pop']
STAT_FIELDS = ['lai_native_mean', 'lai_native_median', 'lai_native_stdDev',
               'lai_native_count', 'lai_native_min', 'lai_native_max',
               'lai_native_p25', 'lai_native_p75']
METADATA_FIELDS = ['boundary_year', 'lai_year', 'batch', 'n_batches', 'batch_seed',
                   'analysis_scale_m', 'analysis_crs', 'lai_units', 'statistical_support',
                   'missing_value_policy', 'lai_source', 'lai_input_band',
                   'lai_input_divisor', 'lai_storage_units', 'script_version']
OUTPUT_FIELDS = ID_FIELDS + ['geometry_area_m2'] + STAT_FIELDS + METADATA_FIELDS


def raw_band():
    return ee.Image(LAI_ASSET).select(LAI_BAND)


def lai_native():
    return raw_band().divide(LAI_INPUT_DIVISOR).rename('lai_native').toFloat()


def statistics_reducer():
    return (ee.Reducer.mean()
            .combine(ee.Reducer.median(), None, True)
            .combine(ee.Reducer.stdDev(), None, True)
            .combine(ee.Reducer.count(), None, True)
            .combine(ee.Reducer.minMax(), None, True)
            .combine(ee.Reducer.percentile([25, 75], ['p25', 'p75']), None, True))


def tracts():
    return (ee.FeatureCollection(TRACT_ASSET)
            .filter(ee.Filter.notNull(['GEOID']))
            .randomColumn('_lai_batch_random', BATCH_SEED)
            .map(lambda f: f.set({
                '_lai_batch': ee.Number(f.get('_lai_batch_random'))
                .multiply(N_BATCHES).floor().int(),
                'geometry_area_m2': f.geometry().area(1)})))


def metadata(script_version, support):
    return {
        'boundary_year': 2020, 'lai_year': 2020, 'batch': BATCH,
        'n_batches': N_BATCHES, 'batch_seed': BATCH_SEED,
        'analysis_scale_m': ANALYSIS_SCALE_M, 'analysis_crs': ANALYSIS_CRS,
        'lai_units': 'm2_leaf_per_m2_ground', 'statistical_support': support,
        'missing_value_policy': 'null for masked pixels; source zeros remain valid',
        'lai_source': LAI_ASSET, 'lai_input_band': LAI_BAND,
        'lai_input_divisor': LAI_INPUT_DIVISOR,
        'lai_storage_units': 'LAI_times_10_uint8', 'script_version': script_version}


def standard_table(collection):
    reduced = lai_native().reduceRegions(
        collection=collection, reducer=statistics_reducer(),
        scale=ANALYSIS_SCALE_M, crs=ANALYSIS_CRS, tileScale=TILE_SCALE,
        maxPixelsPerRegion=MAX_PIXELS_PER_REGION)
    meta = metadata(SCRIPT_VERSION, 'valid pixels in the supplied LAI mosaic')
    return reduced.map(lambda f: ee.Feature(
        None, f.toDictionary(ID_FIELDS + ['geometry_area_m2'] + STAT_FIELDS)).set(meta))


def grid_cells(bounds, cell_w_px=CELL_W_PX, cell_h_px=CELL_H_PX):
    """Client-side rectangles aligned with the native pixel grid covering bounds."""
    xs = [c[0] for c in bounds]
    ys = [c[1] for c in bounds]
    i0 = math.floor((min(xs) - ORIGIN_X) / PIXEL_DEG / cell_w_px)
    i1 = math.ceil((max(xs) - ORIGIN_X) / PIXEL_DEG / cell_w_px)
    j0 = math.floor((ORIGIN_Y - max(ys)) / PIXEL_DEG / cell_h_px)
    j1 = math.ceil((ORIGIN_Y - min(ys)) / PIXEL_DEG / cell_h_px)
    cells = []
    for i in range(i0, i1):
        for j in range(j0, j1):
            x0 = ORIGIN_X + i * cell_w_px * PIXEL_DEG
            x1 = ORIGIN_X + (i + 1) * cell_w_px * PIXEL_DEG
            y1 = ORIGIN_Y - j * cell_h_px * PIXEL_DEG
            y0 = ORIGIN_Y - (j + 1) * cell_h_px * PIXEL_DEG
            cells.append((x0, y0, x1, y1))
    return cells


def tiled_statistics(feature, cells):
    """Statistics dictionary for one feature from merged per-cell histograms."""
    geom = feature.geometry()
    cell_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Rectangle(list(c), 'EPSG:4326', False)) for c in cells])
    pieces = (cell_fc.filterBounds(geom)
              .map(lambda c: ee.Feature(c.geometry().intersection(geom, ee.ErrorMargin(1)))))
    reducer = ee.Reducer.fixedHistogram(0, 256, 256).combine(ee.Reducer.count(), None, True)
    reduced = raw_band().reduceRegions(
        collection=pieces, reducer=reducer, scale=ANALYSIS_SCALE_M, crs=ANALYSIS_CRS,
        tileScale=TILE_SCALE, maxPixelsPerRegion=MAX_PIXELS_PER_REGION)
    reduced = reduced.filter(ee.Filter.gt('count', 0))
    weights = ee.Array(reduced.aggregate_array('histogram').map(
        lambda h: ee.Array(h).slice(1, 1, 2).project([0]).toList()))
    w = ee.Array(ee.Algorithms.If(
        reduced.size().gt(0),
        weights.reduce(ee.Reducer.sum(), [0]).project([1]),
        ee.Array([0] * 256)))
    v = ee.Array(ee.List.sequence(0, 255))
    total = ee.Number(w.reduce(ee.Reducer.sum(), [0]).get([0]))
    divisor = ee.Number(LAI_INPUT_DIVISOR)

    def weighted_sum(arr):
        return ee.Number(arr.reduce(ee.Reducer.sum(), [0]).get([0]))

    mean_raw = weighted_sum(w.multiply(v)).divide(total)
    var_raw = weighted_sum(w.multiply(v).multiply(v)).divide(total).subtract(mean_raw.pow(2)).max(0)
    present = v.mask(w.gt(0))
    cum = w.accum(0)

    def pct(p):
        return ee.Number(v.mask(cum.gte(total.multiply(p))).reduce(ee.Reducer.min(), [0]).get([0]))

    stats = ee.Dictionary({
        'lai_native_mean': mean_raw.divide(divisor),
        'lai_native_median': pct(0.5).divide(divisor),
        'lai_native_stdDev': var_raw.sqrt().divide(divisor),
        'lai_native_count': reduced.aggregate_sum('count'),
        'lai_native_min': ee.Number(present.reduce(ee.Reducer.min(), [0]).get([0])).divide(divisor),
        'lai_native_max': ee.Number(present.reduce(ee.Reducer.max(), [0]).get([0])).divide(divisor),
        'lai_native_p25': pct(0.25).divide(divisor),
        'lai_native_p75': pct(0.75).divide(divisor)})
    return ee.Dictionary(ee.Algorithms.If(total.gt(0), stats, ee.Dictionary({
        'lai_native_count': reduced.aggregate_sum('count')})))


def tiled_table(feature, cells):
    stats = tiled_statistics(feature, cells)
    meta = metadata(TILED_VERSION,
                    'valid pixels in the supplied LAI mosaic; merged 256-bin histograms '
                    'over native-grid-aligned tiles')
    row = feature.toDictionary(ID_FIELDS + ['geometry_area_m2']).combine(stats)
    return ee.FeatureCollection([ee.Feature(None, row).set(meta)])


def save(path, data):
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def validate():
    t = tracts()
    batch = t.filter(ee.Filter.eq('_lai_batch', BATCH))
    info = ee.Dictionary({
        'tract_count': t.size(),
        'batch_count': batch.size(),
        'batch_has_large': batch.filter(ee.Filter.eq('GEOID', LARGE_GEOID)).size(),
        'standard_count': batch.filter(ee.Filter.neq('GEOID', LARGE_GEOID)).size(),
        'large_bounds': t.filter(ee.Filter.eq('GEOID', LARGE_GEOID)).first()
        .geometry().bounds(1).coordinates().get(0)}).getInfo()
    assert info['tract_count'] == 84414, info
    assert info['batch_has_large'] == 1 and info['standard_count'] == info['batch_count'] - 1, info
    large_cells = grid_cells(info.pop('large_bounds'))
    info['large_cell_candidates'] = len(large_cells)
    info['large_cell_bbox_pixels'] = CELL_W_PX * CELL_H_PX

    # Tiled vs standard on a smaller Alaska tract, with small cells so the
    # merge crosses many internal tile boundaries.
    test_geoid = '02020980200'
    test = t.filter(ee.Filter.eq('GEOID', test_geoid)).first()
    test_cells = grid_cells(test.geometry().bounds(1).coordinates().get(0).getInfo(), 1113, 557)
    std = lai_native().reduceRegion(
        statistics_reducer(), test.geometry(), ANALYSIS_SCALE_M, ANALYSIS_CRS, None,
        False, MAX_PIXELS_PER_REGION, TILE_SCALE).getInfo()
    tiled = tiled_statistics(test, test_cells).getInfo()
    comparison = {}
    for k in STAT_FIELDS:
        a, b = std[k], tiled[k]
        comparison[k] = {'standard': a, 'tiled': b,
                         'rel_diff': abs(a - b) / max(abs(a), 1e-12)}
    info['test'] = {'geoid': test_geoid, 'cells': len(test_cells), 'comparison': comparison}
    for k in ['lai_native_median', 'lai_native_min', 'lai_native_max',
              'lai_native_p25', 'lai_native_p75']:
        assert comparison[k]['rel_diff'] < 1e-6, (k, comparison[k])
    assert comparison['lai_native_mean']['rel_diff'] < 1e-4, comparison
    assert comparison['lai_native_stdDev']['rel_diff'] < 1e-4, comparison
    assert comparison['lai_native_count']['rel_diff'] < 1e-4, comparison
    save(OUT / 'validation.json', info)
    print(json.dumps(info, indent=2))
    return info


def submit():
    info = json.loads((OUT / 'validation.json').read_text())
    t = tracts()
    batch = t.filter(ee.Filter.eq('_lai_batch', BATCH))
    large = t.filter(ee.Filter.eq('GEOID', LARGE_GEOID)).first()
    cells = grid_cells(large.geometry().bounds(1).coordinates().get(0).getInfo())
    jobs = [
        (STANDARD_DESC, standard_table(batch.filter(ee.Filter.neq('GEOID', LARGE_GEOID)))),
        (TILED_DESC, tiled_table(large, cells)),
    ]
    log = []
    for desc, table in jobs:
        task = ee.batch.Export.table.toDrive(
            collection=table, description=desc, folder=DRIVE_FOLDER,
            fileNamePrefix=desc, fileFormat='CSV', selectors=OUTPUT_FIELDS)
        task.start()
        entry = {'description': desc, 'task_id': task.id, 'project': PROJECT,
                 'submitted_utc': datetime.datetime.utcnow().isoformat(timespec='seconds')}
        print(json.dumps(entry))
        log.append(entry)
    save(OUT / 'submitted_tasks.json', {'validation_standard_count': info['standard_count'],
                                        'large_cells': len(cells), 'tasks': log})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('action', choices=['validate', 'submit'])
    args = parser.parse_args()
    ee.Initialize(project=PROJECT)
    validate() if args.action == 'validate' else submit()


if __name__ == '__main__':
    main()
