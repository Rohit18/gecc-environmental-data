/**
 * 2020 US CENSUS TRACTS: SVCM-SP air temperature, 2003-2024.
 * Standalone JavaScript for https://code.earthengine.google.com/.
 * Select Cloud project your-ee-project before clicking Run.
 *
 * Same source, units, seasons, pixel-first means and extreme definitions as
 * the CityHealth workflow. This script uses ALL 84,414 2020 tract polygons.
 * Whole tracts are retained, including Alaska/Hawaii and missing-data rows.
 * No PRISM/GRIDMET data, water mask, fallback, clipping to cities or imputation.
 *
 * HOW TO RUN
 * 1. RUN_STAGE='means': creates one CSV task per year for the chosen batch(es).
 *    Each tract/year row contains annual + JJA + DJF means and quality counts.
 *    No threshold assets are needed. Default: batch 0, all 22 years = 22 tasks.
 * 2. Repeat for batches 1 through 9 (change FIRST_BATCH and LAST_BATCH together),
 *    or use FIRST_BATCH=0,LAST_BATCH=9 to create all 220 means tasks at once.
 *    Start the exports you need manually from Tasks. Rerunning the Code Editor
 *    creates duplicate task cards; it does not skip previous completed exports.
 * 3. For hot/cold days as well, set RUN_STAGE='thresholds'. Start the two asset
 *    exports and wait for BOTH to finish. Their region covers full US tracts,
 *    including land outside CityHealth polygons. Do NOT use city-clipped caches.
 * 4. Set RUN_STAGE='extremes' and repeat the same batches. Each export has
 *    two rows per tract/year, one per season. 220 CSV tasks for all years/batches.
 *
 * Method: daily Celsius = b1 * 0.1; temporal mean of available daily values
 * per pixel, then spatial mean across tract pixels at 1000 m / EPSG:4326.
 * Extreme thresholds: each pixel's seasonal daily Tmax P95 / Tmin P05 over
 * 2003-2007; strict > P95 and < P05; count daily exceedances then tract mean.
 * Missing source dates are NOT filled; expected and observed counts are kept.
 * Counts matching the calendar do not guarantee per-pixel completeness.
 *
 * OUTPUTS: Drive/GECC_CensusTract2020_SVCMSP_CityHealthMethod_2003_2024
 * means: GEOID + climate_year keys; extremes: GEOID + climate_year + season.
 * 1,857,108 total means rows and 3,714,216 extreme rows across all batches.
 * Geometry is removed only AFTER reduction. GEOID joins to 2020 tract shapes.
 * Batch assignment is deterministic by GEOID and must not change during a run.
 * Custom export priority is omitted: this noncommercial project rejected it.
 */

// --------------------------- CONFIGURATION ---------------------------
var RUN_STAGE = 'means'; // 'means', 'thresholds', or 'extremes'
var START_YEAR = 2003;
var END_YEAR = 2024;
var N_BATCHES = 10;
var FIRST_BATCH = 0;
var LAST_BATCH = 0; // inclusive; use 9 for all 10 batches in one Run
var BATCH_SEED = 20260914;
var SHOW_PREVIEW = false; // true computes one tract/year sample; no extra export
var PREVIEW_GEOID = '06001400100';

var TRACT_ASSET = 'projects/your-ee-project/assets/GECC/censusTracts2020US';
var SOURCE_ASSET = 'projects/sat-io/open-datasets/global-daily-air-temp/north_america';
var DRIVE_FOLDER = 'GECC_CensusTract2020_SVCMSP_CityHealthMethod_2003_2024';
// New asset names at project root; no folder creation required. Do not overwrite
// the CityHealth threshold assets. Change this root if your account needs it.
var THRESHOLD_ROOT = 'projects/your-ee-project/assets/CensusTract2020_SVCMSP_v1';
var BASELINE_START_YEAR = 2003;
var BASELINE_END_YEAR = 2007;
var SCALE_FACTOR_TO_C = 0.1;
var ANALYSIS_SCALE_M = 1000;
var ANALYSIS_CRS = 'EPSG:4326';
var TILE_SCALE = 4;
var MAX_PIXELS_PER_TRACT = 1e8;
var SEASONS = ['JJA', 'DJF'];
var ID_FIELDS = ['GEOID', 'StFIPS', 'CoFIPS', 'TrFIPS', 'Area_Land', 'Area_Water', 'Pop'];
var EXTREME_METRICS = ['hot_days_tmax_gt_p95', 'cold_days_tmin_lt_p05', 'valid_tmax_days', 'valid_tmin_days'];
var source, tracts, nativeProjection;

// ------------------------------ HELPERS ------------------------------
function validateConfig() {
  if (['means', 'thresholds', 'extremes'].indexOf(RUN_STAGE) < 0) throw new Error('Unknown RUN_STAGE.');
  if (START_YEAR < 2003 || END_YEAR > 2024 || START_YEAR > END_YEAR ||
      Math.floor(START_YEAR) !== START_YEAR || Math.floor(END_YEAR) !== END_YEAR) {
    throw new Error('Use integer years within 2003-2024.');
  }
  if (N_BATCHES < 1 || Math.floor(N_BATCHES) !== N_BATCHES || FIRST_BATCH < 0 ||
      LAST_BATCH >= N_BATCHES || FIRST_BATCH > LAST_BATCH ||
      Math.floor(FIRST_BATCH) !== FIRST_BATCH || Math.floor(LAST_BATCH) !== LAST_BATCH) {
    throw new Error('Invalid batch range.');
  }
}
function pad(value) { return ('000' + value).slice(-3); }
function expectedDays(year, period) {
  var leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  if (period === 'annual') return leap ? 366 : 365;
  if (period === 'JJA') return 92;
  if (period === 'DJF') return leap ? 91 : 90;
  throw new Error('Unknown period: ' + period);
}
function months(period) {
  if (period === 'annual') return 'Jan-Dec';
  if (period === 'JJA') return 'Jun-Aug';
  if (period === 'DJF') return 'Jan-Feb + Dec (same year)';
  throw new Error('Unknown period: ' + period);
}
function daily(year, period, variable) {
  var typed = source.filter(ee.Filter.eq('prop_type', variable));
  var filtered;
  if (period === 'annual') {
    filtered = typed.filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year + 1, 1, 1));
  } else if (period === 'JJA') {
    filtered = typed.filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 9, 1));
  } else if (period === 'DJF') {
    filtered = typed.filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 3, 1))
        .merge(typed.filterDate(ee.Date.fromYMD(year, 12, 1), ee.Date.fromYMD(year + 1, 1, 1)));
  } else { throw new Error('Unknown period: ' + period); }
  return filtered.map(function(image) {
    return image.select('b1').multiply(SCALE_FACTOR_TO_C).rename('temperature_c').toFloat()
        .copyProperties(image, ['system:time_start', 'prop_type']);
  }).sort('system:time_start');
}
// A masked fallback gives empty collections a band schema, without fabricating
// observations or changing counts. It has no effect when real images exist.
function safeReduce(collection, method, name) {
  var empty = ee.Image.constant(0).rename('temperature_c').updateMask(ee.Image.constant(0));
  var usable = ee.ImageCollection(ee.Algorithms.If(collection.size().gt(0), collection, ee.ImageCollection([empty])));
  if (method === 'mean') return usable.mean().rename(name);
  if (method === 'count') return usable.count().rename(name);
  if (method === 'sum') return usable.sum().rename(name);
  throw new Error('Unknown reduction: ' + method);
}
function meanNames(period) {
  if (period === 'annual') return ['annual_mean_daily_tmax_c', 'annual_mean_daily_tmin_c',
      'valid_annual_tmax_days', 'valid_annual_tmin_days'];
  return ['seasonal_mean_daily_tmax_c_' + period, 'seasonal_mean_daily_tmin_c_' + period,
      'valid_seasonal_mean_tmax_days_' + period, 'valid_seasonal_mean_tmin_days_' + period];
}
function baseMetadata(year, batch) {
  return {boundary_year: 2020, climate_year: year, batch: batch, n_batches: N_BATCHES,
    batch_seed: BATCH_SEED, temperature_units: 'degC', analysis_scale_m: ANALYSIS_SCALE_M,
    analysis_crs: ANALYSIS_CRS, source_asset: SOURCE_ASSET, tract_asset: TRACT_ASSET,
    temporal_statistic: 'available daily values per pixel before spatial aggregation',
    spatial_statistic: 'mean across valid tract pixels'};
}
function addCounts(metadata, year, period, high, low) {
  metadata[period + '_expected_days'] = expectedDays(year, period);
  metadata[period + '_source_tmax_image_count'] = high.size();
  metadata[period + '_source_tmin_image_count'] = low.size();
  metadata[period + '_source_counts_match_calendar'] = high.size().eq(expectedDays(year, period))
      .and(low.size().eq(expectedDays(year, period)));
  metadata[period + '_months'] = months(period);
}
function reduceTracts(image, batchTracts, metrics, metadata) {
  return image.reduceRegions({collection: batchTracts, reducer: ee.Reducer.mean(),
    scale: ANALYSIS_SCALE_M, crs: ANALYSIS_CRS, tileScale: TILE_SCALE,
    maxPixelsPerRegion: MAX_PIXELS_PER_TRACT}).map(function(feature) {
      // Retain all source tracts, including those with no valid raster pixels.
      return ee.Feature(null, feature.toDictionary(ID_FIELDS.concat(metrics))).set(metadata);
    });
}
function meansTable(year, batch, batchTracts) {
  var bands = [], metrics = [], metadata = baseMetadata(year, batch);
  ['annual', 'JJA', 'DJF'].forEach(function(period) {
    var high = daily(year, period, 'tmax'), low = daily(year, period, 'tmin');
    var names = meanNames(period);
    bands.push(safeReduce(high, 'mean', names[0]), safeReduce(low, 'mean', names[1]),
        safeReduce(high, 'count', names[2]), safeReduce(low, 'count', names[3]));
    metrics = metrics.concat(names);
    addCounts(metadata, year, period, high, low);
  });
  return reduceTracts(ee.Image.cat(bands).toFloat(), batchTracts, metrics, metadata);
}
function baseline(season, variable) {
  var result = ee.ImageCollection([]);
  for (var year = BASELINE_START_YEAR; year <= BASELINE_END_YEAR; year++) {
    result = result.merge(daily(year, season, variable));
  }
  return result.sort('system:time_start');
}
function thresholdAsset(season) {
  return THRESHOLD_ROOT + '_' + season + '_thresholds_2003_2007';
}
function thresholds(season) {
  return ee.Image.cat([
    baseline(season, 'tmax').reduce(ee.Reducer.percentile([95])).rename('tmax_p95_c'),
    baseline(season, 'tmin').reduce(ee.Reducer.percentile([5])).rename('tmin_p05_c')
  ]).toFloat().set({season: season, months: months(season), baseline_start_year: 2003,
    baseline_end_year: 2007, tmax_percentile: 95, tmin_percentile: 5,
    source_asset: SOURCE_ASSET, tract_asset: TRACT_ASSET, temperature_units: 'degC'});
}
function extremesTable(year, season, batch, batchTracts) {
  var limits = ee.Image(thresholdAsset(season));
  var high = daily(year, season, 'tmax'), low = daily(year, season, 'tmin');
  var hot = high.map(function(image) {
    return image.gt(limits.select('tmax_p95_c').rename('temperature_c')).rename('temperature_c');
  });
  var cold = low.map(function(image) {
    return image.lt(limits.select('tmin_p05_c').rename('temperature_c')).rename('temperature_c');
  });
  var validHigh = safeReduce(high, 'count', EXTREME_METRICS[2]);
  var validLow = safeReduce(low, 'count', EXTREME_METRICS[3]);
  var image = ee.Image.cat([safeReduce(hot, 'sum', EXTREME_METRICS[0]),
    safeReduce(cold, 'sum', EXTREME_METRICS[1]), validHigh, validLow]).toInt16()
    .updateMask(validHigh.gt(0).and(validLow.gt(0)));
  var metadata = baseMetadata(year, batch);
  metadata.season = season; metadata.months = months(season);
  metadata.baseline_start_year = 2003; metadata.baseline_end_year = 2007;
  metadata.tmax_threshold_percentile = 95; metadata.tmin_threshold_percentile = 5;
  metadata.extreme_count_units = 'days'; metadata.threshold_asset = thresholdAsset(season);
  addCounts(metadata, year, season, high, low);
  return reduceTracts(image, batchTracts, EXTREME_METRICS, metadata);
}
function metadataFields() {
  return ['boundary_year', 'climate_year', 'batch', 'n_batches', 'batch_seed',
    'temperature_units', 'analysis_scale_m', 'analysis_crs', 'source_asset', 'tract_asset',
    'temporal_statistic', 'spatial_statistic'];
}
function countFields(period) {
  return [period + '_expected_days', period + '_source_tmax_image_count',
    period + '_source_tmin_image_count', period + '_source_counts_match_calendar', period + '_months'];
}
function outputFields(stage) {
  var fields = ID_FIELDS.concat(metadataFields());
  if (stage === 'means') {
    ['annual', 'JJA', 'DJF'].forEach(function(p) { fields = fields.concat(meanNames(p), countFields(p)); });
  } else {
    fields = fields.concat(['season', 'months', 'baseline_start_year', 'baseline_end_year',
      'tmax_threshold_percentile', 'tmin_threshold_percentile', 'extreme_count_units', 'threshold_asset'],
      EXTREME_METRICS, countFields('JJA'), countFields('DJF'));
  }
  return fields;
}

// MAIN: configurations/helpers above can be tested without launching tasks.
validateConfig();
source = ee.ImageCollection(SOURCE_ASSET);
// Resolve the actual native source grid, rather than relying on a copied grid.
// This small metadata request is also an early source-access check.
nativeProjection = ee.Image(source.first()).select('b1').projection().getInfo();
tracts = ee.FeatureCollection(TRACT_ASSET).filter(ee.Filter.notNull(['GEOID']))
    .map(function(feature) { return ee.Feature(feature.geometry(), feature.toDictionary(ID_FIELDS)); })
    .randomColumn('_batch_random', BATCH_SEED, 'uniform', ['GEOID'])
    .map(function(feature) { return feature.set('_batch', ee.Number(feature.get('_batch_random')).multiply(N_BATCHES).floor().toInt()); });
var inputAudit = ee.Dictionary({count: tracts.size(), unique: tracts.aggregate_count_distinct('GEOID'),
  batch_counts: tracts.aggregate_histogram('_batch')}).getInfo();
if (inputAudit.count !== 84414 || inputAudit.unique !== inputAudit.count) {
  throw new Error('Expected exactly 84,414 unique 2020 tract GEOIDs. Check asset/version.');
}
print('Input audit', inputAudit);
print('Native SVCM-SP grid', nativeProjection);
print('Run stage', RUN_STAGE);

if (RUN_STAGE === 'thresholds') {
  // Near-global longitude range includes US tracts on BOTH sides of the date
  // line. Raster is masked to full tracts and source data, not city polygons.
  var region = ee.Geometry.Rectangle([-179.999999, 15, 179.999999, 72], 'EPSG:4326', false);
  SEASONS.forEach(function(season) {
    Export.image.toAsset({image: thresholds(season).clipToCollection(tracts),
      description: 'CensusTract2020_SVCMSP_v1_' + season + '_thresholds_2003_2007',
      assetId: thresholdAsset(season), region: region, crs: nativeProjection.crs,
      crsTransform: nativeProjection.transform, maxPixels: 1e13,
      pyramidingPolicy: {'.default': 'mean'}});
    print(season + ' baseline source counts', baseline(season, 'tmax').size(), baseline(season, 'tmin').size());
  });
  print('Two threshold export tasks prepared. Start both; wait for completion before extremes.');
} else {
  if (RUN_STAGE === 'extremes') {
    // Fail before creating CSV task cards if either cache is missing/unreadable.
    SEASONS.forEach(function(season) {
      var cache = ee.Image(thresholdAsset(season));
      var info = ee.Dictionary({bands: cache.bandNames(), season: cache.get('season'),
        tract_asset: cache.get('tract_asset'), source_asset: cache.get('source_asset'),
        baseline_start_year: cache.get('baseline_start_year'), baseline_end_year: cache.get('baseline_end_year')}).getInfo();
      if (info.bands.indexOf('tmax_p95_c') < 0 || info.bands.indexOf('tmin_p05_c') < 0 ||
          info.tract_asset !== TRACT_ASSET || info.source_asset !== SOURCE_ASSET || info.season !== season ||
          info.baseline_start_year !== 2003 || info.baseline_end_year !== 2007) {
        throw new Error('Wrong threshold cache for ' + season + ': rerun the full-tract thresholds stage.');
      }
    });
  }
  if (SHOW_PREVIEW) {
    var sample = tracts.filter(ee.Filter.eq('GEOID', PREVIEW_GEOID));
    if (sample.size().getInfo() !== 1) throw new Error('Preview GEOID not found.');
    print('One-tract preview', RUN_STAGE === 'means' ? meansTable(START_YEAR, -1, sample) :
      extremesTable(START_YEAR, 'DJF', -1, sample));
  }
  var tasks = 0;
  for (var batch = FIRST_BATCH; batch <= LAST_BATCH; batch++) {
    var batchTracts = tracts.filter(ee.Filter.eq('_batch', batch));
    if (!inputAudit.batch_counts[String(batch)]) throw new Error('Empty batch: ' + batch);
    for (var year = START_YEAR; year <= END_YEAR; year++) {
      var result = RUN_STAGE === 'means' ? meansTable(year, batch, batchTracts) :
        extremesTable(year, 'JJA', batch, batchTracts).merge(extremesTable(year, 'DJF', batch, batchTracts));
      var name = 'CensusTract2020_SVCMSP_v1_' + RUN_STAGE + '_' + year +
        '_s' + BATCH_SEED + '_b' + pad(batch) + 'of' + pad(N_BATCHES);
      Export.table.toDrive({collection: result, description: name, folder: DRIVE_FOLDER,
        fileNamePrefix: name, fileFormat: 'CSV', selectors: outputFields(RUN_STAGE)});
      tasks++;
    }
  }
  print('Prepared CSV export tasks (start manually in Tasks)', tasks);
}
