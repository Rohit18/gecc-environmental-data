/**
 * GECC 2020 US CENSUS TRACTS: 2020 Leaf Area Index statistics.
 * Paste into https://code.earthengine.google.com/ and select your-ee-project
 * (or another Cloud project with access to the configured assets).
 *
 * Scientific definition
 * ---------------------
 * LAI comes from the supplied 2020 global uint8 mosaic. Its metadata states
 * that the stored values are LAI x 10, so this script divides LAI_uint8 by 10
 * before reduction and exports native LAI units (m2 leaf area per m2 ground
 * area). Statistics use the mosaic's existing valid-pixel mask directly.
 * Missing pixels remain masked and are not changed to zero; source zeros are
 * retained as valid LAI observations.
 *
 * Output statistics per tract: mean, median, population standard deviation,
 * valid-pixel count, minimum, maximum, P25, and P75. All 84,414 source tracts
 * are retained, including tracts with null statistics.
 *
 * EXPORT-ONLY SCRIPT: running it creates one Drive task per selected batch.
 * The default range creates all 120 tasks. Narrow FIRST_BATCH/LAST_BATCH to
 * submit or rerun incrementally. Rerunning a completed batch creates a
 * duplicate task card.
 */

// --------------------------- CONFIGURATION ---------------------------
var N_BATCHES = 120;
var FIRST_BATCH = 0;
var LAST_BATCH = 119; // inclusive; 0-119 creates all 120 exports
var BATCH_SEED = 20260914;

var TRACT_ASSET =
  'projects/your-ee-project/assets/GECC/censusTracts2020US';
var LAI_ASSET =
  'projects/your-ee-project/assets/LAI_2020_Global_Mosaic_uint8';
var LAI_BAND = 'LAI_uint8';
var LAI_INPUT_DIVISOR = 10;
var DRIVE_FOLDER = 'GECC_CensusTract2020_LAI_2020_uint8_source';

var ANALYSIS_SCALE_M = 10;
var ANALYSIS_CRS = 'EPSG:4326';
var TILE_SCALE = 8;
var MAX_PIXELS_PER_REGION = 1e10;
var SCRIPT_VERSION = '1.1.0';
var ID_FIELDS = [
  'GEOID', 'StFIPS', 'CoFIPS', 'TrFIPS',
  'Area_Land', 'Area_Water', 'Pop'
];

// ------------------------------ CHECKS -------------------------------
function validateConfig() {
  if (N_BATCHES < 1 || Math.floor(N_BATCHES) !== N_BATCHES ||
      FIRST_BATCH < 0 || LAST_BATCH >= N_BATCHES ||
      FIRST_BATCH > LAST_BATCH ||
      Math.floor(FIRST_BATCH) !== FIRST_BATCH ||
      Math.floor(LAST_BATCH) !== LAST_BATCH) {
    throw new Error('Invalid batch configuration.');
  }
}

function pad(value) {
  return ('000' + value).slice(-3);
}

validateConfig();

// -------------------------- INPUT IMAGES -----------------------------
var laiSource = ee.Image(LAI_ASSET);
var lai = laiSource.select(LAI_BAND)
  .divide(LAI_INPUT_DIVISOR)
  .rename('lai_native')
  .toFloat();

var statisticsReducer = ee.Reducer.mean()
  .combine({reducer2: ee.Reducer.median(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.count(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.minMax(), sharedInputs: true})
  .combine({
    reducer2: ee.Reducer.percentile([25, 75], ['p25', 'p75']),
    sharedInputs: true
  });

var STAT_FIELDS = [
  'lai_native_mean',
  'lai_native_median',
  'lai_native_stdDev',
  'lai_native_count',
  'lai_native_min',
  'lai_native_max',
  'lai_native_p25',
  'lai_native_p75'
];

var METADATA_FIELDS = [
  'boundary_year', 'lai_year', 'batch', 'n_batches', 'batch_seed',
  'analysis_scale_m', 'analysis_crs', 'lai_units', 'statistical_support',
  'missing_value_policy', 'lai_source', 'lai_input_band',
  'lai_input_divisor', 'lai_storage_units', 'script_version'
];
var OUTPUT_FIELDS = ID_FIELDS.concat(['geometry_area_m2'])
  .concat(STAT_FIELDS).concat(METADATA_FIELDS);

// ------------------------- TRACT BATCHING -----------------------------
var tracts = ee.FeatureCollection(TRACT_ASSET)
  .filter(ee.Filter.notNull(['GEOID']))
  .randomColumn('_lai_batch_random', BATCH_SEED)
  .map(function(feature) {
    var batch = ee.Number(feature.get('_lai_batch_random'))
      .multiply(N_BATCHES).floor().int();
    return feature.set({
      _lai_batch: batch,
      geometry_area_m2: feature.geometry().area(1)
    });
  });

function baseMetadata(batch) {
  return {
    boundary_year: 2020,
    lai_year: 2020,
    batch: batch,
    n_batches: N_BATCHES,
    batch_seed: BATCH_SEED,
    analysis_scale_m: ANALYSIS_SCALE_M,
    analysis_crs: ANALYSIS_CRS,
    lai_units: 'm2_leaf_per_m2_ground',
    statistical_support: 'valid pixels in the supplied LAI mosaic',
    missing_value_policy: 'null for masked pixels; source zeros remain valid',
    lai_source: LAI_ASSET,
    lai_input_band: LAI_BAND,
    lai_input_divisor: LAI_INPUT_DIVISOR,
    lai_storage_units: 'LAI_times_10_uint8',
    script_version: SCRIPT_VERSION
  };
}

function statisticsTable(input, batch) {
  var reduced = lai.reduceRegions({
    collection: input,
    reducer: statisticsReducer,
    scale: ANALYSIS_SCALE_M,
    crs: ANALYSIS_CRS,
    tileScale: TILE_SCALE,
    maxPixelsPerRegion: MAX_PIXELS_PER_REGION
  });
  var metadata = baseMetadata(batch);
  return reduced.map(function(feature) {
    // Geometry-free CSV rows remain joinable through the 11-character GEOID.
    return ee.Feature(null, feature.toDictionary(
      ID_FIELDS.concat(['geometry_area_m2']).concat(STAT_FIELDS)
    )).set(metadata);
  });
}

// ------------------------------ EXPORT -------------------------------
print('Census tract source count (expected 84414)', tracts.size());
print('Distinct GEOIDs', tracts.aggregate_count_distinct('GEOID'));
print('LAI source', laiSource);
print('LAI projection', lai.projection());
print('Batch counts', tracts.aggregate_histogram('_lai_batch'));

for (var batch = FIRST_BATCH; batch <= LAST_BATCH; batch++) {
  var batchTracts = tracts.filter(ee.Filter.eq('_lai_batch', batch));
  var output = statisticsTable(batchTracts, batch);
  var tag = 'b' + pad(batch) + 'of' + pad(N_BATCHES);
  Export.table.toDrive({
    collection: output,
    description: 'GECC_CensusTract2020_LAI_2020_uint8_source_' + tag,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'GECC_CensusTract2020_LAI_2020_uint8_source_' + tag,
    fileFormat: 'CSV',
    selectors: OUTPUT_FIELDS
  });
}
print('Export task cards created for batches', FIRST_BATCH, LAST_BATCH);
