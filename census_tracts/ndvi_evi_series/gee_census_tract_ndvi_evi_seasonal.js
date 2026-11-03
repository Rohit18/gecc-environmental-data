/**
 * SUPERSEDED: historical unmasked pilot, not the NDVI/EVI v1 production method.
 * It omits the water mask, territory add-on, clear-observation depth, and
 * Alaska EPSG:3338 size tiers. Use submit_census_tract_ndvi_evi_exports.py
 * and plans/census_tract_ndvi_evi_v1_HANDOFF.md for the production release.
 *
 * Seasonal Landsat C2 L2 NDVI + EVI statistics for 2020 US Census Tracts.
 *
 * GEE Code Editor JavaScript translation of the GECC FUA workflow, adapted
 * to Census Tracts.  One run creates ONE export for one year, one season, and
 * one spatial batch. Start with YEAR=2022, SEASON='jja', and CHUNK=12.
 *
 * Output: per-tract mean, median, population standard deviation, and valid
 * pixel count for both indices. The seasonal raster is the median of all
 * cloud-masked Landsat observations. DJF deliberately means Jan-Feb-Dec of
 * the same calendar year, matching the existing FUA tables.
 */

var TRACT_ASSET = 'projects/your-ee-project/assets/GECC/censusTracts2020US';
var DRIVE_FOLDER = 'GECC_CensusTract_NDVI_EVI';
var YEAR = 2022;              // Completed seasons: 1985-2025.
var SEASON = 'jja';           // 'jja' or 'djf' (Jan-Feb-Dec in YEAR).
var CHUNK = 12;               // 12 = California; see REGION_STATEFPS.
var SCALE_M = 30;
var TILE_SCALE = 8;
var MAX_PIXELS_PER_REGION = 2e10;
var SHOW_MAP = false;

// Spatially coherent chunks, used to bound the Landsat scene search and keep
// individual exports manageable. These are the production tract groupings.
var REGION_STATEFPS = {
  0: ['09', '23', '25', '33', '44', '50'],
  1: ['34', '36'],
  2: ['10', '11', '24', '42'],
  3: ['17', '18', '39'],
  4: ['26', '55'],
  5: ['19', '20', '27', '29', '31', '38', '46'],
  6: ['12', '13'],
  7: ['37', '45', '51', '54'],
  8: ['01', '21', '28', '47'],
  9: ['48'],
  10: ['05', '22', '40'],
  11: ['04', '08', '16', '30', '32', '35', '49', '56'],
  12: ['06'],
  13: ['02', '15', '41', '53']
};
var REGION_LABELS = {
  0: 'new_england', 1: 'ny_nj', 2: 'mid_atlantic', 3: 'il_in_oh',
  4: 'mi_wi', 5: 'west_north_central', 6: 'fl_ga',
  7: 'carolinas_va_wv', 8: 'east_south_central', 9: 'texas',
  10: 'ar_la_ok', 11: 'mountain', 12: 'california', 13: 'ak_hi_or_wa'
};

function prepL457(image) {
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermal = image.select('ST_B6').multiply(0.00341802).add(149.0);
  var scaled = optical.addBands(thermal).select(
      ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6'],
      ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']);
  var qa = image.select('QA_PIXEL');
  var mask1 = qa.bitwiseAnd(63).eq(0);
  var mask2 = qa.rightShift(8).bitwiseAnd(63).eq(21);
  var mask3 = image.select('QA_RADSAT').eq(0);
  var mask4 = optical.reduce(ee.Reducer.min()).gt(0);
  var mask5 = optical.reduce(ee.Reducer.max()).lt(1);
  var mask6 = image.select('SR_ATMOS_OPACITY').unmask(-1).lt(300);
  return image.select().addBands(scaled).updateMask(
      mask1.and(mask2).and(mask3).and(mask4).and(mask5).and(mask6));
}

function prepL89(image) {
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermal = image.select('ST_B10').multiply(0.00341802).add(149.0);
  var scaled = optical.addBands(thermal).select(
      ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'],
      ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']);
  var qa = image.select('QA_PIXEL');
  var mask1 = qa.bitwiseAnd(63).eq(0);
  var mask2 = qa.rightShift(8).bitwiseAnd(255).eq(85);
  var mask3 = image.select('QA_RADSAT').eq(0);
  var mask4 = optical.reduce(ee.Reducer.min()).gt(0);
  var mask5 = optical.reduce(ee.Reducer.max()).lt(1);
  var mask6 = image.select('SR_QA_AEROSOL').rightShift(6).neq(3);
  return image.select().addBands(scaled).updateMask(
      mask1.and(mask2).and(mask3).and(mask4).and(mask5).and(mask6));
}

function landsatCollection(region) {
  var common = ee.Filter.lt('WRS_ROW', 122);
  var l4 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
      .filter(common).filterBounds(region).map(prepL457);
  var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
      .filter(common).filterBounds(region).map(prepL457);
  var l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
      .filterDate('1984-01-01', '2017-01-01')
      .filter(common).filterBounds(region).map(prepL457);
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterDate('2013-05-01', '2099-01-01')
      .filter(ee.Filter.neq('NADIR_OFFNADIR', 'OFFNADIR'))
      .filter(common).filterBounds(region).map(prepL89);
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
      .filter(ee.Filter.neq('NADIR_OFFNADIR', 'OFFNADIR'))
      .filter(common).filterBounds(region).map(prepL89);
  return l4.merge(l5).merge(l7).merge(l8).merge(l9);
}

function addIndices(image) {
  var ndvi = image.expression(
      'clamp((nir - red) / (nir + red), -1, 1)', {
        nir: image.select('nir'), red: image.select('red')
      }).rename('NDVI');
  var evi = image.expression(
      'clamp(2.5 * ((nir - red) / clamp(nir + 6 * red - 7.5 * blue + 1, 0, 10)), -1, 1)', {
        nir: image.select('nir'), red: image.select('red'), blue: image.select('blue')
      }).rename('EVI');
  return ndvi.addBands(evi).copyProperties(image, ['system:time_start']);
}

function seasonalCollection(sr, year, season) {
  if (season === 'jja') {
    return sr.filterDate(year + '-06-01', year + '-09-01');
  }
  if (season === 'djf') {
    return sr.filterDate(year + '-01-01', year + '-03-01')
        .merge(sr.filterDate(year + '-12-01', (year + 1) + '-01-01'));
  }
  throw new Error("SEASON must be 'jja' or 'djf'.");
}

function statsReducer() {
  return ee.Reducer.mean()
      .combine(ee.Reducer.median(), '', true)
      .combine(ee.Reducer.stdDev(), '', true)
      .combine(ee.Reducer.count(), '', true);
}

if (!REGION_STATEFPS.hasOwnProperty(CHUNK)) {
  throw new Error('CHUNK must be an integer from 0 through 13.');
}
// Use an explicit state-list filter because several batches contain states.
var chunkTracts = ee.FeatureCollection(TRACT_ASSET)
    .filter(ee.Filter.inList('StFIPS', REGION_STATEFPS[CHUNK]));
var region = chunkTracts.geometry(100).bounds(100);
var sr = landsatCollection(region);
var seasonalSr = seasonalCollection(sr, YEAR, SEASON);
var sceneCount = seasonalSr.size();
var empty = ee.Image.constant([0, 0]).rename(['NDVI', 'EVI']).updateMask(0);
var composite = ee.Image(ee.Algorithms.If(
    sceneCount.gt(0), seasonalSr.map(addIndices).median(), empty));

var rawStats = composite.reduceRegions({
  collection: chunkTracts,
  reducer: statsReducer(),
  scale: SCALE_M,
  tileScale: TILE_SCALE,
  maxPixelsPerRegion: MAX_PIXELS_PER_REGION
});

var output = rawStats.map(function(feature) {
  return ee.Feature(null, {
    GEOID: feature.get('GEOID'),
    StFIPS: feature.get('StFIPS'),
    year: YEAR,
    season: SEASON === 'jja' ? 'JJA' : 'DJF_same_year',
    batch: CHUNK,
    batch_label: REGION_LABELS[CHUNK],
    regional_scene_count: sceneCount,
    NDVI_mean: feature.get('NDVI_mean'),
    NDVI_median: feature.get('NDVI_median'),
    NDVI_stdDev: feature.get('NDVI_stdDev'),
    NDVI_count: feature.get('NDVI_count'),
    EVI_mean: feature.get('EVI_mean'),
    EVI_median: feature.get('EVI_median'),
    EVI_stdDev: feature.get('EVI_stdDev'),
    EVI_count: feature.get('EVI_count')
  });
});

var seasonLabel = SEASON === 'jja' ? 'JJA' : 'DJFsameYear';
var description = 'CensusTract_NDVI_EVI_' + seasonLabel + '_' + YEAR +
    '_b' + ('00' + CHUNK).slice(-3);
var selectors = [
  'GEOID', 'StFIPS', 'year', 'season', 'batch', 'batch_label',
  'regional_scene_count', 'NDVI_mean', 'NDVI_median', 'NDVI_stdDev',
  'NDVI_count', 'EVI_mean', 'EVI_median', 'EVI_stdDev', 'EVI_count'
];

print('Tract count', chunkTracts.size());
print('Landsat scenes before seasonal date filter', sr.size());
print('Seasonal Landsat scenes', sceneCount);
print('Output preview', output.limit(5));

Export.table.toDrive({
  collection: output,
  description: description,
  folder: DRIVE_FOLDER,
  fileNamePrefix: description,
  fileFormat: 'CSV',
  selectors: selectors
});

if (SHOW_MAP) {
  Map.centerObject(chunkTracts, 6);
  Map.addLayer(composite.select('NDVI'), {min: -0.2, max: 0.8,
    palette: ['brown', 'yellow', 'green']}, 'Seasonal NDVI');
}
