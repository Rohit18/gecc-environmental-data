/**
 * Exact GEE/JavaScript replacement for:
 * GLC_RemainingTract_c6efe137ca_v1_2021_b005of010
 *
 * Before running, upload
 * outputs/remaining_tract_glc/remaining_tract_batch_lookup.csv as an Earth
 * Engine table asset and set BATCH_LOOKUP_ASSET below.  That file is required:
 * it pins the 3,731 GEOIDs in this historical "remaining tracts" batch.
 *
 * Output: one Drive CSV with one 2021 row per tract and the same columns and
 * definitions used by the original submitted task. Areas are square metres.
 */

// Configuration.
var TRACT_ASSET = 'projects/your-ee-project/assets/GECC/censusTracts2020US';
var BATCH_LOOKUP_ASSET = 'projects/your-project/assets/remaining_tract_batch_lookup';
var OUTPUT_FOLDER = 'GECC_RemainingTracts_GLC_FCS30D';
var YEAR = 2021;
var BATCH = 5;
var N_BATCHES = 10;
var DESCRIPTION = 'GLC_RemainingTract_c6efe137ca_v1_2021_b005of010';
var SCALE_M = 30;
var TILE_SCALE = 8;
var MAX_PIXELS_PER_REGION = 2e10;
var TRACT_BOUNDARY_YEAR = 2020;

var ANNUAL_COLLECTION = 'projects/sat-io/open-datasets/GLC-FCS30D/annual';
var GLC_CLASSES = [
  0, 10, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92,
  120, 121, 122, 130, 140, 150, 152, 153, 181, 182, 183, 184,
  185, 186, 187, 190, 200, 201, 202, 210, 220, 250
];
var TREE_CLASSES = [51, 52, 61, 62, 71, 72, 81, 82, 91, 92];
var SHRUB_CLASSES = [120, 121, 122];
var GRASS_CLASSES = [130];
var CROP_CLASSES = [10, 11, 20];
var VEGETATION_CLASSES = [
  10, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92,
  120, 121, 122, 130, 140, 150, 152, 153, 181, 182, 185, 186
];

function sumClasses(areas, codes) {
  var result = ee.Number(0);
  codes.forEach(function(code) { result = result.add(areas[code]); });
  return result;
}

function summarize(feature) {
  var groups = ee.List(ee.Algorithms.If(feature.get('groups'),
      feature.get('groups'), []));
  var keys = groups.map(function(item) {
    return ee.Number(ee.Dictionary(item).get('class_code')).format('%.0f');
  });
  var values = groups.map(function(item) {
    return ee.Dictionary(item).get('sum');
  });
  var byClass = ee.Dictionary.fromLists(keys, values);
  var areas = {};
  var output = {
    GEOID: feature.get('GEOID'),
    year: YEAR,
    batch: BATCH,
    source_area_m2: feature.geometry().area(1)
  };

  GLC_CLASSES.forEach(function(code) {
    areas[code] = ee.Number(byClass.get(String(code), 0));
    output['area_class_' + code] = areas[code];
  });

  var totalArea = ee.Number(values.iterate(function(value, total) {
    return ee.Number(total).add(value);
  }, 0));
  output.total_area_m2 = totalArea;
  output.glc_coverage_fraction = ee.Algorithms.If(
      ee.Number(output.source_area_m2).gt(0),
      totalArea.divide(output.source_area_m2), 0);

  var groupsToExport = {
    vegetation: VEGETATION_CLASSES,
    tree: TREE_CLASSES,
    shrub: SHRUB_CLASSES,
    grass: GRASS_CLASSES,
    crop: CROP_CLASSES
  };
  Object.keys(groupsToExport).forEach(function(name) {
    var area = sumClasses(areas, groupsToExport[name]);
    output[name + '_area_m2'] = area;
    // The original remaining-tract run used null rather than zero when its
    // GLC raster had no sampled pixels in the tract.
    output[name + '_fraction'] = ee.Algorithms.If(
        totalArea.gt(0), area.divide(totalArea), null);
  });
  output.has_glc_sampled_area = totalArea.gt(0);
  output.selection = 'remaining_2020_tracts_outside_CityHealth_selection';
  output.tract_boundary_year = TRACT_BOUNDARY_YEAR;
  return ee.Feature(null, output);
}

function selectors() {
  var fields = [
    'GEOID', 'year', 'batch', 'source_area_m2', 'total_area_m2',
    'glc_coverage_fraction', 'vegetation_area_m2', 'vegetation_fraction',
    'tree_area_m2', 'tree_fraction', 'shrub_area_m2', 'shrub_fraction',
    'grass_area_m2', 'grass_fraction', 'crop_area_m2', 'crop_fraction'
  ];
  GLC_CLASSES.forEach(function(code) { fields.push('area_class_' + code); });
  return fields.concat([
    'has_glc_sampled_area', 'selection', 'tract_boundary_year'
  ]);
}

// The lookup asset stores the exact prior batch assignment: GEOID,batch.
var lookup = ee.FeatureCollection(BATCH_LOOKUP_ASSET)
    .filter(ee.Filter.eq('batch', BATCH));
var batchGeoids = lookup.aggregate_array('GEOID');
var tracts = ee.FeatureCollection(TRACT_ASSET)
    .filter(ee.Filter.inList('GEOID', batchGeoids))
    .map(function(feature) {
      return ee.Feature(feature.geometry(), {GEOID: feature.get('GEOID')});
    });

// 2021 is annual band b22 because b1 is 2000.
var landCover = ee.ImageCollection(ANNUAL_COLLECTION).select('b22').mosaic()
    .rename('class_code');
var areaAndClass = ee.Image.pixelArea().rename('area_m2').addBands(landCover);
var grouped = areaAndClass.reduceRegions({
  collection: tracts,
  reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class_code'}),
  scale: SCALE_M,
  crs: landCover.projection(),
  tileScale: TILE_SCALE,
  maxPixelsPerRegion: MAX_PIXELS_PER_REGION
});
var output = grouped.map(summarize);

print('Expected exact batch size', 3731);
print('Lookup batch count', lookup.size());
print('Matched source-tract count', tracts.size());
print('Output preview', output.limit(5));

Export.table.toDrive({
  collection: output,
  description: DESCRIPTION,
  folder: OUTPUT_FOLDER,
  fileNamePrefix: DESCRIPTION,
  fileFormat: 'CSV',
  selectors: selectors()
});
