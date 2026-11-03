/**
 * Fully public, paste-and-run GEE example: GLC-FCS30D class areas for
 * 25 Census Tracts in Alameda County, California, for 2021.
 *
 * Uses only public Earth Engine collections.  It is a computationally small
 * analogue of GLC_RemainingTract_c6efe137ca_v1_2021_b005of010, not an attempt
 * to reproduce that private historical tract batch.
 */

var YEAR = 2021;
var STATEFP = '06';       // California
var COUNTYFP = '001';     // Alameda County
var TRACT_LIMIT = 25;     // Raise gradually for a scaling experiment.
var SCALE_M = 30;
var DRIVE_FOLDER = 'GEE_public_GLC_example';
var DESCRIPTION = 'GLC_public_tract_example_2021_CA001_n25';

var GLC_CLASSES = [
  0, 10, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92,
  120, 121, 122, 130, 140, 150, 152, 153, 181, 182, 183, 184,
  185, 186, 187, 190, 200, 201, 202, 210, 220, 250
];

// Public TIGER 2020 tract boundaries and public GLC-FCS30D imagery.
var tracts = ee.FeatureCollection('TIGER/2020/TRACT')
    .filter(ee.Filter.eq('STATEFP', STATEFP))
    .filter(ee.Filter.eq('COUNTYFP', COUNTYFP))
    .sort('GEOID')
    .limit(TRACT_LIMIT);

// GLC-FCS30D annual b1 = 2000; therefore 2021 = b22.
var glcBand = 'b' + (YEAR - 1999);
var glc = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual')
    .select(glcBand)
    .mosaic()
    .rename('class_code');

var areaAndClass = ee.Image.pixelArea().rename('area_m2').addBands(glc);
var grouped = areaAndClass.reduceRegions({
  collection: tracts,
  reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class_code'}),
  scale: SCALE_M,
  crs: glc.projection(),
  tileScale: 4,
  maxPixelsPerRegion: 2e10
});

// Convert GEE's grouped output into normal CSV columns: area_class_<code>.
var output = grouped.map(function(feature) {
  var groups = ee.List(ee.Algorithms.If(feature.get('groups'),
      feature.get('groups'), []));
  var keys = groups.map(function(item) {
    return ee.Number(ee.Dictionary(item).get('class_code')).format('%.0f');
  });
  var values = groups.map(function(item) {
    return ee.Dictionary(item).get('sum');
  });
  var byClass = ee.Dictionary.fromLists(keys, values);
  var row = {
    GEOID: feature.get('GEOID'),
    STATEFP: feature.get('STATEFP'),
    COUNTYFP: feature.get('COUNTYFP'),
    year: YEAR,
    source_area_m2: feature.geometry().area(1),
    total_area_m2: ee.Number(values.iterate(function(value, total) {
      return ee.Number(total).add(value);
    }, 0))
  };
  GLC_CLASSES.forEach(function(code) {
    row['area_class_' + code] = ee.Number(byClass.get(String(code), 0));
  });
  return ee.Feature(null, row);
});

var selectors = [
  'GEOID', 'STATEFP', 'COUNTYFP', 'year', 'source_area_m2', 'total_area_m2'
];
GLC_CLASSES.forEach(function(code) { selectors.push('area_class_' + code); });

print('Selected public tract count', tracts.size());
print('Output preview', output.limit(5));
Map.centerObject(tracts, 10);
Map.addLayer(tracts, {color: 'red'}, '25 public Census Tracts');

Export.table.toDrive({
  collection: output,
  description: DESCRIPTION,
  folder: DRIVE_FOLDER,
  fileNamePrefix: DESCRIPTION,
  fileFormat: 'CSV',
  selectors: selectors
});
