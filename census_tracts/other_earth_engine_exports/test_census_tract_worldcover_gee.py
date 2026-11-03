"""Live Earth Engine checks for association and missing/zero-zone behavior."""
import ee
import submit_census_tract_worldcover_exports as workflow


def run():
    ee.Initialize(project='your-ee-project')
    projection = {'crs': 'EPSG:3857', 'transform': [10, 0, 0, 0, -10, 100]}
    def rect(x0, x1):
        return ee.Geometry.Rectangle([x0, 0, x1, 100], 'EPSG:3857', False)
    tract = ee.Feature(rect(0, 100), {'GEOID': '06001000001', 'StFIPS': '06',
                                    'Area_Land': 10000, 'Area_Water': 0})
    ua1 = ee.Feature(rect(0, 60), {'UACE20': '00001', 'NAME20': 'First'})
    ua2 = ee.Feature(rect(40, 100), {'UACE20': '00002', 'NAME20': 'Second'})
    touch = ee.Feature(rect(100, 120), {'UACE20': '00003', 'NAME20': 'Touch only'})
    wc = ee.Image.constant(10).rename('Map')
    cases = [
        ('no_ua', [], wc, 0),
        ('partial_urban', [ua1], wc, 1),
        ('overlapping_uas', [ua1, ua2, touch], wc, 2),
        ('no_coverage', [ua1], wc.updateMask(ee.Image.constant(0)), 1),
    ]
    for name, features, landcover, expected in cases:
        uas = ee.FeatureCollection(features)
        row = workflow.tract_row(tract, uas, workflow.area_image(landcover, uas), projection).toDictionary().getInfo()
        pairs = workflow.pair_rows(tract, uas).getInfo()['features']
        workflow.check_row(row, pairs)
        assert row['ua_count'] == expected, name
        if name == 'no_ua':
            assert row['primary_ua_id'] == ''
            assert row['urban_zone_area_m2'] == row['urban_tree_area_m2'] == 0
            assert row.get('urban_vegetation_fraction') is None
        if name == 'overlapping_uas':
            assert row['ua_ids'] == '00001|00002'
            assert abs(row['urban_zone_area_m2'] - row['tract_geometry_area_m2']) < 1
            assert row['rural_tree_area_m2'] == 0
        if name == 'no_coverage':
            assert row.get('urban_tree_area_m2') is None
            assert row.get('rural_tree_area_m2') is None
        print(f'PASS {name}', flush=True)
    # Exercise nested map/flatten used by the production crosswalk export.
    pairs = ee.FeatureCollection(ee.List([tract, tract.set('GEOID', '06001000002')]).map(
        lambda f: workflow.pair_rows(f, ee.FeatureCollection([ua1])))).flatten().getInfo()['features']
    assert len(pairs) == 2
    assert {p['properties']['GEOID'] for p in pairs} == {'06001000001', '06001000002'}
    assert {p['properties']['UACE20'] for p in pairs} == {'00001'}
    print('PASS one UA linked to multiple tracts', flush=True)


if __name__ == '__main__':
    run()
