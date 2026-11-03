"""Inventory existing WC2020 caches against frozen tract geometries; prepare jobs."""
import json
import os
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path('/path/to/scratch/gecc_tract_watermask_20261013')
AUDIT = Path('/path/to/scratch/gecc_lai_reproducibility_audit_20261012/rerun_work_20261012')
CACHES = [
    '/path/to/project/gecc/census_tract_uhi/worldcover_2020/tiles',
    '/path/to/project/gecc/fua_uhi/worldcover_2020/tiles',
    '/path/to/scratch/urbveg/data/raw/worldcover_2020/tiles',
    '/path/to/scratch/gecc_tract_addon/uhi/regional/worldcover_tiles',
    str(ROOT / 'inputs/worldcover_missing'),
]

def main():
    geoms = sorted((AUDIT / 'geoms').glob('*.parquet'))
    lai = gpd.GeoDataFrame(pd.concat([gpd.read_parquet(p) for p in geoms], ignore_index=True),
                           geometry='geometry', crs='EPSG:4326')
    assert len(lai) == 85528 and lai.GEOID.is_unique
    lai.to_parquet(ROOT / 'inputs/lai_tracts.parquet')
    # Keep the canopy release's own geometry; do not replace it with LAI geometry.
    chm = gpd.read_file('/path/to/scratch/gecc_tree_height_25m/inputs/Census_2020_US_Tract_Polygons.shp')
    addon = gpd.read_file('/path/to/scratch/gecc_tract_addon/tree_height/tracts_2020.gpkg')
    chm = gpd.GeoDataFrame(pd.concat([chm.to_crs(4326), addon.to_crs(4326)], ignore_index=True),
                           geometry='geometry', crs='EPSG:4326')
    chm.GEOID = chm.GEOID.astype(str)
    assert len(chm) == 85528 and chm.GEOID.is_unique
    chm.to_parquet(ROOT / 'inputs/chm_tracts.parquet')
    footprint = []
    for g in (lai, chm):
        for geom in g.geometry:
            if geom.bounds[2] - geom.bounds[0] > 180:
                footprint.extend([geom.intersection(box(-180,-90,0,90)),
                                  geom.intersection(box(0,-90,180,90))])
            else:
                footprint.append(geom)
    grid = gpd.read_file('/path/to/scratch/urbveg/data/raw/worldcover_2020/aux/esa_worldcover_2020_grid.geojson')
    pairs = grid.sindex.query(gpd.GeoSeries(footprint, crs=4326), predicate='intersects')
    needed = grid.iloc[np.unique(pairs[1])].sort_values('ll_tile')
    files = {}
    cache_counts = {}
    for cache in CACHES:
        paths = sorted(Path(cache).glob('ESA_WorldCover_10m_2020_v100_*_Map.tif'))
        cache_counts[cache] = len(paths)
        for p in paths:
            tile = p.name.split('_')[-2]
            files.setdefault(tile, p.resolve())
    rows = []
    for r in needed.itertuples():
        path = files.get(r.ll_tile)
        rec = {'tile':r.ll_tile, 'path':str(path) if path else '',
               'bytes':path.stat().st_size if path else 0,
               'bounds':json.dumps(list(r.geometry.bounds)),
               'url':f'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{r.ll_tile}_Map.tif'}
        side = Path(str(path) + '.verified.json') if path else None
        if side and side.exists():
            info = json.loads(side.read_text())
            assert info['bytes'] == rec['bytes'], str(path)
            rec['recorded_sha256'] = info.get('sha256','')
        rows.append(rec)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(ROOT / 'inputs/worldcover_inventory.csv', index=False)
    inventory[inventory.path == ''].to_csv(ROOT / 'inputs/worldcover_missing.csv', index=False)
    grid.to_file(ROOT / 'inputs/worldcover_grid.geojson', driver='GeoJSON')
    lai_jobs = []
    for i,p in enumerate(geoms):
        g = gpd.read_parquet(p)
        component = 'conus' if not g.GEOID.str[:2].isin(['02','15','60','66','69','72','78']).any() else 'addon'
        lai_jobs.append({'job':i,'geoms':str(p),'source':str(AUDIT / f'{component}_rerun.vrt')})
    pd.DataFrame(lai_jobs).to_csv(ROOT / 'inputs/lai_jobs.csv',index=False)
    idx = gpd.read_file('/path/to/project/gecc/chm_1m/manifests/tiles.geojson')
    if idx.crs != chm.crs:
        idx = idx.to_crs(chm.crs)
    joined = gpd.sjoin(idx, chm[['GEOID','geometry']], how='inner', predicate='intersects')
    joined['tile'] = idx.loc[joined.index,'tile'].values
    tile_rows=[]
    for tile, group in joined.groupby('tile'):
        paths=[Path('/path/to/project/gecc/chm_1m/chm')/f'{tile}.tif',
               Path('/path/to/scratch/gecc_tract_addon/tree_height/chm1m_source/chm')/f'{tile}.tif']
        source=next((p for p in paths if p.exists()),None)
        if source is None: raise FileNotFoundError(tile)
        tile_rows.append({'tile':tile,'source':str(source),'geoids':json.dumps(sorted(group.GEOID.unique())),
                          'bytes':source.stat().st_size})
    pd.DataFrame(tile_rows).to_parquet(ROOT / 'inputs/chm_jobs.parquet',index=False)
    summary={'cache_counts':cache_counts,'required_worldcover_tiles':len(inventory),
             'existing_required_tiles':int((inventory.path!='').sum()),
             'missing_tiles':inventory.loc[inventory.path=='','tile'].tolist(),
             'lai_tracts':len(lai),'chm_tracts':len(chm),'lai_jobs':len(lai_jobs),'chm_tiles':len(tile_rows)}
    (ROOT / 'inputs/inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
