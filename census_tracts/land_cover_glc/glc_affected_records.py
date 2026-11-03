"""List the tracts and overlaps affected by the GLC-FCS30D annual-tile error (W120N35, W120N40).

Selection rule: a tract is affected when its representative interior point (shapely
representative_point, EPSG:4326) lies in 120-115° W and 30-40° N, the extent of the two
annual tiles. Overlaps are affected when their parent tract is affected and they appear
in vegetation_ua_pieces. Reproduces the 6,424 tracts and 6,380 overlaps of the
2026-11-01 audit exactly.
"""
from pathlib import Path
import csv
import geopandas as gpd
import split_gridmet_prism as split

HERE = Path(__file__).resolve().parent
OUT = HERE / 'glc_affected/GLC_ANNUAL_TILE_AFFECTED.csv'
FIELDS = ['table', 'GEOID', 'piece_id', 'GEOID20', 'glc_tile']

def tile_of(lon, lat):
    if -120 <= lon < -115 and 30 <= lat < 35: return 'W120N35'
    if -120 <= lon < -115 and 35 <= lat < 40: return 'W120N40'
    return None

def build():
    tracts = gpd.read_file(split.source_path('geography/census_tracts_2020.geojson'))
    points = tracts.geometry.representative_point()
    tiles = {g: tile_of(pt.x, pt.y) for g, pt in zip(tracts.GEOID, points)}
    affected = {g: t for g, t in tiles.items() if t}
    with split.source_path('metrics/vegetation_ua_pieces.csv').open(newline='') as f:
        pieces = [r for r in csv.DictReader(f) if r['GEOID'] in affected]
    rows = [{'table': 'land_cover_glc', 'GEOID': g, 'piece_id': '', 'GEOID20': '', 'glc_tile': t} for g, t in sorted(affected.items())]
    rows += [{'table': 'vegetation_ua_pieces', 'GEOID': r['GEOID'], 'piece_id': r['piece_id'], 'GEOID20': r['GEOID20'], 'glc_tile': affected[r['GEOID']]}
             for r in sorted(pieces, key=lambda r: int(r['piece_id']))]
    assert len(affected) == 6424 and sum(g[:2] == '06' for g in affected) == 5715 and sum(g[:2] == '32' for g in affected) == 709
    assert len(pieces) == 6380 and len({r['GEOID20'] for r in pieces}) == 89
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return OUT

DICTIONARY_ROWS = [
    ('table', 'text', 'categorical', 'Affected table: land_cover_glc (one row per tract) or vegetation_ua_pieces (one row per overlap). Filter on this field before joining.'),
    ('GEOID', 'text', 'identifier', 'Eleven-character tract identifier. Keep leading zeros.'),
    ('piece_id', 'integer', 'identifier', 'Tract-Urban Area overlap identifier. Empty for land_cover_glc rows.'),
    ('GEOID20', 'text', 'identifier', 'Five-character 2020 Census Urban Area identifier. Empty for land_cover_glc rows.'),
    ('glc_tile', 'text', 'categorical', 'GLC-FCS30D annual tile containing the representative interior point of the tract: W120N35 (30-35° N) or W120N40 (35-40° N), both 120-115° W.'),
]

def dictionary_rows(template_fields):
    out = []
    for field, serialized, logical, definition in DICTIONARY_ROWS:
        row = dict.fromkeys(template_fields, '')
        row.update({'table': 'glc_annual_tile_affected', 'field': field, 'family': field, 'source_product': 'glc_annual_tile_affected',
                    'source_field': field, 'source_type': 'string' if serialized == 'text' else 'int64', 'units': 'text ID' if logical == 'identifier' and serialized == 'text' else ('integer ID' if logical == 'identifier' else 'category'),
                    'labels': '{}', 'definition': definition, 'null_rule': 'Empty where not applicable to the row\'s table.' if field in ('piece_id', 'GEOID20') else 'Never empty.',
                    'methods': 'glc_annual_tile_affected', 'role': 'quality_list',
                    'statistic_support': 'Records affected by the GLC-FCS30D annual-tile classification error; see the GLC limitations in the codebook.',
                    'serialized_type': serialized, 'logical_type': logical})
        out.append(row)
    return out

if __name__ == '__main__':
    print(build())
