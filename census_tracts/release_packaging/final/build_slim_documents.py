"""Rebuild the codebook and collection guide for the 2026-11-02 copy.

Uses the release builder with the same wording hooks as the original folder, then points it at the
new folder and collapses the variable families that slim_data.py merged or dropped.
Run with PYTHONPATH=.:../census_tracts_release_corrections_20261102:../upload_preparation_2026-10-30
"""
from collections import defaultdict
from pathlib import Path
import copy
import csv
import json
import re

import build_selected_documents as builder
import pi_wording

PACKAGE = Path('/path/to/GECC/uploadGECC/census_tracts_12_metrics_20261102')
HERE = Path(__file__).resolve().parent
changes = json.loads((HERE / 'column_changes.json').read_text())
BUILDER_NAME = {'census_tracts_2020': 'tract_geography', 'tract_urban_area_pieces_2020': 'piece_registry',
                'urban_areas_2020': 'ua_registry'}
TITLES = {'gridmet_cell_count': 'GRIDMET cell count', 'prism_cell_count': 'PRISM cell count',
          'daymet_cell_count': 'Daymet cell count'}

builder.researcher.configure(builder)
pi_wording.configure(builder)
pi_wording.configure_status(builder)

# Point the builder at the new folder and its dictionary.
builder.PACKAGE, builder.OUT = PACKAGE, PACKAGE / 'documentation'
dictionary = list(csv.DictReader((PACKAGE / 'metadata/COLUMN_DICTIONARY.csv').open(encoding='utf-8')))
for r in dictionary:
    r['table'] = BUILDER_NAME.get(r['table'], r['table'])
builder.dictionary[:] = dictionary
builder.mapping = {(r['table'], r['field']): r for r in dictionary}
definitions = {(r['table'], r['family']): r['definition'] for r in dictionary}

# Collapse variable families to match the slimmed tables.
families = builder.families
new_families = {}
for change in changes:
    table = Path(change['member']).stem
    drop = set(change['dropped'])
    sources = {c: new for new, cols in change['collapsed'].items() for c in cols}
    grouped = defaultdict(list)
    for new in change['collapsed']:
        grouped[re.sub(r'_(JJA|DJF)$', '', new)].append(new)
    out, placed = [], set()
    for f in families:
        if f['table'] != table:
            out.append(f); continue
        hit = [c for c in f['columns'] if c in sources]
        for fam in dict.fromkeys(re.sub(r'_(JJA|DJF)$', '', sources[c]) for c in hit):
            if fam in placed:
                continue
            g = copy.deepcopy(f)
            g.update(family=fam, columns=grouped[fam], column_count=len(grouped[fam]), available_years=[], observed_years=[])
            out.append(g); placed.add(fam)
            old = builder.by_family[(table, f['family'])]
            pick = [r for r in old if r['year'] == '2020' and r['period'] in ('JJA', '')] or [r for r in old if r['year'] == '2020'] or old
            seasons = sorted({c.rsplit('_', 1)[1] for c in grouped[fam]} & {'JJA', 'DJF'}, reverse=True) or ['']
            recs = []
            for s in seasons:
                base = next((r for r in pick if r['period'] == s), pick[0] if pick else None)
                if base is None:
                    continue
                r = dict(base, family=fam, field=grouped[fam][0], year='', period=s, measure='')
                recs.append(r)
                if (table, base['field']) in builder.by_category:
                    builder.by_category[(table, r['field'])] = [dict(x, field=r['field'], year='', period=s) for x in builder.by_category[(table, base['field'])]]
            builder.by_family[(table, fam)] = recs
            new_families[(table, fam)] = definitions[(table, fam)]
        rest = [c for c in f['columns'] if c not in sources and c not in drop]
        if rest:
            f['columns'] = rest
            out.append(f)
    families[:] = out
covered_columns = sum(len(f['columns']) for f in families)

# Text: the type paragraph, column counts and the new family definitions.
builder.researcher.TYPE_GUIDANCE = ('For importing the CSV files, use serialized_type, logical_type and null_rule in '
                                    'COLUMN_DICTIONARY.csv. Keep geographic codes as text.')
builder.researcher.TYPE_GUIDANCE_SHORT = builder.researcher.TYPE_GUIDANCE
n_data = len([r for r in dictionary if r['table'] != 'glc_annual_tile_affected'])
previous_clean = builder.clean
builder.clean = lambda s: (previous_clean(s).replace('all 6,426 data-table CSV columns', f'all {n_data:,} data-table CSV columns')
                           .replace('FILE_GUIDE.md', 'FILE_GUIDE.txt').replace('README.md', 'README.txt')
                           .replace('RELEASE_NOTES.md', 'RELEASE_NOTES.txt'))
MISSING = {
    'vegetation_ua_pieces': ('. For the 496 overlaps in Hawaii, Guam, the Northern Mariana Islands and American Samoa, '
                             'GLC-FCS30D has no sampled source area, so their values would be empty. The 31 U.S. Virgin '
                             'Islands overlaps were not processed for this table.'),
    'uhi_ua_pieces': ('. For the 496 overlaps in Hawaii, Guam, the Northern Mariana Islands and American Samoa, SVCM-SP has '
                      'no source data. The 31 U.S. Virgin Islands overlaps were not processed; whole-tract SVCM-SP '
                      'temperature for the Virgin Islands comes from the Latin America archive, which this table does not use.'),
}
OLD_MISSING = (': this table was produced for 2,622 Urban Areas, before the 22 Urban Areas in Hawaii and the Island Areas '
               'were added for NDVI and EVI.')
for tid, text in MISSING.items():
    assert builder.PRODUCTS[tid]['limit'].count(OLD_MISSING) == 1, tid
    builder.PRODUCTS[tid]['limit'] = builder.PRODUCTS[tid]['limit'].replace(OLD_MISSING, text)
OVERVIEW = 'they omit the 527 overlaps in Hawaii and the Island Areas, which are absent from those tables rather than flagged.'
OVERVIEW_NEW = OVERVIEW + (' For the 496 in Hawaii, Guam, the Northern Mariana Islands and American Samoa, the GLC-FCS30D and '
                           'SVCM-SP sources have no data; the 31 U.S. Virgin Islands overlaps were not processed.')
# Reference fixes checked against Crossref and DataCite on 2026-11-03.
REFERENCES = [
    ('A global 1 km resolution daily near-surface air temperature dataset (2003-present), produced with a spatially '
     'varying coefficient model with sign preservation (SVCM-SP). HKU DataHub. DOI 10.25442/hku.c.8414754.',
     'A global 1 km resolution daily near-surface air temperature dataset (2003-present). HKU DataHub. DOI '
     '10.25442/hku.c.8414754. Produced with a spatially varying coefficient model with sign preservation (SVCM-SP).'),
    ('for ecological applications and modeling.', 'for ecological applications and modelling.'),
    ('fine classification system for the period of 1985-2022.',
     'fine classification system for the period from 1985 to 2022 generated using dense-time-series Landsat imagery '
     'and the continuous change-detection method.'),
    ('Daymet Version 4 R1, dataset version 4.1. ORNL DAAC. DOI 10.3334/ORNLDAAC/2129. Source paper DOI '
     '10.1038/s41597-021-00973-0.',
     'Thornton, M.M. et al. (2022). Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 '
     'R1. ORNL DAAC. DOI 10.3334/ORNLDAAC/2129. Source paper: Thornton, P.E. et al. (2021). Gridded daily weather '
     'data for North America with comprehensive uncertainty quantification. Scientific Data 8, 190. DOI '
     '10.1038/s41597-021-00973-0.'),
    ('Wan et al. (2024).', 'Wan, L. et al. (2024).'),
    ('Latin America archive', 'South America archive'),
    ('U.S. Census Bureau. 2020 Census tract and 2020 Census Urban Area boundaries.',
     'U.S. Census Bureau (2020). TIGER/Line Shapefiles, 2020 census tracts, '
     'https://www2.census.gov/geo/tiger/TIGER2020/TRACT/ (states and DC accessed as Earth Engine TIGER/2020/TRACT). '
     'U.S. Census Bureau (2025). TIGER/Line Shapefiles, 2020 Census Urban Areas (tl_2025_us_uac20), '
     'https://www2.census.gov/geo/tiger/TIGER2025/UAC20/.'),
    ('Landsat 8-9 OLI DOI', 'Landsat 8-9 OLI/TIRS DOI'),
]


ACK_ADD = open(HERE / 'ack_addition.txt').read()


def fix(s):
    s = s.replace(OVERVIEW, OVERVIEW_NEW)
    if s.endswith('under award U24AG088894.'):
        s += ACK_ADD
    if s.endswith(('Also cite the source products in section 10.', 'Also cite the source products in codebook section 10.')):
        s += ' Processing code is available at https://github.com/Rohit18/gecc-environmental-data (census_tracts/).'
    for old, new in REFERENCES:
        s = s.replace(old, new)
    return s


cleaned = builder.clean
builder.clean = lambda s: fix(cleaned(s))
previous_definition = builder.definition
builder.definition = lambda f: new_families.get((f['table'], f['family'])) or previous_definition(f)
previous_title = builder.title
builder.title = lambda f: TITLES.get(f['family']) or previous_title(f)

for path in (builder.build_codebook(), builder.build_guide()):
    print('Built', path)
print('families', len(families), 'data columns', n_data)
