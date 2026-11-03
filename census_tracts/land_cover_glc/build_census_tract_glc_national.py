"""Combine CityHealth-selected and remaining tract GLC exports into one national file.

Rebuilds from the raw Earth Engine CSVs rather than the selected delivery:
- selected tracts (47,105), 25 years: GECC_CityHealth2020_SelectedTracts_GLC_FCS30D zip
- remaining tracts (37,309), 25 years: GECC_RemainingTracts_GLC_FCS30D zip
- all tracts, 2022: GECC_CensusTract_GLC_FCS30D zip

Applies the same conventions as build_cityhealth_selected_four_files.py
(zero sampled-area fractions null, has_glc_sampled_area flag), validates batch
membership, exact national coverage and GLC arithmetic, then cross-checks every
selected-tract value (exact round-trip float parsing) against the existing selected delivery CSV and reads the
written CSV back in full.
"""
from pathlib import Path
import hashlib, json, re, zipfile
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DOWNLOADS = Path('/path/to/home/Downloads')
SELECTED_ZIP = DOWNLOADS/'GECC_CityHealth2020_SelectedTracts_GLC_FCS30D-20260914T150846Z-1-001.zip'
REMAINING_ZIP = DOWNLOADS/'GECC_RemainingTracts_GLC_FCS30D-20260914T154403Z-1-001.zip'
NATIONAL_2022_ZIP = ROOT/'cityHealth/GECC_CensusTract_GLC_FCS30D-20260911T142446Z-1-001.zip'
TRACTS_SHP = ROOT/'Census/Census_2020_US_Tract_Polygons/Census_2020_US_Tract_Polygons.shp'
LOOKUP_CSV = ROOT/'cityHealth/tracts_2020_lookup_temperature/cityhealth2020_tract2020_lookup.csv'
SELECTED_DELIVERY = ROOT/'cityHealth/selected_tracts_delivery_2026-09-14/cityHealth2020_selected_tracts_GLC_1985_2022.csv'
OUT_CSV = ROOT/'CensusTracts/GECC_CensusTract_GLC_FCS30D_1985_2022.csv'
AUDIT = ROOT/'outputs/census_tract_glc_national_2026-09-14'
YEARS = [1985, 1990, 1995, *range(2000, 2023)]
EXPORT_YEARS = [y for y in YEARS if y != 2022]
GROUPS = ['vegetation', 'tree', 'shrub', 'grass', 'crop']
FRACTIONS = [f'{g}_fraction' for g in GROUPS]
N_TRACTS, N_SELECTED, N_REMAINING = 84414, 47105, 37309


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def tract_identities():
    tracts = gpd.read_file(TRACTS_SHP, ignore_geometry=True)
    tracts['GEOID'] = tracts.GEOID.astype(str)
    assert len(tracts) == N_TRACTS and tracts.GEOID.is_unique and tracts.GEOID.str.len().eq(11).all()
    lookup = pd.read_csv(LOOKUP_CSV, dtype=str).sort_values(['GEOID', 'city_id'])
    lookup['city_label'] = lookup.city_name+' ('+lookup.city_state+')'
    shared = pd.DataFrame(tracts.drop(columns='geometry', errors='ignore')).set_index('GEOID').sort_index()
    by_tract = lookup.groupby('GEOID')
    shared['cityhealth_selected'] = shared.index.isin(lookup.GEOID).astype(int)
    shared['city_count'] = by_tract.size().reindex(shared.index).fillna(0).astype(int)
    shared['city_ids'] = by_tract.city_id.agg('|'.join)
    shared['city_names'] = by_tract.city_label.agg('|'.join)
    shared['city_match_types'] = by_tract.match_type.agg('|'.join)
    shared['boundary_year'] = 2020
    assert shared.cityhealth_selected.sum() == N_SELECTED
    return shared


def read_zip(path, pattern, expected_batches, expected_years):
    """Read every CSV in an export zip; check year/batch naming and batch membership."""
    frames, seen = [], set()
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith('.csv'):
                continue
            m = re.search(pattern, name)
            assert m, name
            year, batch = int(m['year']), int(m['batch'])
            with archive.open(name) as f:
                d = pd.read_csv(f, dtype={'GEOID': str})
            assert d.year.eq(year).all(), name
            if expected_batches is not None:
                assert d.batch.eq(batch).all(), name
                assert set(d.GEOID) == expected_batches[batch], name
            seen.add((year, batch))
            frames.append(d.drop(columns=['batch', 'selection', 'tract_boundary_year'], errors='ignore'))
    assert seen == {(y, b) for y in expected_years for b in range(10)}, path
    return pd.concat(frames, ignore_index=True)


def batch_sets(csv_path):
    lookup = pd.read_csv(csv_path, dtype={'GEOID': str})
    return {b: set(g.GEOID) for b, g in lookup.groupby('batch')}


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    shared = tract_identities()
    ids = set(shared.index)
    selected_ids = set(shared.index[shared.cityhealth_selected.eq(1)])

    print('Reading raw exports', flush=True)
    selected_batches = batch_sets(ROOT/'outputs/cityhealth_selected_tract_glc/selected_tract_batch_lookup.csv')
    remaining_batches = batch_sets(ROOT/'outputs/remaining_tract_glc/remaining_tract_batch_lookup.csv')
    assert set().union(*selected_batches.values()) == selected_ids
    assert set().union(*remaining_batches.values()) == ids-selected_ids and len(ids-selected_ids) == N_REMAINING
    pat = r'_v1_(?P<year>\d{4})_b(?P<batch>\d{3})of010\.csv$'
    sel = read_zip(SELECTED_ZIP, pat, selected_batches, EXPORT_YEARS)
    rem = read_zip(REMAINING_ZIP, pat, remaining_batches, EXPORT_YEARS)
    nat = read_zip(NATIONAL_2022_ZIP, r'_(?P<year>2022)_b(?P<batch>\d{3})of010\.csv$', None, [2022])
    assert len(nat) == N_TRACTS and set(nat.GEOID) == ids
    assert list(sel.columns) == list(rem.columns)
    zero = nat.total_area_m2.eq(0)
    nat.loc[zero, FRACTIONS] = np.nan
    nat['has_glc_sampled_area'] = (~zero).astype(int)
    assert list(nat.columns) == list(sel.columns)
    d = pd.concat([sel, rem, nat], ignore_index=True)
    del sel, rem, nat

    print('Validating long table', flush=True)
    assert len(d) == N_TRACTS*len(YEARS) and not d.duplicated(['GEOID', 'year']).any()
    for year, part in d.groupby('year'):
        assert set(part.GEOID) == ids, year
    raw = [c for c in d if c.startswith('area_class_')]
    assert len(raw) == 37 and d[raw].ge(0).all().all()
    assert np.allclose(d[raw].sum(axis=1), d.total_area_m2, rtol=1e-10, atol=1e-4)
    assert d.has_glc_sampled_area.eq(d.total_area_m2.gt(0).astype(int)).all()
    nonzero = d.total_area_m2.gt(0)
    assert np.allclose(d.loc[nonzero, 'total_area_m2']/d.loc[nonzero, 'source_area_m2'], d.loc[nonzero, 'glc_coverage_fraction'])
    for g in GROUPS:
        assert d.loc[~nonzero, g+'_fraction'].isna().all()
        assert np.allclose(d.loc[nonzero, g+'_area_m2']/d.loc[nonzero, 'total_area_m2'], d.loc[nonzero, g+'_fraction'])
        assert d.loc[nonzero, g+'_fraction'].between(0, 1+1e-9).all()
    coverage = d.assign(cityhealth_selected=d.GEOID.isin(selected_ids).astype(int)).groupby(['year', 'cityhealth_selected']).agg(
        rows=('GEOID', 'size'), sampled_tracts=('has_glc_sampled_area', 'sum')).reset_index()
    coverage.to_csv(AUDIT/'glc_year_coverage.csv', index=False)

    print('Pivoting to one row per tract', flush=True)
    metrics = [c for c in d if c not in ['GEOID', 'year']]
    parts = []
    for year in YEARS:
        part = d[d.year.eq(year)].set_index('GEOID')[metrics].reindex(shared.index)
        part.columns = [f'{c}_{year}' for c in part]
        parts.append(part)
    del d
    wide = pd.concat([shared, *parts], axis=1)
    del parts
    wide['land_cover_source'] = 'GLC-FCS30D'
    wide['area_units'] = 'm2'
    wide['fraction_denominator'] = 'total sampled GLC raster area; zero-area fractions null'
    wide.index.name = 'GEOID'
    glc_cols = [c for c in wide if re.search(r'_(19|20)\d{2}$', c) and c not in shared.columns]

    print('Cross-checking selected tracts against existing delivery', flush=True)
    checked = 0
    for chunk in pd.read_csv(SELECTED_DELIVERY, dtype={'GEOID': str, 'StFIPS': str, 'CoFIPS': str, 'TrFIPS': str},
                             usecols=lambda c: c == 'GEOID' or c in glc_cols or c in ['Area_Land', 'Area_Water', 'Pop'],
                             chunksize=5000, float_precision='round_trip'):
        chunk = chunk.set_index('GEOID')
        assert list(chunk.columns[chunk.columns.isin(glc_cols)]) == glc_cols
        ours = wide.loc[chunk.index]
        assert np.array_equal(ours[glc_cols].to_numpy(float), chunk[glc_cols].to_numpy(float), equal_nan=True)
        for c in ['Area_Land', 'Area_Water', 'Pop']:
            assert np.allclose(ours[c].astype(float), chunk[c].astype(float)), c
        checked += len(chunk)
    assert checked == N_SELECTED

    print(f'Writing {OUT_CSV} ({len(wide):,} rows, {wide.shape[1]+1:,} columns)', flush=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_CSV)

    print('Reading back written CSV', flush=True)
    back = pd.read_csv(OUT_CSV, dtype={'GEOID': str, 'StFIPS': str, 'CoFIPS': str, 'TrFIPS': str,
                                       'city_ids': str, 'city_names': str, 'city_match_types': str},
                       float_precision='round_trip').set_index('GEOID')
    assert list(back.index) == list(wide.index) and list(back.columns) == list(wide.columns)
    assert np.array_equal(back[glc_cols].to_numpy(float), wide[glc_cols].to_numpy(float), equal_nan=True)
    for c in ['city_ids', 'city_names', 'city_match_types']:
        assert back[c].fillna('').equals(wide[c].fillna('')), c

    result = {
        'status': 'PASS', 'output': str(OUT_CSV.relative_to(ROOT)), 'rows': len(wide), 'columns': wide.shape[1]+1,
        'csv_bytes': OUT_CSV.stat().st_size, 'years': YEARS, 'cityhealth_selected_tracts': N_SELECTED,
        'remaining_tracts': N_REMAINING, 'selected_values_match_existing_delivery': True,
        'conventions': 'zero sampled-area fractions null; has_glc_sampled_area flag; class code 0 separate; areas m2',
        'tract_attributes_source': str(TRACTS_SHP.relative_to(ROOT)),
        'sources': [{'path': str(p), 'sha256': sha256(p)} for p in [SELECTED_ZIP, REMAINING_ZIP, NATIONAL_2022_ZIP]],
    }
    (AUDIT/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
