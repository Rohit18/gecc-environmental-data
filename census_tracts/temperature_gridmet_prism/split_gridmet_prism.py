"""Split the combined GRIDMET/PRISM tables into one table set per product.

Values are copied as serialized text; nothing is recomputed. The combined source
files and metadata are unchanged: split copies are written next to this script.
"""
from collections import Counter
from pathlib import Path
import copy
import csv
import json
import re
import statistics

ROOT = Path('/path/to/GECC')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'stakeholder_packages/census_tracts_20261018'
META = ROOT / 'release_candidates/census_tracts_20261018/metadata'
DATA = HERE / 'split_gridmet_prism_data'
SPLIT_META = HERE / 'split_gridmet_prism_metadata'

OLD = 'temperature_gridmet_prism'
PRODUCTS = {'GRIDMET': 'temperature_gridmet', 'PRISM': 'temperature_prism'}
PREFIX = {'GRIDMET': 'gridmet_', 'PRISM': 'prism_'}
SUFFIXES = ('', '_measurement_support', '_calendar')
OLD_FILES = {'': 'metrics/' + OLD + '.csv',
             '_measurement_support': 'support/' + OLD + '_measurement_support.csv',
             '_calendar': 'metadata/' + OLD + '_calendar.csv'}
FOLDER = {'': 'metrics', '_measurement_support': 'support', '_calendar': 'metadata'}

def new_path(dataset, suffix):
    return f'{FOLDER[suffix]}/{PRODUCTS[dataset]}{suffix}.csv'

SPLIT_FILES = {new_path(d, s): (d, s) for d in PRODUCTS for s in SUFFIXES}

def source_path(member):
    """Where an archive member is read from: split copy or the unchanged source."""
    return DATA / member if member in SPLIT_FILES else SOURCE / member

def suffix_of(table):
    for s in ('_measurement_support', '_calendar'):
        if table == OLD + s: return s
    return '' if table == OLD else None

def datasets_for(table, field):
    """Products a combined (table, field) belongs to: GEOID and calendar fields go to both."""
    s = suffix_of(table)
    if s is None: return []
    if s == '_calendar' or field == 'GEOID': return list(PRODUCTS)
    return [d for d in PRODUCTS if field.startswith(PREFIX[d])]

def column_indices(header, dataset):
    return [i for i, c in enumerate(header) if c == 'GEOID' or c.startswith(PREFIX[dataset])]

def split_files():
    """Write the six split CSVs (skipped if already present). Lines are split on commas as text."""
    if all((DATA / p).is_file() for p in SPLIT_FILES): return
    for s in ('', '_measurement_support'):
        src = SOURCE / OLD_FILES[s]
        with src.open(newline='') as f:
            header = f.readline().rstrip('\r\n').split(',')
            keep = {d: column_indices(header, d) for d in PRODUCTS}
            assert sorted(set(keep['GRIDMET']) | set(keep['PRISM'])) == list(range(len(header)))
            assert set(keep['GRIDMET']) & set(keep['PRISM']) == {header.index('GEOID')}
            outs = {}
            for d in PRODUCTS:
                (DATA / new_path(d, s)).parent.mkdir(parents=True, exist_ok=True)
                outs[d] = (DATA / new_path(d, s)).open('w', newline='')
                outs[d].write(','.join(header[i] for i in keep[d]) + '\n')
            for line in f:
                assert '"' not in line, 'quoted CSV value; text split is not safe'
                values = line.rstrip('\r\n').split(',')
                assert len(values) == len(header)
                for d in PRODUCTS:
                    outs[d].write(','.join(values[i] for i in keep[d]) + '\n')
            for out in outs.values(): out.close()
    with (SOURCE / OLD_FILES['_calendar']).open(newline='') as f:
        lines = f.read().splitlines()
    header = lines[0].split(',')
    for d in PRODUCTS:
        rows = [l for l in lines[1:] if l.split(',')[header.index('dataset')] == d]
        (DATA / new_path(d, '_calendar')).parent.mkdir(parents=True, exist_ok=True)
        (DATA / new_path(d, '_calendar')).write_text('\n'.join([lines[0]] + rows) + '\n')

def verify():
    """Check the split files reproduce every combined source value, column and row."""
    checked = 0
    for s in ('', '_measurement_support'):
        with (SOURCE / OLD_FILES[s]).open(newline='') as src, \
             (DATA / new_path('GRIDMET', s)).open(newline='') as g, \
             (DATA / new_path('PRISM', s)).open(newline='') as pr:
            header = src.readline().rstrip('\r\n').split(',')
            keep = {d: column_indices(header, d) for d in PRODUCTS}
            assert g.readline().rstrip('\r\n').split(',') == [header[i] for i in keep['GRIDMET']]
            assert pr.readline().rstrip('\r\n').split(',') == [header[i] for i in keep['PRISM']]
            for line, gl, pl in zip(src, g, pr, strict=True):
                values = line.rstrip('\r\n').split(',')
                assert gl.rstrip('\r\n').split(',') == [values[i] for i in keep['GRIDMET']]
                assert pl.rstrip('\r\n').split(',') == [values[i] for i in keep['PRISM']]
                checked += 1
    combined = (SOURCE / OLD_FILES['_calendar']).read_text().splitlines()
    parts = [(DATA / new_path(d, '_calendar')).read_text().splitlines() for d in PRODUCTS]
    assert all(p[0] == combined[0] for p in parts)
    assert sorted(combined[1:]) == sorted(parts[0][1:] + parts[1][1:])
    return checked

def file_stats(member):
    path = DATA / member
    with path.open(newline='') as f:
        header = f.readline().split(',')
        rows = sum(1 for _ in f)
    return {'bytes': str(path.stat().st_size), 'rows': str(rows), 'columns': str(len(header))}

def manifest_rows(manifest):
    """Replace each combined inventory row with one row per product, in place."""
    out = []
    for r in manifest:
        s = suffix_of(r['table'])
        if s is None:
            out.append(r); continue
        for d, tid in PRODUCTS.items():
            row = dict(r, path=new_path(d, s), table=tid + s, **file_stats(new_path(d, s)))
            if row.get('parent') == OLD: row['parent'] = tid
            out.append(row)
    return out

def dictionary_rows(rows):
    out = []
    for r in rows:
        s = suffix_of(r['table'])
        if s is None:
            out.append(r); continue
        for d in datasets_for(r['table'], r['field']):
            row = dict(r, table=PRODUCTS[d] + s)
            for k in ('source_product', 'methods'):
                if row.get(k) == OLD: row[k] = PRODUCTS[d]
            out.append(row)
    return out

def _calendar(d):
    with (DATA / new_path(d, '_calendar')).open(newline='') as f:
        return list(csv.DictReader(f))

def _day_count_stats(values, template):
    x = [float(v) for v in values]
    return dict(template, rows=str(len(x)), valid_n=str(len(x)), missing_n='0', mean=str(statistics.fmean(x)),
                sd=str(statistics.stdev(x)), min=str(min(x)), median=str(statistics.median(x)), max=str(max(x)),
                zero_n=str(float(sum(v == 0 for v in x))))

def _year(field):
    m = re.search(r'_(\d{4})$', field)
    return m[1] if m else None

def write_metadata():
    """Write split copies of the metadata the document builder reads."""
    SPLIT_META.mkdir(exist_ok=True)
    # Summary statistics: spatial rows follow their column; calendar rows are recalculated per product.
    with (META / 'FIELD_SUMMARY_STATISTICS.csv').open(newline='') as f:
        summaries = list(csv.DictReader(f))
    combined_calendar = [r for d in PRODUCTS for r in _calendar(d)]
    out = []
    for r in summaries:
        s = suffix_of(r['table'])
        if s is None:
            out.append(r); continue
        for d in datasets_for(r['table'], r['field']):
            row = dict(r, table=PRODUCTS[d] + s)
            if s == '_calendar':
                rows = _calendar(d)
                row.update(rows=str(len(rows)), valid_n=str(len(rows)))
                if r['field'] == 'source_period_day_count':
                    # Same formula must reproduce the published combined statistics.
                    check = _day_count_stats([c['source_period_day_count'] for c in combined_calendar], r)
                    assert all(abs(float(check[k]) - float(r[k])) < 1e-9 for k in ('mean', 'sd', 'min', 'median', 'max')), r
                    row = _day_count_stats([c['source_period_day_count'] for c in rows], row)
            out.append(row)
    valid = {(r['table'], r['field']): r['valid_n'] != '0' for r in out}
    with (SPLIT_META / 'FIELD_SUMMARY_STATISTICS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(out)
    # Categorical counts for calendar fields, recalculated from the split calendars.
    with (META / 'CATEGORICAL_COUNTS.csv').open(newline='') as f:
        categories = list(csv.DictReader(f))
    out = []
    for r in categories:
        s = suffix_of(r['table'])
        if s is None:
            out.append(r); continue
        if any(x['table'] == PRODUCTS['GRIDMET'] + s and x['field'] == r['field'] for x in out): continue
        fields = [x for x in categories if x['table'] == r['table'] and x['field'] == r['field']]
        assert {x['value']: int(x['n']) for x in fields} == dict(Counter(c[r['field']] for c in combined_calendar))
        for d, tid in PRODUCTS.items():
            for value, n in Counter(c[r['field']] for c in _calendar(d)).items():
                out.append(dict(r, table=tid + s, value=value, n=str(n)))
    with (SPLIT_META / 'CATEGORICAL_COUNTS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(categories[0])); w.writeheader(); w.writerows(out)
    # Variable families: columns follow their product; observed years come from nonmissing summaries.
    families = json.loads((META / 'VARIABLE_FAMILIES.json').read_text())
    out = []
    for fam in families:
        s = suffix_of(fam['table'])
        if s is None:
            out.append(fam); continue
        observed = set()
        for d in PRODUCTS:
            row = copy.deepcopy(fam)
            row['table'] = PRODUCTS[d] + s
            if row['source_product'] == OLD: row['source_product'] = PRODUCTS[d]
            if s != '_calendar':
                row['columns'] = [c for c in fam['columns'] if d in datasets_for(fam['table'], c)]
                row['column_count'] = len(row['columns'])
                years = {_year(c) for c in row['columns']} - {None}
                row['available_years'] = sorted(years)
                row['observed_years'] = sorted({_year(c) for c in row['columns'] if valid.get((row['table'], c))} - {None})
                observed |= set(row['observed_years'])
            out.append(row)
        assert s == '_calendar' or observed == set(fam['observed_years']), fam['family']
    (SPLIT_META / 'VARIABLE_FAMILIES.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    # Delivery plan: one table entry per product, with its own files, fields and links.
    plan = json.loads((META / 'DELIVERY_PLAN.json').read_text())
    tables = []
    for t in plan['tables']:
        s = suffix_of(t['product_id'])
        if s is None:
            tables.append(t); continue
        for d, tid in PRODUCTS.items():
            row = copy.deepcopy(t)
            row['product_id'] = tid + s
            row['title'] = d + ' air temperature' + {'': '', '_measurement_support': ' - measurement support', '_calendar': ' - calendar'}[s]
            row['proposed_files'] = [new_path(d, s)]
            if row.get('parent_product_id') == OLD: row['parent_product_id'] = tid
            for fk in row.get('foreign_keys', []):
                if fk['references_product'].startswith(OLD): fk['references_product'] = fk['references_product'].replace(OLD, tid)
            stats = file_stats(new_path(d, s))
            row['proposed_export_rows'] = int(stats['rows'])
            if 'csv_fields' in row and s != '_calendar':
                row['csv_fields'] = [c for c in t['csv_fields'] if d in datasets_for(t['product_id'], c)]
                assert len(row['csv_fields']) == int(stats['columns'])
            if 'proposed_export_column_count' in row: row['proposed_export_column_count'] = int(stats['columns'])
            for key, suffix in (('support_tables', '_measurement_support'), ('calendar_tables', '_calendar')):
                for sub in row.get(key, []):
                    sub_stats = file_stats(new_path(d, suffix))
                    sub.update(product_id=tid + suffix, file=new_path(d, suffix),
                               rows=int(sub_stats['rows']), columns=int(sub_stats['columns']))
            tables.append(row)
    plan['tables'] = tables
    (SPLIT_META / 'DELIVERY_PLAN.json').write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    methods = json.loads((META / 'PRODUCT_METHODS.json').read_text())
    split_methods = {}
    for k, v in methods.items():
        if k == OLD: split_methods.update({tid: v for tid in PRODUCTS.values()})
        else: split_methods[k] = v
    (SPLIT_META / 'PRODUCT_METHODS.json').write_text(json.dumps(split_methods, indent=1, ensure_ascii=False))

if __name__ == '__main__':
    split_files()
    write_metadata()
    print('rows checked', verify())
