"""Update the dictionary, manifest and Markdown files of the 2026-11-02 copy after slim_data.py."""
from pathlib import Path
import csv
import io
import json
import re
import zipfile

PACKAGE = Path('/path/to/GECC/uploadGECC/census_tracts_12_metrics_20261102')
HERE = Path(__file__).resolve().parent
changes = json.loads((HERE / 'column_changes.json').read_text())

SAME = {
    'coverage_pairs': ' The value is the same for every year and season.',
    'total_coverage_weight': ' The value is the same for every year and season.',
    'inside_grid_pairs': ' The value is the same for every year and season.',
    'inside_grid_weight': ' The value is the same for every year and season.',
}
NEW_DEFINITIONS = {
    'gridmet_cell_count': 'Integer number of sampled GRIDMET cells with valid temperature, using cell-center inclusion. '
                          'The count is the same for every year, period and temperature measure. It is not the '
                          'coverage-weight sum used for the statistics.',
    'prism_cell_count': 'Integer number of sampled PRISM cells with valid temperature, using cell-center inclusion. '
                        'The count is the same for every year, period and temperature measure. It is not the '
                        'coverage-weight sum used for the statistics.',
    'daymet_cell_count': 'Coverage-weighted number of Daymet source cells with valid temperature. The count is the '
                         'same for every year, period and temperature measure. Values can be fractional and are not '
                         'day counts.',
    'has_glc_sampled_area': 'Marks whether GLC source sampled area exists for the tract. The value is the same in '
                            'every snapshot.',
}


def family_of(new):
    return re.sub(r'_(JJA|DJF)$', '', new)


def new_definition(new, template):
    if new in NEW_DEFINITIONS:
        return NEW_DEFINITIONS[new]
    fam = family_of(new)
    if fam in SAME:
        return template + SAME[fam]
    return template + ' The value is the same for every observation year.'   # baseline fields


def labels_for(new):
    m = re.search(r'_(JJA|DJF)$', new)
    return json.dumps({'season': m.group(1)}, separators=(',', ':')) if m else '{}'


# Column dictionary
path = PACKAGE / 'metadata/COLUMN_DICTIONARY.csv'
rows = list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8'), newline='')))
header, body = rows[0], rows[1:]
H = {c: i for i, c in enumerate(header)}
for change in changes:
    table = Path(change['member']).stem
    drop = set(change['dropped'])
    sources = {c: new for new, cols in change['collapsed'].items() for c in cols}
    out, done = [], set()
    for row in body:
        if row[H['table']] != table:
            out.append(row); continue
        field = row[H['field']]
        if field in drop:
            continue
        if field in sources:
            new = sources[field]
            if new not in done:
                r = list(row)
                r[H['field']] = new
                r[H['family']] = family_of(new)
                r[H['labels']] = labels_for(new)
                r[H['definition']] = new_definition(new, row[H['definition']])
                out.append(r); done.add(new)
            continue
        out.append(row)
    assert done == set(change['collapsed']), (table, set(change['collapsed']) - done)
    body = out
keep = [i for i, c in enumerate(header) if c not in ('methods', 'source_field', 'source_type')]
buf = io.StringIO(newline='')
csv.writer(buf, lineterminator='\r\n').writerows([[r[i] for i in keep] for r in [header] + body])
path.write_text(buf.getvalue(), encoding='utf-8')
n_dictionary = len(body)
print('dictionary rows', n_dictionary, 'columns', [header[i] for i in keep])

# Check every data CSV header against the dictionary
dict_cols = {}
for r in body:
    dict_cols.setdefault(r[H['table']], []).append(r[H['field']])
manifest_path = PACKAGE / 'FILE_MANIFEST.csv'
manifest = list(csv.DictReader(manifest_path.open(encoding='utf-8', newline='')))
info = {}
for z in sorted(PACKAGE.glob('*.zip')):
    with zipfile.ZipFile(z) as zf:
        for i in zf.infolist():
            with zf.open(i) as f:
                first = f.readline().decode('utf-8').rstrip('\r\n')
            info[i.filename] = (i.file_size, first)
updated = {}
for m in manifest:
    size, first = info[m['path']]
    if m['path'].endswith('.csv'):
        cols = first.split(',')
        assert cols == dict_cols[m['table']], m['path']
        if str(len(cols)) != m['columns']:
            updated[m['path']] = (m['columns'], str(len(cols)))
        m['columns'] = str(len(cols))
    m['bytes'] = str(size)
buf = io.StringIO(newline='')
w = csv.DictWriter(buf, fieldnames=list(manifest[0]), lineterminator='\n' if b'\r\n' not in manifest_path.read_bytes() else '\r\n')
w.writeheader(); w.writerows(manifest)
manifest_path.write_text(buf.getvalue(), encoding='utf-8')
print('manifest column changes', updated)

# Markdown
def replace_once(text, old, new, what):
    assert text.count(old) == 1, (what, text.count(old))
    return text.replace(old, new)

guide = (PACKAGE / 'FILE_GUIDE.md').read_text(encoding='utf-8')
for p, (old, new) in updated.items():
    pat = re.compile(r'(\[' + re.escape(p) + r'\][^\n]*?rows\. )' + old + r'( attribute columns)')
    guide, n = pat.subn(r'\g<1>' + new + r'\g<2>', guide)
    assert n == 1, (p, n)
guide = replace_once(guide, 'all 6,431 data columns', f'all {n_dictionary:,} data columns', 'dictionary count')
guide = replace_once(guide, 'Exact names, definitions, units, types and source mappings', 'Exact names, definitions, units, types and missing-value rules', 'dictionary description')
guide = replace_once(guide, 'Inventory of all extracted files, followed by dataset methods and limitations.', 'Inventory of all extracted files.', 'guide row')
start = guide.index('## Dataset methods and limitations')
end = guide.index('## Authors, license and acknowledgment')
guide = guide[:start] + ('## Dataset methods and limitations\n\nMethods, definitions and limitations for each table are in the '
                         '[codebook](documentation/GECC_CensusTracts_Codebook.pdf). Known source issues are summarized in '
                         '[RELEASE_NOTES.md](RELEASE_NOTES.md).\n\n') + guide[end:]
(PACKAGE / 'FILE_GUIDE.md').write_text(guide, encoding='utf-8')

readme = (PACKAGE / 'README.md').read_text(encoding='utf-8')
start = readme.index('## Column types\n'); end = readme.index('## Missing values and interpretation')
readme = readme[:start] + ('## Column types\n\nWhen importing the CSV files, use serialized_type, logical_type and null_rule in '
                           'COLUMN_DICTIONARY.csv. Keep geographic codes as text.\n\n') + readme[end:]
start = readme.index('Use estimates with their flags and support tables.'); end = readme.index('## Citation and attribution')
readme = readme[:start] + ('Use estimates with their flags and support tables. Known source issues, including the New York City '
                           'SVCM-SP mask, the GRIDMET 2011-2013 offset and the GLC-FCS30D annual-tile issue, are listed in '
                           'RELEASE_NOTES.md. The codebook describes each metric\'s coverage, weighting, thresholds and '
                           'limitations.\n\n') + readme[end:]
(PACKAGE / 'README.md').write_text(readme, encoding='utf-8')
json.dump({'dictionary_rows': n_dictionary, 'manifest_updates': updated}, open(HERE / 'metadata_summary.json', 'w'), indent=1)
print('markdown updated')
