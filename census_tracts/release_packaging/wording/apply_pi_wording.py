"""Apply pi_wording edits to the corrected release's Markdown files and column dictionary."""
from pathlib import Path
import csv
import io

import pi_wording

import sys
PACKAGE = Path(sys.argv[1] if len(sys.argv) > 1 else '/path/to/GECC/uploadGECC/census_tracts_12_metrics_20261030_corrected')

for name in ('README.md', 'FILE_GUIDE.md', 'RELEASE_NOTES.md'):
    path = PACKAGE / name
    before = path.read_text(encoding='utf-8')
    after = pi_wording.text(before)
    path.write_text(after, encoding='utf-8')
    print(name, 'changed' if after != before else 'unchanged')

path = PACKAGE / 'metadata/COLUMN_DICTIONARY.csv'
raw = path.read_bytes().decode('utf-8')
rows = list(csv.reader(io.StringIO(raw, newline='')))
header = rows[0]
t, f, d = header.index('table'), header.index('field'), header.index('definition')
changed = 0
for row in rows[1:]:
    new = pi_wording.text(row[d])
    if row[f] == 'sliver_flag':
        assert row[d] in (pi_wording.OLD_SLIVER, pi_wording.SLIVER[row[t]]), row[d]
        new = pi_wording.SLIVER[row[t]]
    if new != row[d]:
        row[d] = new
        changed += 1
out = io.StringIO(newline='')
csv.writer(out, lineterminator='\r\n' if '\r\n' in raw[:2000] else '\n').writerows(rows)
path.write_text(out.getvalue(), encoding='utf-8')
print('COLUMN_DICTIONARY.csv rows changed:', changed, 'of', len(rows) - 1)

# Item 5: status-code definitions.
rows = list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8'), newline='')))
fam = rows[0].index('family')
n = 0
for row in rows[1:]:
    new = pi_wording.STATUS.get((row[t], row[fam]))
    if new and row[d] != new:
        row[d] = new
        n += 1
out = io.StringIO(newline='')
csv.writer(out, lineterminator='\r\n').writerows(rows)
path.write_text(out.getvalue(), encoding='utf-8')
print('Status definitions set on rows:', n)

# Minor-polish round: apply text replacements to every descriptive dictionary column.
rows = list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8'), newline='')))
cols = [rows[0].index(c) for c in ('units', 'source_type', 'definition', 'statistic_support', 'null_rule')]
n = 0
for row in rows[1:]:
    before = list(row)
    for c in cols:
        row[c] = pi_wording.text(row[c])
    n += row != before
out = io.StringIO(newline='')
csv.writer(out, lineterminator='\r\n').writerows(rows)
path.write_text(out.getvalue(), encoding='utf-8')
print('Polish rows changed:', n)

# Item 4: rename internal table and role names.
def rename(path, columns):
    rows = list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8'), newline='')))
    idx = {c: rows[0].index(c) for c in columns if c in rows[0]}
    n = 0
    for row in rows[1:]:
        before = list(row)
        for c, i in idx.items():
            names = pi_wording.ROLE_RENAME if c == 'role' else pi_wording.TABLE_RENAME
            row[i] = names.get(row[i], row[i])
            if c in ('definition', 'statistic_support'):
                row[i] = pi_wording.text(row[i])
        n += row != before
    raw = path.read_bytes()
    out = io.StringIO(newline='')
    csv.writer(out, lineterminator='\r\n' if b'\r\n' in raw[:3000] else '\n').writerows(rows)
    path.write_text(out.getvalue(), encoding='utf-8')
    print(path.name, 'rename rows changed:', n)

rename(path, ('table', 'role', 'source_product', 'methods', 'definition', 'statistic_support'))
rename(PACKAGE / 'FILE_MANIFEST.csv', ('table', 'role', 'parent'))
