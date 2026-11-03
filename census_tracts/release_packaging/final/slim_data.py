"""Remove redundant columns from the 2026-11-02 release copy and rebuild the affected ZIPs.

Each collapsed column replaces a set of source columns that hold identical values in every row;
this is checked row by row, and the script stops if any row disagrees.
Writes column_changes.json (old -> new mapping) next to this script.
"""
from pathlib import Path
import csv
import io
import json
import os
import re
import sys
import zipfile

PACKAGE = Path('/path/to/GECC/uploadGECC/census_tracts_12_metrics_20261102')
HERE = Path(__file__).resolve().parent
csv.field_size_limit(sys.maxsize)


def plan(member, header):
    """Return (drop, collapse) for one CSV: drop is a set of columns, collapse maps new -> sources."""
    drop, collapse = set(), {}

    def group(new, pattern):
        cols = [c for c in header if re.fullmatch(pattern, c)]
        assert len(cols) > 1, (member, new, pattern)
        collapse[new] = cols

    if member == 'support/temperature_gridmet_measurement_support.csv':
        group('gridmet_cell_count', r'gridmet_t(max|min)_count_.*')
    elif member == 'support/temperature_prism_measurement_support.csv':
        group('prism_cell_count', r'prism_t(max|min)_count_.*')
    elif member == 'support/temperature_daymet_measurement_support.csv':
        drop.add('dataset')
        group('daymet_cell_count', r't(max|min)_count_.*')
    elif member == 'metrics/temperature_daymet.csv':
        drop.add('dataset')
    elif member == 'support/extreme_days_measurement_support.csv':
        for fam in ('coverage_pairs', 'total_coverage_weight', 'inside_grid_pairs', 'inside_grid_weight'):
            group(fam, fam + r'_(JJA|DJF)_\d{4}')
        for t in ('tmax', 'tmin'):
            for k in ('mean_days', 'min_days', 'partial_weight', 'support_weight'):
                for s in ('JJA', 'DJF'):
                    group(f'baseline_{t}_{k}_{s}', f'baseline_{t}_{k}_{s}' + r'_\d{4}')
    elif member == 'metrics/land_cover_glc.csv':
        group('has_glc_sampled_area', r'has_glc_sampled_area_\d{4}')
    elif member in ('metrics/vegetation_ua_pieces.csv', 'support/vegetation_ua_pieces_measurement_support.csv'):
        drop.add('year')
    elif member == 'metrics/ndvi_evi_series.csv':
        drop.add('water_only_tract')
    assert drop <= set(header), (member, drop)
    return drop, collapse


def transform(src, dst, member):
    reader = csv.reader(io.TextIOWrapper(src, encoding='utf-8', newline=''))
    header = next(reader)
    drop, collapse = plan(member, header)
    if not drop and not collapse:
        return None
    first = {cols[0]: new for new, cols in collapse.items()}
    merged = {c for cols in collapse.values() for c in cols}
    out_cols, picks = [], []
    for i, c in enumerate(header):
        if c in first:
            out_cols.append(first[c]); picks.append(i)
        elif c not in drop and c not in merged:
            out_cols.append(c); picks.append(i)
    checks = [[header.index(c) for c in cols] for cols in collapse.values()]
    writer = csv.writer(io.TextIOWrapper(dst, encoding='utf-8', newline='', write_through=True), lineterminator='\n')
    writer.writerow(out_cols)
    rows = 0
    for row in reader:
        for idx in checks:
            v = row[idx[0]]
            if any(row[i] != v for i in idx[1:]):
                raise SystemExit(f'{member}: row {rows + 1} differs within a collapsed group')
        writer.writerow([row[i] for i in picks])
        rows += 1
    return {'member': member, 'rows': rows, 'old_columns': len(header), 'new_columns': len(out_cols),
            'dropped': sorted(drop), 'collapsed': collapse}


ARCHIVES = ['temperature_gridmet.zip', 'temperature_prism.zip', 'temperature_daymet.zip', 'extreme_days.zip',
            'land_cover_glc.zip', 'vegetation_ua_pieces.zip', 'ndvi_evi_series.zip']

changes = []
for name in ARCHIVES:
    path = PACKAGE / name
    tmp = path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            target = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            target.compress_type = zipfile.ZIP_DEFLATED
            target.external_attr = info.external_attr
            with zin.open(info) as src:
                header = src.readline().decode('utf-8').rstrip('\r\n').split(',')
            drop, collapse = plan(info.filename, header) if info.filename.endswith('.csv') else (set(), {})
            with zin.open(info) as src, zout.open(target, 'w', force_zip64=True) as dst:
                if drop or collapse:
                    changes.append(transform(src, dst, info.filename))
                    print('rewrote', info.filename, changes[-1]['old_columns'], '->', changes[-1]['new_columns'], flush=True)
                else:
                    while chunk := src.read(1 << 24):
                        dst.write(chunk)
    os.replace(tmp, path)
    print('rebuilt', name, flush=True)

(HERE / 'column_changes.json').write_text(json.dumps(changes, indent=1) + '\n')
print('done')
