"""Build the selected 12-metric Dataverse delivery without changing source data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import json
import re
import zipfile

ROOT = Path('/path/to/GECC')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'stakeholder_packages/census_tracts_20261018'
DEST = ROOT / 'uploadGECC/census_tracts_12_metrics_20261030'
DEST.mkdir(parents=True, exist_ok=True)
(DEST / 'metadata').mkdir(exist_ok=True)
(DEST / 'documentation').mkdir(exist_ok=True)

def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

RELEASE = '1.0'
manifest = [r for r in read_csv(SOURCE / 'FILE_MANIFEST.csv') if 'era5land' not in r['path']]
dictionary = [r for r in read_csv(SOURCE / 'metadata/COLUMN_DICTIONARY.csv') if 'era5land' not in r['table']]
for row in dictionary:
    row['definition'] = row['definition'].replace('ERA5-Land/Daymet', 'Daymet')
# GRIDMET and PRISM are delivered as separate tables; values are copied unchanged.
import split_gridmet_prism as split
split.split_files()
split.write_metadata()
manifest = split.manifest_rows(manifest)
dictionary = split.dictionary_rows(dictionary)
import editorial_cleanup as editorial
import researcher_documentation as researcher
editorial.rows_cleanup(dictionary)
researcher.update_rows(dictionary)
GRIDMET_2011_2013_LIMIT = ('GRIDMET maximum temperature for 2011, 2012 and 2013 is warmer than PRISM, Daymet and weather-station observations, and minimum temperature is slightly cooler. '
    'For JJA maximum temperature, the median tract difference from Daymet is +0.8 to +1.3 °C, reaching +2 to +3.5 °C in Midwest and Great Plains states. Other years agree within about 0.2 °C. '
    'The same values are in the GRIDMET source files, so they have not been corrected. For analyses that include these years, use PRISM or Daymet or test sensitivity to the GRIDMET values.')
GRIDMET_FIELD_NOTE = ' GRIDMET 2011-2013 values are warmer for maximum and cooler for minimum temperature than PRISM, Daymet and station data; see codebook limitations.'
for row in dictionary:
    if row['table'] == 'temperature_gridmet' and row['field'][-4:] in ('2011', '2012', '2013'):
        row['definition'] += GRIDMET_FIELD_NOTE
GLC_LIST_RULE = ('Tracts are listed when their interior point lies in the two tiles, so tracts crossing the tile edges may be partly affected. '
    'Filter the list by its table column before joining: land_cover_glc rows by GEOID, vegetation_ua_pieces rows by piece_id.')
GLC_ANNUAL_LIMIT = ('The annual GLC-FCS30D tiles W120N35 and W120N40 (115-120° W, 30-40° N) classify most urban land as vegetation or bare land in 2000-2022, unlike the 1985-1995 maps. '
    'For the 6,424 tracts in southern California and Nevada listed in metadata/GLC_ANNUAL_TILE_AFFECTED.csv, 2000-2022 vegetation fractions are too high and the 1995-2000 change is not real. '
    'The values are as in the Earth Engine copy of the source and have not been corrected. ' + GLC_LIST_RULE)
GLC_PIECES_LIMIT = ('The 2020 fractions use the same GLC-FCS30D tiles, so urban fractions for the 6,380 overlaps of the listed tracts (89 Urban Areas) are too high; '
    'metadata/GLC_ANNUAL_TILE_AFFECTED.csv lists them by piece_id.')
GLC_FIELD_NOTE = ' For tracts in GLC-FCS30D annual tiles W120N35 and W120N40 (southern California and Nevada), urban land is classified as vegetation or bare land; metadata/GLC_ANNUAL_TILE_AFFECTED.csv lists the affected records.'
for row in dictionary:
    if (row['table'] == 'land_cover_glc' and row['field'].endswith('_fraction_' + row['field'][-4:]) and row['field'][-4:].isdigit() and int(row['field'][-4:]) >= 2000) or \
       (row['table'] == 'vegetation_ua_pieces' and row['field'] in ('urban_frac', 'rural_frac', 'diff')):
        row['definition'] += GLC_FIELD_NOTE
LAI_SCALE_OLD, LAI_SCALE_NEW = 'Stored LAI divided by 10.', 'The source stores LAI multiplied by 10; the CSV values are already divided by 10.'
OVERLAP_SHARE_NOTE = (' Because overlap and tract areas are computed separately, 18,847 overlaps that cover the whole tract have values slightly above 1 (at most 1.000002). '
    'Treat values within 1 + 1e-5 as 1.')
for row in dictionary:
    if row['table'] == 'lai_water_excluded':
        assert row['statistic_support'].count(LAI_SCALE_OLD) == 1
        row['statistic_support'] = row['statistic_support'].replace(LAI_SCALE_OLD, LAI_SCALE_NEW)
    if row['field'] == 'overlap_share':
        row['definition'] += OVERLAP_SHARE_NOTE
assert sum(r['field'] == 'overlap_share' for r in dictionary) == 1
# Affected-record list for the GLC annual-tile error; a separate download next to the dictionary.
import glc_affected_records as glc_affected
GLC_AFFECTED = glc_affected.build()
(DEST / 'metadata/GLC_ANNUAL_TILE_AFFECTED.csv').write_bytes(GLC_AFFECTED.read_bytes())
dictionary += glc_affected.dictionary_rows(list(dictionary[0]))
write_csv(DEST / 'metadata/COLUMN_DICTIONARY.csv', dictionary)
metrics = [r['table'] for r in manifest if r['role'] == 'main_metric']
assert len(metrics) == 12 and len(manifest) == 34
archives = {tid + '.zip': [] for tid in metrics}
archives['shared_geography.zip'] = []
path_archive = {}
for row in manifest:
    if row['path'].startswith('geography/'):
        archive = 'shared_geography.zip'
    elif row['table'] == 'ndvi_evi_support':
        archive = 'ndvi_evi_series.zip'
    else:
        archive = (row['parent'] or row['table']) + '.zip'
    archives[archive].append(row['path'])
    path_archive[row['path']] = archive
write_csv(DEST / 'FILE_MANIFEST.csv', [{'archive': path_archive[r['path']], **r, 'release_version': RELEASE} for r in manifest])

# Adapt the established document builder; retain scientific definitions/statistics.
original = ROOT / 'reports/public_release_prep_2026-10-19/build_public_documents.py'
code = original.read_text()
code = code.replace("PACKAGE = ROOT / 'stakeholder_packages/census_tracts_20261018'", "PACKAGE = ROOT / 'uploadGECC/census_tracts_12_metrics_20261030'")
code = code.replace("RELEASE = '2026.10.18'", f"RELEASE = '{RELEASE}'")
code = code.replace("META = ROOT / 'release_candidates/census_tracts_20261018/metadata'", f"META = Path({str(split.SPLIT_META)!r})")
code = code.replace("_{STAMP}.pdf'", ".pdf'")
code = code.replace("GECC_LICENCE = None", "GECC_LICENCE = 'CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)'").replace("labelled('Licence.'", "labelled('License.'")
code = code.replace("GECC_CONTACT = None", "GECC_CONTACT = 'TC Chakraborty (tirthankar.chakraborty@pnnl.gov)'")
code = code.replace("author='GECC'", "author='TC Chakraborty; Rohit Mukherjee'")
code = code.replace("if not t['optional_table']}", "if not t['optional_table'] and 'era5land' not in t['product_id']}")
code = code.replace("families = json.loads((META / 'VARIABLE_FAMILIES.json').read_text())", "families = [f for f in json.loads((META / 'VARIABLE_FAMILIES.json').read_text()) if 'era5land' not in f['table']]")
code = code.replace("assert (PACKAGE / filename).is_file(), filename", "assert (ROOT / 'stakeholder_packages/census_tracts_20261018' / filename).is_file(), filename")
code = code.replace(", 'temperature_era5land',", ",")
code = code.replace("    for tid, citation in citations.items():\n", "    for tid, citation in citations.items():\n        if tid not in tables: continue\n")
code = code.replace("STYLES = {", "PRODUCTS.pop('temperature_era5land', None)\n\nSTYLES = {")
code = code.replace('all 8,252 CSV columns', f'all {len(dictionary) - len(glc_affected.DICTIONARY_ROWS):,} data-table CSV columns')
code = code.replace('4 source calendars, the column dictionary and import schema', '4 source calendars, the column dictionary and the GLC tile list')
code = code.replace('metadata/IMPORT_SCHEMA.json gives field types, units and missing-value rules.', 'The column dictionary also gives field types, units and missing-value rules.')
old_formats = 'The data are supplied as 32 CSV tables and 2 GeoJSON boundary files. To map a table, join it to a boundary file by GEOID or piece_id. GeoJSON copies of the 12 metric tables are an optional separate download. They repeat the CSV attributes and add geometry, so choose one format for each analysis to avoid counting the same rows twice.'
new_formats = 'The data are supplied in 13 ZIP downloads: 12 metric archives and shared_geography.zip. They contain 32 CSV data tables and 2 GeoJSON boundary files. Each metric archive includes its measurement-support table and any applicable calendar; ndvi_evi_series.zip also includes the static Landsat support table. Extract the archives into one common folder, preserving their internal metrics/, support/, metadata/ and geography/ paths. To map a table, join it to a boundary file by GEOID or piece_id. The column dictionary, GLC tile list, guides, README, file guide, file inventory and release notes are separate downloads.'
assert old_formats in code
code = code.replace(old_formats, new_formats)
code = code.replace("p(f'{kind}  |  Release {RELEASE}', 'subtitle')", "p(f'{kind}  |  Release {RELEASE}', 'subtitle'), p('TC Chakraborty; Rohit Mukherjee.', 'small')")
code = code.replace("    return out\n\ndef geography_text", "    out.append(labelled('Acknowledgement.', 'This work was funded as a pilot project under the Gateway Exposome Coordinating Center (GECC). This work used analytical resources developed and disseminated by the Gateway Exposome Coordinating Center (GECC) (https://gatewayexposome.org/). The GECC is funded by the National Institute on Aging (NIA), under award U24AG088894.'))\n    return out\n\ndef geography_text")
code = code.replace("CSV: ' + filename", "CSV after extraction: ' + filename")
code = code.replace(".replace('; ', '. ')", ".replace('; ', '. ').replace('(NIA. U24AG088894)', '(NIA; U24AG088894)')")
builder = HERE / 'build_selected_documents.py'
code = editorial.patch_source(code)
code = code.replace("h('3 Geography and shared fields')", "p(researcher.TYPE_GUIDANCE), CondPageBreak(250), h('3 Geography and shared fields')")
code = code.replace("FILE_GUIDE.md lists the files in the collection.')", "FILE_GUIDE.md lists the files in the collection.'), p(researcher.TYPE_GUIDANCE_SHORT)")
code = code.replace("covered = set()", "import researcher_documentation as researcher\ncovered = set()")
old_gridmet_limit = 'A shared analysis scale does not make their estimates interchangeable.'
assert code.count(old_gridmet_limit) == 1
code = code.replace(old_gridmet_limit, old_gridmet_limit + ' ' + GRIDMET_2011_2013_LIMIT.replace("'", "\\'"))
NYC_SVCM_LIMITS = {
    'Some Pacific areas have no source data. DJF omits December 31 in leap years.':
        ' SVCM-SP source cells are masked over parts of New York City. 222 New York State tracts have no SVCM-SP temperature in 2020: 216 in New York City, including 186 of 310 Manhattan tracts (mainly Midtown, Lower Manhattan and the Upper West Side), five in Nassau County and one in Westchester County. 219 of them have no SVCM-SP temperature in any year. Daymet, GRIDMET and PRISM provide values for these tracts.',
    'Pacific source coverage is unavailable.':
        ' The same SVCM-SP source mask leaves the 222 New York State tracts without 2020 extreme-day counts (support_status source_no_common_seasonal_support). Nine further tracts, eight in Brooklyn and one on Staten Island, have no 2020 counts because they lack baseline-threshold support (support_status partial_band_threshold_support). 228 New York tracts have no counts in any year.',
    'Pixel areas use a spherical calculation and can differ from projected polygon areas.':
        ' The SVCM-SP source mask over parts of New York City leaves 175 of the 216 overlaps of the 222 masked tracts without an urban temperature for 2020 JJA maximum temperature; quality_status (urban_no_valid_temperature) identifies them. Puerto Rico has no urban heat island values in any year. Its 987 overlaps lie inside the SVCM-SP North America archive used for this table, which has no valid cells over Puerto Rico, and values from the separate Latin America archive were not combined with it; quality_status identifies these overlaps. Whole-tract SVCM-SP temperature and extreme days for Puerto Rico come from the Latin America archive.',
}
NYC_SVCM_LIMITS['The tree and forest class fraction differs from cover calculated from canopy height.'] = ' ' + GLC_ANNUAL_LIMIT.replace("'", "\\'")
NYC_SVCM_LIMITS['Class groups and denominators differ from the whole-tract GLC product. Check missing values and flags for very small overlaps.'] = ' ' + GLC_PIECES_LIMIT.replace("'", "\\'")
NYC_SVCM_LIMITS['The daily time window is unspecified.'] = ' Extreme-day counts use only cells with a baseline threshold, whereas the valid-day fields also include cells without one. In 71,485 tract-seasons in 4,115 tracts the two supports differ, and a count can exceed the valid-day field; the matched valid-day bounds use the same support as the counts.'
for old_limit, extra in NYC_SVCM_LIMITS.items():
    assert code.count(old_limit) == 1, old_limit
    code = code.replace(old_limit, old_limit + extra)
assert code.count('These values are retained as produced.') == 1
code = code.replace('These values are retained as produced.', 'These values have not been corrected.')
# Wording shared by the PDFs and FILE_GUIDE: old text -> (new text, occurrences expected).
PIECE_SUBSET = ('The table has 75,383 overlaps. The 527 overlaps in Hawaii (413), Guam (52), the U.S. Virgin Islands (31), the Northern Mariana Islands (17) and American Samoa (14) are absent rather than flagged: '
    'this table was produced for 2,622 Urban Areas, before the 22 Urban Areas in Hawaii and the Island Areas were added for NDVI and EVI. To keep every overlap, left-join this table to the overlap geography by piece_id.')
UHI_2023_FILTER = ('For 2023 only, source values outside -95 to 65 °C were removed before averaging. In other years no out-of-range source value falls inside an overlap or rural zone, so the rule would not change them. '
    'In 2023 it removed 19 daily values in 19 cells near Bethel, Alaska, affecting the overlap of tract 02050000200; all other 2023 values are comparable with other years.')
DOC_REPLACEMENTS = {
    'The table has 75,383 overlaps.': (PIECE_SUBSET, 2),
    'The 2023 temperature range filter differs from other years.': (UHI_2023_FILTER, 1),
    'Stored values are divided by 10.': ('The source stores LAI multiplied by 10; the CSV values are already divided by 10 and need no further scaling.', 1),
}
for old, (new, n) in DOC_REPLACEMENTS.items():
    assert code.count(old) == n, old
    code = code.replace(old, new.replace("'", "\\'"))
CODE_ONLY = {
    'UHI and GLC vegetation use 75,383.': 'UHI and GLC vegetation use 75,383: they omit the 527 overlaps in Hawaii and the Island Areas, which are absent from those tables rather than flagged.',
    "'overlap_share': 'Overlap polygon area divided by the parent tract polygon area.',":
        "'overlap_share': 'Overlap polygon area divided by the parent tract polygon area." + OVERLAP_SHARE_NOTE + "',",
    "GECC_CITATION = None": "GECC_CITATION = 'TC Chakraborty; Rohit Mukherjee. GECC Census Tract Environmental Data, release " + RELEASE + ". Cite the released version.'",
}
for old, new in CODE_ONLY.items():
    assert code.count(old) == 1, old
    code = code.replace(old, new)
GRIDMET_PRISM_METHOD = ('The analysis uses a 1,000 m scale. Earth Engine calculates the mean, median and standard deviation with its default mask and coverage weights. '
    'Counts use cell-centre inclusion and are integers. They are not the weight totals used for the statistics. Medians can use histogram approximation for large sample sets.')
SPLIT_PRODUCTS = {
 'temperature_gridmet': {
  'period': '1980-2025. Annual, JJA and same-year DJF.',
  'meaning': 'Tract summaries of period-average daily maximum and minimum air temperature from GRIDMET for the contiguous United States.',
  'method': 'The GRIDMET source grid is 1/24 degree (about 4 km). ' + GRIDMET_PRISM_METHOD,
  'limit': 'GRIDMET and PRISM use different source resolutions and methods, and the shared 1,000 m analysis scale does not make their estimates interchangeable. ' + GRIDMET_2011_2013_LIMIT,
 },
 'temperature_prism': {
  'period': '1981-2025. Annual, JJA and same-year DJF.',
  'meaning': 'Tract summaries of period-average daily maximum and minimum air temperature from PRISM for the contiguous United States.',
  'method': 'The PRISM source grid is 30 arc-seconds (the PRISM 800 m daily product). ' + GRIDMET_PRISM_METHOD,
  'limit': 'PRISM and GRIDMET use different source resolutions and methods, and the shared 1,000 m analysis scale does not make their estimates interchangeable.',
 },
}
block = re.search(r" 'temperature_gridmet_prism': \{\n.*?\n \},\n", code, flags=re.S)
assert block and block.group(0).count("'limit'") == 1
code = code.replace(block.group(0), ''.join(f" {tid!r}: {entry!r},\n" for tid, entry in SPLIT_PRODUCTS.items()))
split_code = {
    "    'temperature_gridmet_prism': 'GRIDMET and PRISM air temperature',\n":
        "    'temperature_gridmet': 'GRIDMET air temperature',\n    'temperature_prism': 'PRISM air temperature',\n",
    "['temperature_svcm', 'temperature_gridmet_prism', 'temperature_daymet', 'extreme_days']":
        "['temperature_svcm', 'temperature_gridmet', 'temperature_prism', 'temperature_daymet', 'extreme_days']",
    "if 'gridmet_prism' in tid else": "if tid.startswith(('temperature_gridmet', 'temperature_prism')) else",
    "        'temperature_gridmet_prism': ('Abatzoglou": "        'temperature_gridmet': ('Abatzoglou",
    "Earth Engine IDAHO_EPSCOR/GRIDMET.\\n'\n                                      'PRISM Group":
        "Earth Engine IDAHO_EPSCOR/GRIDMET.'),\n        'temperature_prism': ('PRISM Group",
    "assert (ROOT / 'stakeholder_packages/census_tracts_20261018' / filename).is_file(), filename":
        "assert split.source_path(filename).is_file(), filename",
    "        for k in ['dataset', 'source', 'measure']:":
        "        for k in (['source', 'measure'] if f['table'].startswith(('temperature_gridmet', 'temperature_prism')) else ['dataset', 'source', 'measure']):",
    "import researcher_documentation as researcher\ncovered = set()":
        "import researcher_documentation as researcher\nimport split_gridmet_prism as split\ncovered = set()",
}
for old, new in split_code.items():
    assert code.count(old) == 1, old
    code = code.replace(old, new)
assert 'temperature_gridmet_prism' not in code
builder.write_text(code)
spec = importlib.util.spec_from_file_location('selected_docs', builder)
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)
researcher.configure(docs)
docs.build_codebook()
docs.build_guide()
assert docs.covered == {(f['table'], f['family']) for f in docs.families}

LICENSE_TEXT = 'This collection is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). Source products keep their own terms; cite them as listed in codebook section 10.'
ack = 'This work was funded as a pilot project under the Gateway Exposome Coordinating Center (GECC). This work used analytical resources developed and disseminated by the Gateway Exposome Coordinating Center (GECC) (https://gatewayexposome.org/). The GECC is funded by the National Institute on Aging (NIA), under award U24AG088894.'
ORDER = ['temperature_svcm', 'temperature_gridmet', 'temperature_prism', 'temperature_daymet', 'extreme_days', 'land_cover_glc', 'ndvi_evi_series',
         'ndvi_evi_ua_pieces', 'vegetation_ua_pieces', 'uhi_ua_pieces', 'canopy_water_excluded', 'lai_water_excluded']
assert sorted(ORDER) == sorted(metrics)
table_lines = []
for tid in ORDER:
    row = next(r for r in manifest if r['table'] == tid)
    scope = '83,776 CONUS tracts' if tid in ('temperature_gridmet', 'temperature_prism') else (f"{int(row['rows']):,} overlaps" if 'pieces' in tid else '85,528 tract rows')
    table_lines.append(f"| [{tid}.zip]({tid}.zip) | {docs.TITLES[tid]} | {docs.PRODUCTS[tid]['period']} | {scope} |")
readme = '''# GECC Census Tract Environmental Data

United States and territories, 2020 boundaries. Release '''+RELEASE+'''.

Authors: TC Chakraborty; Rohit Mukherjee. Contact: TC Chakraborty (tirthankar.chakraborty@pnnl.gov).

This collection contains 12 environmental metric tables covering air temperature, extreme temperature days, land cover, vegetation indices, urban heat island, canopy height and leaf area index. Observation periods range from 1980 to 2025 and differ by metric. The geography includes the 50 states, Washington, DC, Puerto Rico, American Samoa, Guam, the Northern Mariana Islands and the U.S. Virgin Islands. Source coverage varies; a geographic row does not guarantee a valid measurement.

## Start here

- [Collection guide](documentation/GECC_CensusTracts_Collection_Guide.pdf): choose a metric and understand coverage.
- [Codebook](documentation/GECC_CensusTracts_Codebook.pdf): methods, definitions, limitations and source references.
- [Column dictionary](metadata/COLUMN_DICTIONARY.csv): every field name, type, unit and missing-value rule.
- [GLC tile list](metadata/GLC_ANNUAL_TILE_AFFECTED.csv): tracts and overlaps affected by the GLC-FCS30D annual-tile issue; filter its table column before joining.
- [File guide](FILE_GUIDE.md): archive contents and individual table descriptions.
- [File inventory](FILE_MANIFEST.csv): archive membership, extracted paths, row counts, keys and file sizes.
- [Release notes](RELEASE_NOTES.md): release contents and known source issues.

## Downloads

The collection has 21 downloads: 12 metric ZIP archives, one shared-geography ZIP, two PDF guides, the column dictionary, the GLC tile list, three Markdown documents and the file inventory. The archives contain 34 data files: 32 CSV tables and two GeoJSON boundary files. Extracted together with the separate documentation, the collection contains 42 files.

| Download | Metric | Observation period | Row coverage |
| --- | --- | --- | --- |
'''+ '\n'.join(table_lines) + '''

Each metric ZIP contains its main CSV and measurement-support CSV. GRIDMET, PRISM, Daymet and UHI archives also contain their source calendars. The NDVI/EVI series archive includes the static Landsat analysis-scale and water-flag table.

[shared_geography.zip](shared_geography.zip) contains five files: tract attributes and polygons, overlap attributes and polygons, and Urban Area names. The master geography contains 85,528 tracts and 75,910 tract-Urban Area overlaps. UHI and GLC vegetation contrasts use a subset of 75,383 overlaps. The other 527 overlaps, in Hawaii (413), Guam (52), the U.S. Virgin Islands (31), the Northern Mariana Islands (17) and American Samoa (14), are absent from those two tables rather than flagged: they were produced for 2,622 Urban Areas, before the 22 Urban Areas in Hawaii and the Island Areas were added for NDVI and EVI. To keep the full overlap geography, left-join those tables to it by piece_id.

## Extract and join

Extract the required metric ZIPs and shared_geography.zip into the same directory. Preserve the internal metrics/, support/, geography/ and metadata/ folders. The filenames and paths in the codebook and file inventory refer to files after extraction. Individual CSVs remain usable without extracting unrelated metrics.

- Whole-tract tables join to geography/census_tracts_2020.csv or its GeoJSON using GEOID.
- Overlap tables join to geography/tract_urban_area_pieces_2020.csv or its GeoJSON using piece_id. Retain GEOID and GEOID20 to identify the parent tract and Urban Area.
- Join each measurement-support table to its metric using the same key. Calendar rows describe source periods; use the complete key listed in FILE_MANIFEST.csv.
- Read GEOID, GEOID20 and FIPS codes as text to preserve leading zeros. piece_id is an integer. Import CSVs using explicit column types rather than opening them with automatic numeric conversion in a spreadsheet.
- Some tables share column names. For example, temperature_svcm.csv and temperature_daymet.csv share 132 non-key names, including tmax_mean_c_JJA_2020. Before joining such tables, prefix or rename their non-key columns so each value keeps its source.

```python
import pandas as pd
svcm = pd.read_csv('metrics/temperature_svcm.csv', dtype={'GEOID': str}).set_index('GEOID').add_prefix('svcm_')
daymet = pd.read_csv('metrics/temperature_daymet.csv', dtype={'GEOID': str}).set_index('GEOID').add_prefix('daymet_')
tracts = pd.read_csv('geography/census_tracts_2020.csv', dtype={'GEOID': str, 'StFIPS': str, 'CoFIPS': str, 'TrFIPS': str}).set_index('GEOID')
joined = tracts.join(svcm, how='left').join(daymet, how='left')
```

A tract may overlap more than one Urban Area. Overlap measurements describe that intersection and must not be interpreted as whole-tract estimates. Rural references repeat for overlaps in the same Urban Area and are not independent rural observations.

## Missing values and interpretation

CSV text uses UTF-8. Empty numeric cells mean unavailable data; preserve them as missing. Zero is valid unless a quality flag explicitly withholds the estimate. An empty missing_dates value in a source calendar means that no dates are missing. The column dictionary supplies field-specific types and rules.

Years and seasons appear in column names. JJA means June-August. Same-year DJF means January, February and December of the labelled year. Read the calendars for actual source-day support. The 2020 boundary vintage does not identify the acquisition year of every source product.

Use estimates with their flags and support tables. For whole-tract SVCM-SP temperature, sampling cells are included only when the tract covers at least 1/255 of the cell area (approximately 0.392%). The SVCM-SP source is masked over parts of New York City, so 222 New York State tracts (216 in New York City, 186 of them in Manhattan, plus five in Nassau County and one in Westchester County) have no 2020 SVCM-SP temperature or extreme-day counts, and 175 of their 216 overlaps have no 2020 urban heat island value; Daymet, GRIDMET and PRISM cover them. Nine further Brooklyn and Staten Island tracts have no 2020 extreme-day counts because they lack baseline-threshold support. Puerto Rico has no urban heat island values because the SVCM-SP archive used for that table has no valid cells there; whole-tract SVCM-SP values for Puerto Rico come from a separate regional archive. GRIDMET and PRISM are CONUS-only. LAI statistics are withheld for 562 tracts without sampled GLC support. Landsat indices retain an unadjusted sensor-related step during 2013-2017. GRIDMET 2011-2013 maximum temperature is warmer, and minimum temperature slightly cooler, than PRISM, Daymet and station observations; the values have not been corrected (see the codebook). Annual GLC-FCS30D source tiles classify most urban land as vegetation or bare land from 2000 to 2022 in 6,424 tracts in southern California and Nevada; metadata/GLC_ANNUAL_TILE_AFFECTED.csv lists them and their overlaps (see the codebook). Canopy and LAI summaries exclude known permanent water but retain pixels with unknown WorldCover coverage, identified by flags. Read the codebook for each metric's coverage, weighting, thresholds and other limitations.

## Citation and attribution

Cite the released version of this collection: TC Chakraborty; Rohit Mukherjee. GECC Census Tract Environmental Data, release '''+RELEASE+'''. Also cite the source products listed in codebook section 10.

## License

This collection is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). Source products keep their own terms; cite them as listed in codebook section 10.

## Acknowledgment

'''+ack+'\n'
readme = researcher.us_text(readme).replace('## Missing values and interpretation', '## Column types\n\n'+researcher.TYPE_GUIDANCE+'\n\n## Missing values and interpretation')
(DEST / 'README.md').write_text(readme)
guide = (SOURCE / 'FILE_GUIDE.md').read_text()
guide = re.sub(r'### ERA5 Land air temperature\n.*?(?=### Daymet air temperature)', '', guide, flags=re.S)
guide = '\n'.join(line for line in guide.splitlines() if not any(x in line.lower() for x in ['era5land','import_schema','sha256','checksum']))+'\n'
guide = re.sub(r'This inventory lists.*?boundary files\.', 'This inventory describes 34 data files inside 13 ZIP archives. The archives contain 32 CSV tables and 2 GeoJSON boundary files. The collection has 21 downloads; after extraction, it contains 42 files including documentation.', guide)
guide = re.sub(r'GeoJSON copies of the 12 metric tables.*?geography/`\.', 'Extract the archives into a common folder, preserving their internal paths. Paths below refer to extracted files. The file inventory maps each path to its archive.', guide)
guide = guide.replace('all 43 files','all extracted files').replace('8,252',f'{len(dictionary):,}').replace('_20261018.pdf','.pdf')
guide = guide.replace('## Calendars and import metadata','## Calendars and column definitions')
dict_row = re.search(r'^\| \[metadata/COLUMN_DICTIONARY\.csv\].*$', guide, flags=re.M)
assert dict_row
guide = guide.replace(dict_row.group(0), dict_row.group(0) + '\n| [metadata/GLC_ANNUAL_TILE_AFFECTED.csv](metadata/GLC_ANNUAL_TILE_AFFECTED.csv) | Tracts (land_cover_glc rows) and overlaps (vegetation_ua_pieces rows) affected by the GLC-FCS30D annual-tile issue in southern California and Nevada. ' + GLC_LIST_RULE + ' **12,804 rows. 5 columns.** |')
guide = guide.replace('Release 2026.10.18.', 'Release '+RELEASE+'.')
# Direct links to unextracted data point at the owning archive, with the CSV path as label.
for path, archive in path_archive.items():
    guide = guide.replace(']('+path+')', ']('+archive+')')
guide = guide.replace('for Canopy height', 'for canopy height').replace('for Leaf area', 'for leaf area').replace('for Whole tract', 'for whole tract').replace('for Urban heat', 'for urban heat')
guide = guide.replace('Values are sampled on a geographic grid with a 1,000 m step at the equator.', 'Values are sampled on a geographic grid with a 1,000 m step at the equator. Sampling cells are included only when the tract covers at least 1/255 of the cell area (approximately 0.392%).')
# One inventory row and one methods section per temperature product.
def guide_row(row):
    tid = row['table']
    if row['role'] == 'main_metric': text = docs.PRODUCTS[tid]['meaning']
    elif row['role'] == 'measurement_support': text = f"Counts, area or coverage for {docs.TITLES[row['parent']]}. Join to the main table using `GEOID`."
    else: text = 'Source period labels and available day counts. These rows describe periods, not tracts.'
    return f"| [{row['path']}]({path_archive[row['path']]}) | {text} **{int(row['rows']):,} rows. {int(row['columns']):,} attribute columns. Key `{row['primary_key']}`.** |"
for old_path in split.OLD_FILES.values():
    line = re.search(r'^\| \[' + re.escape(old_path) + r'\].*$', guide, flags=re.M)
    assert line, old_path
    new_rows = [r for r in manifest if split.SPLIT_FILES.get(r['path'], (None, None))[1] == split.suffix_of(Path(old_path).stem)]
    assert len(new_rows) == 2
    guide = guide.replace(line.group(0), '\n'.join(guide_row(r) for r in new_rows))
section = re.search(r'### GRIDMET and PRISM air temperature\n.*?(?=### )', guide, flags=re.S)
assert section
guide = guide.replace(section.group(0), ''.join(
    f"### {docs.TITLES[tid]}\n\n{product['meaning']}\n\n**Observation period.** {product['period']}\n\n**Method.** {product['method']}\n\n**Limitations.** {product['limit']}\n\n"
    for tid, product in docs.PRODUCTS.items() if tid in split.PRODUCTS.values()))
assert 'temperature_gridmet_prism' not in guide
guide_limits = {}
guide_limits.update({old: extra.replace("\\'", "'") for old, extra in NYC_SVCM_LIMITS.items()})
for old_limit, extra in guide_limits.items():
    assert guide.count(old_limit) == 1, old_limit
    guide = guide.replace(old_limit, old_limit + extra)
assert guide.count('These values are retained as produced.') == 1
guide = guide.replace('These values are retained as produced.', 'These values have not been corrected.')
for old, (new, n) in DOC_REPLACEMENTS.items():
    assert guide.count(old) == n, old
    guide = guide.replace(old, new)
# Inventory rows follow the codebook order of metrics.
def rank(line):
    stem = Path(re.match(r'\| \[[^/]+/([^\]]+)\.csv\]', line).group(1)).name
    return min(i for i, tid in enumerate(ORDER + ['ndvi_evi_support']) if stem == tid or stem.startswith(tid + '_') or (tid == 'ndvi_evi_series' and stem == 'ndvi_evi_support'))
for prefix in ('| [metrics/', '| [support/'):
    lines = guide.split('\n')
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert idx == list(range(idx[0], idx[-1] + 1))
    lines[idx[0]:idx[-1] + 1] = sorted((lines[i] for i in idx), key=rank)
    guide = '\n'.join(lines)
guide += '\n## Authors, license and acknowledgment\n\nTC Chakraborty; Rohit Mukherjee. Contact: TC Chakraborty (tirthankar.chakraborty@pnnl.gov).\n\n'+LICENSE_TEXT+'\n\n'+ack+'\n'
(DEST / 'FILE_GUIDE.md').write_text(researcher.us_text(guide))
(DEST / 'RELEASE_NOTES.md').write_text('''# Release notes

## Release '''+RELEASE+'''

This release contains 12 environmental metric tables for 2020 Census tracts and tract-Urban Area overlaps in the United States and territories. Each metric is a separate ZIP download with its measurement-support table and, where applicable, a source calendar. Boundaries and Urban Area names are in shared_geography.zip. Observation periods span 1980-2025 and vary by metric.

## Known source issues

- LAI mean, median and standard deviation are withheld for 562 tracts in Hawaii, Guam, the Northern Mariana Islands and American Samoa without sampled GLC support. The lai_source_status field identifies them.
- The SVCM-SP source is masked over parts of New York City. 222 New York State tracts have no 2020 SVCM-SP temperature or extreme-day counts: 216 in New York City (186 in Manhattan), five in Nassau County and one in Westchester County. 175 of their 216 urban heat island overlaps have no 2020 urban temperature. Nine further Brooklyn and Staten Island tracts have no 2020 extreme-day counts because they lack baseline-threshold support. Daymet, GRIDMET and PRISM cover these tracts.
- Puerto Rico has no urban heat island values in any year. The SVCM-SP North America archive used for that table has no valid cells over Puerto Rico, and the separate Latin America archive was not combined with it. Whole-tract SVCM-SP temperature and extreme days for Puerto Rico come from the Latin America archive.
- GRIDMET maximum temperature for 2011-2013 is warmer, and minimum temperature slightly cooler, than PRISM, Daymet and weather-station observations. The same values are in the GRIDMET source files and have not been corrected. The affected GRIDMET field definitions in the column dictionary note the difference.
- The annual GLC-FCS30D source tiles W120N35 and W120N40 classify most urban land as vegetation or bare land from 2000 to 2022. The 2000-2022 GLC fractions for 6,424 tracts in southern California and Nevada and the 2020 urban-rural vegetation fractions for their 6,380 overlaps are too high; metadata/GLC_ANNUAL_TILE_AFFECTED.csv lists them. The values are as in the Earth Engine copy of the source and have not been corrected.
- The UHI and GLC vegetation overlap tables omit the 527 overlaps in Hawaii and the Island Areas (Guam, the U.S. Virgin Islands, the Northern Mariana Islands and American Samoa). These rows are absent rather than flagged; NDVI and EVI overlaps cover all 75,910.
- Landsat sensors were not adjusted to a common scale. A sensor-related step during 2013-2017 must not be read as vegetation change.

Other source gaps, quality flags and methods are described in the codebook.

## License

'''+LICENSE_TEXT+'''

## Authors and acknowledgment

TC Chakraborty; Rohit Mukherjee. Contact: TC Chakraborty (tirthankar.chakraborty@pnnl.gov).

'''+ack+'\n')

for hidden_file in DEST.rglob('.DS_Store'):
    hidden_file.unlink()

def compress(item):
    name, members = item
    target = DEST / name
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for member in members:
            z.write(split.source_path(member), member)
    if target.stat().st_size >= 2_000_000_000:
        raise ValueError(f'Archive needs partitioning: {name}')
    print(f'PACKAGED {name}: {target.stat().st_size:,} bytes', flush=True)
    return {'archive': name, 'members': members, 'bytes': target.stat().st_size}

with ThreadPoolExecutor(max_workers=3) as pool:
    packaged = list(pool.map(compress, archives.items()))
(HERE / 'BUILD_REPORT.json').write_text(json.dumps({'upload_folder': str(DEST), 'metric_tables': len(metrics), 'data_files': len(manifest), 'dictionary_fields': len(dictionary), 'variable_families': len(docs.families), 'covered_families': len(docs.covered), 'archives': packaged}, indent=2)+'\n')
print('BUILD COMPLETE', flush=True)
