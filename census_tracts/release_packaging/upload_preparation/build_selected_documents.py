"""Build the public collection guide and codebook. Adapted from documentation_revision/build_review_documents.py."""
from pathlib import Path
from collections import defaultdict
from functools import partial
from xml.sax.saxutils import escape
import ast
import csv
import hashlib
import json
import re

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, CondPageBreak,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen.canvas import Canvas

ROOT = Path('/path/to/GECC')
META = Path('/path/to/GECC/reports/upload_preparation_2026-10-30/split_gridmet_prism_metadata')
PACKAGE = ROOT / 'uploadGECC/census_tracts_12_metrics_20261030'
OUT = PACKAGE / 'documentation'
HERE = Path(__file__).resolve().parent
RELEASE = '1.0'
STAMP = '20261018'
# Set these before public release. Sections that need them are omitted while they are None.
GECC_CITATION = 'TC Chakraborty; Rohit Mukherjee. GECC Census Tract Environmental Data, release 1.0. Cite the released version.'   # e.g. 'Author A, Author B (2026). GECC Census Tract Environmental Data, release 2026.10.18. DOI ...'
GECC_LICENCE = 'CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)'    # e.g. 'CC BY 4.0'
GECC_CONTACT = 'TC Chakraborty (tirthankar.chakraborty@pnnl.gov)'    # e.g. 'Name, institution, email'
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path('/System/Library/Fonts/Supplemental')
for name, filename in [('Arial', 'Arial.ttf'), ('Arial-Bold', 'Arial Bold.ttf'),
                       ('Arial-Italic', 'Arial Italic.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-Bold')

plan = json.loads((META / 'DELIVERY_PLAN.json').read_text())
tables = {t['product_id']: t for t in plan['tables'] if not t['optional_table'] and 'era5land' not in t['product_id']}
families = [f for f in json.loads((META / 'VARIABLE_FAMILIES.json').read_text()) if 'era5land' not in f['table']]
profiles = json.loads((META / 'PRODUCT_METHODS.json').read_text())
dictionary = list(csv.DictReader((PACKAGE / 'metadata/COLUMN_DICTIONARY.csv').open()))
summaries = list(csv.DictReader((META / 'FIELD_SUMMARY_STATISTICS.csv').open()))
categories = list(csv.DictReader((META / 'CATEGORICAL_COUNTS.csv').open()))
mapping = {(r['table'], r['field']): r for r in dictionary}
by_family = defaultdict(list)
by_category = defaultdict(list)
for r in summaries:
    by_family[(r['table'], r['family'])].append(r)
for r in categories:
    by_category[(r['table'], r['field'])].append(r)
LABELS = {}
for node in ast.parse((ROOT / 'reports/submission_readiness_2026-10-15/review_page/build_page.py').read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'LABELS' for x in node.targets):
        LABELS = ast.literal_eval(node.value)

TITLES = {
    'tract_geography': 'Census tract boundaries and attributes',
    'ua_registry': 'Urban Area names',
    'piece_registry': 'Tract Urban Area overlap table',
    'ndvi_evi_support': 'Landsat analysis grid and water flags',
    'temperature_svcm': 'SVCM SP seasonal air temperature',
    'temperature_gridmet': 'GRIDMET air temperature',
    'temperature_prism': 'PRISM air temperature',
    'temperature_era5land': 'ERA5 Land air temperature',
    'temperature_daymet': 'Daymet air temperature',
    'extreme_days': 'SVCM SP extreme temperature days',
    'land_cover_glc': 'GLC tract land cover',
    'ndvi_evi_series': 'Whole tract Landsat NDVI and EVI',
    'ndvi_evi_ua_pieces': 'NDVI and EVI with a rural reference inside the Urban Area',
    'vegetation_ua_pieces': 'GLC vegetation with a rural reference inside the Urban Area',
    'uhi_ua_pieces': 'Urban heat island with a rural reference inside the Urban Area',
    'canopy_water_excluded': 'Canopy height with permanent water removed',
    'lai_water_excluded': 'Leaf area index with permanent water removed',
}
for tid in tables:
    if tid not in TITLES:
        suffix = '_measurement_support' if tid.endswith('_measurement_support') else '_calendar'
        TITLES[tid] = TITLES[tid.removesuffix(suffix)] + (' measurement support' if suffix == '_measurement_support' else ' calendar')

# Product-specific facts retain the original definitions and calculation rules.
# Editorial and audit history is intentionally excluded from these reading copies.
PRODUCTS = {
 'temperature_svcm': {
  'period': '2003-2024. JJA and same-year DJF.',
  'meaning': 'Seasonal summaries of daily maximum and minimum air temperature within each tract.',
  'method': 'SVCM-SP cells are about 927 m wide. Values are sampled on a geographic grid with a 1,000 m step at the equator. The spatial mean uses cell coverage weights. The median and population standard deviation are unweighted. Support counts are spatial samples, not days. Source nodata and daily values outside -95 to 65 °C are removed.',
  'limit': 'Some Pacific areas have no source data. DJF omits December 31 in leap years. SVCM-SP source cells are masked over parts of New York City. 222 New York State tracts have no SVCM-SP temperature in 2020: 216 in New York City, including 186 of 310 Manhattan tracts (mainly Midtown, Lower Manhattan and the Upper West Side), five in Nassau County and one in Westchester County. 219 of them have no SVCM-SP temperature in any year. Daymet, GRIDMET and PRISM provide values for these tracts.',
 },
 'temperature_gridmet': {'period': '1980-2025. Annual, JJA and same-year DJF.', 'meaning': 'Tract summaries of period-average daily maximum and minimum air temperature from GRIDMET for the contiguous United States.', 'method': 'The GRIDMET source grid is 1/24 degree (about 4 km). The analysis uses a 1,000 m scale. Earth Engine calculates the mean, median and standard deviation with its default mask and coverage weights. Counts use cell-centre inclusion and are integers. They are not the weight totals used for the statistics. Medians can use histogram approximation for large sample sets.', 'limit': 'GRIDMET and PRISM use different source resolutions and methods, and the shared 1,000 m analysis scale does not make their estimates interchangeable. GRIDMET maximum temperature for 2011, 2012 and 2013 is warmer than PRISM, Daymet and weather-station observations, and minimum temperature is slightly cooler. For JJA maximum temperature, the median tract difference from Daymet is +0.8 to +1.3 °C, reaching +2 to +3.5 °C in Midwest and Great Plains states. Other years agree within about 0.2 °C. The same values are in the GRIDMET source files, so they have not been corrected. For analyses that include these years, use PRISM or Daymet or test sensitivity to the GRIDMET values.'},
 'temperature_prism': {'period': '1981-2025. Annual, JJA and same-year DJF.', 'meaning': 'Tract summaries of period-average daily maximum and minimum air temperature from PRISM for the contiguous United States.', 'method': 'The PRISM source grid is 30 arc-seconds (the PRISM 800 m daily product). The analysis uses a 1,000 m scale. Earth Engine calculates the mean, median and standard deviation with its default mask and coverage weights. Counts use cell-centre inclusion and are integers. They are not the weight totals used for the statistics. Medians can use histogram approximation for large sample sets.', 'limit': 'PRISM and GRIDMET use different source resolutions and methods, and the shared 1,000 m analysis scale does not make their estimates interchangeable.'},
 'temperature_era5land': {
  'period': '1950-2025. Annual, JJA and same-year DJF. Days use UTC.',
  'meaning': 'Tract summaries of period-average daily maximum and minimum air temperature from ERA5-Land reanalysis.',
  'method': 'The source grid is 0.1 degree, about 11 km. The mean, median, population standard deviation and valid-cell count use cell coverage weights. Counts can be fractional. Earth Engine daily data are combined with Climate Data Store data for January 1950.',
  'limit': 'January 1, 1950 has 23 hours. The period before 1979 has fewer observations. The source grid does not provide detailed temperature differences within a city.',
 },
 'temperature_daymet': {
  'period': '1980-2025. Annual, JJA and same-year DJF.',
  'meaning': 'Tract summaries of period-average daily maximum and minimum air temperature from Daymet V4 R1.',
  'method': 'The source uses a 1 km Lambert conformal conic grid. The mean, median, population standard deviation and valid-cell count use cell coverage weights. Counts can be fractional. Source fill values are excluded. Every source year has 365 days. Leap years include February 29 and omit December 31.',
  'limit': 'American Samoa, Guam and the Northern Mariana Islands have no source coverage. Other source gaps remain missing. In 11 tract-years in four California coastal tracts, mean summer (JJA) daily maximum temperature is lower than same-year winter (DJF). These values have not been corrected.',
 },
 'extreme_days': {
  'period': '2003-2024. JJA and same-year DJF. Reference period 2003-2007.',
  'meaning': 'Average numbers of unusually hot or cold days within each tract. Fractional days are valid.',
  'method': 'For each source cell and season, hot days have daily maximum temperature above the reference 95th percentile. Cold days have daily minimum temperature below the reference 5th percentile. Tract values average these cell counts with sampling-cell coverage weights. They are not counts based on the tract-average daily temperature. Hot and cold measures can have different support. Source nodata is excluded.',
  'limit': 'The reference period can be incomplete. Pacific source coverage is unavailable. The same SVCM-SP source mask leaves the 222 New York State tracts without 2020 extreme-day counts (support_status source_no_common_seasonal_support). Nine further tracts, eight in Brooklyn and one on Staten Island, have no 2020 counts because they lack baseline-threshold support (support_status partial_band_threshold_support). 228 New York tracts have no counts in any year. JJA uses 92 source days and DJF uses 90. Missing December 31 in leap years is not filled. The daily time window is unspecified. Extreme-day counts use only cells with a baseline threshold, whereas the valid-day fields also include cells without one. In 71,485 tract-seasons in 4,115 tracts the two supports differ, and a count can exceed the valid-day field; the matched valid-day bounds use the same support as the counts. These are relative thresholds, not clinical heat or cold thresholds.',
 },
 'land_cover_glc': {
  'period': '1985, 1990, 1995 and 2000-2022. There are 26 snapshots.',
  'meaning': 'Tract fractions of vegetation, tree and forest cover, shrub, grass and crops from GLC-FCS30D.',
  'method': 'Each fraction is its class-group area divided by the total sampled raw-class area. This denominator includes unclassified codes 0 and 250. The group numerators exclude those codes. Missing years are not interpolated.',
  'limit': 'There is no sampled source area for 562 tracts in any snapshot. The tree and forest class fraction differs from cover calculated from canopy height. The annual GLC-FCS30D tiles W120N35 and W120N40 (115-120° W, 30-40° N) classify most urban land as vegetation or bare land in 2000-2022, unlike the 1985-1995 maps. For the 6,424 tracts in southern California and Nevada listed in metadata/GLC_ANNUAL_TILE_AFFECTED.csv, 2000-2022 vegetation fractions are too high and the 1995-2000 change is not real. The values are as in the Earth Engine copy of the source and have not been corrected. Tracts are listed when their interior point lies in the two tiles, so tracts crossing the tile edges may be partly affected. Filter the list by its table column before joining: land_cover_glc rows by GEOID, vegetation_ua_pieces rows by piece_id.',
 },
 'ndvi_evi_series': {
  'period': '1985-2025. JJA and same-year DJF.',
  'meaning': 'Tract summaries of the Normalized Difference Vegetation Index (NDVI) and Enhanced Vegetation Index (EVI). Both describe vegetation greenness.',
  'method': 'Each Landsat composite pixel is the median of valid observations within the season. The table gives the spatial mean, median and standard deviation of those pixels. The analysis scale is 30 m, with 100 m used for some Alaska groups. Earth Engine weights are retained. Spatial medians can be approximate. Cloud, snow, source validity and GLC water filters are applied. Counts refer to composite cells, not satellite scenes.',
  'limit': 'Landsat sensors were not adjusted to a common scale. A sensor-related step occurs during 2013-2017 and must not be read as vegetation change. Early years and some regions have many missing values.',
 },
 'ndvi_evi_ua_pieces': {
  'period': '2020. JJA and same-year DJF.',
  'meaning': 'NDVI and EVI means for each tract overlap and for rural land inside the same Urban Area. Each difference is the overlap value minus the rural reference.',
  'method': 'ESA WorldCover defines the zones. Its 10 m pixel centres sample Landsat composites on a 30 m geographic grid. Means use valid-pixel area weights. The overlap includes built and unbuilt land. Both zones exclude WorldCover codes 0 and 80. The rural zone also excludes built-up class 50. Snow and ice remain.',
  'limit': 'The table has 75,910 overlaps. A rural reference repeats for overlaps in the same Urban Area. Source-cell counts summed across tiles can count a cell more than once.',
 },
 'vegetation_ua_pieces': {
  'period': '2020 snapshot.',
  'meaning': 'GLC vegetation fractions for each tract overlap and for rural land inside the same Urban Area. The difference is the overlap fraction minus the rural reference fraction.',
  'method': 'GLC supplies vegetation classes. ESA WorldCover defines the zones. Both zones exclude WorldCover codes 0 and 80. The rural zone also excludes built-up class 50. GLC codes 0 and 250 are excluded from each denominator. Snow and ice remain. The overlap includes built and unbuilt land.',
  'limit': 'The table has 75,383 overlaps. The 527 overlaps in Hawaii (413), Guam (52), the U.S. Virgin Islands (31), the Northern Mariana Islands (17) and American Samoa (14) are absent rather than flagged: this table was produced for 2,622 Urban Areas, before the 22 Urban Areas in Hawaii and the Island Areas were added for NDVI and EVI. To keep every overlap, left-join this table to the overlap geography by piece_id. Class groups and denominators differ from the whole-tract GLC product. Check missing values and flags for very small overlaps. The 2020 fractions use the same GLC-FCS30D tiles, so urban fractions for the 6,380 overlaps of the listed tracts (89 Urban Areas) are too high; metadata/GLC_ANNUAL_TILE_AFFECTED.csv lists them by piece_id.',
 },
 'uhi_ua_pieces': {
  'period': '2015-2024. JJA and same-year DJF. Daily maximum and minimum temperature.',
  'meaning': 'Urban heat island (UHI) is the tract overlap air temperature minus the rural air temperature inside the same Urban Area.',
  'method': 'SVCM-SP temperature cells are weighted by retained ESA WorldCover pixel area. Both zones exclude unknown cover, water, snow and ice, using codes 0, 80 and 70. The rural zone also excludes built-up class 50. The overlap includes built and unbuilt land. Rural references match the Urban Area, year, season and temperature measure.',
  'limit': 'The table has 75,383 overlaps. The 527 overlaps in Hawaii (413), Guam (52), the U.S. Virgin Islands (31), the Northern Mariana Islands (17) and American Samoa (14) are absent rather than flagged: this table was produced for 2,622 Urban Areas, before the 22 Urban Areas in Hawaii and the Island Areas were added for NDVI and EVI. To keep every overlap, left-join this table to the overlap geography by piece_id. Many have few effective source cells. For 2020 JJA maximum temperature, half have fewer than 3 and 11,952 have fewer than 1. For 2023 only, source values outside -95 to 65 °C were removed before averaging. In other years no out-of-range source value falls inside an overlap or rural zone, so the rule would not change them. In 2023 it removed 19 daily values in 19 cells near Bethel, Alaska, affecting the overlap of tract 02050000200; all other 2023 values are comparable with other years. Pixel areas use a spherical calculation and can differ from projected polygon areas. The SVCM-SP source mask over parts of New York City leaves 175 of the 216 overlaps of the 222 masked tracts without an urban temperature for 2020 JJA maximum temperature; quality_status (urban_no_valid_temperature) identifies them. Puerto Rico has no urban heat island values in any year. Its 987 overlaps lie inside the SVCM-SP North America archive used for this table, which has no valid cells over Puerto Rico, and values from the separate Latin America archive were not combined with it; quality_status identifies these overlaps. Whole-tract SVCM-SP temperature and extreme days for Puerto Rico come from the Latin America archive.',
 },
 'canopy_water_excluded': {
  'period': 'Source product labelled 2020. ESA WorldCover permanent-water mask.',
  'meaning': 'Tract canopy-height statistics and the fraction of retained cover with height above 1 m.',
  'method': 'The Meta/WRI source has an approximate 1 m pixel size. Statistics use fractional tract-cell coverage weights. Means, medians and population standard deviations include valid heights from 0 to 255 m. The positive-height mean includes the 1 m bin. Cover above 1 m uses bins 2-255 divided by all retained valid weight. ESA WorldCover class 80 is removed at source pixel centres. Unknown WorldCover pixels remain and are flagged.',
  'limit': 'An incomplete water mask can leave water in the result. Pixel size describes spatial resolution, not a minimum tree height.',
 },
 'lai_water_excluded': {
  'period': 'Sentinel-2 source labelled 2020, with GLC support and ESA WorldCover water removal.',
  'meaning': 'Tract mean, median and population standard deviation of leaf area index (LAI), with known permanent water removed.',
  'method': 'The source uses Sentinel-2 imagery with GLC support. The source stores LAI multiplied by 10; the CSV values are already divided by 10 and need no further scaling. Statistics use fractional coverage weights on the source geographic grid. These are cell-coverage fractions, not ground-area weights. ESA WorldCover class 80 is removed at source pixel centres. Pixels with unknown WorldCover coverage remain and are flagged.',
  'limit': 'LAI mean, median and standard deviation are withheld for 562 tracts without sampled GLC support. The source reports zero LAI for 537 of these tracts, and these zeros are not treated as valid. The source-status flag identifies the affected tracts. Water-mask status is separate.',
 },
}

PRODUCTS.pop('temperature_era5land', None)

STYLES = {
 'title': ParagraphStyle('title', fontName='Arial-Bold', fontSize=22, leading=26, spaceAfter=9),
 'subtitle': ParagraphStyle('subtitle', fontName='Arial', fontSize=14, leading=18, spaceAfter=16),
 'h1': ParagraphStyle('h1', fontName='Arial-Bold', fontSize=15, leading=19, spaceBefore=15, spaceAfter=8, keepWithNext=True),
 'h2': ParagraphStyle('h2', fontName='Arial-Bold', fontSize=12, leading=15, spaceBefore=12, spaceAfter=6, keepWithNext=True),
 'h3': ParagraphStyle('h3', fontName='Arial-Bold', fontSize=10.5, leading=13, spaceBefore=9, spaceAfter=4, keepWithNext=True),
 'body': ParagraphStyle('body', fontName='Arial', fontSize=10.5, leading=14, spaceAfter=7),
 'small': ParagraphStyle('small', fontName='Arial', fontSize=9, leading=11.5, spaceAfter=5),
 'field': ParagraphStyle('field', fontName='Arial', fontSize=9, leading=11.5, spaceAfter=4, splitLongWords=True),
 'cell': ParagraphStyle('cell', fontName='Arial', fontSize=8.5, leading=11, spaceAfter=0),
 'headcell': ParagraphStyle('headcell', fontName='Arial-Bold', fontSize=8.5, leading=11, spaceAfter=0),
 'number': ParagraphStyle('number', fontName='Arial', fontSize=8, leading=10.5, alignment=TA_RIGHT),
}

def clean(s):
    return str(s).replace('–', '-').replace('—', '-').replace('‑', '-')

def p(s, style='body'):
    return Paragraph(escape(clean(s)).replace('\n', '<br/>'), STYLES[style])

def labelled(label, s):
    return Paragraph('<b>' + escape(label) + '</b> ' + escape(clean(s)), STYLES['body'])

class Document(BaseDocTemplate):
    def __init__(self, filename, kind):
        super().__init__(str(filename), pagesize=(612, 792), leftMargin=50, rightMargin=50,
                         topMargin=49, bottomMargin=48, title='GECC Census Tract Environmental Data ' + kind,
                         author='TC Chakraborty; Rohit Mukherjee', pageCompression=1)
        self.kind = kind
        self.addPageTemplates(PageTemplate(id='page', frames=[Frame(50, 48, 512, 695, leftPadding=0,
            rightPadding=0, topPadding=0, bottomPadding=0)], onPage=self.decorate))
    def decorate(self, canvas, doc):
        canvas.setFont('Arial', 8)
        canvas.drawString(50, 763, 'GECC Census Tract Environmental Data')
        canvas.drawString(50, 27, self.kind)
        canvas.drawRightString(562, 27, str(doc.page))
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ('h1', 'h2'):
            if flowable.getPlainText() == 'Contents':
                return
            level = 0 if flowable.style.name == 'h1' else 1
            key = 'section-' + str(self.seq.nextf('heading'))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level)
            if level == 0:
                self.notify('TOCEntry', (0, flowable.getPlainText(), self.page, key))

def h(s, level=1):
    return p(s, f'h{level}')

def table(headers, rows, widths, numeric=False):
    data = [[p(x, 'headcell') for x in headers]]
    for row in rows:
        data.append([p(v, 'number' if numeric and i > 0 and re.fullmatch(r'[-+\d,.eE]+', str(v)) else 'cell') for i, v in enumerate(row)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E9EDF0')),
        ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#D9D9D9')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return t

def fmt(v):
    if v in ('', None): return '-'
    x = float(v)
    if x.is_integer() and abs(x) < 1e7: return f'{int(x):,}'
    return f'{x:,.4g}'

def years(seq):
    vals = sorted({int(x) for x in seq})
    if not vals: return 'Static or calendar'
    groups = []
    start = last = vals[0]
    for v in vals[1:]:
        if v == last + 1: last = v
        else:
            groups.append(str(start) if start == last else f'{start}-{last}')
            start = last = v
    groups.append(str(start) if start == last else f'{start}-{last}')
    return ', '.join(groups)

def patterns(f):
    result = []
    for col in f['columns']:
        labels = json.loads(mapping[(f['table'], col)]['labels'])
        pattern = col
        for k in ['year', 'observation_year']:
            if k in labels:
                pattern = re.sub(r'(?<!\d)' + str(labels[k]) + r'(?!\d)', '{year}', pattern)
        for k in ['season', 'period']:
            if k in labels: pattern = pattern.replace(str(labels[k]).replace('DJF_same_year', 'DJF'), '{period}')
        for k in (['source', 'measure'] if f['table'].startswith(('temperature_gridmet', 'temperature_prism')) else ['dataset', 'source', 'measure']):
            if k in labels:
                pattern = pattern.replace(str(labels[k]), '{' + k + '}').replace(str(labels[k]).lower(), '{' + k + '}')
        if pattern not in result: result.append(pattern)
    return '\n'.join(result)

def snapshot(f):
    records = by_family[(f['table'], f['family'])]
    if f['table'].startswith('uhi_ua_pieces') and any(r['measure'] == 'TMAX' for r in records):
        records = [r for r in records if r['measure'] == 'TMAX']
    if not records: return []
    if not any(r['year'] for r in records): return records[:1]
    out = []
    groups = defaultdict(list)
    for r in records: groups[(r['source'], r['measure'])].append(r)
    for group in groups.values():
        available = [r for r in group if r['valid_n'] != '0'] or group
        last = max(int(r['year']) for r in available)
        for year in sorted({2020, last}):
            candidates = [r for r in group if int(r['year']) == year]
            if candidates: out.append(next((r for r in candidates if r['period'] == 'JJA'), candidates[0]))
    return out

def context(r):
    return ' '.join(str(r[k]) for k in ['source', 'measure', 'year', 'period'] if r[k]) or 'Snapshot'

def title(f):
    name = f['family']
    simple = {
        'NDVI_stdDev': 'NDVI spatial standard deviation',
        'EVI_stdDev': 'EVI spatial standard deviation',
        'chm_median_m': 'Median canopy height',
        'chm_stdev_m': 'Canopy height spatial standard deviation',
        'urban_ndvi': 'Overlap NDVI', 'rural_ndvi': 'Rural reference NDVI', 'delta_ndvi': 'NDVI difference',
        'urban_evi': 'Overlap EVI', 'rural_evi': 'Rural reference EVI', 'delta_evi': 'EVI difference',
        'urban_temperature_c': 'Overlap air temperature', 'rural_temperature_c': 'Rural reference air temperature',
        'tmax_valid_count': 'Maximum temperature sample count', 'tmin_valid_count': 'Minimum temperature sample count',
        'tmax_count': 'Maximum temperature cell count', 'tmin_count': 'Minimum temperature cell count',
    }
    simple.update({'coverage_pairs': 'Contributing tract–sampling-cell pairs', 'total_coverage_weight': 'Total sampling-cell coverage weight', 'inside_grid_pairs': 'Contributing pairs inside the grid', 'inside_grid_weight': 'Sampling-cell coverage weight inside the grid', 'hot_days_tmax_gt_p95_support_weight': 'Support weight for hot-day estimates', 'cold_days_tmin_lt_p05_support_weight': 'Support weight for cold-day estimates', 'valid_tmax_days_support_weight': 'Support weight for valid maximum-temperature days', 'valid_tmin_days_support_weight': 'Support weight for valid minimum-temperature days', 'baseline_tmax_mean_days': 'Mean valid days in the maximum-temperature baseline', 'baseline_tmax_min_days': 'Minimum valid days in the maximum-temperature baseline', 'baseline_tmax_partial_weight': 'Incomplete maximum-temperature baseline weight', 'baseline_tmax_support_weight': 'Maximum-temperature baseline support weight', 'baseline_tmin_mean_days': 'Mean valid days in the minimum-temperature baseline', 'baseline_tmin_min_days': 'Minimum valid days in the minimum-temperature baseline', 'baseline_tmin_partial_weight': 'Incomplete minimum-temperature baseline weight', 'baseline_tmin_support_weight': 'Minimum-temperature baseline support weight', 'hot_matched_valid_days_bound': 'Matched valid-day bound for hot-day estimates', 'cold_matched_valid_days_bound': 'Matched valid-day bound for cold-day estimates', 'pixel_nonwater_count': 'Non-water composite-cell support count', 'ndvi_urban_valid_coverage': 'Overlap NDVI valid coverage', 'ndvi_rural_valid_coverage': 'Rural reference NDVI valid coverage', 'evi_urban_valid_coverage': 'Overlap EVI valid coverage', 'evi_rural_valid_coverage': 'Rural reference EVI valid coverage', 'expected_days': 'Expected days', 'observed_days': 'Observed days', 'missing_dates': 'Missing dates'})
    if name in simple: return simple[name]
    if name.startswith(('tmax_', 'tmin_')) and name.endswith(('_stddev_c', '_median_c')):
        measure = 'Maximum' if name.startswith('tmax') else 'Minimum'
        statistic = 'spatial standard deviation' if 'stddev' in name else 'spatial median'
        return measure + ' temperature ' + statistic
    return clean(LABELS.get(name, (name.replace('_', ' '),))[0]).replace('Tract-UA piece', 'Tract Urban Area overlap').replace('piece', 'overlap').replace('Piece', 'Overlap')

CALENDAR_DEFS = {
    'dataset': 'Source product for this calendar row.', 'year': 'Year used to label the observation period.',
    'period': 'Annual or seasonal observation period.', 'season': 'Season used for the observation period.',
    'temperature_measure': 'Daily maximum or minimum temperature measure.',
    'source_period_day_count': 'Number of source days in the observation period.',
    'expected_days': 'Number of days expected in the observation period.',
    'observed_days': 'Number of source days available in the observation period.',
    'missing_dates': 'Dates missing from the source period. An empty string means no dates are missing.',
    'period_complete': 'True when the source has all expected days for the period.',
}
DEFS = {
 'GEOID': 'Eleven-character Census tract identifier.',
 'StFIPS': 'Two-character state or territory FIPS code.',
 'CoFIPS': 'Three-character county component of the tract identifier.',
 'TrFIPS': 'Six-character tract component of the tract identifier.',
 'GEOID20': 'Five-character 2020 Census Urban Area identifier.',
 'piece_id': 'Identifier for the area where one tract overlaps one Urban Area.',
 'NAME20': 'Urban Area name.',
 'dataset': 'Source product name.',
 'Area_Land': 'Census land area of the tract. This is separate from the sampled raster area.',
 'Area_Water': 'Census water area of the tract. This is separate from the ESA WorldCover water mask.',
 'water_only_tract': 'True when the Census land area is zero.',
 'analysis_scale_m': 'Analysis scale in metres. It is 30 m outside Alaska and either 30 or 100 m in Alaska.',
 'overlap_share': 'Overlap polygon area divided by the parent tract polygon area. Because overlap and tract areas are computed separately, 18,847 overlaps that cover the whole tract have values slightly above 1 (at most 1.000002). Treat values within 1 + 1e-5 as 1.',
 'piece_area_m2': 'Area of the tract overlap in the source equal-area projection.',
 'tract_area_m2': 'Area of the parent tract used to calculate the overlap share.',
 'sliver_flag': 'Flags a very small tract overlap under the product rule. Flagged rows remain in the table.',
 'quality_status': 'Shows whether an estimate is available and gives the reason for a missing estimate.',
 'lai_source_status': 'glc_2021_unavailable identifies tracts without sampled GLC support. Their LAI statistics are withheld. not_flagged_for_glc_gap means this rule did not apply.',
 'lai_native_mean': 'Coverage-weighted mean LAI after known water removal. Withheld where the source-status flag shows absent GLC support.',
 'chm_mean_m': 'Coverage-weighted mean canopy height after known water removal. Valid heights of 0 and 1 m remain.',
 'chm_mean_gt0_m': 'Mean of retained canopy heights above 0 m. The 1 m height bin is included.',
 'chm_cover_gt1m_frac': 'Weight of retained heights in bins 2-255 m divided by all retained valid weight. The 1 m bin is excluded.',
 'chm_median_m': 'Coverage-weighted spatial median of retained canopy heights, including 0 and 1 m.',
 'chm_stdev_m': 'Coverage-weighted population standard deviation of retained canopy heights, including 0 and 1 m.',
 'vegetation_fraction': 'Fraction of sampled tract area in the GLC vegetation class group.',
 'tree_fraction': 'Fraction in the GLC tree and forest class group. This differs from canopy-height cover.',
 'shrub_fraction': 'Fraction in the GLC shrub class group.',
 'grass_fraction': 'Fraction in the GLC grass class group.',
 'crop_fraction': 'Fraction in the GLC crop class group. Woody-crop class 12 is excluded.',
 'urban_frac': 'GLC vegetation fraction in the tract overlap after the ESA WorldCover water filter. Built and unbuilt land are included.',
 'rural_frac': 'GLC vegetation fraction on non-built-up, non-water land inside the same Urban Area. The reference repeats for all overlaps in that Urban Area.',
 'diff': 'urban_frac minus rural_frac. A positive value means the overlap has a larger vegetation fraction than its rural reference.',
 'water_mask_status': 'Shows whether the water mask is complete, incomplete, all water or without source data.',
 'worldcover_unknown_fraction': 'Share of source support with unknown ESA WorldCover coverage. These pixels remain in the result.',
 'tmax_valid_count': 'Number of valid spatial samples for maximum temperature. This is not a day count.',
 'tmin_valid_count': 'Number of valid spatial samples for minimum temperature. This is not a day count.',
 'EVI_count': 'Number of valid EVI composite cells. EVI can be missing when NDVI is valid.',
 'urban_clear_obs_mean': 'Mean number of clear observations over valid overlap pixels.',
}

def definition(f):
    tid, name = f['table'], f['family']
    if tid.endswith('_calendar'): return CALENDAR_DEFS[name]
    if name in DEFS: return DEFS[name]
    if name in ('tmax_count', 'tmin_count'):
        return ('Number of sampled cells for this temperature measure.' if tid.startswith(('temperature_gridmet', 'temperature_prism')) else
                'Coverage-weighted number of valid source cells for this temperature measure. Values can be fractional.')
    if name in ('baseline_tmax_partial_weight', 'baseline_tmin_partial_weight'):
        return 'Sum of sampling-cell coverage weights for cells with an incomplete reference period. These are not physical areas.'
    if name in ('baseline_tmax_support_weight', 'baseline_tmin_support_weight'):
        return 'Sum of sampling-cell coverage weights for cells with reference-period data. These are not physical areas.'
    if name.startswith(('tmax_', 'tmin_')) and name.endswith(('_mean_c', '_median_c', '_stddev_c')):
        measure = 'maximum' if name.startswith('tmax') else 'minimum'
        statistic = 'mean' if '_mean_' in name else 'median' if '_median_' in name else 'population standard deviation'
        return f'Spatial {statistic} of period-average daily {measure} air temperature within the tract. The product method defines the weights.'
    if name in ('NDVI_mean', 'EVI_mean', 'NDVI_median', 'EVI_median', 'NDVI_stdDev', 'EVI_stdDev'):
        index, statistic = name.split('_', 1)
        statistic = 'standard deviation' if statistic == 'stdDev' else statistic
        return f'Spatial {statistic} of the seasonal Landsat {index} composite over retained non-water pixels.'
    if name in ('urban_ndvi', 'rural_ndvi', 'delta_ndvi', 'urban_evi', 'rural_evi', 'delta_evi'):
        zone, index = name.split('_')
        index = index.upper()
        if zone == 'urban': return f'Area-weighted mean {index} for valid pixels in the tract overlap.'
        if zone == 'rural': return f'Area-weighted mean {index} for rural land inside the same Urban Area. The reference repeats across its overlaps.'
        return f'Tract overlap {index} minus the matching rural reference {index}.'
    return clean(editorial.prose(f['definition'])).strip()

import editorial_cleanup as editorial
DEFS.update(editorial.DEFINITIONS)
CALENDAR_DEFS.update(editorial.CALENDAR)

PRODUCTS['temperature_svcm']['method'] += ' Sampling cells are included only when the tract covers at least 1/255 of the cell area (approximately 0.392%).'
PRODUCTS['temperature_svcm']['guide_note'] = 'Sampling cells are included only when the tract covers at least 1/255 of the cell area (approximately 0.392%).'
import researcher_documentation as researcher
import split_gridmet_prism as split
covered = set()
blocks = []
def family_block(f):
    tid, name = f['table'], f['family']
    covered.add((tid, name))
    if name in {'GEOID', 'piece_id', 'GEOID20'} and tid not in {'tract_geography', 'ua_registry', 'piece_registry'}:
        return []
    records = by_family[(tid, name)]
    typ = records[0]['type'] if records else 'categorical'
    typename = {'continuous': 'Continuous', 'binary': 'Flag', 'categorical': 'Category', 'identifier': 'Identifier', 'period': 'Calendar'}.get(typ, typ)
    units = clean(f['units'])
    year_label = years(f['observed_years']) if f['observed_years'] else ('Calendar' if tid.endswith('_calendar') else 'Static')
    start = [h(title(f), 3), p(patterns(f), 'field'),
             p(f'{year_label}  |  {typename}  |  {units}', 'small'), p(definition(f))]
    sample = snapshot(f)
    result = start
    if typ == 'continuous' and sample:
        rows = [[context(r)] + [fmt(r[k]) for k in ('valid_n','missing_n','mean','sd','min','median','max')] for r in sample]
        result.append(table(['Period', 'N', 'Missing', 'Mean', 'SD', 'Min', 'Median', 'Max'], rows,
                            [113, 45, 45, 62, 62, 62, 62, 61], True))
    elif typ in ('categorical', 'binary') and sample:
        counts = by_category[(tid, sample[0]['field'])]
        if len(counts) <= 12:
            result.append(table(['Period', 'Recorded value', 'Count'], [[context(sample[0]), r['value'], fmt(r['n'])] for r in counts], [128, 310, 74], True))
        else:
            result.append(p(f'{fmt(sample[0]["valid_n"])} nonmissing values and {fmt(sample[0]["missing_n"])} missing values in {context(sample[0])}.', 'small'))
    elif records:
        result.append(p(f'{fmt(records[0]["valid_n"])} nonmissing records and {fmt(records[0]["missing_n"])} missing records.', 'small'))
    result.append(Spacer(1, 5))
    blocks.append({'table': tid, 'family': name, 'pattern': patterns(f), 'definition': definition(f)})
    return [KeepTogether(result)]

def files(tid):
    t = tables[tid]
    filename = next(x for x in t['proposed_files'] if x.endswith('.csv'))
    assert split.source_path(filename).is_file(), filename
    return p('CSV after extraction: ' + filename + '. Primary key: ' + ', '.join(t['candidate_export_primary_key']) + '.', 'small')

def opening(kind):
    return [p('GECC Census Tract Environmental Data', 'title'), p(f'{kind}  |  Release {RELEASE}', 'subtitle'), p('TC Chakraborty; Rohit Mukherjee.', 'small')]

def terms_section(heading, sources_where, short=False):
    """Version, citation, terms and contact. GECC-specific lines appear once they are set."""
    out = [h(heading), labelled('Release.', f'{RELEASE}. Cite the release number.')]
    if GECC_CITATION:
        out.append(labelled('How to cite.', GECC_CITATION + f' Also cite the source products in {sources_where}.'))
    else:
        out.append(labelled('How to cite.', f'Cite this collection with its release number, and cite the source products in {sources_where}.'))
    if GECC_LICENCE:
        out.append(labelled('License.', GECC_LICENCE + '.'))
    if short:
        out.append(labelled('Source terms.', f'Each source product keeps its own terms of use and attribution requirements. See {sources_where}.'))
    else:
        out.append(labelled('Source terms.', 'The derived values depend on third-party source products. Each source keeps its own terms of use and attribution requirements. ESA WorldCover is distributed under CC BY 4.0, which requires attribution. PRISM data require acknowledgement of the PRISM Group, Oregon State University. Landsat data are courtesy of the U.S. Geological Survey.'))
    if GECC_CONTACT:
        out.append(labelled('Contact.', GECC_CONTACT + '.'))
    out.append(labelled('Acknowledgement.', 'This work was funded as a pilot project under the Gateway Exposome Coordinating Center (GECC). This work used analytical resources developed and disseminated by the Gateway Exposome Coordinating Center (GECC) (https://gatewayexposome.org/). The GECC is funded by the National Institute on Aging (NIA), under award U24AG088894.'))
    return out

def geography_text():
    return [p('The collection uses 2020 Census tract boundaries. Whole-tract tables have one row per tract, identified by GEOID. The master table has 85,528 tracts. GRIDMET and PRISM cover 83,776 tracts in the contiguous United States.'),
        p('Some measures describe only the area where a tract overlaps an Urban Area. The tract-Urban Area overlap table links each area to its tract and Urban Area. The field piece_id identifies the overlap. GEOID identifies the tract and GEOID20 identifies the Urban Area.'),
        p('A tract can overlap more than one Urban Area. These rows remain separate. NDVI and EVI use 75,910 overlaps. UHI and GLC vegetation use 75,383: they omit the 527 overlaps in Hawaii and the Island Areas, which are absent from those tables rather than flagged. Rural references are shared by overlaps in the same Urban Area. They do not describe rural land outside that Urban Area.')]

def build_codebook():
    story = opening('Codebook')
    story += [p('Use this codebook to find variable definitions, calculation methods and summary statistics. The collection guide helps you choose a dataset. The column dictionary gives the exact name of every field.'), h('Contents')]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('toc', fontName='Arial', fontSize=10.5, leading=16, spaceBefore=5)]
    story += [toc, PageBreak(), h('1 Data and geography')]
    story += geography_text()
    story += [p('Keep GEOID, FIPS and Urban Area codes as text. piece_id is an integer. Join whole-tract tables by GEOID and overlap tables by piece_id. Join each measurement-support table with the same key as its main table.'),
        h('2 How to read the codebook'),
        p('The codebook covers all 6,426 data-table CSV columns through variable groups and shared identifier definitions. In a field pattern, {year}, {period}, {dataset}, {source} or {measure} stands for the value used in that column name. Use metadata/COLUMN_DICTIONARY.csv for each exact field name, unit and missing-value rule.'),
        p('JJA means June, July and August. Same-year DJF means January, February and December of the labelled year. Observation years differ from the 2020 boundary year. Read the source calendar for day counts and missing dates.'),
        p('Empty numeric fields mean unavailable data. They do not mean zero. Keep valid zeros unless a source flag withholds the estimate. An empty missing_dates value in a calendar means no dates are missing.'),
        p('Years listed under a variable have at least one nonmissing value. They do not indicate complete coverage. Summary tables show 2020 JJA where available and the latest year with a value. Static fields show their snapshot distribution. UHI examples use maximum temperature.'),
        p('N is the number of nonmissing rows. Summary means and medians give each row equal weight. Summary SD is the sample standard deviation across rows. A variable describing spatial SD instead measures variation among cells within a tract or overlap. Rural reference values repeat across overlap rows and are not independent observations.'),
        p(researcher.TYPE_GUIDANCE), CondPageBreak(250), h('3 Geography and shared fields')]
    groups = [
        ('3 Geography and shared fields', ['tract_geography', 'ua_registry', 'piece_registry', 'ndvi_evi_support']),
        ('4 Temperature', ['temperature_svcm', 'temperature_gridmet', 'temperature_prism', 'temperature_daymet', 'extreme_days']),
        ('5 Land cover and vegetation indices', ['land_cover_glc', 'ndvi_evi_series']),
        ('6 Comparisons with rural land inside Urban Areas', ['ndvi_evi_ua_pieces', 'vegetation_ua_pieces', 'uhi_ua_pieces']),
        ('7 Canopy height and leaf area index', ['canopy_water_excluded', 'lai_water_excluded']),
        ('8 Measurement support and calendars', [tid for tid, t in tables.items() if t['role'] in ('measurement_support', 'calendar_lookup')]),
    ]
    for gi, (section, tids) in enumerate(groups):
        if gi: story += [CondPageBreak(250), h(section)]
        if gi == 3:
            story += [p('Each comparison uses rural land inside the same Urban Area. A positive difference means the overlap value is higher than its rural reference. The rural land filters differ by product, as described below.')]
        if gi == 5:
            story += [p('Support tables describe counts, sampled area and coverage. Use them with the main tables. A positive count does not by itself establish a successful estimate. The calendars describe shared source periods and use the full key shown for each file.')]
        for tid in tids:
            story += [h(TITLES[tid], 2), files(tid)]
            if tid in PRODUCTS:
                product = PRODUCTS[tid]
                # Keep the product title, file reference and method on one page.
                intro = story[-2:] + [p(product['meaning']), labelled('Observation period.', product['period']), labelled('Method.', product['method']), labelled('Limitations.', product['limit'])]
                story[-2:] = [KeepTogether(intro)]
                if tid == 'land_cover_glc':
                    story += [table(['Group', 'GLC class codes'], [
                        ['Tree and forest', '51, 52, 61, 62, 71, 72, 81, 82, 91, 92'],
                        ['Shrub', '120, 121, 122'], ['Grass', '130'], ['Crop', '10, 11, 20'],
                        ['Vegetation', '10, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 120, 121, 122, 130, 140, 150, 152, 153, 181, 182, 185, 186'],
                    ], [110, 402]), p('The whole-tract vegetation group includes woody-crop class 12. Classes 150, 152 and 153 are included here but excluded from the overlap vegetation group.', 'small')]
                if tid == 'vegetation_ua_pieces':
                    story += [p('The vegetation group uses GLC codes 10, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 120, 121, 122, 130, 140, 181, 182, 185 and 186.', 'small')]
            entries = []
            for f in families:
                if f['table'] == tid: entries.extend(family_block(f))
            if tid not in PRODUCTS and entries:
                # Keep a shared-table title with its first variable definition.
                story[-2:] = [KeepTogether(story[-2:] + entries[0]._content)]
                story.extend(entries[1:])
            else:
                story.extend(entries)
    story += [CondPageBreak(250), h('9 Practical use'),
        p('Use the main estimates with their quality flags and measurement-support tables. Missing values must remain missing. Spatial standard deviation describes variation within an area, not model uncertainty. These raster summaries are not population-weighted exposure estimates.'),
        p('For LAI, the source flag withholds estimates in 562 tracts without sampled GLC support. These include 461 tracts in Hawaii, 57 in Guam, 26 in the Northern Mariana Islands and 18 in American Samoa. Water-mask flags describe a separate issue.'),
        p('Avoid counting a tract more than once when it has several overlap rows. Do not treat repeated rural references as separate rural observations.'),
        h('10 Source references')]
    worldcover = 'Zanaga, D. et al. (2021). ESA WorldCover 10 m 2020 v100. DOI 10.5281/zenodo.5571936. CC BY 4.0.'
    landsat = ('Landsat Collection 2 Level-2 surface reflectance, courtesy of the U.S. Geological Survey. Landsat 4-5 TM DOI 10.5066/P9IAXOVV. '
               'Landsat 7 ETM+ DOI 10.5066/P9C7I13B. Landsat 8-9 OLI DOI 10.5066/P9OGBGM6. Processed in Earth Engine.')
    glc = 'Zhang, X. et al. (2024). GLC_FCS30D: the first global 30 m land-cover dynamics monitoring product with a fine classification system for the period of 1985-2022. Earth System Science Data 16, 1353-1381. DOI 10.5194/essd-16-1353-2024.'
    citations = {
        'tract_geography': 'U.S. Census Bureau. 2020 Census tract and 2020 Census Urban Area boundaries.',
        'temperature_svcm': 'SVCM-SP collection. DOI 10.25442/hku.c.8323119.',
        'temperature_gridmet': ('Abatzoglou, J. T. (2013). Development of gridded surface meteorological data for ecological applications and modelling. '
                                      'International Journal of Climatology 33, 121-131. DOI 10.1002/joc.3413. Earth Engine IDAHO_EPSCOR/GRIDMET.'),
        'temperature_prism': ('PRISM Group, Oregon State University. Daily 800 m data, https://prism.oregonstate.edu. '
                                      'Earth Engine projects/sat-io/open-datasets/OREGONSTATE/PRISM_800_DAILY.'),
        'temperature_era5land': 'Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth System Science Data 13, 4349-4383. DOI 10.5194/essd-13-4349-2021. Copernicus Climate Change Service, ERA5-Land hourly data, DOI 10.24381/cds.e2161bac.',
        'temperature_daymet': 'Daymet Version 4 R1, dataset version 4.1. ORNL DAAC. DOI 10.3334/ORNLDAAC/2129. Source paper DOI 10.1038/s41597-021-00973-0.',
        'extreme_days': 'SVCM-SP collection. DOI 10.25442/hku.c.8323119. GECC extreme-day calculations.',
        'land_cover_glc': glc + ' Earth Engine projects/sat-io/open-datasets/GLC-FCS30D/five-years-map and /annual.',
        'ndvi_evi_series': landsat + ' GECC NDVI and EVI calculations.',
        'ndvi_evi_ua_pieces': landsat + '\n' + worldcover,
        'vegetation_ua_pieces': glc + '\n' + worldcover,
        'uhi_ua_pieces': 'SVCM-SP collection. DOI 10.25442/hku.c.8323119. GECC calculations.\n' + worldcover,
        'canopy_water_excluded': ('Tolan, J. et al. (2024). Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. '
                                  'Remote Sensing of Environment 300, 113888. DOI 10.1016/j.rse.2023.113888. Meta and World Resources Institute, '
                                  'https://registry.opendata.aws/dataforgood-fb-forests/ (forests/v1/alsgedi_global_v6_float/).\n' + worldcover),
        'lai_water_excluded': 'Wan et al. (2024). Correcting confounding canopy structure, biochemistry and soil background effects improves leaf area index estimates across diverse ecosystems from Sentinel-2 imagery. Remote Sensing of Environment 309, 114224. DOI 10.1016/j.rse.2024.114224. Local adaptation with GLC support.\n' + worldcover,
    }
    for tid, citation in citations.items():
        if tid not in tables: continue
        story += [h(TITLES[tid], 2), p(citation)]
    story += terms_section('11 Release, citation and terms', 'section 10')
    path = OUT / f'GECC_CensusTracts_Codebook.pdf'
    Document(path, 'Codebook').multiBuild(story, canvasmaker=partial(Canvas, initialFontName='Arial'))
    assert covered == {(f['table'], f['family']) for f in families}
    return path

def build_guide():
    story = opening('Collection Guide')
    story += [p('Use this guide to choose a dataset and link it to Census tract geography. The collection contains 12 environmental datasets for temperature, land cover and vegetation. Each main dataset has a measurement-support table with counts, area or coverage.'),
        h('1 Choose a dataset')]
    rows = []
    for tid, product in PRODUCTS.items():
        rows.append([TITLES[tid], f'{tables[tid]["proposed_export_rows"]:,}', product['period']])
    story += [table(['Dataset', 'Rows', 'Observation period'], rows, [218, 56, 238]),
        p('The data are supplied in 13 ZIP downloads: 12 metric archives and shared_geography.zip. They contain 32 CSV data tables and 2 GeoJSON boundary files. Each metric archive includes its measurement-support table and any applicable calendar; ndvi_evi_series.zip also includes the static Landsat support table. Extract the archives into one common folder, preserving their internal metrics/, support/, metadata/ and geography/ paths. To map a table, join it to a boundary file by GEOID or piece_id. The column dictionary, GLC tile list, guides, README, file guide, file inventory and release notes are separate downloads.', 'small'),
        PageBreak(), h('2 Understand the geography')]
    story += geography_text()
    story += [h('3 Open and join the files'), table(['Folder', 'Contents', 'How to link'], [
        ['geography/', 'Tract boundaries, overlap boundaries and Urban Area names', 'GEOID for tracts. piece_id for overlaps. GEOID20 for Urban Area names.'],
        ['metrics/', '12 main datasets and a static Landsat support table', 'GEOID or piece_id, as listed in the codebook.'],
        ['support/', '12 measurement-support tables', 'Use the same spatial key as the main table.'],
        ['metadata/', '4 source calendars, the column dictionary and the GLC tile list', 'Use the full calendar key shown in the codebook.'],
        ['documentation/', 'Collection guide and codebook', 'Use the codebook for variable definitions.'],
    ], [83, 217, 212]),
        p('Keep GEOID, FIPS and Urban Area codes as text. piece_id is an integer. Years and seasons appear in column names. CSV files use UTF-8. Empty numeric fields mean missing data, not zero. GeoJSON coordinates are longitude and latitude.'),
        p('JJA means June through August. Same-year DJF means January, February and December of the labelled year. Historical measurements use 2020 boundaries. Use the source calendars to check available days.'),
        h('4 Check coverage and missing values'),
        p('A tract row can exist without a usable measurement. Read quality flags and the matching support table before analysis. Row counts are counts of areas, not people.'),
        p('LAI statistics are withheld for 562 tracts without sampled GLC support. The source-status flag identifies them. For canopy height and LAI, known permanent water is removed with ESA WorldCover. Pixels with unknown WorldCover coverage remain and are flagged. Some water can therefore remain.'),
        p('Spatial standard deviation describes variation within an area. It does not measure model uncertainty. The data are raster summaries, not population-weighted exposure estimates.'),
        PageBreak(), h('5 Interpret the measures')]
    for tid, product in PRODUCTS.items():
        story += [KeepTogether([h(TITLES[tid], 2), p(product['meaning'] + (' ' + product['guide_note'] if 'guide_note' in product else '')), p(product['limit'])])]
    story += [h('6 Find more detail'),
        p('The codebook gives field patterns, definitions, methods and example summary statistics. metadata/COLUMN_DICTIONARY.csv lists every exact field name. The column dictionary also gives field types, units and missing-value rules. FILE_GUIDE.md lists the files in the collection.'), p(researcher.TYPE_GUIDANCE_SHORT)]
    story += terms_section('7 Release, citation and terms', 'codebook section 10', short=True)
    path = OUT / f'GECC_CensusTracts_Collection_Guide.pdf'
    Document(path, 'Collection Guide').build(story, canvasmaker=partial(Canvas, initialFontName='Arial'))
    return path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__ == '__main__':
    outputs = [build_codebook(), build_guide()]
    audit = {'release': RELEASE, 'physical_columns': len(dictionary),
             'variable_families': len(families), 'covered_families': len(covered),
             'printed_variable_blocks': len(blocks),
             'gecc_citation_set': bool(GECC_CITATION), 'gecc_licence_set': bool(GECC_LICENCE),
             'gecc_contact_set': bool(GECC_CONTACT),
             'outputs': {p.name: sha(p) for p in outputs}, 'blocks': blocks}
    (HERE / 'content_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'blocks'}, indent=2))
