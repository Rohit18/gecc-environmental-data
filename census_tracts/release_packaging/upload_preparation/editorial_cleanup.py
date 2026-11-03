"""Public documentation edits only; never edits data tables or archives."""
import re
LABELS = {
'coverage_pairs':'Contributing tract–sampling-cell pairs','total_coverage_weight':'Total sampling-cell coverage weight','inside_grid_pairs':'Contributing pairs inside the grid','inside_grid_weight':'Sampling-cell coverage weight inside the grid',
'hot_days_tmax_gt_p95_support_weight':'Support weight for hot-day estimates','cold_days_tmin_lt_p05_support_weight':'Support weight for cold-day estimates','valid_tmax_days_support_weight':'Support weight for valid maximum-temperature days','valid_tmin_days_support_weight':'Support weight for valid minimum-temperature days',
'baseline_tmax_mean_days':'Mean valid days in the maximum-temperature baseline','baseline_tmax_min_days':'Minimum valid days in the maximum-temperature baseline','baseline_tmax_partial_weight':'Incomplete maximum-temperature baseline weight','baseline_tmax_support_weight':'Maximum-temperature baseline support weight',
'baseline_tmin_mean_days':'Mean valid days in the minimum-temperature baseline','baseline_tmin_min_days':'Minimum valid days in the minimum-temperature baseline','baseline_tmin_partial_weight':'Incomplete minimum-temperature baseline weight','baseline_tmin_support_weight':'Minimum-temperature baseline support weight',
'hot_matched_valid_days_bound':'Matched valid-day bound for hot-day estimates','cold_matched_valid_days_bound':'Matched valid-day bound for cold-day estimates','pixel_nonwater_count':'Non-water composite-cell support count','ndvi_urban_valid_coverage':'Overlap NDVI valid coverage','ndvi_rural_valid_coverage':'Rural reference NDVI valid coverage','evi_urban_valid_coverage':'Overlap EVI valid coverage','evi_rural_valid_coverage':'Rural reference EVI valid coverage','expected_days':'Expected days','observed_days':'Observed days','missing_dates':'Missing dates'}
CALENDAR = {'dataset':'Source product for this calendar row.','year':'Year used to label the observation period.','period':'Annual or seasonal observation period.','season':'Season used for the observation period.','temperature_measure':'Daily maximum or minimum temperature measure.','source_period_day_count':'Number of source days in the observation period.','expected_days':'Number of days expected in the observation period.','observed_days':'Number of source days available in the observation period.','missing_dates':'Dates missing from the source period. An empty string means no dates are missing.','period_complete':'True when the source has all expected days for the period.'}
DEFINITIONS = {
'water_only_tract':'True when the Census land area is zero.',
'chm_cover_gt1m_frac':'Weight of retained heights in bins 2-255 m divided by all retained valid weight. The 1 m bin is excluded.',
'total_area_m2':'GLC sampled-area denominator, including retained unclassified/fill classes 0 and 250.',
'unclassified_fraction':'GLC class 0 plus class 250 area divided by the total sampled GLC area.',
'pixel_nonwater_count':'Non-water composite-cell count used as the denominator of observed_fraction for the tract, year and season.',
'rural_valid_coverage':'Share of rural reference weight with valid temperature.',
'tract_area_m2':'Area of the parent tract used to calculate the overlap share.'}
def prose(s):
    s=s.replace('No source medians or SDs are invented.', 'Medians and standard deviations are not supplied for these overlap vegetation indices.')
    s=s.replace('Existing urban_frac, rural_frac and diff copied exactly.', 'Urban and rural vegetation fractions and their difference.')
    s=s.replace('Published class-group fractions are copied exactly.', 'GLC class-group area fractions.')
    s=s.replace('joined from the existing diagnostic', '').replace('joined from the diagnostic', '')
    s=re.sub(r'\b[Pp]ublished\s+', '', s)
    s=re.sub(r'\b[Ee]xisting\s+', '', s)
    s=re.sub(r'\b[Pp]ieces\b', 'overlaps', s)
    s=re.sub(r'\b[Pp]iece\b', lambda m:'Overlap' if m[0][0].isupper() else 'overlap', s)
    s=re.sub(r'\bUA\b', 'Urban Area', s)
    s=s.replace('Retained-overlap', 'Retained overlap').replace('retained-overlap','retained overlap')
    s=s.strip()
    if s and s[0].islower() and not s.startswith(('urban_frac','glc_','not_flagged')): s=s[0].upper()+s[1:]
    return s

def rows_cleanup(rows):
    for row in rows:
        family=row['family']
        row['methods'] = row['methods'].lower()
        if row['table'] in {'temperature_svcm', 'temperature_svcm_measurement_support'} and '1/255' not in row['statistic_support']:
            row['statistic_support'] += ' Sampling cells are included only when the tract covers at least 1/255 of the cell area (approximately 0.392%).'
        if row['table'].endswith('_calendar'):
            row['definition']=CALENDAR[row['field']]
        elif family in DEFINITIONS:
            row['definition']=DEFINITIONS[family]
        for key in ['definition','statistic_support']:
            row[key]=prose(row[key])
    return rows

def patch_source(code):
    code=re.sub(r"def clean\(s\):\n    return .*?\n", "def clean(s):\n    return str(s).replace('–', '-').replace('—', '-').replace('‑', '-')\n", code)
    code=code.replace("    if name in simple: return simple[name]", "    simple.update("+repr(LABELS)+")\n    if name in simple: return simple[name]")
    start=code.index("    s = f['definition'].replace")
    end=code.index('\ncovered = set()',start)
    code=code[:start]+"    return clean(editorial.prose(f['definition'])).strip()\n"+code[end:]
    marker='covered = set()'
    setup="import editorial_cleanup as editorial\nDEFS.update(editorial.DEFINITIONS)\nCALENDAR_DEFS.update(editorial.CALENDAR)\n\n"
    setup += "PRODUCTS['temperature_svcm']['method'] += ' Sampling cells are included only when the tract covers at least 1/255 of the cell area (approximately 0.392%).'\nPRODUCTS['temperature_svcm']['guide_note'] = 'Sampling cells are included only when the tract covers at least 1/255 of the cell area (approximately 0.392%).'\n"
    code = code.replace("p(product['meaning']), p(product['limit'])", "p(product['meaning'] + (' ' + product['guide_note'] if 'guide_note' in product else '')), p(product['limit'])")
    return code.replace(marker,setup+marker)
