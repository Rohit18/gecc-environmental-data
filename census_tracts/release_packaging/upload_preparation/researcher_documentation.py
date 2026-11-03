"""Field-specific public descriptions and US spelling; no data transformations."""
import re

US_WORDS = {
    'labelled':'labeled', 'labelling':'labeling', 'centre':'center', 'centres':'centers',
    'metre':'meter', 'metres':'meters', 'kilometre':'kilometer', 'kilometres':'kilometers',
    'colour':'color', 'colours':'colors', 'behaviour':'behavior', 'behavioural':'behavioral',
    'modelling':'modeling', 'modelled':'modeled', 'neighbour':'neighbor', 'neighbours':'neighbors',
    'neighbourhood':'neighborhood', 'neighbourhoods':'neighborhoods',
    'licence':'license', 'licences':'licenses', 'acknowledgement':'acknowledgment',
    'acknowledgements':'acknowledgments', 'organised':'organized', 'organisation':'organization',
    'organisations':'organizations', 'normalised':'normalized', 'normalisation':'normalization',
    'summarised':'summarized', 'summarise':'summarize', 'harmonised':'harmonized',
    'harmonisation':'harmonization', 'analysed':'analyzed', 'analyse':'analyze',
    'visualisation':'visualization', 'visualise':'visualize', 'recognised':'recognized',
}
US_PATTERN = re.compile(r'\b('+'|'.join(US_WORDS)+r')\b',re.I)
def us_text(s):
    def replace(m):
        value=US_WORDS[m[0].lower()]
        return value.upper() if m[0].isupper() else value.capitalize() if m[0][0].isupper() else value
    return US_PATTERN.sub(replace,s).replace('SVCM SP','SVCM-SP').replace('Whole tract','Whole-tract').replace('whole tract','whole-tract')

def field_definition(table, family):
    fixed={
        'EVI_count':'Number of valid EVI composite cells. EVI can be unavailable where NDVI is valid, so this count can differ from NDVI_count.',
        'piece_id':'Integer identifier for one Census tract-Urban Area intersection in this release. Join to the shared overlap registry using piece_id; identifiers are specific to that registry version.',
        'dataset':'Name of the temperature source product.',
        'analysis_scale_m':'Analysis scale in meters: 30 m outside Alaska, and 30 m or 100 m in Alaska. The value in each tract row identifies the scale used.',
        'piece_area_m2':'Area of the tract-Urban Area intersection measured in an equal-area projection.',
        'vegetation_fraction':'Area in the GLC vegetation class group divided by the total sampled GLC area. The class-group members are listed in codebook section 5.',
        'sliver_flag':'Source flag identifying a small tract-Urban Area overlap. Flagged rows are retained. The numerical cutoff is not specified in the available producer documentation; this flag should not be treated as a universal area threshold.',
        'single_scene':'True when clear_obs_mean is below 1.5; otherwise false when observation depth is available. This flag describes mean clear-observation depth, rather than a count of satellite scenes.',
        'valid_tmax_days':'Spatially averaged number of valid daily maximum-temperature observations over all sampling cells with at least one valid maximum- and minimum-temperature day in the season. This support can include cells without a 2003-2007 baseline threshold, so it can differ from the extreme-day support. Compare extreme-day counts with the matched valid-day bounds, not with this field.',
        'valid_tmin_days':'Spatially averaged number of valid daily minimum-temperature observations over all sampling cells with at least one valid maximum- and minimum-temperature day in the season. This support can include cells without a 2003-2007 baseline threshold, so it can differ from the extreme-day support. Compare extreme-day counts with the matched valid-day bounds, not with this field.',
        'coverage_pairs':'Number of contributing tract-to-sampling-cell pairs across the seasonal sampling support. Repeated mappings to the same native cell are retained.',
        'inside_grid_pairs':'Number of contributing tract-to-sampling-cell pairs inside the source grid. Repeated mappings to the same native cell are retained.',
    }
    if table.endswith('_calendar'): return None
    if family in fixed: return fixed[family]
    if family in ('chm_median_m','chm_stdev_m'):
        stat='median' if family=='chm_median_m' else 'population standard deviation'
        return f'Coverage-weighted spatial {stat} of retained valid canopy heights, including 0 and 1 m.'
    if family in ('NDVI_median','EVI_median','NDVI_stdDev','EVI_stdDev'):
        index,stat=family.split('_',1)
        stat='median' if stat=='median' else 'standard deviation'
        approximation=' Spatial medians can be histogram-approximate.' if stat=='median' else ''
        return f'Spatial {stat} of the seasonal per-pixel temporal median {index} composite over retained non-water pixels. Earth Engine mask and coverage weighting is retained.{approximation} See codebook section 5 for sensor and support limitations.'
    if re.fullmatch(r't(?:max|min)_(?:median|stddev)_c',family):
        measure='maximum' if family.startswith('tmax') else 'minimum'
        stat='median' if '_median_' in family else 'population standard deviation'
        if table.startswith(('temperature_gridmet', 'temperature_prism')):
            weighting='Earth Engine mask and cell-coverage weighting is retained.'
            if stat=='median': weighting+=' The median can be histogram-approximate.'
        elif 'daymet' in table:
            weighting='Source-cell coverage weights are used.'
        else:
            weighting='Included sampling cells have equal weight.'
            if stat=='median': weighting+=' The median is unweighted.'
        return f'Spatial {stat} of period-average daily {measure} air temperature within the tract. {weighting}'
    if family in ('tmax_count','tmin_count'):
        measure='maximum' if family.startswith('tmax') else 'minimum'
        if table.startswith(('temperature_gridmet', 'temperature_prism')):
            return f'Integer number of sampled cells with valid {measure} temperature, using cell-center inclusion. This count is not the coverage-weight sum used by the statistical reducers.'
        return f'Coverage-weighted number of source cells with valid {measure} temperature. Values can be fractional and are not day counts.'
    if re.fullmatch(r'(urban|rural|delta)_(ndvi|evi)',family):
        zone,index=family.split('_'); index=index.upper()
        if zone=='urban':return f'Valid-pixel area-weighted mean {index} in the tract-Urban Area overlap.'
        if zone=='rural':return f'Valid-pixel area-weighted mean {index} on rural land inside the same Urban Area. This reference repeats for overlaps in that Urban Area.'
        return f'Overlap {index} minus the matching rural reference {index} inside the same Urban Area. Both means use valid-pixel area weights.'
    if re.fullmatch(r'baseline_t(?:max|min)_(partial|support)_weight',family):
        measure='maximum' if family.startswith('baseline_tmax') else 'minimum'
        support='an incomplete' if '_partial_' in family else 'an available'
        return f'Sum of geographic sampling-cell coverage weights for cells with {support} 2003-2007 {measure}-temperature baseline. These weights are not physical areas.'
    if family in ('ndvi_urban_valid_coverage','ndvi_rural_valid_coverage','evi_urban_valid_coverage','evi_rural_valid_coverage'):
        index,zone,_=family.split('_',2)
        area='tract-Urban Area overlap' if zone=='urban' else 'rural reference inside the same Urban Area'
        return f'Valid {index.upper()} pixel weight divided by total retained pixel weight for the {area}. This is a coverage fraction, distinct from the vegetation-index value.'
    band_weights={
        'total_coverage_weight':'the seasonal sampling support',
        'inside_grid_weight':'the seasonal sampling support inside the source grid',
        'hot_days_tmax_gt_p95_support_weight':'the hot-day exceedance estimates',
        'cold_days_tmin_lt_p05_support_weight':'the cold-day exceedance estimates',
        'valid_tmax_days_support_weight':'the valid maximum-temperature day counts',
        'valid_tmin_days_support_weight':'the valid minimum-temperature day counts',
    }
    if family in band_weights:
        return 'Sum of geographic sampling-cell intersection-fraction weights supporting '+band_weights[family]+'. Different temperature bands can have different denominators.'
    if family in ('hot_matched_valid_days_bound','cold_matched_valid_days_bound'):
        band='hot-day' if family.startswith('hot_') else 'cold-day'
        return f'Coverage-weighted available-day upper bound calculated over the same sampling support as the {band} exceedance estimate.'
    return None

TYPE_GUIDANCE = ('In COLUMN_DICTIONARY.csv, source_type records the upstream representation: '
    'large_string/string/text are text types; int32/int64/integer are integer types; double is '
    'a floating-point numeric type; bool/boolean are Boolean types. CSV text identifies an input '
    'whose logical type is specified by the retained dictionary columns; GeoJSON property identifies '
    'a value stored in feature properties. For importing the distributed CSV files, use serialized_type '
    'and logical_type together with null_rule. These describe the output representation, field meaning '
    'and missing-value handling; geographic codes remain text.')

def update_rows(rows):
    for r in rows:
        definition=field_definition(r['table'],r['family'])
        if definition: r['definition']=definition
        for key in ('definition','statistic_support'):
            r[key]=us_text(r[key])
        if 'logical type requires schema' in r['source_type']:
            r['source_type']='CSV text; see serialized_type and logical_type'
    return rows

TYPE_GUIDANCE_SHORT = ('For CSV import, use serialized_type, logical_type and null_rule in the column dictionary. '
    'source_type records upstream storage types such as int64 or double. The README explains these '
    'type labels; geographic codes must remain text.')

def configure(builder):
    for name in ('h1','h2'):
        builder.STYLES[name].keepWithNext=True
    original_clean=builder.clean
    builder.clean=lambda s:us_text(original_clean(s))
    original_labelled=builder.labelled
    builder.labelled=lambda label, s:original_labelled(us_text(label), s)
    original_definition=builder.definition
    builder.definition=lambda f:field_definition(f['table'],f['family']) or original_definition(f)
    if hasattr(builder,'TITLES'):
        builder.TITLES={k:us_text(v) for k,v in builder.TITLES.items()}
