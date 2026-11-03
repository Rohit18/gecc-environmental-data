"""Wording edits agreed before PI review (2 Nov 2026), shared by the PDF builder and apply_pi_wording.py."""

SLIVER_BASE = ("True when the overlap covers less than 1% of its parent tract's area (overlap_share < 0.01). "
               "Flagged rows are retained, not removed. Many are slivers along boundaries where the tract and "
               "Urban Area edges nearly coincide")
SLIVER = {
    'ndvi_evi_ua_pieces': SLIVER_BASE + '; 1,779 of the NDVI/EVI overlaps contain no 10 m pixel center and have no '
                          'estimate (quality_status piece_no_pixels).',
    'vegetation_ua_pieces': SLIVER_BASE + '.',
}
OLD_SLIVER = ('Source flag identifying a small tract-Urban Area overlap. Flagged rows are retained. The numerical '
              'cutoff is not specified in the available producer documentation; this flag should not be treated as '
              'a universal area threshold.')

# (old, new) pairs applied to every public text string.
REPLACEMENTS = [
    (' The daily time window is unspecified.', ''),
    (': this table was produced for 2,622 Urban Areas, before the 22 Urban Areas in Hawaii and the Island Areas '
     'were added for NDVI and EVI.', '. This table does not cover Hawaii or the Island Areas.'),
    (': this table was produced for 2,622 Urban Areas, before 22 Urban Areas in Hawaii and the Island Areas '
     'were added for NDVI and EVI.', '. This table does not cover Hawaii or the Island Areas.'),
    (': they were produced for 2,622 Urban Areas, before the 22 Urban Areas in Hawaii and the Island Areas '
     'were added for NDVI and EVI.', '. These two tables do not cover Hawaii or the Island Areas.'),
]


def text(s):
    for old, new in REPLACEMENTS:
        s = s.replace(old, new)
    return s


def configure(builder):
    """Wrap the builder's text and definition hooks; call after researcher.configure."""
    previous_clean = builder.clean
    builder.clean = lambda s: text(previous_clean(s))
    previous_definition = builder.definition
    builder.definition = lambda f: SLIVER.get(f['table']) if f['family'] == 'sliver_flag' else previous_definition(f)

# Status-code definitions (item 5), keyed by (table, family).
_WATER = ('Shows how the ESA WorldCover permanent-water mask applied, checked in this order. {extra}'
          'worldcover_incomplete: some source pixels have no WorldCover class; they are kept in the estimate. '
          'water_only: no pixel remains after permanent water is removed, so statistics are empty. '
          'ok: every source pixel has a WorldCover class and some pixels remain after water removal.')
_PIECE = 'piece_no_pixels: the overlap contains no 10 m WorldCover pixel center, so it has no estimate. '
STATUS = {
    ('extreme_days', 'support_status'):
        'Shows whether extreme-day counts are available and why they may be missing. ok: counts available. '
        'outside_selected_source_grid: the tract lies outside the SVCM-SP archive grid used for its region (mainly Hawaii). '
        'unsupported_pacific_source_route: no SVCM-SP archive covers the tract (American Samoa, Guam and the Northern '
        'Mariana Islands). source_no_common_seasonal_support: the tract is inside the grid, but no sampling cell has '
        'valid source data for the season; in 2020 JJA these are the 222 masked New York tracts and 61 others, '
        'including 15 in Puerto Rico. partial_band_threshold_support: at least one of hot days, cold days and the '
        'valid-day counts has support and at least one does not, usually because some cells have no baseline threshold.',
    ('ndvi_evi_ua_pieces', 'quality_status'):
        'Shows whether the overlap estimate is available and why it may be missing. ok: overlap, rural reference and '
        'difference are available. ' + _PIECE + 'ua_no_valid_value: the Urban Area\'s rural reference has no valid pixel '
        'in this season; rural and difference values are empty and the overlap value is kept if available. '
        'urban_no_valid_value: the overlap has pixels, but none with a valid value.',
    ('uhi_ua_pieces', 'quality_status'):
        'Shows whether the estimate is available and why it may be missing. ok: overlap temperature, rural reference '
        'and UHI are available. ' + _PIECE + 'ua_no_valid_temperature: no retained pixel in the Urban Area has a valid '
        'temperature (all Puerto Rico Urban Areas and a few others). urban_no_valid_temperature: the overlap has '
        'retained pixels, but none in a cell with valid temperature. The suffix +incomplete_period is added when the '
        'season is missing a source date: DJF 2016, 2020 and 2024 lack December 31 (see metadata/uhi_ua_pieces_calendar.csv).',
    ('canopy_water_excluded', 'water_mask_status'): _WATER.format(extra=''),
    ('lai_water_excluded', 'water_mask_status'):
        _WATER.format(extra='source_no_valid_pixels: the source has no valid LAI pixel in the tract. '),
}


def configure_status(builder):
    previous_definition = builder.definition
    builder.definition = lambda f: STATUS.get((f['table'], f['family'])) or previous_definition(f)

# Citation, references, units and jargon (minor-polish round, 2 Nov 2026).
SVCM_REF = ('Zhou, Y., Zhang, T. and Zhou, W. (2026). A global 1 km resolution daily near-surface air temperature '
            'dataset (2003-present), produced with a spatially varying coefficient model with sign preservation '
            '(SVCM-SP). HKU DataHub. DOI 10.25442/hku.c.8414754. Input land surface temperature: Zhang, T. et al. '
            '(2026). A global seamless 1 km resolution daily land surface temperature dataset (2003-present). HKU '
            'DataHub. DOI 10.25442/hku.c.8323119.')
REPLACEMENTS += [
    ('SVCM-SP collection. DOI 10.25442/hku.c.8323119.', SVCM_REF),
    ('SVCM-SP cells are about 927 m wide.',
     'SVCM-SP (spatially varying coefficient model with sign preservation) cells are about 927 m wide.'),
    (' Local adaptation with GLC support.', ' Applied by GECC using this method, with GLC-FCS30D land cover.'),
    ('Cite the released version of this collection:', 'Cite this collection by its DOI and version:'),
    ('release 1.0. Cite the released version.', 'release 1.0. Cite the DOI and the version.'),
    (' Cite the release number.', ''),
    ('degrees C', '°C'),
    ('used by the statistical reducers', 'used for the statistics'),
    ('native-cell', 'source-cell'),
    ('native cell', 'source cell'),
    ('source-specific cells', 'source cells'),
    ('CSV text; see serialized_type and logical_type', 'CSV text'),
    ('so urban fractions for the 6,380 overlaps', 'so overlap vegetation fractions (urban_frac) for the 6,380 overlaps'),
]

# Item 4: public table and role names in the dictionary and manifest.
TABLE_RENAME = {'tract_geography': 'census_tracts_2020', 'piece_registry': 'tract_urban_area_pieces_2020',
                'ua_registry': 'urban_areas_2020', 'uhi_ua_summary': 'uhi_ua_pieces',
                'extreme_days_support': 'extreme_days', 'land_cover_glc_unclassified': 'land_cover_glc'}
ROLE_RENAME = {'master_tract_registry': 'geography', 'spatial_registry': 'geography'}
REPLACEMENTS += [
    ('Join to the shared overlap registry using piece_id; identifiers are specific to that registry version.',
     'Join other overlap tables to it by piece_id.'),
    ('Identifier/static registry attribute.', 'Identifier or static attribute.'),
]
_BUILDER_NAME = {'census_tracts_2020': 'tract_geography', 'tract_urban_area_pieces_2020': 'piece_registry',
                 'urban_areas_2020': 'ua_registry'}


def configure_tables(builder):
    """The builder keys dictionary rows by its internal table ids; map renamed rows back on load."""
    for r in builder.dictionary:
        r['table'] = _BUILDER_NAME.get(r['table'], r['table'])
    builder.mapping = {(r['table'], r['field']): r for r in builder.dictionary}
REPLACEMENTS.append(('Readable name stored once in the Urban Area registry.', 'Urban Area name.'))

# Final-scan leftovers (2 Nov 2026).
REPLACEMENTS += [
    ('The Earth Engine extraction computes mean, stdDev, median and count with shared inputs. Statistical reducers '
     'retain default EE mask/coverage weighting; count uses centroid inclusion and is an integer sample count, so it '
     'is not the weighted denominator. EE median is direct for small input sets and histogram-based for larger sets.',
     'Earth Engine computes the mean, standard deviation, median and count from the same inputs. The statistics use '
     "Earth Engine's default mask and coverage weighting; the count uses cell-center inclusion and is an integer, so "
     'it is not the weighted denominator. The median is exact for small sample sets and histogram-based for larger ones.'),
    ('Mean/median use production Earth Engine weighting;', 'Mean and median use Earth Engine coverage weighting;'),
    ('Native-grid mean after known water exclusion.', 'Source-grid mean after known water exclusion.'),
    ('native temperature cell area', 'source temperature cell area'),
    ('All retained native pixel support', 'All retained source-pixel support'),
    ('cover the whole-tract have', 'cover the whole tract have'),
]
REPLACEMENTS += [
    ('withheld where GLC 2021 support is unavailable', 'withheld where GLC support is unavailable'),
    ('with no sampled GLC 2021 support;', 'with no sampled GLC support;'),
    ('derived from observed GLC 2021 availability', 'derived from observed GLC availability'),
]
