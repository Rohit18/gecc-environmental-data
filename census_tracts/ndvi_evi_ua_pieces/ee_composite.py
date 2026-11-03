"""Landsat C2 L2 seasonal median NDVI/EVI composite: GECC NDVI/EVI v1 code minus its GLC water mask.

The functions below are copied verbatim from the Stage 1 prompt (user, 2026-10-13). Do not edit them.
In 2020 only Landsat 8 contributes (L7 stops in 2017, L9 starts in 2021); the full code is kept for fidelity.
"""
import ee

NDVI_EXPR = 'clamp((b("nir") - b("red")) / (b("nir") + b("red")), -1, 1)'
EVI_EXPR = ('clamp(2.5 * ((b("nir") - b("red")) / '
            'clamp(b("nir") + 6 * b("red") - 7.5 * b("blue") + 1, 0, 10)), -1.0, 1.0)')

def prep_c2sr_l4l5l7(image):
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B6").multiply(0.00341802).add(149.0)
    scaled = optical.addBands(thermal).select(
        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6"],
        ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"])
    qa_clear = image.select("QA_PIXEL").bitwiseAnd(0x3F3F).eq(0x1500)
    radsat = image.select("QA_RADSAT").eq(0)
    rng = optical.reduce(ee.Reducer.minMax())
    opacity = image.select("SR_ATMOS_OPACITY").unmask(-1).lt(300)
    return image.select().addBands(scaled).updateMask(
        qa_clear.And(radsat).And(rng.select("min").gt(0)).And(rng.select("max").lt(1)).And(opacity))

def prep_c2sr_l8l9(image):
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B10").multiply(0.00341802).add(149.0)
    scaled = optical.addBands(thermal).select(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"],
        ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"])
    qa_clear = image.select("QA_PIXEL").bitwiseAnd(0xFF3F).eq(0x5500)
    radsat = image.select("QA_RADSAT").eq(0)
    rng = optical.reduce(ee.Reducer.minMax())
    aerosol = image.select("SR_QA_AEROSOL").rightShift(6).neq(3)
    return image.select().addBands(scaled).updateMask(
        qa_clear.And(radsat).And(rng.select("min").gt(0)).And(rng.select("max").lt(1)).And(aerosol))

def season_date_ranges(year, season):
    if season == "JJA":
        return [(f"{year}-06-01", f"{year}-09-01")]
    if season == "DJF_same_year":
        return [(f"{year}-01-01", f"{year}-03-01"), (f"{year}-12-01", f"{year + 1}-01-01")]
    raise ValueError(season)

def landsat_sr(region, date_ranges):
    dates = ee.Filter.Or(*[ee.Filter.date(a, b) for a, b in date_ranges])
    def sensor(cid):
        return ee.ImageCollection(cid).filter(dates).filterBounds(region).filter("WRS_ROW < 122")
    l4 = sensor("LANDSAT/LT04/C02/T1_L2").map(prep_c2sr_l4l5l7)
    l5 = sensor("LANDSAT/LT05/C02/T1_L2").map(prep_c2sr_l4l5l7)
    l7 = sensor("LANDSAT/LE07/C02/T1_L2").filterDate("1984-01-01", "2017-01-01").map(prep_c2sr_l4l5l7)
    l8 = (sensor("LANDSAT/LC08/C02/T1_L2").filterDate("2013-05-01", "2099-01-01")
          .filter(ee.Filter.neq("NADIR_OFFNADIR", "OFFNADIR")).map(prep_c2sr_l8l9))
    l9 = sensor("LANDSAT/LC09/C02/T1_L2").filter(ee.Filter.neq("NADIR_OFFNADIR", "OFFNADIR")).map(prep_c2sr_l8l9)
    return l4.merge(l5).merge(l7).merge(l8).merge(l9)

def add_indices(image):
    return (image.expression(NDVI_EXPR).rename("NDVI")
            .addBands(image.expression(EVI_EXPR).rename("EVI"))
            .copyProperties(image, ["system:time_start"]))

def seasonal_composite(region, year, season):
    """Per-pixel median NDVI/EVI and the number of clear observations behind it."""
    sr = landsat_sr(region, season_date_ranges(year, season))
    reduced = sr.map(add_indices).reduce(ee.Reducer.median().combine(ee.Reducer.count(), sharedInputs=True))
    comp = reduced.select(["NDVI_median", "EVI_median", "NDVI_count"], ["NDVI", "EVI", "clear_obs"])
    empty = ee.Image.constant([0, 0, 0]).rename(["NDVI", "EVI", "clear_obs"]).updateMask(0)
    comp = ee.Image(ee.Algorithms.If(sr.size().gt(0), comp, empty))
    return comp.addBands(comp.select("clear_obs").updateMask(comp.select("NDVI").mask()), overwrite=True)


# ----------------------------------------------------------------------------- additions (not from v1)
def init_ee(project, url):
    """Initialize Earth Engine on the given project. The stored EE token is rejected on NERSC (2026-10-13), so
    gcloud application-default credentials are used with the same project."""
    import google.auth
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                           "https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=creds, project=project, opt_url=url)


def chunk_region(bounds):
    """Planar EPSG:4326 rectangle (west, south, east, north)."""
    return ee.Geometry.Rectangle(list(bounds), proj="EPSG:4326", geodesic=False)


def fetch_image(region, year, season, fill_float, fill_clear):
    """The composite prepared for computePixels: NDVI/EVI unmasked with fill_float as float32, clear_obs unmasked
    with fill_clear as int16; plus the scene count of the collection."""
    comp = seasonal_composite(region, year, season)
    vi = comp.select(["NDVI", "EVI"]).unmask(fill_float).toFloat()
    co = comp.select("clear_obs").unmask(fill_clear).toInt16()
    n = landsat_sr(region, season_date_ranges(year, season)).size()
    return vi.addBands(co), comp, n
