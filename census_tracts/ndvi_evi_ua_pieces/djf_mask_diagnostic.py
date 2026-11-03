"""Stage 1 diagnostic (information only): why the Columbus DJF_same_year 2020 composite is sparse.
For each Landsat 8 C2 L2 scene in the season intersecting the Columbus UA, the share of the UA's pixels (native 30 m
sampling, mean of 0/1 masks) passing each component of prep_c2sr_l8l9's mask:
  qa_bits05 (fill, dilated cloud, cirrus, cloud, shadow, snow all 0), qa_conf (cloud/shadow/snow/cirrus confidence
  all low: bits 8-15 = 01010101), radsat, reflectance range, aerosol not high, and all combined;
  also the snow bit (5) and cloud bit (3) shares. Denominator: the part of the box each scene covers. Output: qa/stage1_djf_mask_diagnostic.json
Usage (s1aef-train): python djf_mask_diagnostic.py [GEOID20 ...]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ee
import ee_composite as ec
from ndvi_common import load_config, utc_now, write_json_atomic
cfg = load_config()
ec.init_ee(cfg["earth_engine"]["project"], cfg["earth_engine"]["url"])
BOXES = {"19234": [-83.231, 39.77, -82.595, 40.332]}
out = {"utc": utc_now(), "note": "share of pixels in the UA bounding box at native 30 m (a 300 m run was wrong: QA bands are averaged in pyramids)", "uas": {}}
for g20, b in BOXES.items():
    reg = ee.Geometry.Rectangle(b, proj="EPSG:4326", geodesic=False)
    for season in ("DJF_same_year", "JJA"):
        dates = ee.Filter.Or(*[ee.Filter.date(x, y) for x, y in ec.season_date_ranges(2020, season)])
        col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filter(dates).filterBounds(reg).filter("WRS_ROW < 122")
               .filter(ee.Filter.neq("NADIR_OFFNADIR", "OFFNADIR")))
        def stats(im):
            qa = im.select("QA_PIXEL")
            opt = im.select("SR_B.").multiply(0.0000275).add(-0.2)
            rng = opt.reduce(ee.Reducer.minMax())
            m = {"qa_bits05": qa.bitwiseAnd(0x3F).eq(0), "qa_conf": qa.rightShift(8).bitwiseAnd(0xFF).eq(0x55),
                 "radsat": im.select("QA_RADSAT").eq(0), "range": rng.select("min").gt(0).And(rng.select("max").lt(1)),
                 "aerosol": im.select("SR_QA_AEROSOL").rightShift(6).neq(3),
                 "snow_bit": qa.bitwiseAnd(1 << 5).neq(0), "cloud_bit": qa.bitwiseAnd(1 << 3).neq(0),
                 "cirrus_conf_high": qa.rightShift(14).bitwiseAnd(3).eq(3), "snow_conf_high": qa.rightShift(12).bitwiseAnd(3).eq(3)}
            m["all"] = qa.bitwiseAnd(0xFF3F).eq(0x5500).And(m["radsat"]).And(m["range"]).And(m["aerosol"])
            img = ee.Image.cat([v.rename(k) for k, v in m.items()]).unmask(0)
            r = img.reduceRegion(ee.Reducer.mean(), reg, scale=30, maxPixels=1e9)   # native 30 m: no pyramid averaging of bit-packed QA bands
            return ee.Feature(None, r).set("id", im.get("system:index"), "date", im.date().format("YYYY-MM-dd"),
                                           "cloud_cover", im.get("CLOUD_COVER"))
        fs = col.map(stats).getInfo()["features"]
        out["uas"].setdefault(g20, {})[season] = [f["properties"] for f in fs]
write_json_atomic(os.path.join(cfg["paths"]["qa"], "stage1_djf_mask_diagnostic.json"), out)
for g20, d in out["uas"].items():
    for season, rows in d.items():
        print(g20, season, len(rows), "scenes")
        for r in sorted(rows, key=lambda r: r["date"]):
            print("  ", r["date"], r["id"][-20:], "cc=%s" % r["cloud_cover"], " ".join(f"{k}={r[k]:.2f}" for k in
                  ("all", "qa_bits05", "qa_conf", "aerosol", "range", "snow_bit", "cloud_bit", "cirrus_conf_high", "snow_conf_high")))
