"""Stage 1: verify every downloaded GLC_FCS30D annual tile and write manifests/glc_tiles.csv.

Checks per tile: present; size == zip central-directory uncompressed size; CRC-32 recomputed from the file
== CRC-32 in the member's zip local header (fetched with a 30-byte range request); opens; 23 bands; uint8;
EPSG:4326; pixel size == config; 18554 x 18554; bounds consistent with the name (UPPER-LEFT corner).
Zenodo publishes md5 only per zip archive, not per member, so md5 cannot be checked for extracted members.
Usage: python verify_glc_tiles.py [n_procs]
"""
import json, os, struct, sys, urllib.request, zlib
from multiprocessing import Pool

import pandas as pd
import rasterio

from urbveg_common import ROOT, load_config

cfg = load_config()
g = cfg["glc_fcs30d"]
REC = g["zenodo_record"]
import sys as _sys
_man = _sys.argv[_sys.argv.index("--manifest") + 1] if "--manifest" in _sys.argv else f"{ROOT}/manifests/glc_tiles_{REC}.csv"
_tag = _sys.argv[_sys.argv.index("--tag") + 1] if "--tag" in _sys.argv else ""
sel = pd.read_csv(_man)
mem = pd.read_csv(f"{ROOT}/manifests/glc_zip_members_{REC}.csv").set_index("member")


def check(r):
    try:
        return _check(r)
    except Exception as e:             # record, do not crash the pool
        return {"tile": r["tile"], "present": True, "all_ok": False, "error": f"{type(e).__name__}: {e}"}


def _check(r):
    member = r["member_Annual"]
    path = os.path.join(cfg["paths"]["glc_dir"], member.split("/")[-1])
    m = mem.loc[member]
    out = {"tile": r["tile"], "path": path, "zip": r["zip_Annual"], "member": member,
           "expected_bytes": int(m.file_size), "zip_compressed_bytes": int(m.compress_size)}
    out["present"] = os.path.exists(path)
    if not out["present"]:
        return out
    out["bytes"] = os.path.getsize(path)
    out["size_ok"] = out["bytes"] == out["expected_bytes"]
    url = f"https://zenodo.org/records/{REC}/files/{r['zip_Annual']}?download=1"
    off = int(m.header_offset)
    import time as _t
    for _a in range(6):
        try:
            hdr = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + 29}"}), timeout=120).read()
            break
        except Exception:
            if _a == 5:
                raise
            _t.sleep(10 * (_a + 1))
    flag, crc = struct.unpack("<H", hdr[6:8])[0], struct.unpack("<I", hdr[14:18])[0]
    c = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            c = zlib.crc32(b, c)
    out.update({"zip_crc32": f"{crc:08x}", "file_crc32": f"{c:08x}", "crc_ok": (crc == c) and not (flag & 8)})
    with rasterio.open(path) as d:
        lon = int(r["tile"][1:r["tile"].index("N" if "N" in r["tile"] else "S")]) * (1 if r["tile"][0] == "E" else -1)
        ns = "N" if "N" in r["tile"] else "S"
        lat = int(r["tile"].split(ns)[1]) * (1 if ns == "N" else -1)
        out.update({"count": d.count, "dtype": d.dtypes[0], "crs": str(d.crs), "res": d.res[0],
                    "width": d.width, "height": d.height, "left": d.bounds.left, "top": d.bounds.top,
                    "right": d.bounds.right, "bottom": d.bounds.bottom, "nodata": d.nodata})
        out["opens"] = True
    out["bands_ok"] = out["count"] == g["annual_band_count"]
    out["dtype_ok"] = out["dtype"] == "uint8"
    out["crs_ok"] = out["crs"] == "EPSG:4326"
    out["res_ok"] = abs(out["res"] - g["pixel_size_deg"]) < 1e-15
    out["shape_ok"] = out["width"] == out["height"] == g["tile_size_px"]
    out["bounds_ok"] = abs(out["left"] - lon) < 1e-9 and abs(out["top"] - lat) < 1e-9
    out["all_ok"] = all(out[k] for k in ("size_ok", "crc_ok", "opens", "bands_ok", "dtype_ok", "crs_ok",
                                         "res_ok", "shape_ok", "bounds_ok"))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 16
    with Pool(n) as p:
        res = p.map(check, [r for _, r in sel.iterrows()])
    df = pd.DataFrame(res)
    df.to_csv(f"{ROOT}/manifests/glc_tiles{_tag}.csv", index=False)
    summ = {"n_selected": len(df), "n_present": int(df.present.sum()), "n_all_ok": int(df.get("all_ok", pd.Series(dtype=bool)).fillna(False).sum()),
            "total_bytes": int(df.get("bytes", pd.Series(dtype=float)).fillna(0).sum()),
            "failures": df.loc[~df.get("all_ok", pd.Series(False, index=df.index)).fillna(False).astype(bool), "tile"].tolist()}
    json.dump(summ, open(f"{ROOT}/manifests/glc_tiles_verify_summary{_tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
