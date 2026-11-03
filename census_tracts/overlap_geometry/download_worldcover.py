"""Download the required ESA WorldCover 2020 v100 tiles to CFS and verify them.

Tiles are retained permanently: a tile already present with the manifest's byte
size and a matching recorded checksum is never re-downloaded.  Each tile gets a
<tile>.verified.json sidecar recording URL, bytes, ETag, sha256, CRS, transform
and the classes observed, so build_worldcover_fractions.py can prove the Map
band and classes 50 and 80 without re-reading every tile.

Usage:
  python download_worldcover.py --manifest worldcover_manifest.csv \
      --dest DIR [--workers 8] [--only-temperature-grid]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import common

log = common.setup_logging("download_worldcover")


def verify_tile(path: Path) -> dict:
    """Open the tile and record what it actually contains."""
    import numpy as np
    import rasterio
    with rasterio.open(path) as ds:
        band_desc = ds.descriptions[0] if ds.descriptions else None
        # Sample classes without reading 1.3 GB: decimated overview read.
        arr = ds.read(1, out_shape=(1, min(2048, ds.height), min(2048, ds.width)))
        vals, counts = np.unique(arr, return_counts=True)
        return {
            "width": ds.width, "height": ds.height, "count": ds.count,
            "dtype": ds.dtypes[0], "nodata": ds.nodata,
            "crs": ds.crs.to_string(), "transform": list(ds.transform)[:6],
            "bounds": list(ds.bounds), "band_description": band_desc,
            "compression": str(ds.compression),
            "sampled_classes": {int(v): int(c) for v, c in zip(vals, counts)},
            "has_class_50": bool((arr == common.WC_BUILT).any()),
            "has_class_80": bool((arr == common.WC_WATER).any()),
        }


def fetch(row: dict, dest: Path, retries: int = 5) -> dict:
    tile = row["tile"]
    final = dest / f"ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
    side = Path(str(final) + ".verified.json")
    if final.exists() and side.exists():
        rec = json.loads(side.read_text())
        if rec.get("bytes") == final.stat().st_size == int(row["bytes"]):
            return {"tile": tile, "action": "already_present", "bytes": rec["bytes"]}
    last = None
    for attempt in range(retries):
        try:
            with common.atomic_path(final) as tmp:
                with urllib.request.urlopen(row["url"], timeout=600) as r, \
                        open(tmp, "wb") as fh:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        fh.write(chunk)
            got = final.stat().st_size
            if got != int(row["bytes"]):
                raise IOError(f"{tile}: got {got} bytes, manifest says {row['bytes']}")
            rec = {"tile": tile, "url": row["url"], "bytes": got,
                   "etag_from_manifest": row.get("etag"),
                   "sha256": common.sha256_file(final),
                   "verified_utc": common.utc_now()}
            rec.update(verify_tile(final))
            common.write_json_atomic(side, rec)
            return {"tile": tile, "action": "downloaded", "bytes": got}
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("%s attempt %d failed: %s", tile, attempt + 1, exc)
    raise RuntimeError(f"{tile}: download failed after {retries} attempts: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-temperature-grid", action="store_true",
                    help="skip tiles that cannot carry a temperature value")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype={"tile": str})
    if args.only_temperature_grid and "intersects_temperature_grid" in df.columns:
        before = len(df)
        df = df[df["intersects_temperature_grid"].astype(str).str.lower() == "true"]
        log.info("%d of %d tiles intersect the temperature grid", len(df), before)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("fetching %d tiles (%.1f GB) into %s", len(df),
             df["bytes"].sum() / 1e9, dest)

    rows = df.to_dict("records")
    done = 0
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(lambda r: fetch(r, dest), rows):
            results.append(res)
            done += 1
            if done % 10 == 0:
                log.info("  %d/%d tiles", done, len(rows))

    n_new = sum(1 for r in results if r["action"] == "downloaded")
    log.info("complete: %d downloaded, %d already present, %.1f GB total",
             n_new, len(results) - n_new,
             sum(r["bytes"] for r in results) / 1e9)


if __name__ == "__main__":
    main()
