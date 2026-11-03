"""Download selected GLC_FCS30D tiles by extracting single members from the Zenodo zips with HTTP
range requests (one streaming GET per member; the whole zip is never downloaded).

Reads manifests/glc_tiles_<record>.csv (from select_glc_tiles.py). Idempotent and resumable:
skips tiles already present with the expected uncompressed size; writes to .part then renames;
checks the zip CRC-32. Never overwrites. Intended to run on a DTN (dtn01-04) for the full set.
Usage: python fetch_glc_tiles.py <record> [--kind Annual|5years] [--tiles W85N40,W80N40] [--limit N]
"""
import argparse, logging, os, struct, sys, time, urllib.request, zlib

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("record")
ap.add_argument("--kind", default="Annual")
ap.add_argument("--tiles", default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--shard", default=None, help="k/n: process every n-th tile starting at k (0-based)")
ap.add_argument("--manifest", default=None, help="tile list csv (default manifests/glc_tiles_<record>.csv)")
ap.add_argument("--outdir", default=None, help="default data/raw/glc_fcs30d/tiles")
ap.add_argument("--root", default=os.environ.get("URBVEG_ROOT", "/path/to/scratch/urbveg"))
a = ap.parse_args()
ROOT = a.root
logging.basicConfig(filename=f"{ROOT}/logs/fetch_glc_{a.record}" + (f"_shard{a.shard.replace('/', 'of')}" if a.shard else "") + ".log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger()

sel = pd.read_csv(a.manifest or f"{ROOT}/manifests/glc_tiles_{a.record}.csv")
mem = pd.read_csv(f"{ROOT}/manifests/glc_zip_members_{a.record}.csv").set_index("member")
if a.tiles:
    sel = sel[sel.tile.isin(a.tiles.split(","))]
if a.limit:
    sel = sel.head(a.limit)
if a.shard:
    k, n = map(int, a.shard.split("/"))
    sel = sel.sort_values("tile").iloc[k::n]
outdir = a.outdir or f"{ROOT}/data/raw/glc_fcs30d/tiles"
os.makedirs(outdir, exist_ok=True)


def fetch(zipname, member, dest):
    r = mem.loc[member]
    url = f"https://zenodo.org/records/{a.record}/files/{zipname}?download=1"
    off, csize, usize = int(r.header_offset), int(r.compress_size), int(r.file_size)
    # local file header: 30 bytes + name + extra; read it to find the data start
    hdr = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + 29}"}), timeout=120).read()
    sig, _, _, method, _, _, crc, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", hdr)
    assert sig == 0x04034B50, "bad local header signature"
    start = off + 30 + nlen + xlen
    resp = urllib.request.urlopen(urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{start + csize - 1}"}), timeout=300)
    d = zlib.decompressobj(-15) if method == 8 else None
    got, crc_calc = 0, 0
    with open(dest + ".part", "wb") as f:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            got += len(chunk)
            data = d.decompress(chunk) if d else chunk
            crc_calc = zlib.crc32(data, crc_calc)
            f.write(data)
        if d:
            tail = d.flush(); crc_calc = zlib.crc32(tail, crc_calc); f.write(tail)
    if got != csize or os.path.getsize(dest + ".part") != usize:
        raise IOError(f"size mismatch got={got}/{csize} out={os.path.getsize(dest + '.part')}/{usize}")
    if crc and crc_calc != crc:   # crc may be 0 in the local header when a data descriptor is used
        raise IOError("CRC mismatch")
    os.rename(dest + ".part", dest)
    return usize


ok = skip = err = 0
for r in sel.itertuples():
    zipname, member = getattr(r, f"zip_{a.kind}"), getattr(r, f"member_{a.kind}")
    if not isinstance(member, str):
        log.warning("no %s member for %s", a.kind, r.tile); continue
    dest = f"{outdir}/{member.split('/')[-1]}"
    usize = int(mem.loc[member].file_size)
    if os.path.exists(dest):
        if os.path.getsize(dest) == usize:
            skip += 1; log.info("SKIP %s", dest); continue
        log.error("present with wrong size, not overwriting: %s", dest); err += 1; continue
    for attempt in range(4):
        try:
            t = time.time(); n = fetch(zipname, member, dest)
            ok += 1; log.info("OK %s %d B %.1fs", dest, n, time.time() - t); break
        except Exception as e:
            log.warning("attempt %d failed %s: %s", attempt, member, e); time.sleep(20 * (attempt + 1))
    else:
        err += 1; log.error("FAILED %s", member)
msg = f"done ok={ok} skip={skip} err={err}"
log.info(msg); print(msg)
sys.exit(1 if err else 0)
