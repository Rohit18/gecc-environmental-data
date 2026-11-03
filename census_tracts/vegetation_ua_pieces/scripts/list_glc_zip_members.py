"""List members of the GLC_FCS30D Zenodo zip archives without downloading them.

Reads each zip's central directory with HTTP range requests and writes
manifests/glc_zip_members_<record>.csv (zip, member, compressed bytes, uncompressed bytes,
member header offset). Idempotent: skips zips already listed in the output CSV.
"""
import csv, io, json, logging, os, sys, time, urllib.request, zipfile

ROOT = "/path/to/scratch/urbveg"
record = sys.argv[1]  # e.g. 15063683
zips = sys.argv[2:] or None
logging.basicConfig(filename=f"{ROOT}/logs/list_glc_zip_members_{record}.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

class HttpRangeFile:
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                data = urllib.request.urlopen(req, timeout=120).read(); break
            except Exception as e:
                logging.warning("retry %s %s", attempt, e); time.sleep(5 * (attempt + 1))
        else: raise IOError("range read failed")
        self.pos += len(data); return data

meta = json.load(open(f"{ROOT}/data/interim/zenodo_{record}.json"))
files = {f["key"]: f["size"] for f in meta["files"] if f["key"].endswith(".zip")}
out = f"{ROOT}/manifests/glc_zip_members_{record}.csv"
done = set()
if os.path.exists(out):
    done = {r["zip"] for r in csv.DictReader(open(out))}
new = not os.path.exists(out)
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new: w.writerow(["zip", "member", "compress_size", "file_size", "header_offset", "compress_type"])
    for z in sorted(zips or files):
        if z in done: logging.info("SKIP %s", z); continue
        url = f"https://zenodo.org/records/{record}/files/{z}?download=1"
        zf = zipfile.ZipFile(HttpRangeFile(url, files[z]))
        for i in zf.infolist():
            w.writerow([z, i.filename, i.compress_size, i.file_size, i.header_offset, i.compress_type])
        fh.flush(); logging.info("OK %s members=%d", z, len(zf.infolist())); print("OK", z, len(zf.infolist()), flush=True)
