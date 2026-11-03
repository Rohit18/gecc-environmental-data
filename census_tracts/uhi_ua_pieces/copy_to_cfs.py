"""Copy outputs, config, scripts, tests, qa, README and REPORT to CFS (/path/to/project/urbveg/uhi/) with
sha256 verification. Never overwrites: a destination file with identical content is skipped; one with different
content is first renamed to <name>.bak-<UTC stamp>. Test subsets (data/output/test_*) are not copied.
Writes manifests/cfs_copy_<stamp>.csv (path, bytes, sha256, action) in the destination and in logs/.
"""
import csv
import os
import shutil
import time

from uhi_common import UHI_ROOT, load_config, setup_logging, sha256_file

log = setup_logging("copy_to_cfs")
cfg = load_config()
DST = cfg["paths"]["cfs_copy"]
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
items = ["data/output", "config", "scripts", "tests", "qa", "README.md", "REPORT.md"]
files = []
for it in items:
    src = os.path.join(UHI_ROOT, it)
    if os.path.isfile(src):
        files.append(it)
        continue
    for root, dirs, fs in os.walk(src):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache") and not d.startswith("test_")]
        files += [os.path.relpath(os.path.join(root, f), UHI_ROOT) for f in fs if not f.endswith(".part")]
rows = []
for rel in sorted(files):
    s, d = os.path.join(UHI_ROOT, rel), os.path.join(DST, rel)
    h = sha256_file(s)
    action = "copied"
    if os.path.exists(d):
        if sha256_file(d) == h:
            rows.append({"path": rel, "bytes": os.path.getsize(s), "sha256": h, "action": "identical_skipped"})
            continue
        os.replace(d, f"{d}.bak-{stamp}")
        action = "replaced_old_kept_as_bak"
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d + ".part")
    if sha256_file(d + ".part") != h:
        raise SystemExit(f"checksum mismatch after copy: {rel}")
    os.replace(d + ".part", d)
    rows.append({"path": rel, "bytes": os.path.getsize(s), "sha256": h, "action": action})
for out in (os.path.join(DST, "manifests", f"cfs_copy_{stamp}.csv"), os.path.join(cfg["paths"]["logs"], f"cfs_copy_{stamp}.csv")):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "action"])
        w.writeheader()
        w.writerows(rows)
log.info("%d files: %s", len(rows), {a: sum(r["action"] == a for r in rows) for a in {r["action"] for r in rows}})
