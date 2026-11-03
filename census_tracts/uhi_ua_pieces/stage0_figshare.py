"""Stage 0, item 4a: fetch the figshare metadata of the 2015-2024 source releases (article IDs from the
September source manifest) and summarize method / input / version differences between the Iowa State
(2003-2020) and University of Hong Kong (2021-2024) releases. Raw API responses are kept in qa/figshare/.
Idempotent: an existing raw response is reused unless --refetch.
"""
import argparse
import html
import json
import os
import re
import urllib.request

import pandas as pd

from uhi_common import UHI_ROOT, load_config, setup_logging, utc_now, write_json_atomic

log = setup_logging("stage0_figshare")
ap = argparse.ArgumentParser()
ap.add_argument("--refetch", action="store_true")
a = ap.parse_args()
cfg = load_config()
man = pd.read_csv(cfg["paths"]["source_manifest"], dtype=str)
man = man[man.year.astype(int).isin(cfg["years"])].drop_duplicates("year")
raw_dir = f"{UHI_ROOT}/qa/figshare"
os.makedirs(raw_dir, exist_ok=True)


def text(s):
    s = re.sub(r"<br\s*/?>|</p>|</li>", "\n", s or "")
    return re.sub(r"[ \t]+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()


arts = {}
for _, r in man.iterrows():
    p = f"{raw_dir}/article_{r.article_id}.json"
    if a.refetch or not os.path.exists(p):
        with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{r.article_id}", timeout=60) as f:
            d = json.load(f)
        write_json_atomic(p, d)
    d = json.load(open(p))
    vers = None
    try:
        vp = f"{raw_dir}/article_{r.article_id}_versions.json"
        if a.refetch or not os.path.exists(vp):
            with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{r.article_id}/versions", timeout=60) as f:
                write_json_atomic(vp, json.load(f))
        vers = json.load(open(vp))
    except Exception as e:  # noqa: BLE001
        log.warning("versions for %s: %s", r.article_id, e)
    arts[int(r.year)] = {
        "article_id": int(r.article_id), "manifest_doi": r.article_doi, "api_doi": d.get("doi"),
        "title": d.get("title"), "version": d.get("version"), "n_versions": len(vers) if vers else None,
        "published_date": d.get("published_date"), "created_date": d.get("created_date"),
        "modified_date": d.get("modified_date"), "group_id": d.get("group_id"),
        "authors": [x.get("full_name") for x in d.get("authors", [])],
        "license": (d.get("license") or {}).get("name"),
        "description": text(d.get("description")),
        "references": d.get("references"), "related_materials": [
            {k: m.get(k) for k in ("title", "identifier", "relation", "identifier_type")}
            for m in d.get("related_materials", [])],
        "custom_fields": {c.get("name"): c.get("value") for c in d.get("custom_fields", [])},
        "funding": [f.get("title") for f in d.get("funding_list", [])],
        "keywords": d.get("tags"), "files": [{"name": f.get("name"), "size": f.get("size"), "md5": f.get("computed_md5")}
                                            for f in d.get("files", []) if "NorthAmerica" in f.get("name", "")],
        "n_files": len(d.get("files", [])), "file_names": sorted(f.get("name") for f in d.get("files", [])),
    }

# Normalise descriptions (drop the year) to see whether the text differs between releases
norm = {y: re.sub(r"\b(19|20)\d\d\b", "YEAR", v["description"]) for y, v in arts.items()}
groups = {}
for y, t in norm.items():
    groups.setdefault(t, []).append(y)
summary = {"generated_utc": utc_now(), "articles": arts,
           "description_variants": [{"years": ys, "text": t} for t, ys in groups.items()],
           "file_name_sets": {str(y): v["file_names"] for y, v in arts.items()}}
write_json_atomic(f"{UHI_ROOT}/qa/stage0_figshare.json", summary)
for t, ys in groups.items():
    log.info("description variant for years %s:\n%s\n", ys, t)
for y, v in arts.items():
    log.info("%d id=%s v%s doi=%s published=%s modified=%s authors=%s refs=%s related=%s custom=%s n_files=%s",
             y, v["article_id"], v["version"], v["api_doi"], v["published_date"], v["modified_date"],
             v["authors"], v["references"], v["related_materials"], v["custom_fields"], v["n_files"])
