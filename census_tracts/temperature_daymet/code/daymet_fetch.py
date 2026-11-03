#!/usr/bin/env python3
"""Plan, stage, archive, verify and safely purge the locked ORNL Daymet sources.

Only requests.Session loads Earthdata credentials. No credential file is
opened by this code and no server response bodies or tokens are logged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import threading
import time

import requests

COLLECTION = "C2532426483-ORNL_CLOUD"
BASE_URL = "https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data/"
CFS = Path("/path/to/project/gecc/daymet")
ROOT = Path("/path/to/scratch/daymet_tract_temperature")
_mutex = threading.RLock()
_stop = threading.Event()


def immutable_read_path(path):
    """Same immutable CFS bytes through NERSC's read-only mount when available."""
    path = Path(path)
    if path.is_relative_to('/global/cfs'):
        candidate = Path('/dvs_ro/cfs') / path.relative_to('/global/cfs')
        if candidate.is_file() and candidate.stat().st_size == path.stat().st_size:
            return candidate
    return path


class AuthenticationFailure(RuntimeError):
    pass


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_manifest(path):
    return json.loads(Path(path).read_text())


def update_manifest(path, filename, fields):
    with _mutex:
        with Path(str(path) + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest = read_manifest(path)
            record = next(r for r in manifest["files"] if r["filename"] == filename)
            record.update(fields)
            manifest["updated_utc"] = utc_now()
            atomic_json(path, manifest)


def parse_years(value):
    years = set()
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    if not years or not years.issubset(set(range(1980, 2026))):
        raise ValueError("Only 1980–2025 are allowed")
    return sorted(years)


def plan(args):
    raw_path = Path(args.manifest).parent / "cmr_granules_raw.json"
    if raw_path.exists() and not args.refresh:
        raw = json.loads(raw_path.read_text())
    else:
        items = []
        for page in range(1, 100):
            response = requests.get(
                "https://cmr.earthdata.nasa.gov/search/granules.umm_json",
                params={"collection_concept_id": COLLECTION, "page_size": 2000, "page_num": page},
                timeout=90,
            )
            response.raise_for_status()
            batch = response.json()["items"]
            items.extend(batch)
            if len(batch) < 2000:
                break
        raw = {"items": items}
        atomic_json(raw_path, raw)
    previous = {r["filename"]: r for r in read_manifest(args.manifest)["files"]} if Path(args.manifest).exists() else {}
    records = {}
    for item in raw["items"]:
        umm = item["umm"]
        match = re.search(r"(daymet_v4_daily_(na|hi|pr)_(tmax|tmin)_(\d{4})\.nc)$", umm["GranuleUR"])
        if not match or not 1980 <= int(match[4]) <= 2025:
            continue
        name, domain, measure, year = match.groups()
        if umm["CollectionReference"] != {"ShortName": "Daymet_Daily_V4R1_2129", "Version": "4.1"}:
            raise ValueError(f"{name}: collection version mismatch")
        links = [u["URL"] for u in umm["RelatedUrls"] if u.get("Type") == "GET DATA" and u["URL"] == BASE_URL + name]
        if len(links) != 1 or name in records:
            raise ValueError(f"{name}: missing/duplicate locked HTTPS URL or granule")
        distribution = umm["DataGranule"]["ArchiveAndDistributionInformation"]
        if len(distribution) != 1 or "SizeInBytes" not in distribution[0]:
            raise ValueError(f"{name}: ambiguous CMR byte size")
        metadata = distribution[0]
        checksum = metadata.get("Checksum", {})
        record = dict(filename=name, domain=domain, measure=measure, year=int(year),
                      url=links[0], cmr_bytes=int(metadata["SizeInBytes"]),
                      cmr_checksum=checksum.get("Value"), cmr_checksum_algorithm=checksum.get("Algorithm"),
                      cmr_concept_id=item["meta"]["concept-id"], cmr_revision=item["meta"]["revision-id"],
                      cfs_path=str(CFS / "daily" / year / name), status="planned")
        if name in previous:
            old = previous[name]
            if any(old.get(k) != record[k] for k in ["url", "cmr_bytes", "cmr_checksum", "cmr_checksum_algorithm"]):
                raise ValueError(f"{name}: CMR metadata changed; refusing silent archive replacement")
            record.update(old)
        records[name] = record
    expected = {f"daymet_v4_daily_{domain}_{measure}_{year}.nc"
                for year in range(1980, 2026) for domain in ["na", "hi", "pr"] for measure in ["tmax", "tmin"]}
    if set(records) != expected:
        raise ValueError(f"CMR inventory mismatch: {len(records)} files; missing {sorted(expected-set(records))[:6]}")
    manifest = dict(source="Daymet_Daily_V4R1_2129", version="4.1", doi="10.3334/ORNLDAAC/2129",
                    collection=COLLECTION, updated_utc=utc_now(), files=[records[n] for n in sorted(records)])
    atomic_json(args.manifest, manifest)
    total = sum(r["cmr_bytes"] for r in records.values())
    print(f"CMR verified: 276 tmax/tmin files, 1980–2025, {total} bytes ({total/1e12:.3f} TB)", flush=True)


def verify_file(path, record, expected_sha=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != record["cmr_bytes"]:
        raise ValueError(f"{record['filename']}: file missing or CMR byte-size mismatch")
    sha = sha256_of(path)
    algorithm = (record.get("cmr_checksum_algorithm") or "").lower().replace("-", "")
    if record.get("cmr_checksum"):
        if algorithm == "sha256":
            actual = sha
        elif algorithm in ["md5", "sha1", "sha512"]:
            digest = hashlib.new(algorithm)
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            actual = digest.hexdigest()
        else:
            raise ValueError(f"{record['filename']}: unsupported supplied checksum algorithm")
        if actual.lower() != record["cmr_checksum"].lower():
            raise ValueError(f"{record['filename']}: CMR checksum mismatch")
    if expected_sha and sha != expected_sha:
        raise ValueError(f"{record['filename']}: archived SHA-256 mismatch")
    return sha


def fetch_one(record, args):
    directory=(CFS/'manifests/file_locks') if args.command=='fetch-cfs' else Path(args.staging)/'.locks'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/(record['filename']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=next(r for r in read_manifest(args.manifest)['files'] if r['filename']==record['filename'])
        return _fetch_one(current,args)


def _fetch_one(record, args):
    name = record["filename"]
    if record["status"] == "verified_cfs":
        verify_file(immutable_read_path(record["cfs_path"]), record, record["sha256"])
        print(f"{name}: verified CFS archive; download skipped", flush=True)
        return
    direct = args.command == 'fetch-cfs'
    target = Path(record['cfs_path']) if direct else Path(args.staging) / str(record["year"]) / name
    if direct and not target.resolve().is_relative_to((CFS/'daily').resolve()):
        raise ValueError('Direct archive destination outside locked CFS daily directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verification_path = immutable_read_path(target) if direct else target
        sha = verify_file(verification_path, record, record.get("sha256"))
        update_manifest(args.manifest, name, dict(status="verified_staging", bytes=target.stat().st_size, sha256=sha,
                                                staging_path=str(target), staging_verification_time=utc_now()))
        if direct:
            update_manifest(args.manifest,name,dict(status='verified_cfs',archive_time=utc_now(),verification_time=utc_now(),staging_path=None,download_destination='CFS',archive_verification_read_path=str(verification_path)))
        return
    temporary = target.with_name(name + ".part")
    for attempt in range(1, 6):
        if _stop.is_set():
            raise AuthenticationFailure("Stopped after authentication failure")
        started = time.monotonic()
        try:
            with requests.Session() as session:
                # requests loads ~/.netrc internally; never inspect credentials here.
                with session.get(record["url"], stream=True, timeout=(30, 120)) as response:
                    if response.status_code in [401, 403] or "text/html" in response.headers.get("Content-Type", "").lower():
                        _stop.set()
                        raise AuthenticationFailure(f"{name}: Earthdata authentication/authorization failed (HTTP {response.status_code})")
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(chunk_size=8 << 20):
                            if _stop.is_set():
                                raise AuthenticationFailure("Stopped after authentication failure")
                            stream.write(block)
                        stream.flush()
                        os.fsync(stream.fileno())
            elapsed = time.monotonic() - started
            sha = verify_file(temporary, record)
            os.replace(temporary, target)
            update_manifest(args.manifest, name, dict(status="verified_staging", bytes=target.stat().st_size,
                            sha256=sha, staging_path=str(target), download_time=utc_now(), download_seconds=elapsed,
                            throughput_MB_s=target.stat().st_size/1e6/elapsed, staging_verification_time=utc_now(), attempts=attempt))
            if direct:
                verification_path = immutable_read_path(target)
                verify_file(verification_path,record,sha)
                update_manifest(args.manifest,name,dict(status='verified_cfs',archive_time=utc_now(),verification_time=utc_now(),staging_path=None,download_destination='CFS',archive_verification_read_path=str(verification_path)))
            location='CFS archive' if direct else 'scratch staging'
            print(f"{name}: {location} checksum verified; {elapsed:.1f}s, {target.stat().st_size/1e6/elapsed:.2f} MB/s", flush=True)
            return
        except AuthenticationFailure:
            raise
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            if status is not None and status not in [408, 429, 500, 502, 503, 504]:
                raise RuntimeError(f"{name}: ORNL HTTP {status}; stopped") from None
            if attempt == 5:
                raise RuntimeError(f"{name}: transient network failure after 5 attempts") from None
            delay = min(60, 2**attempt) + random.random()
            print(f"{name}: transient network failure; retry in {delay:.1f}s", flush=True)
            time.sleep(delay)


def select_records(args):
    years = parse_years(args.years)
    domains = args.domains.split(",")
    measures = args.measures.split(",")
    if not set(domains).issubset({"na", "hi", "pr"}) or not set(measures).issubset({"tmax", "tmin"}):
        raise ValueError("Only na/hi/pr and tmax/tmin are allowed")
    if args.command in ["fetch", "fetch-cfs", "archive"] and years != [2020] and not args.approved_post_pilot:
        raise ValueError("Post-pilot years require user approval and --approved-post-pilot")
    return sorted([r for r in read_manifest(args.manifest)["files"]
            if r["year"] in years and r["domain"] in domains and r["measure"] in measures],
            key=lambda r:(r['year'],r['domain'],r['measure']))


def archive_one(record, args):
    target = Path(record["cfs_path"])
    if not target.resolve().is_relative_to((CFS / "daily").resolve()):
        raise ValueError("Archive destination is outside the locked CFS daily directory")
    if target.exists():
        sha = verify_file(target, record, record.get("sha256"))
    else:
        source = Path(args.staging) / str(record["year"]) / record["filename"]
        sha = verify_file(source, record, record.get("sha256"))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        shutil.copyfile(source, temporary)
        verify_file(temporary, record, sha)
        os.replace(temporary, target)
    # Re-read final CFS bytes rather than trusting a staging checksum.
    verify_file(target, record, sha)
    update_manifest(args.manifest, record["filename"], dict(status="verified_cfs", sha256=sha,
                    bytes=target.stat().st_size, archive_time=utc_now(), verification_time=utc_now()))
    print(f"{record['filename']}: CFS archive verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "fetch", "fetch-cfs", "archive", "verify", "purge"])
    parser.add_argument("--manifest", default=str(CFS / "manifests/daymet_manifest.json"))
    parser.add_argument("--staging", default=str(ROOT / "work/staging"))
    parser.add_argument("--years", default="2020")
    parser.add_argument("--domains", default="na,hi,pr")
    parser.add_argument("--measures", default="tmax,tmin")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scratch-budget-gb", type=float, default=150)
    parser.add_argument("--approved-post-pilot", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "plan":
            plan(args)
            return
        records = select_records(args)
        if not records:
            raise ValueError("No matching granules")
        if args.command in ['fetch','fetch-cfs']:
            if args.command=='fetch-cfs' and not os.environ.get('SLURM_JOB_ID'):
                raise ValueError('Direct CFS downloads must run inside the xfer Slurm job')
            if not 1 <= args.workers <= 8:
                raise ValueError('Download streams must be between 1 and 8 (default 4)')
            expected_peak = sum(r["cmr_bytes"] for r in records if r["status"] != "verified_cfs")
            staging = Path(args.staging)
            present = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file()) if staging.exists() else 0
            if args.command=='fetch' and present + expected_peak > args.scratch_budget_gb * 1e9:
                raise ValueError("Staging could exceed scratch budget; process fewer years")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(fetch_one, record, args) for record in records]
                for future in as_completed(futures):
                    future.result()
        elif args.command == "archive":
            if any(r["cmr_bytes"] > 1e9 for r in records) and not os.environ.get("SLURM_JOB_ID"):
                raise ValueError("Large source copies must run in the xfer Slurm job")
            for record in records:
                archive_one(record, args)
        elif args.command == "verify":
            for record in records:
                if record["status"] != "verified_cfs":
                    raise ValueError(f"{record['filename']}: not verified on CFS")
                verify_file(record["cfs_path"], record, record["sha256"])
                update_manifest(args.manifest, record["filename"], dict(verification_time=utc_now()))
            print(f"Verified {len(records)} CFS files", flush=True)
        else:
            staging = Path(args.staging).resolve()
            if not staging.is_relative_to((ROOT / "work").resolve()):
                raise ValueError("Purge must be inside the Daymet scratch work directory")
            for record in records:
                audit = CFS / "composites" / str(record["year"]) / f"Daymet_seasonal_{record['domain']}_{record['year']}.json"
                side = json.loads(audit.read_text())
                if not side.get("passed") or record["status"] != "verified_cfs":
                    raise ValueError("Purge blocked: CFS verification and composite audit must pass")
                verify_file(record["cfs_path"], record, record["sha256"])
                verify_file(audit.with_suffix(".tif"), {"filename":audit.name, "cmr_bytes":side["composite_bytes"]}, side["composite_sha256"])
                if side["inputs"][record["filename"]]["sha256"] != record["sha256"]:
                    raise ValueError("Purge blocked: composite input checksum mismatch")
                target = staging / str(record["year"]) / record["filename"]
                if target.resolve().is_relative_to(staging):
                    target.unlink(missing_ok=True)
            print("Verified scratch staging purged; CFS daily archive retained", flush=True)
    except AuthenticationFailure as error:
        atomic_json(CFS / "audits/authentication_preflight.json", dict(passed=False, created_utc=utc_now(), error=str(error)))
        raise SystemExit(f"STOPPED: {error}. No workaround attempted.") from None


if __name__ == "__main__":
    main()
