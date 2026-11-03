#!/usr/bin/env python3
"""Run PR and USVI GLC exports one task at a time with an EECU guard."""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/glc_nonconus"
LEDGER = OUT / "eecu_ledger.json"
PROJECT = "your-ee-project"
CAP_HOURS = 950.0
POLL_SECONDS = 30
GLC_PREFIX = "GLC_NONCONUS_CensusTract_v1"
WC_PREFIX = "WC2020_NONCONUS_tract_"
ACTIVE = {"PENDING", "READY", "RUNNING", "CANCEL_REQUESTED"}
SUPERSEDED_FAILURES = {"GLC_NONCONUS_CensusTract_v1_2020_s72_n1000_b000"}


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    ee.Initialize(project=PROJECT)
    last_message = None
    while True:
        operations = ee.data.listOperations(f"projects/{PROJECT}")
        active_glc, failed_glc = [], []
        for operation in operations:
            meta = operation.get("metadata", {})
            name = meta.get("description", "")
            if name.startswith((WC_PREFIX, GLC_PREFIX)):
                ledger[name] = max(ledger.get(name, 0),
                                   float(meta.get("batchEecuUsageSeconds", 0)))
            if name.startswith(GLC_PREFIX):
                if meta.get("state") in ACTIVE:
                    active_glc.append(operation)
                if (name not in SUPERSEDED_FAILURES
                        and meta.get("state") in {"FAILED", "CANCELLED"}):
                    failed_glc.append((name, meta.get("state"), operation.get("error")))
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True))
        hours = sum(ledger.values()) / 3600
        if failed_glc:
            log(f"STOP: GLC task failure: {failed_glc}")
            return 2
        if hours >= CAP_HOURS:
            for operation in active_glc:
                ee.data.cancelOperation(operation["name"])
            log(f"STOP: combined measured usage {hours:.4f} EECU-hours")
            return 3
        if active_glc:
            message = f"WAIT: {len(active_glc)} GLC active; combined {hours:.4f} EECU-hours"
            if message != last_message:
                log(message)
                last_message = message
            time.sleep(POLL_SECONDS)
            continue
        command = [sys.executable, str(ROOT / "submit_nonconus_census_tract_glc_exports.py"),
                   "--project", PROJECT, "--states", "72", "78",
                   "--batch-size", "100", "--max-new-tasks", "1",
                   "--eecu-cap-hours", str(CAP_HOURS), "--submit"]
        result = subprocess.run(command, text=True, capture_output=True)
        started = [line for line in result.stdout.splitlines() if line.startswith("STARTED ")]
        for line in started:
            log(line)
        if result.returncode:
            log(f"STOP: submitter returned {result.returncode}: {result.stderr.strip()} {result.stdout[-2000:]}")
            return 4
        if not started:
            log(f"DONE: all planned GLC exports submitted and completed; combined {hours:.4f} EECU-hours")
            return 0
        last_message = None
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
