#!/usr/bin/env python3
"""Complete the bounded 10 m Aleutian tile repair with the shared EECU guard."""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee


ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "outputs/worldcover_2020_nonconus/eecu_ledger.json"
PROJECT = "your-ee-project"
PREFIX = "WC2020_NONCONUS_tract_v4_areas_ak_aleutian_02016000100_tile_"
ACTIVE = {"PENDING", "READY", "RUNNING", "CANCEL_REQUESTED"}


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def main():
    ee.Initialize(project=PROJECT)
    last_message = None
    while True:
        operations = ee.data.listOperations(f"projects/{PROJECT}")
        tile_tasks = [op for op in operations if op.get("metadata", {}).get(
            "description", "").startswith(PREFIX)]
        active = [op for op in tile_tasks if op.get("metadata", {}).get("state") in ACTIVE]
        failed = [op for op in tile_tasks if op.get("metadata", {}).get("state") in
                  {"FAILED", "CANCELLED"}]
        if failed:
            log(f"STOP: tile export failed: {[(x['metadata']['description'], x.get('error')) for x in failed]}")
            return 2
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        for op in operations:
            meta = op.get("metadata", {})
            name = meta.get("description", "")
            if name.startswith(("WC2020_NONCONUS_tract_", "GLC_NONCONUS_CensusTract_v1")):
                ledger[name] = max(ledger.get(name, 0),
                                   float(meta.get("batchEecuUsageSeconds", 0)))
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True))
        hours = sum(ledger.values()) / 3600
        if hours >= 950:
            for op in active:
                ee.data.cancelOperation(op["name"])
            log(f"STOP: shared 950-hour guard reached at {hours:.4f}")
            return 3
        if len(active) >= 2:
            message = f"WAIT: {len(active)} repair tiles active; {hours:.4f} EECU-hours combined"
            if message != last_message:
                log(message)
                last_message = message
            time.sleep(30)
            continue
        command = [sys.executable, str(ROOT / "submit_worldcover_ak_aleutian_tile_repair.py"),
                   "--submit", "--max-new-tasks", str(2 - len(active))]
        result = subprocess.run(command, capture_output=True, text=True)
        started = [line for line in result.stdout.splitlines() if line.startswith("STARTED ")]
        for line in started:
            log(line)
        if result.returncode:
            log(f"STOP: submitter returned {result.returncode}: {result.stderr.strip()} {result.stdout[-1000:]}")
            return 4
        if not started and not active:
            log(f"DONE: all tile exports complete; {hours:.4f} combined EECU-hours")
            return 0
        if not started:
            message = f"WAIT: {len(active)} repair tile active; {hours:.4f} EECU-hours combined"
            if message != last_message:
                log(message)
                last_message = message
        else:
            last_message = None
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
