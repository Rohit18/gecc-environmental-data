#!/usr/bin/env python3
"""Run the authorized WorldCover add-on queue with measured EECU controls.

The process can be restarted safely. It submits at most two exports, waits for
both to finish, and records the largest reported EECU usage for every task.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/worldcover_2020_nonconus"
LEDGER = OUT / "eecu_ledger.json"
PROJECT = "your-ee-project"
WC_PREFIX = "WC2020_NONCONUS_tract_"
GLC_PREFIX = "GLC_NONCONUS_CensusTract_v1"
STOP_HOURS = 1000.0
RESERVE_HOURS = 50.0
POLL_SECONDS = 30
MAX_ACTIVE_TASKS = 5
STATES = ("72", "15", "60", "66", "69", "78", "02")
ACTIVE = {"PENDING", "READY", "RUNNING", "CANCEL_REQUESTED"}
SUPERSEDED_FAILURES = {
    "WC2020_NONCONUS_tract_v2_areas_s02_n25_a20000_b000",
    "WC2020_NONCONUS_tract_v3_areas_s02_n25_a20000_b001",
    "WC2020_NONCONUS_tract_v3_areas_s02_n25_a20000_b031",
    "WC2020_NONCONUS_tract_v3_areas_s02_n25_a20000_b033",
    "WC2020_NONCONUS_tract_v3_areas_s02_n25_a20000_b037",
    "WC2020_NONCONUS_tract_v3_areas_s02_n25_a20000_b041",
    "WC2020_NONCONUS_tract_v3_areas_s02_n25_a20000_b042",
}


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def snapshot(ledger):
    operations = ee.data.listOperations(f"projects/{PROJECT}")
    active, failed = [], []
    for operation in operations:
        meta = operation.get("metadata", {})
        name = meta.get("description", "")
        if not name.startswith((WC_PREFIX, GLC_PREFIX)):
            continue
        seconds = float(meta.get("batchEecuUsageSeconds", 0))
        ledger[name] = max(seconds, ledger.get(name, 0))
        state = meta.get("state")
        if name.startswith(WC_PREFIX) and state in ACTIVE:
            active.append(operation)
        if (name.startswith(WC_PREFIX) and name not in SUPERSEDED_FAILURES
                and state in {"FAILED", "CANCELLED"}):
            failed.append((name, state, operation.get("error")))
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    return sum(ledger.values()) / 3600, active, failed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    ee.Initialize(project=PROJECT)
    last_message = None
    while True:
        hours, active, failed = snapshot(ledger)
        if failed:
            log(f"STOP: {len(failed)} failed/cancelled task(s): {failed}")
            return 2
        if hours >= STOP_HOURS - RESERVE_HOURS:
            for operation in active:
                ee.data.cancelOperation(operation["name"])
            log(f"STOP: {hours:.4f} EECU-hours reached the {STOP_HOURS-RESERVE_HOURS:.0f}-hour guard")
            return 3
        if len(active) >= MAX_ACTIVE_TASKS:
            message = f"WAIT: {len(active)} active, {hours:.4f} EECU-hours measured"
            if message != last_message:
                log(message)
                last_message = message
            time.sleep(POLL_SECONDS)
            continue
        command = [sys.executable, str(ROOT / "submit_nonconus_census_tract_worldcover_exports.py"),
                   "--project", PROJECT, "--states", *STATES, "--allow-alaska",
                   "--batch-size", "25", "--alaska-max-area-km2", "20000",
                   "--max-new-tasks", str(min(2, MAX_ACTIVE_TASKS - len(active))),
                   "--eecu-cap-hours", str(STOP_HOURS),
                   "--allow-active", "--submit"]
        result = subprocess.run(command, text=True, capture_output=True)
        started = [line for line in result.stdout.splitlines() if line.startswith("STARTED ")]
        for line in started:
            log(line)
        if result.returncode:
            log(f"STOP: submitter returned {result.returncode}: {result.stderr.strip()} {result.stdout[-2000:]}")
            return 4
        if not started:
            if active:
                message = f"WAIT: remaining {len(active)} active, {hours:.4f} EECU-hours measured"
                if message != last_message:
                    log(message)
                    last_message = message
                time.sleep(POLL_SECONDS)
                continue
            log(f"DONE: no pending batches; {hours:.4f} EECU-hours measured")
            return 0
        last_message = None
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
