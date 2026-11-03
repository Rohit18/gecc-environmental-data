#!/usr/bin/env python3
"""Run the version-1 Census Tract NDVI/EVI export queue with EECU controls.

Restart-safe. Phase 1 submits the 19 per-batch pixel-total exports, phase 2 the
NDVI/EVI exports (1985-2025, JJA and same-year DJF, 19 batches), always through
submit_census_tract_ndvi_evi_exports.py, which skips tasks that are active,
completed or downloaded, retries a failed task once at a higher tileScale, and
gives up after two failures.

Controls:
- keeps at most MAX_ACTIVE_TASKS queued or running;
- records the largest reported EECU usage of every task in a ledger;
- stops submitting when measured EECU-hours plus a projection for active tasks
  would pass CAP_HOURS; if measured usage alone passes the cap, cancels queued
  (not running) tasks and stops;
- pauses (stops submitting) if a finished task used more than ANOMALY_FACTOR
  times the EECU its batch area predicts, from the median rate of finished
  tasks of the same season;
- stops if more than MAX_GIVEN_UP tasks have failed twice.

Earth Engine reports usage asynchronously, so the cap is an operational guard,
not a strict limit.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ee

import submit_census_tract_ndvi_evi_exports as wf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/census_tract_ndvi_evi_v1"
LEDGER = OUT / "eecu_ledger.json"
PROJECT = "your-ee-project"
CAP_HOURS = 75000.0
MAX_ACTIVE_TASKS = 40
POLL_SECONDS = 120
ANOMALY_FACTOR = 4.0
MIN_TASKS_FOR_ANOMALY = 10
MIN_GROUP_TASKS_FOR_ANOMALY = 5
GROUPED_ANOMALY_FACTOR = 10.0
# Grouped tasks are flagged only above this absolute cost too: many Alaska
# winter groups have no scenes (0 EECU-h), so ratios to a near-zero median are
# meaningless (a normal 1.2 EECU-h task read as 13,550x). Largest Alaska task
# seen so far: 11.3 EECU-h.
GROUPED_ANOMALY_MIN_HOURS = 25.0
GROUP_KEY_RE = re.compile(r"_b\d{3}_(\d+m_p\d{2})$")
MAX_GIVEN_UP = 20
MAX_API_FAILURES = 60  # consecutive failed polls (~2 h) before giving up
DEFAULT_TASK_HOURS = 40.0  # projection for an active task before any finish
ACTIVE = {"PENDING", "READY", "RUNNING", "CANCEL_REQUESTED", "CANCELLING"}
QUEUED = {"PENDING", "READY"}
TASK_RE = re.compile(r"_(JJA|DJFsameYear)_(\d{4})_b(\d{3})(?:_\d+m_p\d{2})?$")
CHUNKS: list[str] = []  # empty = submitter default (all batches except deferred)
ALSO_CHECK: list[str] = []  # other projects whose finished tasks count as done


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def batch_areas_km2(ledger: dict) -> dict:
    if "areas_km2" not in ledger:
        areas = {}
        for chunk in wf.REGION_STATEFPS:
            fc = wf.chunk_source(chunk)
            areas[str(chunk)] = (
                ee.Number(fc.aggregate_sum("Area_Land"))
                .add(fc.aggregate_sum("Area_Water"))
                .divide(1e6)
                .getInfo()
            )
        ledger["areas_km2"] = areas
    return ledger["areas_km2"]


def snapshot(ledger: dict) -> tuple[float, list, dict]:
    tasks = ledger.setdefault("tasks", {})
    active = []
    for operation in ee.data.listOperations(f"projects/{PROJECT}"):
        meta = operation.get("metadata", {})
        name = meta.get("description", "")
        if not name.startswith(wf.TASK_PREFIX):
            continue
        seconds = float(meta.get("batchEecuUsageSeconds", 0) or 0)
        key = operation["name"]  # one entry per attempt
        rec = tasks.get(key, {"description": name})
        rec["eecu_s"] = max(seconds, rec.get("eecu_s", 0.0))
        rec["state"] = meta.get("state")
        rec["start"] = meta.get("startTime")
        rec["end"] = meta.get("endTime")
        tasks[key] = rec
        if rec["state"] in ACTIVE:
            active.append(operation)
    ledger["measured_hours"] = sum(r["eecu_s"] for r in tasks.values()) / 3600
    ledger["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LEDGER.write_text(json.dumps(ledger, indent=1, sort_keys=True))
    return ledger["measured_hours"], active, tasks


def finished_hours(tasks: dict) -> list[float]:
    return [r["eecu_s"] / 3600 for r in tasks.values() if r.get("state") == "SUCCEEDED"]


def anomalies(tasks: dict, areas: dict) -> list[str]:
    """Flag finished tasks that cost far more than comparable ones.

    Whole batches: EECU per km2 of the batch, against the median of the same
    season. Grouped batches (Alaska tiers/parts differ ~1000x in cost): EECU
    against the same group's median in the same season over other years, with a
    looser GROUPED_ANOMALY_FACTOR because early years have far fewer scenes.
    """
    batch_rates: dict[str, list[float]] = {}
    group_costs: dict[tuple[str, str], list[float]] = {}
    done = []
    for rec in tasks.values():
        match = TASK_RE.search(rec["description"])
        if rec.get("state") != "SUCCEEDED" or not match:
            continue
        season, chunk = match.group(1), match.group(3).lstrip("0") or "0"
        group = GROUP_KEY_RE.search(rec["description"])
        if not group and rec["eecu_s"] <= 0:
            continue
        if group:  # zero-cost tasks count: empty winters are real data points
            key = (season, group.group(1))
            group_costs.setdefault(key, []).append(rec["eecu_s"])
            done.append((rec["description"], "group", key, rec["eecu_s"]))
        else:
            rate = rec["eecu_s"] / areas[chunk]
            batch_rates.setdefault(season, []).append(rate)
            done.append((rec["description"], "batch", season, rate))
    flagged = []
    for description, kind, key, value in done:
        pool = group_costs[key] if kind == "group" else batch_rates[key]
        need = MIN_GROUP_TASKS_FOR_ANOMALY if kind == "group" else MIN_TASKS_FOR_ANOMALY
        factor = GROUPED_ANOMALY_FACTOR if kind == "group" else ANOMALY_FACTOR
        if len(pool) < need:
            continue
        typical = statistics.median(pool)
        if kind == "group" and value / 3600 < GROUPED_ANOMALY_MIN_HOURS:
            continue
        if value > factor * typical:
            label = f"{key[1]} {key[0]}" if kind == "group" else key
            flagged.append(f"{description}: {value / typical:.1f}x the median {label}")
    return flagged


def run_submitter(pixel_totals: bool, max_new: int) -> tuple[int, list[str], int, int, str]:
    command = [sys.executable, str(ROOT / "submit_census_tract_ndvi_evi_exports.py"),
               "--project", PROJECT, "--submit", "--max-new-tasks", str(max_new)]
    if ALSO_CHECK:
        command += ["--also-check-projects", ",".join(ALSO_CHECK)]
    if pixel_totals:
        command.append("--pixel-totals")
    if CHUNKS:
        command += ["--chunks", ",".join(CHUNKS)]
    result = subprocess.run(command, text=True, capture_output=True)
    out = result.stdout
    started = [line for line in out.splitlines() if line.startswith("STARTED: ")]
    available = re.search(r"available to submit: (\d+)", out)
    given_up = re.search(r"failed \d\+ times, not resubmitted: (\d+)", out)
    return (result.returncode, started,
            int(available.group(1)) if available else -1,
            int(given_up.group(1)) if given_up else 0,
            (result.stderr or "")[-2000:] + out[-2000:])


def main() -> int:
    global LEDGER, CHUNKS, PROJECT, CAP_HOURS, ALSO_CHECK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="",
                        help="Comma-separated batches to run (default: submitter default, "
                             "i.e. all except deferred batches such as Alaska 14).")
    parser.add_argument("--project", default=PROJECT,
                        help="Earth Engine project that runs (and is billed for) the tasks.")
    parser.add_argument("--also-check-projects", default="",
                        help="Other projects whose finished/active tasks count as done.")
    parser.add_argument("--cap", type=float, default=CAP_HOURS,
                        help="EECU-hour cap, measured over this project's tasks with the run's prefix.")
    parser.add_argument("--ledger", default=str(LEDGER),
                        help="Ledger JSON; use a separate file for a separate run (e.g. Alaska).")
    args = parser.parse_args()
    CHUNKS = [c.strip() for c in args.chunks.split(",") if c.strip()]
    PROJECT, CAP_HOURS = args.project, args.cap
    ALSO_CHECK = [p.strip() for p in args.also_check_projects.split(",") if p.strip()]
    LEDGER = Path(args.ledger)
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    ee.Initialize(project=PROJECT)
    areas = batch_areas_km2(ledger)
    log(f"START: project {PROJECT} (also checking {ALSO_CHECK or 'none'}), "
        f"cap {CAP_HOURS:.0f} EECU-h, max {MAX_ACTIVE_TASKS} active, prefix {wf.TASK_PREFIX}, "
        f"chunks {CHUNKS or 'default'}, ledger {LEDGER.name}")
    phases = [("pixel totals", True), ("NDVI/EVI", False)]
    phase = ledger.get("phase", 0)
    last = None
    api_failures = 0
    while phase < len(phases):
        label, pixel_totals = phases[phase]
        try:
            hours, active, tasks = snapshot(ledger)
            api_failures = 0
        except Exception as exc:  # network or API outage: wait and retry
            api_failures += 1
            if api_failures > MAX_API_FAILURES:
                log(f"STOP: Earth Engine unreachable for {api_failures} polls: {exc}")
                return 7
            log(f"WAIT: Earth Engine unreachable (attempt {api_failures}): {str(exc)[:200]}")
            time.sleep(POLL_SECONDS)
            continue
        if hours >= CAP_HOURS:
            queued = [op for op in active if op["metadata"].get("state") in QUEUED]
            for op in queued:
                ee.data.cancelOperation(op["name"])
            log(f"STOP: measured {hours:,.1f} EECU-h reached the cap; cancelled {len(queued)} queued task(s)")
            return 3
        flagged = anomalies(tasks, areas)
        if flagged:
            log("PAUSE: unexpectedly expensive task(s): " + "; ".join(flagged[:5]))
            return 5
        per_task = statistics.median(finished_hours(tasks)) if finished_hours(tasks) else DEFAULT_TASK_HOURS
        projected = hours + len(active) * per_task
        room = MAX_ACTIVE_TASKS - len(active)
        affordable = int(max(0.0, CAP_HOURS - projected) // max(per_task, 1e-6))
        to_submit = min(room, affordable)
        status = (f"{label}: {len(active)} active, measured {hours:,.1f} EECU-h, "
                  f"projected {projected:,.1f}, median task {per_task:.2f} EECU-h")
        if to_submit <= 0 and not active:
            log(f"STOP: the cap leaves no room for another task; {status}")
            return 3
        if to_submit <= 0:
            if status != last:
                log(f"WAIT: {status}")
                last = status
            time.sleep(POLL_SECONDS)
            continue
        code, started, available, given_up, tail = run_submitter(pixel_totals, to_submit)
        for line in started:
            log(line)
        if code:  # usually a transient network/API error; retry, give up after ~2 h
            api_failures += 1
            if api_failures > MAX_API_FAILURES:
                log(f"STOP: submitter kept failing ({api_failures}x), last exit {code}: {tail}")
                return 4
            log(f"WAIT: submitter exited {code} (attempt {api_failures}): {tail[-300:]}")
            time.sleep(POLL_SECONDS)
            continue
        if given_up > MAX_GIVEN_UP:
            log(f"STOP: {given_up} tasks failed twice; investigate before continuing")
            return 6
        if not started and available == 0:
            if active:
                if status != last:
                    log(f"WAIT: {status} (nothing left to submit in this phase)")
                    last = status
                time.sleep(POLL_SECONDS)
                continue
            log(f"PHASE DONE: {label}; measured {hours:,.1f} EECU-h; {given_up} task(s) given up")
            phase += 1
            ledger["phase"] = phase
            LEDGER.write_text(json.dumps(ledger, indent=1, sort_keys=True))
            continue
        last = None
        time.sleep(POLL_SECONDS)
    hours, _, _ = snapshot(ledger)
    log(f"DONE: all phases complete; measured {hours:,.1f} EECU-h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
