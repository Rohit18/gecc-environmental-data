"""Download native-grid tiles of LAI_Global_Mosaic_2020 from Earth Engine with an EECU budget.

Every computePixels request runs with the Earth Engine profiler on. Profiles are
summed in batches and appended to a per-project ledger (JSONL). The run stops
submitting tiles when either
  * this run's measured EECU-seconds reach --max-eecu-hours, or
  * the ledger total for the project reaches --ledger-cap-hours,
with measured values multiplied by --margin and tiles whose profile has not been
read yet charged at --pending-cost seconds each.

Tiles are 2048x2048 px on the asset's own EPSG:4326 grid, written as int32
DEFLATE GeoTIFFs with nodata=-1 (pixels masked in Earth Engine). Each request
returns the LAI band plus the image mask, and -1 is written locally where the
mask is 0. Filling the mask on the server with Image.unmask cost ~5x more EECU
per tile, so it is not used. Existing tiles are skipped, so the script can be
rerun to resume.
"""
import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import re
import sys
import time
import warnings

import numpy as np

ASSET = 'projects/your-ee-project/assets/LAI_Global_Mosaic_2020'
SX = 8.983152841195215e-05
TX, TY = -180.00001488697754, 90.00005235925298
W, H = 4007502, 2003752
T = 2048
NODATA = -1
PROFILE_CALL_COST = 0.05  # EECU-s charged for each Profile.getProfiles call (not itself profiled)
UNREADABLE_COST = 1.0     # EECU-s charged for a request whose profile cannot be read


def tile_path(out, r, c):
    return os.path.join(out, 'tiles', f'r{r:04d}', f'r{r:04d}_c{c:04d}.tif')


def _init_worker(project):
    import ee
    warnings.filterwarnings('ignore', message='.*restricted mode.*')
    ee.Initialize(project=project)


def fetch_tile(args):
    """Fetch one tile; returns (r, c, status, profile_ids, seconds)."""
    import ee
    import rasterio
    from rasterio.transform import Affine

    r, c, out = args
    path = tile_path(out, r, c)
    if os.path.exists(path):
        return r, c, 'exists', [], 0.0
    w, h = min(T, W - c * T), min(T, H - r * T)
    x0, y0 = TX + c * T * SX, TY - r * T * SX
    img = ee.Image(ASSET)
    img = img.addBands(img.mask().rename('m'))
    ids = []
    t0 = time.time()
    for attempt in range(6):
        try:
            with ee.data.profiling(ids.append):
                arr = ee.data.computePixels({
                    'expression': img,
                    'fileFormat': 'NUMPY_NDARRAY',
                    'grid': {
                        'dimensions': {'width': w, 'height': h},
                        'affineTransform': {'scaleX': SX, 'shearX': 0, 'translateX': x0,
                                            'shearY': 0, 'scaleY': -SX, 'translateY': y0},
                        'crsCode': 'EPSG:4326',
                    },
                })
            break
        except Exception as e:  # transient: quota/concurrency/5xx/timeouts
            msg = str(e)
            if attempt == 5 or not re.search(r'429|Too many|concurren|500|502|503|504|timed out|Timeout|reset|Deadline', msg, re.I):
                return r, c, f'error: {msg[:300]}', ids, time.time() - t0
            time.sleep(min(60, 2 ** attempt * 2))
    a = np.ascontiguousarray(arr['LAI'], dtype=np.int32)
    a[arr['m'] == 0] = NODATA
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with rasterio.open(tmp, 'w', driver='GTiff', width=w, height=h, count=1, dtype='int32',
                       crs='EPSG:4326', transform=Affine(SX, 0, x0, 0, -SX, y0), nodata=NODATA,
                       compress='DEFLATE', predictor=2, zlevel=6, tiled=True,
                       blockxsize=512, blockysize=512) as dst:
        dst.write(a, 1)
    os.replace(tmp, path)
    return r, c, 'ok', ids, time.time() - t0


def _sum_profile(text):
    total = 0.0
    for line in text.splitlines():
        tok = line.split()
        if tok and re.fullmatch(r'\d+(\.\d+)?', tok[0]):
            total += float(tok[0])
    return total


def profile_eecu_seconds(ids, attempts=8):
    """Sum the EECU-s column of the aggregated profile for these request ids.

    Returns (eecu_s, n_calls, unreadable_ids). If the whole batch cannot be read,
    it is split in halves; a single id that still cannot be read is returned in
    unreadable_ids so the caller can charge it at UNREADABLE_COST.
    """
    import ee
    get_profiles = ee.ApiFunction.lookup('Profile.getProfiles').call
    for attempt in range(attempts):
        try:
            return _sum_profile(get_profiles(ids=ids).getInfo()), attempt + 1, []
        except Exception:
            time.sleep(3 * (attempt + 1))
    if len(ids) == 1:
        return 0.0, attempts, list(ids)
    half = len(ids) // 2
    e1, n1, u1 = profile_eecu_seconds(ids[:half], attempts=3)
    e2, n2, u2 = profile_eecu_seconds(ids[half:], attempts=3)
    return e1 + e2, attempts + n1 + n2, u1 + u2


def ledger_total(path, project):
    tot = 0.0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('project') == project:
                    tot += rec['eecu_s']
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True)
    ap.add_argument('--tiles', required=True, help='text file of "row col" lines')
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='only the first N tiles (pilot)')
    ap.add_argument('--max-eecu-hours', type=float, required=True, help='cap for this run')
    ap.add_argument('--ledger-cap-hours', type=float, default=100.0, help='cap across all runs on this project')
    ap.add_argument('--margin', type=float, default=1.25, help='multiplier on measured EECU for cap checks')
    ap.add_argument('--pending-cost', type=float, default=0.3, help='EECU-s charged per tile not yet profiled')
    ap.add_argument('--batch', type=int, default=200, help='tiles per profile read / ledger entry')
    a = ap.parse_args()

    import ee
    warnings.filterwarnings('ignore', message='.*restricted mode.*')
    ee.Initialize(project=a.project)

    ledger = os.path.join(a.out, f'eecu_ledger_{a.project}.jsonl')
    prior = ledger_total(ledger, a.project)
    run_cap = a.max_eecu_hours * 3600
    global_cap = a.ledger_cap_hours * 3600

    tiles = np.loadtxt(a.tiles, dtype=int, ndmin=2)
    todo = [(int(r), int(c), a.out) for r, c in tiles if not os.path.exists(tile_path(a.out, r, c))]
    if a.limit:
        todo = todo[:a.limit]
    print(f'{dt.datetime.now():%F %T} project={a.project} tiles_in_list={len(tiles)} to_fetch={len(todo)} '
          f'ledger_prior={prior/3600:.3f} EECU-h run_cap={a.max_eecu_hours} h ledger_cap={a.ledger_cap_hours} h '
          f'margin={a.margin}', flush=True)
    if prior * a.margin >= global_cap:
        sys.exit('ledger cap already reached; not starting')

    run_measured = 0.0
    pending_ids, pending_tiles = [], 0
    n_ok = n_err = n_done = 0
    t_start = time.time()
    stopped = None

    def flush_batch():
        nonlocal run_measured, pending_ids, pending_tiles
        if not pending_tiles and not pending_ids:
            return
        measured, n_calls, unreadable = profile_eecu_seconds(pending_ids) if pending_ids else (0.0, 0, [])
        eecu = measured + n_calls * PROFILE_CALL_COST + len(unreadable) * UNREADABLE_COST
        run_measured += eecu
        with open(ledger, 'a') as f:
            f.write(json.dumps({'time': dt.datetime.now().isoformat(timespec='seconds'), 'project': a.project,
                                'asset': ASSET, 'tiles': pending_tiles, 'requests': len(pending_ids),
                                'profiled_eecu_s': round(measured, 4), 'profile_calls': n_calls,
                                'unreadable_profiles': len(unreadable), 'eecu_s': round(eecu, 4),
                                'profile_ids': pending_ids}) + '\n')
        if unreadable:
            print(f'  {len(unreadable)} profile(s) unreadable, charged {UNREADABLE_COST} EECU-s each', flush=True)
        pending_ids, pending_tiles = [], 0

    def over_cap():
        committed = (run_measured * a.margin) + pending_tiles * a.pending_cost
        if committed >= run_cap:
            return f'run cap: committed {committed/3600:.3f} EECU-h >= {a.max_eecu_hours} h'
        if (prior + run_measured) * a.margin + pending_tiles * a.pending_cost >= global_cap:
            return f'ledger cap: {(prior + run_measured)/3600:.3f} measured EECU-h x margin >= {a.ledger_cap_hours} h'
        return None

    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(a.project,))
    try:
        for r, c, status, ids, secs in pool.imap_unordered(fetch_tile, todo, chunksize=1):
            n_done += 1
            pending_ids += ids
            if status == 'ok':
                n_ok += 1
                pending_tiles += 1
            elif status.startswith('error'):
                n_err += 1
                pending_tiles += 1
                print(f'  tile r{r} c{c} {status}', flush=True)
            if pending_tiles >= a.batch:
                flush_batch()
                el = time.time() - t_start
                rate = n_done / el
                print(f'{dt.datetime.now():%F %T} done={n_done}/{len(todo)} ok={n_ok} err={n_err} '
                      f'run_eecu={run_measured/3600:.4f} h ({run_measured/max(n_ok + n_err, 1):.3f} s/tile) '
                      f'ledger={(prior + run_measured)/3600:.3f} h rate={rate:.2f} tiles/s '
                      f'eta={(len(todo) - n_done)/max(rate, 1e-9)/60:.0f} min', flush=True)
            stopped = over_cap()
            if stopped or n_err > max(50, 0.02 * n_done):
                stopped = stopped or f'too many errors ({n_err})'
                pool.terminate()
                break
    finally:
        pool.terminate()
        pool.join()
        flush_batch()
    el = time.time() - t_start
    print(f'{dt.datetime.now():%F %T} FINISHED ok={n_ok} err={n_err} of {len(todo)} in {el/60:.1f} min; '
          f'run_eecu={run_measured:.1f} s = {run_measured/3600:.4f} h; '
          f'ledger_total={(prior + run_measured)/3600:.4f} h' + (f'; STOPPED: {stopped}' if stopped else ''),
          flush=True)


if __name__ == '__main__':
    main()
