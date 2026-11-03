"""Recover the audited state/DC production index from a raw positive-overlap index.

Filter after float32 storage, matching the production produce.py step. Never
rewrite an input or an existing destination. This module does not alter rasters.
"""
import argparse
from pathlib import Path
import numpy as np

COVERAGE_MIN = 1.0 / 255.0
REQUIRED = ('tract', 'cell', 'coverage', 'native_idx')

def filter_arrays(arrays):
    for name in REQUIRED:
        if name not in arrays:
            raise ValueError(f'Missing index array: {name}')
    cov = arrays['coverage']
    if cov.dtype != np.float32 or cov.ndim != 1:
        raise ValueError('Expected the production raw index coverage as a 1-D float32 array')
    if any(a.ndim != 1 or len(a) != len(cov) for a in arrays.values()):
        raise ValueError('Index arrays must be aligned one-dimensional arrays')
    if not np.isfinite(cov).all() or np.any(cov <= 0) or np.any(cov > 1):
        raise ValueError('Raw coverage must be finite, positive and no greater than one')
    keep = cov >= np.float32(COVERAGE_MIN)
    return {name: values[keep] for name, values in arrays.items()}

def filter_index(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Source and destination must be different')
    with np.load(source, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    result = filter_arrays(arrays)
    # Exclusive creation prevents replacing any existing production artifact.
    with destination.open('xb') as f:
        np.savez(f, **result)
    return len(arrays['coverage']), len(result['coverage'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_index', type=Path)
    parser.add_argument('output_index', type=Path)
    args = parser.parse_args()
    before, after = filter_index(args.raw_index, args.output_index)
    print(f'Kept {after} of {before} pairs with coverage >= 1/255')
