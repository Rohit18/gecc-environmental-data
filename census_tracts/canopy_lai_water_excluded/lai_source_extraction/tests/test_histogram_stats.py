#!/usr/bin/env python3
"""Tests for lai_common.histogram_stats. Run:  python tests/test_histogram_stats.py"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lai_common import STAT_COLUMNS, histogram_stats  # noqa: E402


def close(a, b, tol=1e-12):
    return a is not None and abs(a - b) <= tol


def test_equal_weights():
    s = histogram_stats([0, 1, 2, 3], [1, 1, 1, 1], 4)
    assert close(s["lai_native_mean"], 0.15)
    assert close(s["lai_native_stdDev"], math.sqrt(1.25) / 10)          # population, not sample
    assert s["lai_native_p25"] == 0.0 and s["lai_native_median"] == 0.1 and s["lai_native_p75"] == 0.2
    assert s["lai_native_min"] == 0.0 and s["lai_native_max"] == 0.3 and s["lai_native_count"] == 4


def test_zero_is_valid():
    s = histogram_stats([0], [1.0], 7)
    assert s["lai_native_mean"] == 0.0 and s["lai_native_median"] == 0.0 and s["lai_native_count"] == 7


def test_no_valid_pixels_all_null():
    s = histogram_stats([], [], 0)
    assert s["lai_native_count"] == 0
    assert all(s[c] is None for c in STAT_COLUMNS if c != "lai_native_count")


def test_zero_weight_values_have_no_support():
    s = histogram_stats([0, 5, 9], [0.0, 1.0, 1.0], 2)
    assert s["lai_native_min"] == 0.5 and s["lai_native_max"] == 0.9


def test_unsorted_input():
    a = histogram_stats([3, 0, 2, 1], [1, 1, 1, 1], 4)
    b = histogram_stats([0, 1, 2, 3], [1, 1, 1, 1], 4)
    assert a == b


def test_exact_tie_with_float_fractions():
    # ten equal shares of 0.1: cumulative 0.5 is reached exactly at value 4 (float sums give 0.5000000000000001)
    s = histogram_stats(list(range(10)), [0.1] * 10, 10)
    assert s["lai_native_median"] == 0.4
    assert s["lai_native_p25"] == 0.2   # cum 0.3 >= 0.25 first at value 2
    # a cumulative sum that falls just short of p*total by round-off must still select the value
    w = [1 / 3, 1 / 6, 1 / 2]
    assert histogram_stats([1, 2, 3], w, 3)["lai_native_median"] == 0.2


def test_matches_brute_force_expansion():
    rng = np.random.default_rng(0)
    for _ in range(500):
        k = rng.integers(1, 12)
        v = rng.choice(np.arange(0, 70), size=k, replace=False)
        w = rng.integers(1, 20, size=k)
        s = histogram_stats(v, w / w.sum(), int(w.sum()))   # frac-style normalized weights
        x = np.sort(np.repeat(v, w))
        n = x.size
        assert close(s["lai_native_mean"], x.mean() / 10, 1e-12)
        assert close(s["lai_native_stdDev"], x.std(ddof=0) / 10, 1e-12)
        for p, col in ((0.25, "lai_native_p25"), (0.5, "lai_native_median"), (0.75, "lai_native_p75")):
            first = x[max(math.ceil(p * n) - 1, 0)]       # first value with cumulative count >= p*n
            assert s[col] == first / 10, (col, v, w, s[col], first)
        assert s["lai_native_min"] == x[0] / 10 and s["lai_native_max"] == x[-1] / 10


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
