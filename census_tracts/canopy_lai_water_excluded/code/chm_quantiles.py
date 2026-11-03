"""Stable inverse-CDF quantiles from coverage-weighted 256-bin canopy histograms.

Replaces the quantile step of chm1m_stats.stats_from_hist and merge_chm1m.py,
which computed ``c = np.cumsum(h / h.sum())`` and ``np.searchsorted(c, q, "left")``.
Normalising and then summing in float64 can leave an exact cumulative boundary
just below ``q`` and so select the next occupied bin.

Definition (unchanged from the release intent; inverse empirical CDF, lower at ties)

    Q(q) = min{ k : C_k >= q * N - eps }      C_k = sum_{j<=k} h_j,  N = C_255

evaluated in exact rational arithmetic on the stored float64 weights:

* ``q`` is a rational (1/2, 19/20), never a binary float.
* Every stored weight is converted exactly with ``float.as_integer_ratio`` and
  scaled to a common power-of-two denominator, so C_k and N are exact integers
  and no rounding happens inside the reducer.
* ``eps = N / 2**TIE_BAND_LOG2`` with TIE_BAND_LOG2 = 44. The band exists only
  for rounding already present in the stored weights: exactextract stores each
  bin as fl(fl(w/cnt) * cnt) (<= 2u relative), per-tile pooling adds <= T*u*N and
  per-chunk pooling <= P*u*N, u = 2**-53. With 2 + T + P <= 512 that bound is
  <= 2**-44 * N. The band cannot merge distinct integer cumulative weights while
  N < 2**44 (then eps < 1 pixel); ``quantile`` raises if N >= 2**44.
* A cumulative weight inside [q*N - eps, q*N) counts as reaching q*N, so an
  exact tie resolves to the lower bin, as the inverse-CDF definition requires.
  It is not the NumPy interpolated median, and never averages two bins.

No-data rules: an empty histogram (N == 0) returns NaN. Negative, non-finite,
or wrongly sized histograms raise; they are never treated as zero canopy.
"""
from fractions import Fraction
import math
import numpy as np

NBINS = 256
TIE_BAND_LOG2 = 44
QUANTILES = {"chm_median_m": Fraction(1, 2), "chm_p95_m": Fraction(19, 20)}


def _validate(h):
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 1 or h.size != NBINS:
        raise ValueError(f"histogram must have {NBINS} bins, got shape {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError("histogram has non-finite weights")
    if np.any(h < 0):
        raise ValueError("histogram has negative weights")
    return h


def exact_weights(h):
    """Stored float64 weights as exact integers over one common 2**k denominator."""
    h = _validate(h)
    ratios = [float(x).as_integer_ratio() for x in h]
    den = max(d for _, d in ratios)          # all denominators are powers of two
    return [n * (den // d) for n, d in ratios], den


def select_bin(w, q, band_log2=TIE_BAND_LOG2):
    """Inverse-CDF bin from exact integer weights ``w``; returns (bin, exact gap).

    gap = (C_k - q*N) / N at the chosen bin (negative only inside the tie band).
    ``band_log2=None`` means no band: the exact quantile of the stored weights.
    """
    q = Fraction(q)
    if not (0 < q <= 1):
        raise ValueError("q must be in (0, 1]")
    n = sum(w)
    if n == 0:
        return float("nan"), None
    # comparisons are scaled by q.denominator * 2**band_log2 to stay in integers
    a, b = q.numerator, q.denominator
    band = 0 if band_log2 is None else band_log2
    scale = b << band
    target = a * n << band                    # q*N scaled
    eps = 0 if band_log2 is None else b * n   # (N / 2**band) scaled
    c = 0
    for k, x in enumerate(w):
        c += x
        if x and c * scale >= target - eps:
            return float(k), Fraction(c * b - a * n, b * n)
    raise AssertionError("cumulative weight never reached q*N")  # impossible for q <= 1


def quantile_with_gap(h, q, band_log2=TIE_BAND_LOG2):
    """(bin, gap) for a float64 histogram; (nan, None) when it is empty."""
    w, _ = exact_weights(h)
    return select_bin(w, q, band_log2)


def quantile(h, q, band_log2=TIE_BAND_LOG2):
    """Inverse-CDF quantile (lower at exact ties) of a 256-bin histogram."""
    h = _validate(h)
    if h.sum() >= 2.0 ** band_log2:
        raise ValueError("total weight >= 2**band_log2: the tie band could merge distinct pixels")
    return quantile_with_gap(h, q, band_log2)[0]


def stats_from_hist(h, total_weight=None):
    """Drop-in replacement for chm1m_stats.stats_from_hist.

    Every non-quantile statistic is computed by exactly the same float64
    expressions as the archived function, so those values are bit-identical.
    Only chm_median_m and chm_p95_m change, and only where the archived
    searchsorted crossed an exact cumulative boundary.
    """
    h = _validate(h)
    n = h.sum()
    if n <= 0:
        return None
    v = np.arange(NBINS, dtype=np.float64)
    p = h / n
    mean = float((v * p).sum())
    var = float((v * v * p).sum() - mean * mean)
    gt0 = h[1:].sum()
    mean_gt0 = float((v[1:] * h[1:]).sum() / gt0) if gt0 > 0 else np.nan
    return dict(
        chm_mean_m=mean,
        chm_mean_gt0_m=mean_gt0,
        chm_px_gt0_frac=float(gt0 / n),
        chm_cover_gt1m_frac=float(h[2:].sum() / n),
        chm_cover_ge1m_frac=float(h[1:].sum() / n),
        chm_cover_gt5m_frac=float(h[6:].sum() / n),
        chm_median_m=quantile(h, QUANTILES["chm_median_m"]),
        chm_p95_m=quantile(h, QUANTILES["chm_p95_m"]),
        chm_max_m=float(np.max(np.nonzero(h)[0])),
        chm_stdev_m=float(np.sqrt(max(var, 0.0))),
        n_px=float(n),
    )


def archived_quantile(h, q):
    """The archived computation, kept verbatim for regression comparison."""
    h = np.asarray(h, dtype=np.float64)
    c = np.cumsum(h / h.sum())
    return float(np.searchsorted(c, q, side="left"))
