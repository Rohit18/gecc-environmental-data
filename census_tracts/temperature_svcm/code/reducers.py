"""Per-tract reductions of a native-grid composite through a sampling index.

Index arrays: tract (int), native_idx (int, -1 = outside source grid), and for the
touched rule a coverage fraction per sample. Samples whose composite value is nodata
are not valid samples."""
import numpy as np

NODATA = -9999.0

def reduce(composite, tract, native_idx, n_tracts, coverage=None):
    ok = native_idx >= 0
    v = np.full(tract.shape, np.nan)
    v[ok] = composite.ravel()[native_idx[ok]]
    ok &= (v != NODATA)
    t = tract[ok]; x = v[ok]
    w = coverage[ok].astype(np.float64) if coverage is not None else None
    cnt = np.bincount(t, minlength=n_tracts)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_u = np.bincount(t, x, n_tracts) / cnt
        dev = x - mean_u[t]
        var_pop = np.bincount(t, dev * dev, n_tracts) / cnt
        out = {"count": cnt.astype(np.int64), "mean_unweighted": mean_u,
               "sd_pop": np.sqrt(var_pop), "sd_sample": np.sqrt(var_pop * cnt / (cnt - 1))}
        if w is not None:
            sw = np.bincount(t, w, n_tracts)
            mean_w = np.bincount(t, w * x, n_tracts) / sw
            dw = x - mean_w[t]
            out["mean_weighted"] = mean_w
            out["sd_weighted"] = np.sqrt(np.bincount(t, w * dw * dw, n_tracts) / sw)
    order = np.lexsort((x, t)); xs = x[order]
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    med = np.full(n_tracts, np.nan); has = cnt > 0
    med[has] = (xs[starts[has] + (cnt[has] - 1) // 2] + xs[starts[has] + cnt[has] // 2]) / 2
    out["median"] = med
    for k in out:
        if k != "count":
            out[k][~has] = np.nan
    out["sd_sample"][cnt < 2] = np.nan
    return out
