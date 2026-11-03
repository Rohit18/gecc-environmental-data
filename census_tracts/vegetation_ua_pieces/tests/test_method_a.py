"""Unit tests: toy case with known answers, and the GLC overlap (name-cell) rule."""
import os, sys

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from finalize_core import fractions  # noqa: E402
from urbveg_worker import GLCLookup, accumulate, row_areas_m2  # noqa: E402

VEG, IMP, BUILT, WATER, OTHER = 130, 190, 50, 80, 30


def toy():
    """Three GLC pixels, each 3x3 WorldCover pixels, side by side (3 rows x 9 cols)."""
    wc = np.full((3, 9), OTHER, np.uint8)
    b1 = wc[:, 0:3].reshape(-1); b1[:3] = BUILT                      # GLC#1: 3 built, 6 unbuilt
    wc[:, 0:3] = b1.reshape(3, 3)
    b2 = wc[:, 3:6].reshape(-1); b2[:5] = BUILT                      # GLC#2: 5 built, 4 unbuilt
    wc[:, 3:6] = b2.reshape(3, 3)
    b3 = wc[:, 6:9].reshape(-1); b3[:2] = WATER; b3[2] = BUILT       # GLC#3: 2 water, 1 built, 6 unbuilt
    wc[:, 6:9] = b3.reshape(3, 3)
    glc = np.zeros((1, 3, 9), np.uint8)
    glc[0, :, 0:3], glc[0, :, 3:6], glc[0, :, 6:9] = VEG, IMP, VEG
    return wc, glc


def test_toy_case_known_answers():
    wc, glc = toy()
    piece = np.ones((3, 9), np.int32)
    ua = np.ones((3, 9), np.int32)
    res = accumulate(wc, glc, piece, ua, np.ones(3), 1, 1)          # equal areas
    u = res["urban"][1, 0]; r = res["rural"][1, 0]
    assert u.sum() == 25 and u[VEG] == 16
    assert r.sum() == 16 and r[VEG] == 12
    urban = pd.DataFrame({"piece_id": 1, "year": 2020, "glc_class": np.nonzero(u)[0], "area_m2": u[u > 0]})
    rural = pd.DataFrame({"ua": 1, "year": 2020, "glc_class": np.nonzero(r)[0], "area_m2": r[r > 0]})
    uf, rf = fractions(urban, rural, [VEG])
    assert uf.frac.iloc[0] == pytest.approx(0.64)
    assert rf.frac.iloc[0] == pytest.approx(0.75)
    assert uf.frac.iloc[0] - rf.frac.iloc[0] == pytest.approx(-0.11)
    # diagnostics
    assert res["water_m2"][1] == 2 and res["built_m2"][1] == 9 and res["valid_m2"][1] == 27


def test_fill_classes_excluded_and_empty_list_refused():
    urban = pd.DataFrame({"piece_id": 1, "year": 2020, "glc_class": [VEG, IMP, 0, 250], "area_m2": [1.0, 1.0, 5.0, 5.0]})
    rural = pd.DataFrame({"ua": 1, "year": 2020, "glc_class": [VEG], "area_m2": [1.0]})
    uf, _ = fractions(urban, rural, [VEG])
    assert uf.frac.iloc[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fractions(urban, rural, [])


def test_wc_nodata_excluded():
    wc, glc = toy()
    wc[0, 0] = 0                                                       # one built pixel -> nodata
    res = accumulate(wc, glc, np.ones((3, 9), np.int32), np.ones((3, 9), np.int32), np.ones(3), 1, 1)
    assert res["urban"][1, 0].sum() == 24 and res["nodata_m2"][1] == 1


def _write_tile(path, west, north, res, n, value):
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(west, north, res, res)) as d:
        d.write(np.full((1, n, n), value, np.uint8))


def test_overlap_rule_takes_name_cell_tile(tmp_path):
    """Four neighbouring synthetic tiles that, like GLC_FCS30D, extend 0.0002 deg past their 5-deg name-cell
    (east and south overhang). Pixel centres inside an overhang strip must take the tile whose name-cell
    contains them, not the tile whose overhang covers them."""
    n = 50; res = 5.0002 / n
    vals = {"W10N10": 1, "W5N10": 2, "W10N5": 3, "W5N5": 4}
    paths = {}
    for name, v in vals.items():
        lon = -int(name[1:name.index("N")]); lat = int(name[name.index("N") + 1:])
        paths[name] = str(tmp_path / f"{name}.tif")
        _write_tile(paths[name], lon, lat, res, n, v)
    out = GLCLookup(paths, [1]).lookup(np.array([-5.0001, -4.9999, -7.5, -2.5]),   # W10 side, W5 side (in W10 overhang)
                                       np.array([5.0001, 4.9999, 7.5, 2.5]))[0]    # N10 side, N5 side (in N10 overhang)
    assert out[0, 0] == 1 and out[0, 1] == 2            # lon -4.9999 is inside W10N10's overhang -> must be W5N10
    assert out[1, 0] == 3 and out[1, 1] == 4            # lat 4.9999 is inside the N10 tiles' overhang -> N5 tiles
    assert out[2, 2] == 1 and out[2, 3] == 2 and out[3, 2] == 3 and out[3, 3] == 4   # interiors
    # sanity: the overhang really covers those points
    with rasterio.open(paths["W10N10"]) as d:
        assert d.bounds.right > -4.9999 and d.bounds.bottom < 4.9999


def test_missing_glc_tile_is_fill(tmp_path):
    lut = GLCLookup({}, [1])
    out = lut.lookup(np.array([-82.5]), np.array([40.5]))
    assert out.shape == (1, 1, 1) and out[0, 0, 0] == 0


def test_row_area_sums_to_sphere_band():
    # one row of 1 deg x 1 deg at the equator: R^2 * dlon * sin(1deg)
    R = 6371007.181
    a = row_areas_m2(1.0, 1.0, 1.0, 1, R)[0]
    assert a == pytest.approx(R ** 2 * np.deg2rad(1.0) * np.sin(np.deg2rad(1.0)))


def test_overhang_strict_vs_clamp(tmp_path):
    """A centre inside its own name-cell but within the tile's last row (the overhang row) gets the overhang
    value under 'strict' and the adjacent (second-to-last) row under 'clamp'."""
    n = 50; res = 5.0002 / n                         # last row spans [4.9998, 5.0998): covers lat 5.00005
    p = str(tmp_path / "W10N10.tif")
    a = np.full((1, n, n), 7, np.uint8); a[0, -1, :] = 99; a[0, :, -1] = 99
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(-10, 10, res, res)) as d:
        d.write(a)
    lons, lats = np.array([-7.5, -5.00005]), np.array([7.5, 5.00005])
    s = GLCLookup({"W10N10": p}, [1], "strict").lookup(lons, lats)[0]
    c = GLCLookup({"W10N10": p}, [1], "clamp").lookup(lons, lats)[0]
    assert s[0, 0] == 7 and s[1, 0] == 99 and s[0, 1] == 99
    assert c[0, 0] == 7 and c[1, 0] == 7 and c[0, 1] == 7 and c[1, 1] == 7
