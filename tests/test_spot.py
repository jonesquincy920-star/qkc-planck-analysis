"""Unit tests for spot.py using a synthetic HEALPix map."""

import numpy as np
import healpy as hp
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qkc.spot import extract_patch, patch_stats, galactic_to_healpy


def _make_map(nside=64, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 50, hp.nside2npix(nside))


def test_galactic_to_healpy_returns_valid_angles():
    theta, phi = galactic_to_healpy(-57.0, -27.0)
    assert 0 <= theta <= np.pi
    assert 0 <= phi <= 2 * np.pi


def test_extract_patch_returns_nonempty():
    hmap = _make_map()
    ipix, vals = extract_patch(hmap, l_deg=-57.0, b_deg=-27.0, radius_deg=10.0)
    assert len(ipix) > 0
    assert len(vals) == len(ipix)


def test_extract_patch_values_match_map():
    hmap = _make_map()
    ipix, vals = extract_patch(hmap, radius_deg=5.0)
    np.testing.assert_array_equal(vals, hmap[ipix])


def test_patch_stats_keys():
    vals = np.random.normal(0, 1, 200)
    stats = patch_stats(vals)
    expected = {"n_pixels", "mean", "std", "min", "max", "skewness", "kurtosis"}
    assert set(stats.keys()) == expected


def test_patch_stats_known_values():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = patch_stats(vals)
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["n_pixels"] == 5
