"""Unit tests for stats.py using synthetic data."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qkc.stats import anomaly_score, monte_carlo_pvalue


def test_anomaly_score_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    scores = anomaly_score(x, x.copy())
    assert scores["ks_pvalue"] > 0.05, "Identical distributions should not be flagged"
    assert abs(scores["z_score"]) < 1.0


def test_anomaly_score_different_distributions():
    rng = np.random.default_rng(1)
    obs = rng.normal(50, 1, 200)   # offset mean
    ref = rng.normal(0, 1, 5000)
    scores = anomaly_score(obs, ref)
    assert scores["ks_pvalue"] < 1e-10
    assert scores["z_score"] > 10


def test_anomaly_score_keys():
    x = np.ones(100)
    y = np.zeros(100)
    result = anomaly_score(x, y)
    assert set(result.keys()) == {"ks_statistic", "ks_pvalue", "z_score", "z_pvalue"}
