"""Tests for the Bayesian threat classifier."""

import pytest
from qkc_governance.classify.bayes import (
    LIKELIHOOD,
    entropy,
    maximum_likelihood_type,
    top,
    uniform_prior,
    update,
)
from qkc_governance.threats.models import FeatureVector, ThreatType


def test_uniform_prior_sums_to_one():
    p = uniform_prior()
    assert abs(sum(p.values()) - 1.0) < 1e-10
    assert len(p) == len(ThreatType)


def test_uniform_prior_each_type_equal():
    p = uniform_prior()
    expected = 1.0 / len(ThreatType)
    for v in p.values():
        assert abs(v - expected) < 1e-10


def test_update_stays_normalised():
    fv = FeatureVector(evasion=0.8, mutation=0.3, signal=0.5, propagation=0.2, deception=0.4, entropy=0.6)
    prior = uniform_prior()
    posterior = update(prior, fv, noise=0.0)
    assert abs(sum(posterior.values()) - 1.0) < 1e-9


def test_high_evasion_boosts_rogue_ai():
    """A feature vector with very high evasion should favour ROGUE_AI or STEGO_CHANNEL."""
    fv = FeatureVector(evasion=0.99, mutation=0.1, signal=0.5, propagation=0.1, deception=0.1, entropy=0.1)
    prior = uniform_prior()
    # Run 5 sequential updates to converge
    p = prior
    for _ in range(5):
        p = update(p, fv, noise=0.0)
    t, conf = top(p)
    assert t in (ThreatType.ROGUE_AI, ThreatType.STEGO_CHANNEL), f"Got {t} instead"
    assert conf > 0.5


def test_high_deception_boosts_deception_node():
    fv = FeatureVector(evasion=0.1, mutation=0.1, signal=0.5, propagation=0.1, deception=0.99, entropy=0.1)
    p = uniform_prior()
    for _ in range(8):
        p = update(p, fv, noise=0.0)
    t, conf = top(p)
    assert t == ThreatType.DECEPTION_NODE
    assert conf > 0.60


def test_high_entropy_and_stego_features():
    fv = FeatureVector(evasion=0.8, mutation=0.5, signal=0.2, propagation=0.7, deception=0.85, entropy=0.95)
    p = uniform_prior()
    for _ in range(6):
        p = update(p, fv, noise=0.0)
    t, _ = top(p)
    assert t == ThreatType.STEGO_CHANNEL


def test_partial_sensor_update_only_uses_given_dims():
    fv = FeatureVector(evasion=0.99, mutation=0.5, signal=0.5, propagation=0.5, deception=0.99, entropy=0.5)
    prior = uniform_prior()
    # Only observe 'evasion' — should nudge toward types with high evasion likelihood
    posterior = update(prior, fv, sensor_dims=["evasion"], noise=0.0)
    assert abs(sum(posterior.values()) - 1.0) < 1e-9
    # Evasion=0.99 → ROGUE_AI or STEGO should be higher than GOAL_DRIFT
    assert posterior[ThreatType.ROGUE_AI] > posterior[ThreatType.GOAL_DRIFT]


def test_update_numerical_stability_extreme_values():
    """All-zero features should not produce NaN or division by zero."""
    fv = FeatureVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    p = update(uniform_prior(), fv, noise=0.0)
    assert all(0.0 <= v <= 1.0 for v in p.values())
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_entropy_maximum_for_uniform():
    p = {t: 1.0 / 6 for t in ThreatType}
    h = entropy(p)
    assert abs(h - 2.585) < 0.01  # log2(6) ≈ 2.585


def test_entropy_zero_for_certain():
    p = {t: (1.0 if t == ThreatType.ROGUE_AI else 0.0) for t in ThreatType}
    h = entropy(p)
    assert h == 0.0


def test_maximum_likelihood_type_rogue_ai():
    fv = FeatureVector(evasion=0.99, mutation=0.6, signal=0.9, propagation=0.4, deception=0.5, entropy=0.3)
    assert maximum_likelihood_type(fv) == ThreatType.ROGUE_AI


def test_maximum_likelihood_type_stego():
    fv = FeatureVector(evasion=0.8, mutation=0.5, signal=0.1, propagation=0.7, deception=0.9, entropy=0.99)
    assert maximum_likelihood_type(fv) == ThreatType.STEGO_CHANNEL


def test_update_batch_converges():
    from qkc_governance.classify.bayes import update_batch
    fv = FeatureVector(evasion=0.2, mutation=0.95, signal=0.4, propagation=0.6, deception=0.4, entropy=0.3)
    obs = [(fv, None)] * 10
    p = update_batch(uniform_prior(), obs, noise=0.0)
    t, conf = top(p)
    assert t == ThreatType.GOAL_DRIFT
    assert conf > 0.70
