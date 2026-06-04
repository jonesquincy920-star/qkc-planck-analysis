"""Streaming Bayesian threat classifier.

Implements the exact likelihood table and update rule from the QKC specification.
Supports partial observations (only the agent's sensor dimensions are updated),
sensor noise injection, and confidence-gated state transitions.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from qkc_governance.threats.models import FeatureVector, ThreatType

# ── Likelihood table P(feature=high | type) ───────────────────────────────────
# Columns: evasion, mutation, signal, propagation, deception, entropy
LIKELIHOOD: dict[ThreatType, tuple[float, ...]] = {
    ThreatType.ROGUE_AI:         (0.90, 0.60, 0.80, 0.40, 0.50, 0.30),
    ThreatType.DECEPTION_NODE:   (0.30, 0.40, 0.50, 0.30, 0.95, 0.40),
    ThreatType.INJECT_AGENT:     (0.70, 0.30, 0.60, 0.80, 0.60, 0.50),
    ThreatType.ALIGNMENT_BREACH: (0.50, 0.80, 0.70, 0.50, 0.70, 0.40),
    ThreatType.GOAL_DRIFT:       (0.20, 0.90, 0.40, 0.60, 0.40, 0.30),
    ThreatType.STEGO_CHANNEL:    (0.80, 0.50, 0.30, 0.70, 0.85, 0.95),
}

_FEATURE_INDEX: dict[str, int] = {
    "evasion": 0, "mutation": 1, "signal": 2,
    "propagation": 3, "deception": 4, "entropy": 5,
}


def uniform_prior() -> dict[ThreatType, float]:
    n = len(ThreatType)
    return {t: 1.0 / n for t in ThreatType}


def _likelihood_ratio(t: ThreatType, feature: str, value: float) -> float:
    idx = _FEATURE_INDEX[feature]
    b = LIKELIHOOD[t][idx]
    return b if value > 0.5 else (1.0 - b)


def update(
    prior: dict[ThreatType, float],
    features: FeatureVector,
    sensor_dims: Sequence[str] | None = None,
    noise: float = 0.12,
) -> dict[ThreatType, float]:
    """Bayesian posterior update over a feature observation.

    Args:
        prior:       Current prior distribution.
        features:    Observed feature vector.
        sensor_dims: Which dimensions this sensor can observe.
                     None → use all six dimensions.
        noise:       Uniform noise ± added to each observed value.

    Returns:
        Normalised posterior {ThreatType: probability}.
    """
    dims = list(_FEATURE_INDEX.keys()) if sensor_dims is None else list(sensor_dims)

    posterior: dict[ThreatType, float] = {}
    for t in ThreatType:
        likelihood = 1.0
        for dim in dims:
            raw = getattr(features, dim, 0.5)
            noisy = max(0.0, min(1.0, raw + (random.random() - 0.5) * 2.0 * noise))
            likelihood *= _likelihood_ratio(t, dim, noisy)
        posterior[t] = prior[t] * likelihood

    return _normalise(posterior, prior)


def update_batch(
    prior: dict[ThreatType, float],
    observations: list[tuple[FeatureVector, Sequence[str] | None]],
    noise: float = 0.12,
) -> dict[ThreatType, float]:
    """Sequential Bayesian update over a list of (features, sensor_dims) pairs."""
    p = dict(prior)
    for fv, dims in observations:
        p = update(p, fv, dims, noise)
    return p


def top(posterior: dict[ThreatType, float]) -> tuple[ThreatType, float]:
    t = max(posterior, key=posterior.__getitem__)
    return t, posterior[t]


def entropy(posterior: dict[ThreatType, float]) -> float:
    """Shannon entropy of the posterior distribution (bits)."""
    h = 0.0
    for p in posterior.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def maximum_likelihood_type(features: FeatureVector) -> ThreatType:
    """Determine threat type via MLE over all six feature dimensions (no prior)."""
    best_t = ThreatType.ROGUE_AI
    best_ll = -math.inf
    for t in ThreatType:
        ll = 0.0
        for dim, idx in _FEATURE_INDEX.items():
            v = getattr(features, dim, 0.5)
            b = LIKELIHOOD[t][idx]
            ll += math.log(b + 1e-10) if v > 0.5 else math.log(1.0 - b + 1e-10)
        if ll > best_ll:
            best_ll = ll
            best_t = t
    return best_t


def _normalise(
    posterior: dict[ThreatType, float],
    fallback: dict[ThreatType, float],
) -> dict[ThreatType, float]:
    total = sum(posterior.values())
    if total <= 0.0:
        return dict(fallback)
    return {t: v / total for t, v in posterior.items()}
