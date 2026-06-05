"""Belief message format, log-opinion pooling, and divergence metrics.

Mathematical basis
------------------
Log-opinion pooling (Genest & Zidek 1986):

    consensus[t] ∝ ∏ p_i[t]^w_i

where w_i are normalised agent trust weights.  This is the unique pooling
operator that is externally Bayesian — i.e. the same result is reached
whether agents pool first and then update on new evidence, or update
individually and then pool.  It is strictly more robust than arithmetic
pooling because it cannot be dominated by a single overconfident agent.

KL divergence between agents is used as a disagreement signal:

    KL(p ‖ q) = Σ p(t) log(p(t)/q(t))

Symmetric KL = 0.5 * (KL(p‖q) + KL(q‖p)) is used to avoid asymmetry
artefacts when comparing two agent posteriors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from qkc_governance.threats.models import ThreatType

_N_TYPES = len(ThreatType)
_UNIFORM: dict[ThreatType, float] = {t: 1.0 / _N_TYPES for t in ThreatType}


@dataclass(frozen=True)
class BeliefMessage:
    """A posterior distribution broadcast by one agent to its neighbours."""

    from_agent: str
    threat_id: str
    posterior: dict[ThreatType, float]   # normalised, sums to 1.0
    confidence: float                     # max(posterior.values())
    sensor_reliability: float = 0.80      # historical accuracy weight
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def log_opinion_pool(
    posteriors: list[tuple[dict[ThreatType, float], float]],
) -> dict[ThreatType, float]:
    """Logarithmic opinion pool over a list of (posterior, weight) pairs.

    Returns the normalised geometric weighted mean.  Falls back to the
    uniform distribution if no posteriors are provided or weights sum to zero.
    """
    if not posteriors:
        return dict(_UNIFORM)

    total_w = sum(w for _, w in posteriors)
    if total_w <= 0.0:
        return dict(_UNIFORM)

    log_pool: dict[ThreatType, float] = {t: 0.0 for t in ThreatType}
    for posterior, w in posteriors:
        nw = w / total_w
        for t in ThreatType:
            p = max(posterior.get(t, 1e-12), 1e-12)
            log_pool[t] += nw * math.log(p)

    pool = {t: math.exp(v) for t, v in log_pool.items()}
    total = sum(pool.values())
    if total <= 0.0:
        return dict(_UNIFORM)
    return {t: v / total for t, v in pool.items()}


def kl_divergence(
    p: dict[ThreatType, float],
    q: dict[ThreatType, float],
) -> float:
    """KL(p ‖ q) in nats.  Clips zeros to 1e-12 to avoid -inf."""
    kl = 0.0
    for t in ThreatType:
        p_t = max(p.get(t, 1e-12), 1e-12)
        q_t = max(q.get(t, 1e-12), 1e-12)
        kl += p_t * math.log(p_t / q_t)
    return max(0.0, kl)


def symmetric_kl(
    p: dict[ThreatType, float],
    q: dict[ThreatType, float],
) -> float:
    """Symmetric KL = 0.5*(KL(p‖q) + KL(q‖p)) in nats."""
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))


def top_type(posterior: dict[ThreatType, float]) -> tuple[ThreatType, float]:
    t = max(posterior, key=posterior.__getitem__)
    return t, posterior[t]
