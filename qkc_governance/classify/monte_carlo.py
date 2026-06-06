"""Monte Carlo methods for threat assessment uncertainty quantification.

Uses MC sampling to:
  1. Estimate posterior credible intervals given noisy observations.
  2. Compute probability that effective confidence exceeds a threshold.
  3. Select the optimal governance agent path in a dependency graph.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from qkc_governance.classify.bayes import LIKELIHOOD, update, uniform_prior
from qkc_governance.threats.models import FeatureVector, ThreatType


@dataclass(frozen=True)
class MCResult:
    """Summary statistics from a Monte Carlo assessment run."""

    mean_confidence: float
    std_confidence: float
    ci_low: float          # 5th percentile
    ci_high: float         # 95th percentile
    p_above_threshold: float  # P(confidence > threshold)
    top_type: ThreatType
    top_type_prob: float
    n_samples: int


def assess_threat(
    features: FeatureVector,
    sensor_dims: list[str] | None = None,
    prior: dict[ThreatType, float] | None = None,
    *,
    n_samples: int = 2000,
    noise: float = 0.12,
    threshold: float = 0.42,
) -> MCResult:
    """Monte Carlo posterior sampling with per-sample noise injection.

    Each sample independently adds noise to the feature vector and runs
    a full Bayesian update.  Returns summary statistics over all samples.
    """
    if prior is None:
        prior = uniform_prior()

    confidences: list[float] = []
    type_counts: dict[ThreatType, int] = {t: 0 for t in ThreatType}

    for _ in range(n_samples):
        noisy = FeatureVector(
            evasion=_clip(features.evasion + _noise(noise)),
            mutation=_clip(features.mutation + _noise(noise)),
            signal=_clip(features.signal + _noise(noise)),
            propagation=_clip(features.propagation + _noise(noise)),
            deception=_clip(features.deception + _noise(noise)),
            entropy=_clip(features.entropy + _noise(noise)),
        )
        posterior = update(prior, noisy, sensor_dims, noise=0.0)
        t = max(posterior, key=posterior.__getitem__)
        conf = posterior[t]
        type_counts[t] += 1
        confidences.append(conf)

    confidences.sort()
    mean_c = sum(confidences) / n_samples
    var_c  = sum((c - mean_c) ** 2 for c in confidences) / n_samples
    std_c  = var_c ** 0.5
    ci_low  = confidences[int(0.05 * n_samples)]
    ci_high = confidences[int(0.95 * n_samples)]
    p_above = sum(1 for c in confidences if c > threshold) / n_samples
    top_t   = max(type_counts, key=type_counts.__getitem__)

    return MCResult(
        mean_confidence=mean_c,
        std_confidence=std_c,
        ci_low=ci_low,
        ci_high=ci_high,
        p_above_threshold=p_above,
        top_type=top_t,
        top_type_prob=type_counts[top_t] / n_samples,
        n_samples=n_samples,
    )


def multi_step_posterior(
    initial_features: FeatureVector,
    n_steps: int,
    sensor_schedule: list[list[str]],
    noise: float = 0.10,
    n_samples: int = 500,
) -> list[dict[ThreatType, float]]:
    """Simulate how the posterior evolves over n_steps Bayesian updates.

    Returns one posterior dict per step.  Useful for predicting how quickly
    a sensor combination will converge to a confident classification.
    """
    posteriors: list[dict[ThreatType, float]] = []
    prior = uniform_prior()
    for step in range(n_steps):
        dims = sensor_schedule[step % len(sensor_schedule)]
        # Aggregate n_samples noisy updates into a single mean posterior
        accumulated: dict[ThreatType, float] = {t: 0.0 for t in ThreatType}
        for _ in range(n_samples):
            p = update(prior, initial_features, dims, noise)
            for t in ThreatType:
                accumulated[t] += p[t]
        prior = {t: accumulated[t] / n_samples for t in ThreatType}
        posteriors.append(dict(prior))
    return posteriors


def steps_to_confidence(
    features: FeatureVector,
    sensor_dims: list[str],
    target_confidence: float = 0.55,
    max_steps: int = 20,
    n_samples: int = 200,
    noise: float = 0.10,
) -> int:
    """Estimate the number of Bayesian update steps needed to exceed target_confidence.

    Returns max_steps if the target is never reached.
    """
    prior = uniform_prior()
    for step in range(1, max_steps + 1):
        accumulated: dict[ThreatType, float] = {t: 0.0 for t in ThreatType}
        for _ in range(n_samples):
            p = update(prior, features, sensor_dims, noise)
            for t in ThreatType:
                accumulated[t] += p[t]
        prior = {t: accumulated[t] / n_samples for t in ThreatType}
        best = max(prior.values())
        if best >= target_confidence:
            return step
    return max_steps


def _noise(scale: float) -> float:
    return (random.random() - 0.5) * 2.0 * scale


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))
