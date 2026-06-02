"""
Planck CMB → EnneractBrain interface.

Maps the 9 cognitive axes to physically motivated features extracted from
CMB anomaly statistics and patch descriptors, then drives the brain with
that stimulus.  Results can be compared across sky positions (e.g.  the
QKC cold-spot at l=−57°, b=−27° vs. random locations).

Axis → CMB feature mapping
---------------------------
    0  perception     ← normalised mean temperature   (patch mean / std)
    1  memory         ← KS-test significance          (1 − ks_pvalue)
    2  reasoning      ← σ(z-score)                    (tanh of z)
    3  emotion        ← σ(excess kurtosis)
    4  language       ← σ(skewness)
    5  spatial        ← fractional disc coverage      (n_pixels / full_sky)
    6  temporal       ← low-ℓ power suppression       (C_ℓ<10 / mean C_ℓ)
    7  social         ← z-score significance           (1 − z_pvalue)
    8  metacognition  ← combined significance         (1 − ks_p · z_p)
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from .enneract import COGNITIVE_AXES, N_DIM

if TYPE_CHECKING:
    from .brain import EnneractBrain


def cmb_stats_to_stimulus(
    anomaly_stats: dict,
    patch_stats:   dict  = None,
    low_ell_ratio: float = 0.5,
    disc_coverage: float = 0.5,
) -> np.ndarray:
    """
    Convert CMB analysis outputs into a (9,) stimulus vector in [0, 1].

    Parameters
    ----------
    anomaly_stats : dict from qkc.stats.anomaly_score()
                    Keys: ks_statistic, ks_pvalue, z_score, z_pvalue
    patch_stats   : dict from qkc.spot.patch_stats(), optional
                    Keys: mean, std, skewness, kurtosis, n_pixels
    low_ell_ratio : float — ratio of low-ℓ power to mean (from power spectrum)
    disc_coverage : float — fractional sky coverage of the analysis disc
    """
    def _get(d, key, default=0.0):
        return float((d or {}).get(key, default))

    def _sigmoid(x):
        return float(1.0 / (1.0 + np.exp(-float(x))))

    ks_p  = _get(anomaly_stats, "ks_pvalue",  0.5)
    z     = _get(anomaly_stats, "z_score",    0.0)
    z_p   = _get(anomaly_stats, "z_pvalue",   0.5)

    mean_T = _get(patch_stats, "mean",     0.0)
    std_T  = _get(patch_stats, "std",      1.0)
    skew   = _get(patch_stats, "skewness", 0.0)
    kurt   = _get(patch_stats, "kurtosis", 0.0)

    stimulus = np.array([
        _sigmoid(mean_T / max(std_T, 1e-12)),    # 0 perception
        float(np.clip(1.0 - ks_p, 0, 1)),        # 1 memory
        _sigmoid(z),                              # 2 reasoning
        _sigmoid(kurt),                           # 3 emotion
        _sigmoid(skew),                           # 4 language
        float(np.clip(disc_coverage, 0, 1)),      # 5 spatial
        float(np.clip(low_ell_ratio, 0, 1)),      # 6 temporal
        float(np.clip(1.0 - z_p, 0, 1)),          # 7 social
        float(np.clip(1.0 - ks_p * z_p, 0, 1)),  # 8 metacognition
    ], dtype=np.float64)

    return np.clip(stimulus, 0.0, 1.0)


def analyze_with_brain(
    brain:         "EnneractBrain",
    anomaly_stats: dict,
    patch_stats:   dict  = None,
    label:         str   = "CMB_patch",
    low_ell_ratio: float = 0.5,
    disc_coverage: float = 0.5,
) -> dict:
    """
    Drive the EnneractBrain with CMB patch statistics and return structured
    analysis results.

    Parameters
    ----------
    brain         : EnneractBrain instance
    anomaly_stats : dict from qkc.stats.anomaly_score()
    patch_stats   : dict from qkc.spot.patch_stats(), optional
    label         : string tag stored with the memory pattern
    low_ell_ratio : float — low-ℓ / mean power ratio
    disc_coverage : float — fractional sky coverage

    Returns
    -------
    dict with:
        stimulus        — (9,) array fed into the brain
        thought_summary — dominant shell, axis, energies, coherence
        interpretation  — human-readable sentence
        memory_index    — slot index in associative memory
    """
    stimulus = cmb_stats_to_stimulus(
        anomaly_stats,
        patch_stats   = patch_stats,
        low_ell_ratio = low_ell_ratio,
        disc_coverage = disc_coverage,
    )
    thought    = brain.think(stimulus)
    mem_index  = brain.remember(thought.raw_state, label=label)

    return {
        "stimulus": stimulus.tolist(),
        "thought_summary": {
            "dominant_shell":  thought.dominant_shell,
            "dominant_axis":   thought.dominant_axis,
            "shell_energies":  thought.shell_energies.tolist(),
            "axis_coherence":  thought.axis_coherence.tolist(),
            "memory_match":    thought.memory_match,
            "memory_overlap":  thought.memory_overlap,
        },
        "memory_index":  mem_index,
        "interpretation": _interpret(thought),
    }


def compare_patches(
    brain:          "EnneractBrain",
    anomaly_stats_a: dict,
    anomaly_stats_b: dict,
    patch_stats_a:   dict = None,
    patch_stats_b:   dict = None,
    label_a:         str  = "patch_A",
    label_b:         str  = "patch_B",
) -> dict:
    """
    Process two CMB patches through the brain and compute their cognitive
    distance in the 9D hypercube activation space.

    Returns a dict including stimulus vectors, dominant axes, and the
    inner-product overlap between the two induced activation patterns.
    """
    stim_a = cmb_stats_to_stimulus(anomaly_stats_a, patch_stats_a)
    stim_b = cmb_stats_to_stimulus(anomaly_stats_b, patch_stats_b)

    enc_a = brain.encode(stim_a)
    enc_b = brain.encode(stim_b)

    overlap = float(np.abs(np.vdot(enc_a, enc_b)) /
                    (np.linalg.norm(enc_a) * np.linalg.norm(enc_b) + 1e-12))

    axis_diff = np.abs(stim_a - stim_b)

    return {
        label_a:      {"stimulus": stim_a.tolist()},
        label_b:      {"stimulus": stim_b.tolist()},
        "overlap":    overlap,
        "distance":   float(1.0 - overlap),
        "axis_delta": {COGNITIVE_AXES[d]: float(axis_diff[d]) for d in range(N_DIM)},
        "most_different_axis": COGNITIVE_AXES[int(np.argmax(axis_diff))],
    }


def _interpret(thought) -> str:
    _shell_roles = [
        "core integration", "primary axes", "feature binding", "association",
        "working memory", "abstract reasoning", "metacognition layer",
        "self-model", "consciousness edge", "unified awareness",
    ]
    return (
        f"CMB signal peaked in shell {thought.dominant_shell} "
        f"({_shell_roles[thought.dominant_shell]}) "
        f"with dominant coherence along the '{thought.dominant_axis}' axis. "
        f"Nearest memory overlap: {thought.memory_overlap:.4f}."
    )
