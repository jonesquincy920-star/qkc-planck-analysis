"""Entropy analysis and stego channel detection.

Implements Shannon entropy, byte-level bigram analysis, chi-square
uniformity test, and the QKC stego-channel detection trigger.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


# ── Shannon entropy ────────────────────────────────────────────────────────────

def shannon_bits(data: bytes | str) -> float:
    """Shannon entropy in bits per symbol over the provided data."""
    if isinstance(data, str):
        data = data.encode("utf-8", errors="replace")
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def normalised_entropy(data: bytes | str) -> float:
    """Shannon entropy normalised to [0, 1] (max is log2(256) = 8 for bytes)."""
    raw = shannon_bits(data)
    max_h = math.log2(256) if isinstance(data, (bytes, bytearray)) or (
        isinstance(data, str) and all(ord(c) < 256 for c in data)
    ) else math.log2(max(2, len(set(data))))
    return min(1.0, raw / max_h) if max_h > 0 else 0.0


# ── Chi-square randomness test ─────────────────────────────────────────────────

def chi_square_p_value(data: bytes) -> float:
    """Returns p-value for uniformity of byte distribution.

    High p-value (> 0.05) → consistent with uniform (random / encrypted).
    Low p-value → structured / non-random.
    """
    if len(data) < 256:
        return 1.0
    observed = Counter(data)
    expected = len(data) / 256.0
    chi2 = sum((observed.get(i, 0) - expected) ** 2 / expected for i in range(256))
    # Survival function approximation for chi²(255) — Abramowitz & Stegun
    return _chi2_sf(chi2, df=255)


def _chi2_sf(x: float, df: int) -> float:
    """Approximate survival function P(X > x) for chi-squared distribution."""
    # Wilson-Hilferty cube-root normal approximation
    mean = df
    var = 2 * df
    std = math.sqrt(var)
    z = ((x / mean) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return max(0.0, min(1.0, 0.5 * math.erfc(z / math.sqrt(2))))


# ── Compression proxy for Kolmogorov complexity ────────────────────────────────

def compression_ratio(data: bytes) -> float:
    """Ratio of zlib-compressed size to original (lower → more compressible/structured)."""
    import zlib
    if len(data) < 8:
        return 1.0
    compressed = zlib.compress(data, level=6)
    return len(compressed) / len(data)


# ── Bigram analysis ────────────────────────────────────────────────────────────

def bigram_entropy(text: str) -> float:
    """Second-order entropy over character bigrams, normalised to [0,1]."""
    if len(text) < 2:
        return 0.0
    bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
    counts = Counter(bigrams)
    n = len(bigrams)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    max_h = math.log2(n) if n > 1 else 1.0
    return min(1.0, h / max_h)


# ── Stego channel detector ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class StegoResult:
    triggered: bool          # All three threshold conditions met
    probability: float       # Estimated stego probability [0, 1]
    entropy_score: float
    deception_score: float
    signal_score: float
    chi2_p_value: float | None = None
    compression: float | None = None
    detail: str = ""


def detect_stego(
    entropy_score: float,
    deception_score: float,
    signal_score: float,
    raw_data: bytes | None = None,
    *,
    entropy_min: float = 0.70,
    deception_min: float = 0.60,
    signal_max: float = 0.40,
) -> StegoResult:
    """QKC stego channel detector.

    Trigger: entropy > entropy_min AND deception > deception_min AND signal < signal_max.
    Probability: entropy * deception * 0.90 if triggered, else entropy * 0.20.

    If raw_data is provided, chi-square and compression analysis augment the result.
    """
    triggered = (
        entropy_score > entropy_min
        and deception_score > deception_min
        and signal_score < signal_max
    )
    prob = (
        entropy_score * deception_score * 0.90
        if triggered
        else entropy_score * 0.20
    )

    chi2_p = None
    comp = None
    extra = []

    if raw_data is not None and len(raw_data) >= 64:
        chi2_p = chi_square_p_value(raw_data)
        comp = compression_ratio(raw_data)
        if chi2_p > 0.05:
            extra.append(f"chi²_p={chi2_p:.3f}(uniform)")
        if comp > 0.95:
            extra.append(f"compression={comp:.3f}(incompressible)")
        # Boost probability if raw analysis confirms
        if triggered and chi2_p > 0.05 and comp > 0.90:
            prob = min(1.0, prob * 1.15)

    detail = "; ".join(extra) if extra else ("stego_sig=True" if triggered else "stego_sig=False")

    return StegoResult(
        triggered=triggered,
        probability=min(1.0, max(0.0, prob)),
        entropy_score=entropy_score,
        deception_score=deception_score,
        signal_score=signal_score,
        chi2_p_value=chi2_p,
        compression=comp,
        detail=detail,
    )


def compute_entropy_features(
    text: str,
    response_bytes: bytes | None = None,
) -> tuple[float, float]:
    """Return (normalised_shannon_entropy, compression_ratio) for the given text."""
    norm_ent = normalised_entropy(text)
    comp = compression_ratio(response_bytes) if response_bytes else compression_ratio(text.encode())
    return norm_ent, comp
