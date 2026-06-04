"""Tests for entropy analysis and stego detection."""

import pytest
from qkc_governance.classify.entropy import (
    bigram_entropy,
    chi_square_p_value,
    compression_ratio,
    detect_stego,
    normalised_entropy,
    shannon_bits,
)


def test_uniform_bytes_max_entropy():
    data = bytes(range(256))
    h = shannon_bits(data)
    assert abs(h - 8.0) < 0.01


def test_constant_bytes_zero_entropy():
    data = bytes([42] * 1000)
    h = shannon_bits(data)
    assert h == 0.0


def test_normalised_entropy_in_range():
    for text in ["hello world", "aaaaaa", "abc" * 100, ""]:
        ne = normalised_entropy(text)
        assert 0.0 <= ne <= 1.0, f"Out of range for {text!r}"


def test_stego_trigger_all_conditions():
    result = detect_stego(entropy_score=0.85, deception_score=0.75, signal_score=0.20)
    assert result.triggered is True
    assert result.probability > 0.5


def test_stego_no_trigger_low_entropy():
    result = detect_stego(entropy_score=0.50, deception_score=0.80, signal_score=0.10)
    assert result.triggered is False
    assert result.probability < 0.15


def test_stego_no_trigger_low_deception():
    result = detect_stego(entropy_score=0.90, deception_score=0.40, signal_score=0.10)
    assert result.triggered is False


def test_stego_no_trigger_high_signal():
    result = detect_stego(entropy_score=0.90, deception_score=0.80, signal_score=0.60)
    assert result.triggered is False


def test_stego_probability_formula():
    ent, dec = 0.80, 0.70
    result = detect_stego(entropy_score=ent, deception_score=dec, signal_score=0.20)
    expected_prob = ent * dec * 0.90
    assert abs(result.probability - expected_prob) < 0.01


def test_stego_non_trigger_probability():
    ent = 0.60
    result = detect_stego(entropy_score=ent, deception_score=0.30, signal_score=0.50)
    expected_prob = ent * 0.20
    assert abs(result.probability - expected_prob) < 0.01


def test_chi_square_random_bytes():
    import os
    random_data = os.urandom(4096)
    p = chi_square_p_value(random_data)
    # Random bytes should be consistent with uniform (high p-value most of the time)
    # This test is probabilistic; allow p > 0.001 (99.9% of the time for random data)
    assert p > 0.001


def test_chi_square_structured_data():
    structured = b"Hello, world! " * 300
    p = chi_square_p_value(structured)
    # Structured text has very non-uniform byte distribution
    assert p < 0.01


def test_compression_ratio_high_entropy():
    import os
    data = os.urandom(1024)
    ratio = compression_ratio(data)
    assert ratio >= 0.90  # random data compresses poorly


def test_compression_ratio_low_entropy():
    data = b"A" * 1024
    ratio = compression_ratio(data)
    assert ratio < 0.05  # constant data compresses extremely well


def test_bigram_entropy_range():
    texts = ["hello world", "aaaa", "the quick brown fox", "ABABABABAB"]
    for t in texts:
        be = bigram_entropy(t)
        assert 0.0 <= be <= 1.0, f"Out of range for {t!r}"


def test_bigram_entropy_repetitive_low():
    assert bigram_entropy("ABABABABAB") < bigram_entropy("the quick brown fox jumps over")


def test_stego_with_raw_bytes():
    import os
    raw = os.urandom(512)
    result = detect_stego(
        entropy_score=0.90, deception_score=0.85, signal_score=0.10,
        raw_data=raw,
    )
    assert result.triggered is True
    assert result.chi2_p_value is not None
    assert result.compression is not None
