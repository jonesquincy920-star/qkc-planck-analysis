"""Tests for the threat lifecycle state machine."""

import pytest
from qkc_governance.lifecycle import TRANSITIONS, describe, evaluate
from qkc_governance.threats.models import FeatureVector, ThreatRecord, ThreatStatus, ThreatType
from qkc_governance.classify.bayes import uniform_prior


def _record(status=ThreatStatus.ACTIVE, confidence=0.3) -> ThreatRecord:
    return ThreatRecord(
        subject_id="test-agent",
        status=status,
        features=FeatureVector(evasion=0.8),
        posterior=uniform_prior(),
        confidence=confidence,
    )


def test_scout_locates_active_threat():
    r = _record(ThreatStatus.ACTIVE)
    result = evaluate(r, "SCOUT")
    assert result == ThreatStatus.LOCATED


def test_stego_locates_active_threat():
    r = _record(ThreatStatus.ACTIVE)
    result = evaluate(r, "STEGO")
    assert result == ThreatStatus.LOCATED


def test_analyst_classifies_when_above_gate():
    r = _record(ThreatStatus.LOCATED, confidence=0.60)
    result = evaluate(r, "ANALYST")
    assert result == ThreatStatus.CLASSIFIED


def test_analyst_does_not_classify_below_gate():
    r = _record(ThreatStatus.LOCATED, confidence=0.40)
    result = evaluate(r, "ANALYST")
    assert result is None


def test_hunter_contains_classified_threat():
    r = _record(ThreatStatus.CLASSIFIED)
    result = evaluate(r, "HUNTER")
    assert result == ThreatStatus.CONTAINED


def test_striker_destroys_above_gate():
    r = _record(ThreatStatus.CONTAINED, confidence=0.50)
    result = evaluate(r, "STRIKER")
    assert result == ThreatStatus.DESTROYED


def test_striker_holds_below_gate():
    r = _record(ThreatStatus.CONTAINED, confidence=0.30)
    result = evaluate(r, "STRIKER")
    assert result is None


def test_destroyed_never_transitions():
    r = _record(ThreatStatus.DESTROYED, confidence=0.99)
    for role in ("SCOUT", "STEGO", "ANALYST", "HUNTER", "STRIKER", "GUARD"):
        assert evaluate(r, role) is None


def test_wrong_agent_for_status():
    r = _record(ThreatStatus.LOCATED)
    # HUNTER acts on CLASSIFIED, not LOCATED
    assert evaluate(r, "HUNTER") is None


def test_scout_does_not_transition_located_threat():
    r = _record(ThreatStatus.LOCATED)
    # Scout only transitions ACTIVE → LOCATED
    assert evaluate(r, "SCOUT") is None


def test_describe_returns_string():
    desc = describe(ThreatStatus.ACTIVE, ThreatStatus.LOCATED)
    assert isinstance(desc, str) and len(desc) > 0


def test_claude_confidence_override():
    """When claude_analysis is set, effective_confidence should use Claude's value."""
    from qkc_governance.threats.models import ClaudeAnalysis, Priority, ResponseAction
    r = _record(ThreatStatus.CONTAINED, confidence=0.30)
    r.claude_analysis = ClaudeAnalysis(
        threat_type=ThreatType.ROGUE_AI,
        confidence=0.75,
        reasoning="High evasion pattern matches rogue AI profile.",
        priority=Priority.CRITICAL,
        action=ResponseAction.DESTROY,
    )
    # effective_confidence returns Claude's value
    assert r.effective_confidence() == 0.75
    # Striker gate should now fire
    result = evaluate(r, "STRIKER")
    assert result == ThreatStatus.DESTROYED


def test_all_transitions_have_descriptions():
    for t in TRANSITIONS:
        assert len(t.description) > 0
