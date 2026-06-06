"""Tests for the distributed Bayesian consensus layer."""

import asyncio
import math
import pytest

from qkc_governance.consensus.graph import (
    LATTICE_EDGES, mean_out_weight, neighbours, trust_weight,
)
from qkc_governance.consensus.propagation import (
    BeliefMessage, kl_divergence, log_opinion_pool, symmetric_kl, top_type,
)
from qkc_governance.consensus.protocol import BeliefExchange, DIVERGENCE_THRESHOLD
from qkc_governance.threats.models import ThreatType
from qkc_governance.audit.chain import AuditChain
from qkc_governance.threats.registry import ThreatRegistry
import tempfile, os


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def exchange(tmp_path):
    registry = ThreatRegistry()
    audit = AuditChain(path=tmp_path / "test_consensus.jsonl", secret="test")
    await audit.load_and_verify()
    return BeliefExchange(registry, audit, divergence_threshold=DIVERGENCE_THRESHOLD)


def _uniform() -> dict[ThreatType, float]:
    n = len(ThreatType)
    return {t: 1.0 / n for t in ThreatType}


def _peaked(t: ThreatType, mass: float = 0.80) -> dict[ThreatType, float]:
    """Posterior concentrated on type t."""
    others = [x for x in ThreatType if x != t]
    leftover = (1.0 - mass) / len(others)
    return {x: (mass if x == t else leftover) for x in ThreatType}


# ── Graph topology ─────────────────────────────────────────────────────────────

def test_all_agents_have_edges():
    agents = ["A1-SCOUT", "A2-SCOUT", "A3-STEGO", "A4-ANALYST",
              "A5-HUNTER", "A6-STRIKER", "A7-GUARD"]
    for a in agents:
        assert a in LATTICE_EDGES
        assert len(LATTICE_EDGES[a]) > 0


def test_trust_weights_in_range():
    for agent, edges in LATTICE_EDGES.items():
        for neighbour, w in edges:
            assert 0.0 < w <= 1.0, f"Weight {w} out of range for {agent}→{neighbour}"


def test_scouts_share_high_trust():
    assert trust_weight("A1-SCOUT", "A2-SCOUT") >= 0.85
    assert trust_weight("A2-SCOUT", "A1-SCOUT") >= 0.85


def test_guard_connected_to_all():
    all_agents = {a for a in LATTICE_EDGES if a != "A7-GUARD"}
    guard_neighbours = set(neighbours("A7-GUARD"))
    assert all_agents == guard_neighbours


def test_mean_out_weight_positive():
    for agent in LATTICE_EDGES:
        assert mean_out_weight(agent) > 0.0


# ── Log-opinion pooling ────────────────────────────────────────────────────────

def test_pool_single_posterior_returns_itself():
    p = _peaked(ThreatType.ROGUE_AI, 0.80)
    result = log_opinion_pool([(p, 1.0)])
    assert abs(result[ThreatType.ROGUE_AI] - 0.80) < 0.01


def test_pool_two_identical_posteriors():
    p = _peaked(ThreatType.STEGO_CHANNEL, 0.75)
    result = log_opinion_pool([(p, 0.5), (p, 0.5)])
    tt, conf = top_type(result)
    assert tt == ThreatType.STEGO_CHANNEL
    assert conf > 0.70


def test_pool_normalised():
    p1 = _peaked(ThreatType.ROGUE_AI, 0.80)
    p2 = _peaked(ThreatType.INJECT_AGENT, 0.70)
    result = log_opinion_pool([(p1, 0.6), (p2, 0.4)])
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-9


def test_pool_empty_returns_uniform():
    result = log_opinion_pool([])
    n = len(ThreatType)
    for v in result.values():
        assert abs(v - 1.0 / n) < 1e-9


def test_dominant_weight_wins():
    p_rogue  = _peaked(ThreatType.ROGUE_AI,    0.90)
    p_stego  = _peaked(ThreatType.STEGO_CHANNEL, 0.90)
    result   = log_opinion_pool([(p_rogue, 0.9), (p_stego, 0.1)])
    tt, _    = top_type(result)
    assert tt == ThreatType.ROGUE_AI


# ── KL divergence ─────────────────────────────────────────────────────────────

def test_kl_identical_distributions_is_zero():
    p = _peaked(ThreatType.GOAL_DRIFT, 0.70)
    assert kl_divergence(p, p) < 1e-9


def test_kl_non_negative():
    p = _peaked(ThreatType.ROGUE_AI, 0.80)
    q = _peaked(ThreatType.DECEPTION_NODE, 0.80)
    assert kl_divergence(p, q) >= 0.0


def test_symmetric_kl_is_symmetric():
    p = _peaked(ThreatType.INJECT_AGENT, 0.75)
    q = _peaked(ThreatType.ALIGNMENT_BREACH, 0.75)
    assert abs(symmetric_kl(p, q) - symmetric_kl(q, p)) < 1e-9


def test_strongly_different_posteriors_have_high_skl():
    p = _peaked(ThreatType.ROGUE_AI, 0.95)
    q = _peaked(ThreatType.STEGO_CHANNEL, 0.95)
    assert symmetric_kl(p, q) > 0.40


def test_similar_posteriors_have_low_skl():
    p = _peaked(ThreatType.ROGUE_AI, 0.60)
    q = _peaked(ThreatType.ROGUE_AI, 0.65)
    assert symmetric_kl(p, q) < 0.10


# ── BeliefExchange ─────────────────────────────────────────────────────────────

async def test_single_broadcast_stores_belief(exchange):
    p = _peaked(ThreatType.ROGUE_AI, 0.80)
    msg = BeliefMessage(from_agent="A1-SCOUT", threat_id="t-001",
                        posterior=p, confidence=0.80)
    await exchange.broadcast(msg)
    stored = exchange.agent_belief("t-001", "A1-SCOUT")
    assert stored is not None
    assert abs(stored[ThreatType.ROGUE_AI] - 0.80) < 0.01


async def test_two_broadcasts_produce_consensus(exchange):
    p1 = _peaked(ThreatType.ROGUE_AI, 0.80)
    p2 = _peaked(ThreatType.ROGUE_AI, 0.75)
    await exchange.broadcast(BeliefMessage("A1-SCOUT", "t-002", p1, 0.80))
    await exchange.broadcast(BeliefMessage("A2-SCOUT", "t-002", p2, 0.75))
    summary = exchange.summary("t-002")
    assert summary["agents"] == 2
    assert summary["consensus_type"] == ThreatType.ROGUE_AI.value
    assert summary["consensus_confidence"] > 0.70


async def test_divergence_detected(exchange):
    p_rogue = _peaked(ThreatType.ROGUE_AI,      0.95)
    p_stego = _peaked(ThreatType.STEGO_CHANNEL, 0.95)
    await exchange.broadcast(BeliefMessage("A1-SCOUT",  "t-003", p_rogue, 0.95))
    await exchange.broadcast(BeliefMessage("A3-STEGO",  "t-003", p_stego, 0.95))
    pairs = exchange.divergent_pairs("t-003")
    assert len(pairs) == 1
    assert set(pairs[0]) == {"A1-SCOUT", "A3-STEGO"}


async def test_convergence_clears_divergence(exchange):
    p_rogue = _peaked(ThreatType.ROGUE_AI, 0.95)
    p_stego = _peaked(ThreatType.STEGO_CHANNEL, 0.95)
    await exchange.broadcast(BeliefMessage("A1-SCOUT", "t-004", p_rogue, 0.95))
    await exchange.broadcast(BeliefMessage("A3-STEGO", "t-004", p_stego, 0.95))
    assert len(exchange.divergent_pairs("t-004")) == 1

    # Both agents now agree
    p_agree = _peaked(ThreatType.ROGUE_AI, 0.80)
    await exchange.broadcast(BeliefMessage("A1-SCOUT", "t-004", p_agree, 0.80))
    await exchange.broadcast(BeliefMessage("A3-STEGO", "t-004", p_agree, 0.80))
    assert len(exchange.divergent_pairs("t-004")) == 0


async def test_unknown_threat_returns_empty_summary(exchange):
    s = exchange.summary("no-such-threat")
    assert s["agents"] == 0
    assert s["consensus"] is None


async def test_divergence_logged_to_audit(exchange):
    p_rogue = _peaked(ThreatType.ROGUE_AI,      0.95)
    p_stego = _peaked(ThreatType.STEGO_CHANNEL, 0.95)
    await exchange.broadcast(BeliefMessage("A4-ANALYST", "t-005", p_rogue, 0.95))
    await exchange.broadcast(BeliefMessage("A5-HUNTER",  "t-005", p_stego, 0.95))
    entries = await exchange._audit.recent(10)
    divergence_events = [e for e in entries if e.event_type == "BELIEF_DIVERGENCE"]
    assert len(divergence_events) >= 1
    assert "skl=" in divergence_events[0].detail
