"""BeliefExchange — distributed Bayesian consensus coordinator.

Each governance agent calls broadcast() after every Bayesian update.
The exchange:

  1. Stores the agent's private posterior for the threat.
  2. Detects divergence between any two agents that have both observed
     the same threat (symmetric KL > threshold → BELIEF_DIVERGENCE event).
  3. When ≥ 2 agents have posted beliefs, recomputes the consensus
     posterior via log-opinion pooling and writes it back to the registry.

The result is that the registry's posterior for any threat converges to a
weighted collective judgment rather than whichever agent last wrote to it.
Disagreement itself becomes an actionable signal — strong divergence means
the threat is ambiguous or actively evading consistent characterisation.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qkc_governance.consensus.graph import mean_out_weight
from qkc_governance.consensus.propagation import (
    BeliefMessage,
    log_opinion_pool,
    symmetric_kl,
    top_type,
)
from qkc_governance.threats.models import ThreatType

if TYPE_CHECKING:
    from qkc_governance.audit.chain import AuditChain
    from qkc_governance.threats.registry import ThreatRegistry

log = logging.getLogger(__name__)

# Symmetric KL (nats) above which two-agent disagreement triggers an audit event
DIVERGENCE_THRESHOLD = 0.40


@dataclass
class _ThreatBeliefState:
    agent_beliefs: dict[str, dict[ThreatType, float]] = field(default_factory=dict)
    divergence_pairs: set[frozenset[str]] = field(default_factory=set)


class BeliefExchange:
    """Coordinates belief propagation across the governance lattice.

    Thread-safety: all mutation happens inside a single asyncio.Lock.
    """

    def __init__(
        self,
        registry: "ThreatRegistry",
        audit: "AuditChain",
        divergence_threshold: float = DIVERGENCE_THRESHOLD,
    ) -> None:
        self._registry = registry
        self._audit = audit
        self._threshold = divergence_threshold
        self._lock = asyncio.Lock()
        self._state: dict[str, _ThreatBeliefState] = defaultdict(_ThreatBeliefState)

    # ── Public API ────────────────────────────────────────────────────────────

    async def broadcast(self, msg: BeliefMessage) -> None:
        """Receive a belief from an agent, check divergence, update consensus."""
        async with self._lock:
            state = self._state[msg.threat_id]
            state.agent_beliefs[msg.from_agent] = dict(msg.posterior)

            await self._check_divergence(msg, state)

            if len(state.agent_beliefs) >= 2:
                await self._update_consensus(msg.threat_id, state)

    def agent_belief(
        self, threat_id: str, agent_id: str
    ) -> dict[ThreatType, float] | None:
        state = self._state.get(threat_id)
        if state is None:
            return None
        b = state.agent_beliefs.get(agent_id)
        return dict(b) if b else None

    def all_beliefs(
        self, threat_id: str
    ) -> dict[str, dict[ThreatType, float]]:
        """Return all agent private posteriors for a threat."""
        state = self._state.get(threat_id)
        if state is None:
            return {}
        return {k: dict(v) for k, v in state.agent_beliefs.items()}

    def divergent_pairs(self, threat_id: str) -> list[tuple[str, str]]:
        state = self._state.get(threat_id)
        if state is None:
            return []
        return [tuple(sorted(pair)) for pair in state.divergence_pairs]

    def summary(self, threat_id: str) -> dict:
        """Snapshot of consensus state for API exposure."""
        beliefs = self.all_beliefs(threat_id)
        if not beliefs:
            return {"agents": 0, "consensus": None, "divergent_pairs": []}

        pooled = log_opinion_pool([
            (p, mean_out_weight(aid)) for aid, p in beliefs.items()
        ])
        tt, conf = top_type(pooled)

        return {
            "agents": len(beliefs),
            "consensus_type": tt.value,
            "consensus_confidence": round(conf, 4),
            "consensus_posterior": {k.value: round(v, 4) for k, v in pooled.items()},
            "agent_beliefs": {
                aid: {k.value: round(v, 4) for k, v in post.items()}
                for aid, post in beliefs.items()
            },
            "divergent_pairs": self.divergent_pairs(threat_id),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _update_consensus(
        self, threat_id: str, state: _ThreatBeliefState
    ) -> None:
        pooled = log_opinion_pool([
            (post, mean_out_weight(aid))
            for aid, post in state.agent_beliefs.items()
        ])
        tt, conf = top_type(pooled)
        await self._registry.update_posterior(threat_id, pooled, conf)
        log.debug(
            "[CONSENSUS] threat=%s n=%d top=%s conf=%.3f",
            threat_id[:8], len(state.agent_beliefs), tt.value, conf,
        )

    async def _check_divergence(
        self, msg: BeliefMessage, state: _ThreatBeliefState
    ) -> None:
        for other_id, other_post in state.agent_beliefs.items():
            if other_id == msg.from_agent:
                continue
            pair = frozenset([msg.from_agent, other_id])
            skl = symmetric_kl(msg.posterior, other_post)
            if skl > self._threshold and pair not in state.divergence_pairs:
                state.divergence_pairs.add(pair)
                log.warning(
                    "[DIVERGENCE] threat=%s %s↔%s skl=%.3f",
                    msg.threat_id[:8], msg.from_agent, other_id, skl,
                )
                await self._audit.log(
                    event_type="BELIEF_DIVERGENCE",
                    agent_id=msg.from_agent,
                    threat_id=msg.threat_id,
                    subject_id="",
                    detail=(
                        f"{msg.from_agent}↔{other_id} "
                        f"skl={skl:.3f} > threshold={self._threshold:.2f}"
                    ),
                    severity="HIGH",
                )
            elif skl <= self._threshold and pair in state.divergence_pairs:
                # Agents converged — remove from divergence set
                state.divergence_pairs.discard(pair)
