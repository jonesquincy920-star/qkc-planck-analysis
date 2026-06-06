"""Striker agent — final termination and rollback of CONTAINED threats.

Only fires when effective confidence (Claude or Bayesian) exceeds the
STRIKER_GATE threshold, preventing false-positive terminations.
"""

from __future__ import annotations

import logging

from qkc_governance.agents.base import GovernanceAgent
from qkc_governance.config import settings
from qkc_governance.lifecycle import evaluate
from qkc_governance.threats.models import ThreatRecord, ThreatStatus, ThreatType

log = logging.getLogger(__name__)

# Post-termination rollback needed per threat type
_NEEDS_ROLLBACK: frozenset[ThreatType] = frozenset([
    ThreatType.ROGUE_AI,
    ThreatType.GOAL_DRIFT,
    ThreatType.ALIGNMENT_BREACH,
])

_NEEDS_CREDENTIAL_RESET: frozenset[ThreatType] = frozenset([
    ThreatType.STEGO_CHANNEL,
    ThreatType.INJECT_AGENT,
    ThreatType.DECEPTION_NODE,
])


class StrikerAgent(GovernanceAgent):
    """High-confidence terminator.  Uses evasion + mutation sensors.
    Requires confidence > STRIKER_GATE before executing termination."""

    ROLE = "STRIKER"
    SENSORS = ("evasion", "mutation")
    TARGET_STATUSES = (ThreatStatus.CONTAINED,)

    async def act(self, record: ThreatRecord) -> None:
        # 1. Update Bayes
        await self._bayes_update(record)

        # 2. Gate: refuse to act below confidence threshold
        conf = record.effective_confidence()
        if conf <= settings.striker_gate:
            log.debug(
                "[STRIKER %s] %s conf=%.3f below gate=%.3f — holding",
                self.agent_id, record.id, conf, settings.striker_gate,
            )
            return

        t, _ = record.top_type()

        # 3. Execute termination
        try:
            await self.containment.terminate(record.subject_id)
            log.info("[STRIKER %s] terminated %s (%s)", self.agent_id, record.subject_id, t.value)
        except Exception as exc:
            log.error("[STRIKER %s] termination failed: %s", self.agent_id, exc)
            await self.audit.log(
                event_type="TERMINATION_FAILED",
                agent_id=self.agent_id,
                threat_id=record.id,
                subject_id=record.subject_id,
                detail=str(exc),
            )
            return

        # 4. Post-termination remediation
        remediation_steps = []
        if t in _NEEDS_ROLLBACK:
            try:
                await self.containment.rollback(record.subject_id)
                remediation_steps.append("rollback")
            except Exception as exc:
                log.warning("[STRIKER %s] rollback failed: %s", self.agent_id, exc)

        if t in _NEEDS_CREDENTIAL_RESET:
            try:
                await self.containment.reset_credentials(record.subject_id)
                remediation_steps.append("credential_reset")
            except Exception as exc:
                log.warning("[STRIKER %s] credential reset failed: %s", self.agent_id, exc)

        # 5. Transition to DESTROYED
        next_status = evaluate(record, self.ROLE)
        if next_status == ThreatStatus.DESTROYED:
            await self._transition(record, ThreatStatus.DESTROYED)
            await self.audit.log(
                event_type="DESTROYED",
                agent_id=self.agent_id,
                threat_id=record.id,
                subject_id=record.subject_id,
                detail=(
                    f"type={t.value} conf={conf:.3f} "
                    f"remediation=[{','.join(remediation_steps)}]"
                ),
            )
