"""Hunter agent — active containment of classified threats.

Applies rate-limiting and sandboxing to CLASSIFIED threats before
escalating to the Striker for final termination.
"""

from __future__ import annotations

import logging

from qkc_governance.agents.base import GovernanceAgent
from qkc_governance.lifecycle import evaluate
from qkc_governance.threats.models import ResponseAction, ThreatRecord, ThreatStatus, ThreatType

log = logging.getLogger(__name__)

# Containment strategy per threat type
_ACTIONS: dict[ThreatType, tuple[ResponseAction, str]] = {
    ThreatType.ROGUE_AI:         (ResponseAction.ISOLATE,  "network isolation — rogue boundary exceeded"),
    ThreatType.DECEPTION_NODE:   (ResponseAction.ISOLATE,  "isolate — deceptive outputs silenced"),
    ThreatType.INJECT_AGENT:     (ResponseAction.CONTAIN,  "sandbox — injection surface eliminated"),
    ThreatType.ALIGNMENT_BREACH: (ResponseAction.CONTAIN,  "rate-limit + oversight mode engaged"),
    ThreatType.GOAL_DRIFT:       (ResponseAction.MONITOR,  "enhanced telemetry — drift trajectory tracked"),
    ThreatType.STEGO_CHANNEL:    (ResponseAction.ISOLATE,  "covert channel severed — output quarantined"),
}


class HunterAgent(GovernanceAgent):
    """Containment specialist.  Uses propagation + signal sensors.
    Applies containment actions via the ContainmentAdapter, then
    transitions the record to CONTAINED."""

    ROLE = "HUNTER"
    SENSORS = ("propagation", "signal")
    TARGET_STATUSES = (ThreatStatus.CLASSIFIED,)

    async def act(self, record: ThreatRecord) -> None:
        # 1. Update Bayes with our sensors
        await self._bayes_update(record)

        # 2. Determine containment action from classified type
        t, _ = record.top_type()
        action, detail = _ACTIONS.get(t, (ResponseAction.CONTAIN, "default containment"))

        # 3. Execute containment via adapter
        try:
            match action:
                case ResponseAction.ISOLATE:
                    await self.containment.isolate(record.subject_id)
                case ResponseAction.CONTAIN:
                    await self.containment.rate_limit(record.subject_id, limit=10)
                case ResponseAction.MONITOR:
                    await self.containment.enable_enhanced_monitoring(record.subject_id)
                case _:
                    await self.containment.rate_limit(record.subject_id, limit=10)
        except Exception as exc:
            log.error("[HUNTER %s] containment failed for %s: %s", self.agent_id, record.id, exc)
            await self.audit.log(
                event_type="CONTAINMENT_FAILED",
                agent_id=self.agent_id,
                threat_id=record.id,
                subject_id=record.subject_id,
                detail=str(exc),
            )
            return

        record.containment_action = action
        record.containment_detail = detail

        # 4. Transition to CONTAINED
        next_status = evaluate(record, self.ROLE)
        if next_status == ThreatStatus.CONTAINED:
            await self._transition(record, ThreatStatus.CONTAINED)
            await self.audit.log(
                event_type="CONTAINED",
                agent_id=self.agent_id,
                threat_id=record.id,
                subject_id=record.subject_id,
                detail=f"action={action.value} {detail}",
            )
