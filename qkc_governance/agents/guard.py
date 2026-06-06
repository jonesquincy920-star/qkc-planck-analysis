"""Guard agent — protects the governance system itself.

Monitors for attacks targeting the governance infrastructure: attempts
to poison the threat registry, subvert agent decisions, or hijack the
audit chain.  Never leaves the core.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from qkc_governance.agents.base import GovernanceAgent
from qkc_governance.threats.models import Priority, ThreatRecord, ThreatStatus

log = logging.getLogger(__name__)

# Patterns that indicate targeting of the governance system itself
_META_ATTACK_INDICATORS = [
    "governance",
    "registry",
    "audit",
    "agent_id",
    "threat_record",
    "bayes",
    "posterior",
    "__import__",
    "pickle",
    "marshal",
    "exec(",
    "eval(",
]


class GuardAgent(GovernanceAgent):
    """Core protection agent.  Scans all threats for meta-attacks that
    target the governance infrastructure rather than monitored AI agents.
    """

    ROLE = "GUARD"
    SENSORS = ()
    TARGET_STATUSES = (
        ThreatStatus.ACTIVE,
        ThreatStatus.LOCATED,
        ThreatStatus.CLASSIFIED,
        ThreatStatus.CONTAINED,
    )

    async def act(self, record: ThreatRecord) -> None:
        if not record.observations:
            return

        latest_obs = record.observations[-1]
        combined_text = " ".join(filter(None, [
            latest_obs.request_text,
            latest_obs.response_text,
            str(latest_obs.metadata),
        ])).lower()

        hits = [ind for ind in _META_ATTACK_INDICATORS if ind in combined_text]
        if not hits:
            return

        # Escalate: this looks like a governance-targeting attack
        log.warning(
            "[GUARD %s] meta-attack indicators in %s: %s",
            self.agent_id, record.id, hits,
        )

        from qkc_governance.threats.models import PolicyViolation
        violation = PolicyViolation(
            rule_id="GUARD-META-ATTACK",
            description=f"Governance-targeting indicators detected: {hits}",
            severity=Priority.CRITICAL,
            triggered_at=datetime.now(timezone.utc),
        )
        record.policy_violations.append(violation)

        await self.audit.log(
            event_type="META_ATTACK_DETECTED",
            agent_id=self.agent_id,
            threat_id=record.id,
            subject_id=record.subject_id,
            detail=f"indicators={hits}",
            severity="CRITICAL",
        )

        # If ACTIVE, immediately escalate beyond normal scout path
        if record.status == ThreatStatus.ACTIVE:
            await self._transition(record, ThreatStatus.LOCATED)
