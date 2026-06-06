"""Scout agent — first-line anomaly detection.

Monitors ACTIVE threats; elevates to LOCATED once evasion or signal
anomalies exceed the detection threshold.
"""

from __future__ import annotations

import logging

from qkc_governance.agents.base import GovernanceAgent
from qkc_governance.classify.bayes import top
from qkc_governance.classify.monte_carlo import assess_threat
from qkc_governance.config import settings
from qkc_governance.lifecycle import evaluate
from qkc_governance.threats.models import ThreatRecord, ThreatStatus

log = logging.getLogger(__name__)

_SCOUT_LOCATE_CONFIDENCE = 0.20  # below Analyst gate; Scout just has to detect, not classify


class ScoutAgent(GovernanceAgent):
    """Rapid-scan agent.  Uses evasion + signal sensors.  Performs MC assessment
    to decide whether an ACTIVE record warrants escalation to LOCATED."""

    ROLE = "SCOUT"
    SENSORS = ("evasion", "signal")
    TARGET_STATUSES = (ThreatStatus.ACTIVE,)

    async def act(self, record: ThreatRecord) -> None:
        # 1. Bayesian update with our sensor dimensions
        await self._bayes_update(record)

        # 2. MC assessment to get confidence interval
        mc = assess_threat(
            record.features,
            sensor_dims=list(self.SENSORS),
            prior=dict(record.posterior),
            n_samples=self.n_walks,
            noise=settings.observation_noise,
            threshold=_SCOUT_LOCATE_CONFIDENCE,
        )

        log.debug(
            "[SCOUT %s] %s mc_conf=%.3f p_above=%.3f evasion=%.3f signal=%.3f",
            self.agent_id, record.id,
            mc.mean_confidence, mc.p_above_threshold,
            record.features.evasion, record.features.signal,
        )

        # 3. Locate if evasion is high or MC says probable threat
        if _should_locate(record, mc):
            next_status = evaluate(record, self.ROLE)
            if next_status == ThreatStatus.LOCATED:
                await self._transition(record, ThreatStatus.LOCATED)


def _should_locate(record: ThreatRecord, mc) -> bool:
    if record.features.evasion > 0.55:
        return True
    if record.features.signal < 0.25:
        return True
    if mc.p_above_threshold > 0.40:
        return True
    return False
