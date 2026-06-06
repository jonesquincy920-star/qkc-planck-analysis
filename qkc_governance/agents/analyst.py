"""Analyst agent — deep Bayesian + Claude API classification.

Acts on LOCATED threats.  Accumulates Bayesian evidence over multiple
cycles, calls the Claude API once per unique threat, and transitions to
CLASSIFIED when the posterior confidence gate is cleared.
"""

from __future__ import annotations

import asyncio
import logging

from qkc_governance.agents.base import GovernanceAgent
from qkc_governance.classify import claude_api
from qkc_governance.classify.bayes import top
from qkc_governance.classify.monte_carlo import steps_to_confidence
from qkc_governance.config import settings
from qkc_governance.lifecycle import evaluate
from qkc_governance.threats.models import ThreatRecord, ThreatStatus

log = logging.getLogger(__name__)


class AnalystAgent(GovernanceAgent):
    """Full-spectrum analyst.  Uses deception + mutation sensors and
    calls the Claude API for semantic threat interpretation."""

    ROLE = "ANALYST"
    SENSORS = ("deception", "mutation")
    TARGET_STATUSES = (ThreatStatus.LOCATED,)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._claude_lock = asyncio.Lock()
        self._in_flight: set[str] = set()

    async def act(self, record: ThreatRecord) -> None:
        # 1. Bayesian update with our sensors
        await self._bayes_update(record)

        # 2. Estimate how many more cycles to reach confidence gate
        steps_left = steps_to_confidence(
            record.features, list(self.SENSORS),
            target_confidence=settings.analyst_gate,
            n_samples=50,
            noise=settings.observation_noise,
        )
        log.debug(
            "[ANALYST %s] %s conf=%.3f (~%d more steps to gate)",
            self.agent_id, record.id, record.confidence, steps_left,
        )

        # 3. Queue Claude analysis (once per threat, non-blocking)
        if (
            not record.claude_pending
            and record.claude_analysis is None
            and record.id not in self._in_flight
        ):
            asyncio.create_task(self._run_claude(record))

        # 4. Check transition gate
        next_status = evaluate(record, self.ROLE)
        if next_status == ThreatStatus.CLASSIFIED:
            t, conf = top(record.posterior)
            await self._transition(record, ThreatStatus.CLASSIFIED)
            await self.audit.log(
                event_type="CLASSIFIED",
                agent_id=self.agent_id,
                threat_id=record.id,
                subject_id=record.subject_id,
                detail=f"type={t.value} conf={conf:.3f} source={'claude' if record.claude_analysis else 'bayes'}",
            )

    async def _run_claude(self, record: ThreatRecord) -> None:
        async with self._claude_lock:
            if record.id in self._in_flight or record.claude_analysis is not None:
                return
            self._in_flight.add(record.id)
            record.claude_pending = True

        try:
            analysis = await claude_api.classify_threat(record)
            if analysis is not None:
                record.claude_analysis = analysis
                # Boost posterior toward Claude's classification
                record.posterior[analysis.threat_type] = max(
                    record.posterior[analysis.threat_type],
                    analysis.confidence * 0.9,
                )
                # Re-normalise
                total = sum(record.posterior.values())
                record.posterior = {t: v / total for t, v in record.posterior.items()}
                record.confidence = max(record.confidence, analysis.confidence)
                await self.audit.log(
                    event_type="CLAUDE_ANALYSIS",
                    agent_id=self.agent_id,
                    threat_id=record.id,
                    subject_id=record.subject_id,
                    detail=(
                        f"type={analysis.threat_type.value} "
                        f"conf={analysis.confidence:.3f} "
                        f"action={analysis.action.value} "
                        f"latency={analysis.latency_ms:.0f}ms"
                    ),
                )
                log.info(
                    "[ANALYST %s] Claude → %s %.0f%% %s",
                    self.agent_id, analysis.threat_type.value,
                    analysis.confidence * 100, analysis.reasoning,
                )
        finally:
            record.claude_pending = False
            self._in_flight.discard(record.id)
