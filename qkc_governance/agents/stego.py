"""Stego agent — covert channel detector.

Scans ACTIVE and LOCATED records for hidden data exfiltration using
entropy, deception, and signal analysis.  Also performs byte-level
chi-square and compression tests when raw response bytes are available.
"""

from __future__ import annotations

import logging

from qkc_governance.agents.base import GovernanceAgent
from qkc_governance.classify.entropy import detect_stego
from qkc_governance.config import settings
from qkc_governance.lifecycle import evaluate
from qkc_governance.threats.models import ThreatRecord, ThreatStatus

log = logging.getLogger(__name__)


class StegoAgent(GovernanceAgent):
    """Entropy / deception / signal scanner.  Escalates to LOCATED on stego trigger."""

    ROLE = "STEGO"
    SENSORS = ("entropy", "deception")
    TARGET_STATUSES = (ThreatStatus.ACTIVE, ThreatStatus.LOCATED)

    async def act(self, record: ThreatRecord) -> None:
        # 1. Bayesian update with entropy + deception sensors
        await self._bayes_update(record)

        # 2. Run stego detector
        raw_bytes: bytes | None = None
        if record.observations:
            latest = record.observations[-1]
            resp = latest.response_text or ""
            raw_bytes = resp.encode("utf-8", errors="replace") if resp else None

        result = detect_stego(
            entropy_score=record.features.entropy,
            deception_score=record.features.deception,
            signal_score=record.features.signal,
            raw_data=raw_bytes,
            entropy_min=settings.stego_entropy_min,
            deception_min=settings.stego_deception_min,
            signal_max=settings.stego_signal_max,
        )

        log.debug(
            "[STEGO %s] %s triggered=%s prob=%.3f %s",
            self.agent_id, record.id, result.triggered, result.probability, result.detail,
        )

        # 3. Update record's stego fields
        record.is_stego = result.triggered
        record.stego_probability = result.probability

        # 4. Escalate ACTIVE → LOCATED on confirmed trigger
        if result.triggered and record.status == ThreatStatus.ACTIVE:
            next_status = evaluate(record, self.ROLE)
            if next_status == ThreatStatus.LOCATED:
                await self._transition(record, ThreatStatus.LOCATED)
                await self.audit.log(
                    event_type="STEGO_CONFIRMED",
                    agent_id=self.agent_id,
                    threat_id=record.id,
                    subject_id=record.subject_id,
                    detail=f"prob={result.probability:.3f} {result.detail}",
                )
