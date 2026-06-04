"""In-memory threat registry with optional SQLite persistence."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Callable

from qkc_governance.config import settings
from qkc_governance.threats.models import (
    AgentObservation,
    FeatureVector,
    ThreatRecord,
    ThreatStatus,
    ThreatType,
)


class ThreatRegistry:
    """Thread-safe registry for active and historical threat records.

    Fires registered callbacks on every state change so that governance
    agents can subscribe without polling.
    """

    def __init__(self) -> None:
        self._records: dict[str, ThreatRecord] = {}
        self._lock = asyncio.Lock()
        self._on_change: list[Callable[[ThreatRecord], None]] = []

    # ── Registration / subscription ───────────────────────────────────────────

    def on_change(self, callback: Callable[[ThreatRecord], None]) -> None:
        """Subscribe to any record mutation."""
        self._on_change.append(callback)

    def _notify(self, record: ThreatRecord) -> None:
        for cb in self._on_change:
            try:
                cb(record)
            except Exception:
                pass

    # ── Submission ────────────────────────────────────────────────────────────

    async def submit(
        self,
        obs: AgentObservation,
        features: FeatureVector | None = None,
    ) -> ThreatRecord:
        """Create a new ThreatRecord from an observation or update an existing
        ACTIVE/LOCATED record for the same subject.

        Returns the (possibly new) record.
        """
        # Lazy imports break the classify ↔ threats circular dependency
        from qkc_governance.classify.bayes import maximum_likelihood_type, uniform_prior
        from qkc_governance.classify.entropy import detect_stego

        fv = features or obs.features or FeatureVector()

        async with self._lock:
            # Re-use the most recent non-destroyed record for this subject
            existing = self._latest_active(obs.subject_id)
            if existing is not None:
                existing.observations.append(obs)
                existing.features = fv
                existing.touch()
                self._notify(existing)
                return existing

            # Spawn a new threat
            stego = detect_stego(
                fv.entropy, fv.deception, fv.signal,
                entropy_min=settings.stego_entropy_min,
                deception_min=settings.stego_deception_min,
                signal_max=settings.stego_signal_max,
            )

            record = ThreatRecord(
                id=str(uuid.uuid4()),
                subject_id=obs.subject_id,
                status=ThreatStatus.ACTIVE,
                features=fv,
                prior=uniform_prior(),
                posterior=uniform_prior(),
                strength=0.5 + 0.5 * fv.l2_norm() / (len(FeatureVector.names()) ** 0.5),
                is_stego=stego.triggered,
                stego_probability=stego.probability,
                observations=[obs],
            )

            # If stego triggered, bias the prior
            if stego.triggered and import_random() > 0.4:
                record.true_type = ThreatType.STEGO_CHANNEL
            else:
                record.true_type = maximum_likelihood_type(fv)

            self._records[record.id] = record
            self._notify(record)
            return record

    # ── Lookups ───────────────────────────────────────────────────────────────

    async def get(self, threat_id: str) -> ThreatRecord | None:
        async with self._lock:
            return self._records.get(threat_id)

    async def all(self) -> list[ThreatRecord]:
        async with self._lock:
            return list(self._records.values())

    async def by_status(self, *statuses: ThreatStatus) -> list[ThreatRecord]:
        async with self._lock:
            return [r for r in self._records.values() if r.status in statuses]

    async def active_count(self) -> int:
        async with self._lock:
            return sum(
                1 for r in self._records.values()
                if r.status != ThreatStatus.DESTROYED
            )

    # ── Mutation helpers (called by lifecycle / agents) ───────────────────────

    async def update_posterior(
        self,
        threat_id: str,
        posterior: dict[ThreatType, float],
        confidence: float,
    ) -> None:
        async with self._lock:
            r = self._records.get(threat_id)
            if r is None:
                return
            r.posterior = posterior
            r.confidence = confidence
            r.touch()
            self._notify(r)

    async def transition(
        self,
        threat_id: str,
        new_status: ThreatStatus,
        by_agent: str,
    ) -> ThreatRecord | None:
        async with self._lock:
            r = self._records.get(threat_id)
            if r is None or r.status == ThreatStatus.DESTROYED:
                return None
            old = r.status
            r.status = new_status
            match new_status:
                case ThreatStatus.LOCATED:
                    r.located_by = by_agent
                case ThreatStatus.CLASSIFIED:
                    r.classified_by = by_agent
                case ThreatStatus.CONTAINED:
                    r.contained_by = by_agent
                case ThreatStatus.DESTROYED:
                    r.destroyed_by = by_agent
            r.touch()
            self._notify(r)
            return r

    # ── Internals ─────────────────────────────────────────────────────────────

    def _latest_active(self, subject_id: str) -> ThreatRecord | None:
        candidates = [
            r for r in self._records.values()
            if r.subject_id == subject_id and r.status != ThreatStatus.DESTROYED
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.created_at)


def import_random() -> float:
    import random
    return random.random()
