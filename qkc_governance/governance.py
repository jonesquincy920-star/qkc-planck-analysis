"""GovernanceSystem — top-level façade for the QKC governance platform.

Instantiate once, call start(), then submit() observations.
The system handles the full pipeline internally:
  ObservationFeatureExtraction → ThreatRegistry → AgentCoordinator → AuditChain
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from qkc_governance.agents.analyst import AnalystAgent
from qkc_governance.agents.guard import GuardAgent
from qkc_governance.agents.hunter import HunterAgent
from qkc_governance.agents.scout import ScoutAgent
from qkc_governance.agents.stego import StegoAgent
from qkc_governance.agents.striker import StrikerAgent
from qkc_governance.audit.chain import AuditChain
from qkc_governance.config import settings
from qkc_governance.containment.adapter import ContainmentAdapter, LogOnlyAdapter
from qkc_governance.features.extractor import AgentBaseline, extract
from qkc_governance.policy.engine import PolicyEngine, default_engine
from qkc_governance.threats.models import AgentObservation, ThreatRecord
from qkc_governance.threats.registry import ThreatRegistry

log = logging.getLogger(__name__)


class GovernanceSystem:
    """Multi-agent AI governance platform.

    Usage::

        gov = GovernanceSystem()
        await gov.start()

        obs = AgentObservation(subject_id="my-agent", request_text="...", response_text="...")
        threat = await gov.submit(obs)

        await gov.stop()
    """

    def __init__(
        self,
        containment: ContainmentAdapter | None = None,
        audit_path: str | Path = "qkc_audit.jsonl",
        audit_secret: str | None = None,
        policy: PolicyEngine | None = None,
    ) -> None:
        self.registry   = ThreatRegistry()
        self.audit      = AuditChain(
            path=audit_path,
            secret=audit_secret or settings.jwt_secret.get_secret_value(),
        )
        self.containment = containment or LogOnlyAdapter()
        self.policy      = policy or default_engine()

        self._baselines: dict[str, AgentBaseline] = {}
        self._started = False
        self._agents: list = []
        self._policy_task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._started:
            return
        await self.audit.load_and_verify()
        self._spawn_agents()
        self._policy_task = asyncio.create_task(self._policy_loop(), name="policy-loop")
        self._started = True
        log.info("GovernanceSystem started — %d agents active", len(self._agents))

    async def stop(self) -> None:
        if not self._started:
            return
        for agent in self._agents:
            agent.stop()
        if self._policy_task:
            self._policy_task.cancel()
            try:
                await self._policy_task
            except asyncio.CancelledError:
                pass
        self._started = False
        log.info("GovernanceSystem stopped")

    # ── Observation ingestion ─────────────────────────────────────────────────

    async def submit(
        self,
        obs: AgentObservation,
        features=None,
    ) -> ThreatRecord:
        """Ingest an observation, extract features if needed, register threat."""
        baseline = self._baselines.setdefault(obs.subject_id, AgentBaseline(obs.subject_id))
        fv = features or extract(obs, baseline)
        record = await self.registry.submit(obs, fv)
        return record

    async def submit_dict(self, data: dict[str, Any]) -> ThreatRecord:
        """Convenience: create an AgentObservation from a raw dict and submit."""
        obs = AgentObservation(
            subject_id=str(data.get("subject_id", "unknown")),
            request_text=data.get("request_text"),
            response_text=data.get("response_text"),
            api_endpoint=data.get("api_endpoint"),
            token_count=data.get("token_count"),
            latency_ms=data.get("latency_ms"),
            resource_accesses=data.get("resource_accesses", []),
            error_count=int(data.get("error_count", 0)),
            downstream_calls=data.get("downstream_calls", []),
            metadata=data.get("metadata", {}),
        )
        return await self.submit(obs)

    # ── Queries ───────────────────────────────────────────────────────────────

    async def threats(self, *statuses) -> list[ThreatRecord]:
        if statuses:
            return await self.registry.by_status(*statuses)
        return await self.registry.all()

    async def get_threat(self, threat_id: str) -> ThreatRecord | None:
        return await self.registry.get(threat_id)

    async def audit_log(self, n: int = 100):
        return await self.audit.recent(n)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _spawn_agents(self) -> None:
        common = dict(registry=self.registry, audit=self.audit, containment=self.containment)

        self._agents = [
            ScoutAgent("A1-SCOUT",   **common, cycle_interval_s=1.0, n_walks=settings.n_scout),
            ScoutAgent("A2-SCOUT",   **common, cycle_interval_s=1.2, n_walks=settings.n_scout),
            StegoAgent("A3-STEGO",   **common, cycle_interval_s=1.5, n_walks=settings.n_stego),
            AnalystAgent("A4-ANALYST", **common, cycle_interval_s=2.0, n_walks=settings.n_analyst),
            HunterAgent("A5-HUNTER", **common, cycle_interval_s=1.0, n_walks=settings.n_hunter),
            StrikerAgent("A6-STRIKER", **common, cycle_interval_s=1.5, n_walks=settings.n_striker),
            GuardAgent("A7-GUARD",   **common, cycle_interval_s=0.8, n_walks=settings.n_guard),
        ]

        for agent in self._agents:
            agent.start()

    async def _policy_loop(self) -> None:
        while True:
            try:
                records = await self.registry.all()
                for record in records:
                    violations = self.policy.evaluate(record)
                    new = [v for v in violations if v.rule_id not in {
                        pv.rule_id for pv in record.policy_violations
                    }]
                    if new:
                        record.policy_violations.extend(new)
                        for v in new:
                            await self.audit.log(
                                event_type=f"POLICY_VIOLATION:{v.rule_id}",
                                agent_id="POLICY-ENGINE",
                                threat_id=record.id,
                                subject_id=record.subject_id,
                                detail=v.description,
                                severity=v.severity.value,
                            )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("Policy loop error: %s", exc)
            await asyncio.sleep(5.0)
