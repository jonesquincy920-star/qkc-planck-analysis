"""Base class for all governance agents."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qkc_governance.audit.chain import AuditChain
    from qkc_governance.consensus.protocol import BeliefExchange
    from qkc_governance.containment.adapter import ContainmentAdapter
    from qkc_governance.threats.models import ThreatRecord
    from qkc_governance.threats.registry import ThreatRegistry

log = logging.getLogger(__name__)


@dataclass
class AgentState:
    agent_id: str
    role: str
    active: bool = False
    cycles: int = 0
    threats_handled: int = 0
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    errors: int = 0


class GovernanceAgent(ABC):
    """Abstract base for SCOUT, STEGO, ANALYST, HUNTER, STRIKER, and GUARD agents.

    Each agent runs a continuous async loop, polling the threat registry and
    acting on records in its target lifecycle stage.
    """

    ROLE: str = "BASE"
    SENSORS: tuple[str, ...] = ()  # feature dimensions this agent observes
    TARGET_STATUSES: tuple = ()    # ThreatStatus values this agent acts on

    def __init__(
        self,
        agent_id: str,
        registry: "ThreatRegistry",
        audit: "AuditChain",
        containment: "ContainmentAdapter",
        cycle_interval_s: float = 1.0,
        n_walks: int = 60,
        exchange: "BeliefExchange | None" = None,
    ) -> None:
        self.agent_id = agent_id
        self.registry = registry
        self.audit = audit
        self.containment = containment
        self.cycle_interval_s = cycle_interval_s
        self.n_walks = n_walks
        self.exchange = exchange
        self.state = AgentState(agent_id=agent_id, role=self.ROLE)
        self._task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=self.agent_id)
            self.state.active = True
            log.info("%s started", self.agent_id)

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self.state.active = False
        log.info("%s stopped", self.agent_id)

    async def join(self) -> None:
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while True:
            try:
                await self._cycle()
                self.state.cycles += 1
                self.state.last_active = datetime.now(timezone.utc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.errors += 1
                log.error("%s cycle error: %s", self.agent_id, exc)
            await asyncio.sleep(self.cycle_interval_s)

    async def _cycle(self) -> None:
        from qkc_governance.threats.models import ThreatStatus
        targets = await self.registry.by_status(*self.TARGET_STATUSES)
        for record in targets:
            await self.act(record)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    async def act(self, record: "ThreatRecord") -> None:
        """Examine a record and take appropriate action."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _bayes_update(self, record: "ThreatRecord") -> None:
        """Apply a Bayesian update, then broadcast to the belief exchange."""
        from qkc_governance.classify.bayes import top, update
        posterior = update(
            record.posterior, record.features,
            sensor_dims=list(self.SENSORS),
            noise=0.10,
        )
        t, conf = top(posterior)
        record.posterior = posterior
        record.confidence = conf

        if self.exchange is not None and self.SENSORS:
            from qkc_governance.consensus.propagation import BeliefMessage
            await self.exchange.broadcast(BeliefMessage(
                from_agent=self.agent_id,
                threat_id=record.id,
                posterior=dict(posterior),
                confidence=conf,
            ))
        else:
            await self.registry.update_posterior(record.id, posterior, conf)

    async def _transition(self, record: "ThreatRecord", new_status) -> None:
        from qkc_governance.lifecycle import describe
        desc = describe(record.status, new_status)
        updated = await self.registry.transition(record.id, new_status, self.agent_id)
        if updated is not None:
            self.state.threats_handled += 1
            await self.audit.log(
                event_type=f"TRANSITION:{record.status.value}→{new_status.value}",
                agent_id=self.agent_id,
                threat_id=record.id,
                subject_id=record.subject_id,
                detail=desc,
            )
            log.info("[%s] %s → %s (%s)", self.agent_id, record.id, new_status.value, desc)

    def __repr__(self) -> str:
        return f"<{self.ROLE} {self.agent_id} cycles={self.state.cycles}>"
