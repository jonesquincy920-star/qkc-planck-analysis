"""Threat lifecycle state machine.

ACTIVE → LOCATED → CLASSIFIED → CONTAINED → DESTROYED

Each transition has a guard predicate; guards return True when the
transition should fire.  Transitions are evaluated in order by the
coordinator on every governance cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from qkc_governance.config import settings
from qkc_governance.threats.models import ThreatRecord, ThreatStatus


@dataclass(frozen=True)
class Transition:
    from_status: ThreatStatus
    to_status:   ThreatStatus
    guard:       Callable[[ThreatRecord, str], bool]  # (record, agent_role) → bool
    description: str


def _scout_locates(record: ThreatRecord, agent_role: str) -> bool:
    return agent_role in ("SCOUT", "STEGO")


def _analyst_classifies(record: ThreatRecord, agent_role: str) -> bool:
    if agent_role != "ANALYST":
        return False
    conf = record.effective_confidence()
    return conf > settings.analyst_gate


def _hunter_contains(record: ThreatRecord, agent_role: str) -> bool:
    return agent_role == "HUNTER"


def _striker_destroys(record: ThreatRecord, agent_role: str) -> bool:
    if agent_role != "STRIKER":
        return False
    return record.effective_confidence() > settings.striker_gate


TRANSITIONS: list[Transition] = [
    Transition(
        from_status=ThreatStatus.ACTIVE,
        to_status=ThreatStatus.LOCATED,
        guard=_scout_locates,
        description="Scout or Stego agent detected anomalous activity",
    ),
    Transition(
        from_status=ThreatStatus.LOCATED,
        to_status=ThreatStatus.CLASSIFIED,
        guard=_analyst_classifies,
        description="Analyst achieved classification confidence above gate",
    ),
    Transition(
        from_status=ThreatStatus.CLASSIFIED,
        to_status=ThreatStatus.CONTAINED,
        guard=_hunter_contains,
        description="Hunter applied containment to classified threat",
    ),
    Transition(
        from_status=ThreatStatus.CONTAINED,
        to_status=ThreatStatus.DESTROYED,
        guard=_striker_destroys,
        description="Striker terminated threat with sufficient confidence",
    ),
]

_FROM_MAP: dict[ThreatStatus, list[Transition]] = {}
for _t in TRANSITIONS:
    _FROM_MAP.setdefault(_t.from_status, []).append(_t)


def evaluate(record: ThreatRecord, agent_role: str) -> ThreatStatus | None:
    """Return the next status if a transition fires, else None.

    Checks only transitions that originate from record.status.
    Does not mutate the record — the caller is responsible for
    calling registry.transition() when acting on this result.
    """
    if record.status == ThreatStatus.DESTROYED:
        return None
    for t in _FROM_MAP.get(record.status, []):
        if t.guard(record, agent_role):
            return t.to_status
    return None


def describe(from_status: ThreatStatus, to_status: ThreatStatus) -> str:
    for t in TRANSITIONS:
        if t.from_status == from_status and t.to_status == to_status:
            return t.description
    return f"{from_status.value} → {to_status.value}"
