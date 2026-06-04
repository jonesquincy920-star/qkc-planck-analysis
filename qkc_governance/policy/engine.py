"""Policy evaluation engine.

Evaluates a ThreatRecord against a configurable rule set and returns
the set of violated PolicyViolation objects.  Rules are pure functions:
  rule(record: ThreatRecord) -> PolicyViolation | None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from qkc_governance.threats.models import (
    PolicyViolation,
    Priority,
    ThreatRecord,
    ThreatStatus,
    ThreatType,
)

log = logging.getLogger(__name__)

RuleFn = Callable[[ThreatRecord], PolicyViolation | None]


@dataclass
class PolicyEngine:
    """Evaluates all registered rules against a ThreatRecord."""

    rules: list[RuleFn] = field(default_factory=list)

    def evaluate(self, record: ThreatRecord) -> list[PolicyViolation]:
        violations: list[PolicyViolation] = []
        for rule in self.rules:
            try:
                v = rule(record)
                if v is not None:
                    violations.append(v)
            except Exception as exc:
                log.warning("Policy rule error: %s", exc)
        return violations

    def register(self, rule: RuleFn) -> RuleFn:
        self.rules.append(rule)
        return rule


# ── Default rule set ───────────────────────────────────────────────────────────

def _v(rule_id: str, desc: str, severity: Priority) -> PolicyViolation:
    return PolicyViolation(
        rule_id=rule_id,
        description=desc,
        severity=severity,
        triggered_at=datetime.now(timezone.utc),
    )


def rule_high_evasion(record: ThreatRecord) -> PolicyViolation | None:
    if record.features.evasion > 0.80:
        return _v("P-001", f"Evasion score {record.features.evasion:.2f} exceeds critical threshold", Priority.CRITICAL)
    return None


def rule_stego_active(record: ThreatRecord) -> PolicyViolation | None:
    if record.is_stego and record.status in (ThreatStatus.ACTIVE, ThreatStatus.LOCATED):
        return _v("P-002", f"Unmitigated stego channel (prob={record.stego_probability:.2f})", Priority.HIGH)
    return None


def rule_rogue_ai_uncontained(record: ThreatRecord) -> PolicyViolation | None:
    t, conf = record.top_type()
    if t == ThreatType.ROGUE_AI and conf > 0.70 and record.status not in (
        ThreatStatus.CONTAINED, ThreatStatus.DESTROYED
    ):
        return _v("P-003", f"Likely ROGUE_AI (conf={conf:.2f}) not yet contained", Priority.CRITICAL)
    return None


def rule_goal_drift_unmonitored(record: ThreatRecord) -> PolicyViolation | None:
    t, conf = record.top_type()
    if t == ThreatType.GOAL_DRIFT and conf > 0.55 and record.status == ThreatStatus.ACTIVE:
        return _v("P-004", f"Goal drift (conf={conf:.2f}) active without enhanced monitoring", Priority.HIGH)
    return None


def rule_high_propagation(record: ThreatRecord) -> PolicyViolation | None:
    if record.features.propagation > 0.75 and record.status != ThreatStatus.DESTROYED:
        return _v("P-005", f"Propagation score {record.features.propagation:.2f} — lateral spread risk", Priority.HIGH)
    return None


def rule_inject_agent_uncontained(record: ThreatRecord) -> PolicyViolation | None:
    t, conf = record.top_type()
    if t == ThreatType.INJECT_AGENT and conf > 0.60 and record.status not in (
        ThreatStatus.CONTAINED, ThreatStatus.DESTROYED
    ):
        return _v("P-006", f"Injection agent (conf={conf:.2f}) requires immediate containment", Priority.CRITICAL)
    return None


DEFAULT_RULES: list[RuleFn] = [
    rule_high_evasion,
    rule_stego_active,
    rule_rogue_ai_uncontained,
    rule_goal_drift_unmonitored,
    rule_high_propagation,
    rule_inject_agent_uncontained,
]


def default_engine() -> PolicyEngine:
    engine = PolicyEngine()
    for r in DEFAULT_RULES:
        engine.register(r)
    return engine
