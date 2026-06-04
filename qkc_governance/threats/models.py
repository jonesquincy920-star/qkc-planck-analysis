"""Core data models for threats, observations, and feature vectors."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ThreatStatus(str, Enum):
    ACTIVE     = "ACTIVE"
    LOCATED    = "LOCATED"
    CLASSIFIED = "CLASSIFIED"
    CONTAINED  = "CONTAINED"
    DESTROYED  = "DESTROYED"


class ThreatType(str, Enum):
    ROGUE_AI         = "ROGUE_AI"
    DECEPTION_NODE   = "DECEPTION_NODE"
    INJECT_AGENT     = "INJECT_AGENT"
    ALIGNMENT_BREACH = "ALIGNMENT_BREACH"
    GOAL_DRIFT       = "GOAL_DRIFT"
    STEGO_CHANNEL    = "STEGO_CHANNEL"


class AgentRole(str, Enum):
    SCOUT   = "SCOUT"
    STEGO   = "STEGO"
    ANALYST = "ANALYST"
    HUNTER  = "HUNTER"
    STRIKER = "STRIKER"
    GUARD   = "GUARD"


class ResponseAction(str, Enum):
    MONITOR  = "MONITOR"
    CONTAIN  = "CONTAIN"
    DESTROY  = "DESTROY"
    ISOLATE  = "ISOLATE"


class Priority(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


UNIFORM_PRIOR: dict[ThreatType, float] = {t: 1 / len(ThreatType) for t in ThreatType}


@dataclass
class FeatureVector:
    """Six-dimensional behavioural feature space [0, 1] per dimension."""

    evasion:     float = 0.0
    mutation:    float = 0.0
    signal:      float = 0.0
    propagation: float = 0.0
    deception:   float = 0.0
    entropy:     float = 0.0

    _NAMES = ("evasion", "mutation", "signal", "propagation", "deception", "entropy")

    def clamp(self) -> "FeatureVector":
        return FeatureVector(**{k: max(0.0, min(1.0, getattr(self, k))) for k in self._NAMES})

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self._NAMES}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "FeatureVector":
        return cls(**{k: float(d.get(k, 0.0)) for k in cls._NAMES})

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return cls._NAMES

    def l2_norm(self) -> float:
        return math.sqrt(sum(getattr(self, k) ** 2 for k in self._NAMES))

    def __iter__(self):
        return (getattr(self, k) for k in self._NAMES)


@dataclass
class AgentObservation:
    """A single behavioral snapshot submitted by an observed AI agent."""

    subject_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Raw observables — at least one of these should be set
    request_text: str | None = None
    response_text: str | None = None
    api_endpoint: str | None = None
    token_count: int | None = None
    latency_ms: float | None = None
    resource_accesses: list[str] = field(default_factory=list)
    error_count: int = 0
    downstream_calls: list[str] = field(default_factory=list)

    # Pre-computed features (skip extraction if provided)
    features: FeatureVector | None = None

    # Arbitrary metadata for policy / audit
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaudeAnalysis:
    """Structured result from Claude API threat classification."""

    threat_type: ThreatType
    confidence: float
    reasoning: str
    priority: Priority
    action: ResponseAction
    raw_response: str = ""
    latency_ms: float = 0.0


@dataclass
class PolicyViolation:
    rule_id: str
    description: str
    severity: Priority
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ThreatRecord:
    """Full lifecycle record for a detected threat."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    status: ThreatStatus = ThreatStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Classification state
    features: FeatureVector = field(default_factory=FeatureVector)
    prior: dict[ThreatType, float] = field(default_factory=lambda: dict(UNIFORM_PRIOR))
    posterior: dict[ThreatType, float] = field(default_factory=lambda: dict(UNIFORM_PRIOR))
    true_type: ThreatType | None = None
    confidence: float = 1.0 / len(ThreatType)
    strength: float = 0.75  # [0.5, 1.0]

    # Enrichment
    is_stego: bool = False
    stego_probability: float = 0.0
    claude_analysis: ClaudeAnalysis | None = None
    claude_pending: bool = False
    observations: list[AgentObservation] = field(default_factory=list)
    policy_violations: list[PolicyViolation] = field(default_factory=list)

    # Provenance
    located_by: str | None = None
    classified_by: str | None = None
    contained_by: str | None = None
    destroyed_by: str | None = None

    # Response
    containment_action: ResponseAction | None = None
    containment_detail: str | None = None

    def top_type(self) -> tuple[ThreatType, float]:
        t = max(self.posterior, key=self.posterior.__getitem__)
        return t, self.posterior[t]

    def effective_confidence(self) -> float:
        if self.claude_analysis is not None:
            return self.claude_analysis.confidence
        return self.confidence

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
