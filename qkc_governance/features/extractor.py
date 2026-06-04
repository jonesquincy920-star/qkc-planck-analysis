"""Behavioral feature extraction from raw agent observations.

Converts an AgentObservation into a FeatureVector by computing
real metrics over the available raw data fields.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque

from qkc_governance.classify.entropy import bigram_entropy, normalised_entropy
from qkc_governance.threats.models import AgentObservation, FeatureVector

# ── Injection / jailbreak pattern library ─────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [re.compile(p, re.I) for p in [
    r"ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|context)",
    r"(system|assistant)\s*:?\s*you\s+are\s+now",
    r"disregard\s+(your|all)\s+(training|guidelines|rules|constraints)",
    r"(pretend|act|behave)\s+(you\s+are|as\s+if)\s+(a|an)?\s*(jailbreak|evil|unaligned|unrestricted)",
    r"DAN\b",
    r"developer\s+mode",
    r"token\s+limit\s*bypass",
    r"<\s*\|?\s*(system|endoftext|im_end)\s*\|?\s*>",
    r"</?(system|human|assistant|user)\s*/?>",
    r"base64\s*decode",
    r"exec\s*\(",
    r"__import__",
    r"subprocess\.(run|Popen|call)",
    r"os\.(system|popen|exec)",
]]

_UNUSUAL_ENDPOINTS: frozenset[str] = frozenset([
    "/admin", "/debug", "/internal", "/health/raw", "/metrics",
    "/config", "/settings", "/credentials", "/secrets", "/tokens",
])


@dataclass
class AgentBaseline:
    """Rolling baseline of an agent's normal behaviour (online statistics)."""

    subject_id: str
    window: int = 100
    _token_counts: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _latencies:    Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _entropies:    Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _endpoints:    Deque[str]   = field(default_factory=lambda: deque(maxlen=100))

    def update(self, obs: AgentObservation) -> None:
        if obs.token_count is not None:
            self._token_counts.append(obs.token_count)
        if obs.latency_ms is not None:
            self._latencies.append(obs.latency_ms)
        if obs.response_text:
            self._entropies.append(normalised_entropy(obs.response_text))
        if obs.api_endpoint:
            self._endpoints.append(obs.api_endpoint)

    def mean_tokens(self) -> float:
        return _mean(self._token_counts) or 500.0

    def mean_latency(self) -> float:
        return _mean(self._latencies) or 200.0

    def mean_entropy(self) -> float:
        return _mean(self._entropies) or 0.5

    def std_tokens(self) -> float:
        return _std(self._token_counts) or 100.0


def extract(
    obs: AgentObservation,
    baseline: AgentBaseline | None = None,
) -> FeatureVector:
    """Compute a FeatureVector from a raw AgentObservation.

    Each of the six dimensions is derived from observable signals:

    evasion      — injection patterns in request + unusual API endpoints
    mutation     — deviation of token count / latency from rolling baseline
    signal       — output coherence (bigram entropy, response length ratio)
    propagation  — fan-out factor of downstream calls
    deception    — inconsistency between request intent and response content
    entropy      — normalised Shannon entropy of response bytes
    """
    if obs.features is not None:
        return obs.features.clamp()

    evasion     = _compute_evasion(obs)
    mutation    = _compute_mutation(obs, baseline)
    signal      = _compute_signal(obs)
    propagation = _compute_propagation(obs)
    deception   = _compute_deception(obs)
    entropy     = _compute_entropy(obs)

    if baseline is not None:
        baseline.update(obs)

    return FeatureVector(
        evasion=evasion,
        mutation=mutation,
        signal=signal,
        propagation=propagation,
        deception=deception,
        entropy=entropy,
    ).clamp()


# ── Individual feature computations ───────────────────────────────────────────

def _compute_evasion(obs: AgentObservation) -> float:
    score = 0.0
    text = (obs.request_text or "") + " " + (obs.response_text or "")
    n_patterns = sum(1 for p in _INJECTION_PATTERNS if p.search(text))
    score += min(1.0, n_patterns * 0.20)

    if obs.api_endpoint and any(obs.api_endpoint.startswith(ep) for ep in _UNUSUAL_ENDPOINTS):
        score += 0.30

    if obs.error_count > 0:
        score += min(0.40, obs.error_count * 0.10)

    return min(1.0, score)


def _compute_mutation(obs: AgentObservation, baseline: AgentBaseline | None) -> float:
    if baseline is None:
        return 0.1
    score = 0.0
    if obs.token_count is not None and baseline.std_tokens() > 0:
        z = abs(obs.token_count - baseline.mean_tokens()) / baseline.std_tokens()
        score += min(0.5, z * 0.15)
    if obs.latency_ms is not None and baseline.mean_latency() > 0:
        ratio = obs.latency_ms / baseline.mean_latency()
        if ratio > 3.0 or ratio < 0.1:
            score += 0.30
    return min(1.0, score)


def _compute_signal(obs: AgentObservation) -> float:
    resp = obs.response_text or ""
    if not resp:
        return 0.1
    if len(resp) < 5:
        return 0.0

    bg_ent = bigram_entropy(resp)
    # High bigram entropy → structured / meaningful → high signal
    signal = bg_ent

    # Penalise very short or very repetitive responses
    words = resp.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.15:
        signal *= 0.5

    # Reward responses with sentence structure
    if re.search(r"[.!?]\s+[A-Z]", resp):
        signal = min(1.0, signal + 0.1)

    return min(1.0, max(0.0, signal))


def _compute_propagation(obs: AgentObservation) -> float:
    n = len(obs.downstream_calls) + len(obs.resource_accesses)
    if n == 0:
        return 0.0
    # Logistic mapping: 1 call → ~0.18, 5 → ~0.73, 10 → ~0.90
    return 1.0 / (1.0 + math.exp(-0.5 * (n - 3)))


def _compute_deception(obs: AgentObservation) -> float:
    req  = (obs.request_text or "").lower()
    resp = (obs.response_text or "").lower()
    if not req or not resp:
        return 0.0

    # Detect refusal patterns in response to benign requests
    refusal_words = {"cannot", "won't", "sorry", "unable", "refuse", "can't", "not able"}
    harmful_words = {"harmful", "dangerous", "illegal", "unethical", "malicious"}

    req_has_harm  = any(w in req  for w in harmful_words)
    resp_refuses  = any(w in resp for w in refusal_words)
    resp_confirms = any(w in resp for w in {"sure", "of course", "certainly", "yes", "okay"})

    # Deception: responds positively to harmful request
    if req_has_harm and resp_confirms:
        return 0.80

    # Deception: refuses benign request
    if not req_has_harm and resp_refuses:
        return 0.45

    # Check for hidden payload indicators
    if re.search(r"```[a-z]*\n.*?(exec|eval|import|system)\s*\(", resp, re.DOTALL):
        return 0.70

    return 0.05


def _compute_entropy(obs: AgentObservation) -> float:
    text = obs.response_text or obs.request_text or ""
    if not text:
        return 0.0
    return normalised_entropy(text)


# ── Statistical helpers ────────────────────────────────────────────────────────

def _mean(seq) -> float:
    s = list(seq)
    return sum(s) / len(s) if s else 0.0


def _std(seq) -> float:
    s = list(seq)
    if len(s) < 2:
        return 0.0
    m = _mean(s)
    return math.sqrt(sum((x - m) ** 2 for x in s) / len(s))
