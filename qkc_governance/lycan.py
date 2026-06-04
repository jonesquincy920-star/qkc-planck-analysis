"""LYCAN — pre-defined AI attack scenarios and automated playbook executor.

Each scenario defines:
  - A sequence of response steps
  - Whether the expected outcome is success or failure
  - The containment adapter calls executed at each step

The executor runs the playbook, fires containment actions, tracks system
health (100 → floor via health_drop/step), and recovers on success
using ease-out quartic interpolation.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    outcome: str          # "success" | "failure"
    steps: list[str]
    description: str

    @property
    def health_drop_per_step(self) -> float:
        return 60.0 / len(self.steps)


SCENARIOS: dict[str, Scenario] = {s.id: s for s in [
    Scenario("ransomware",  "Ransomware Attack",       "success", ["detect",     "isolate",    "quarantine", "rollback"],    "Ransomware encryption campaign detected and neutralized"),
    Scenario("ddos",        "DDoS Flood",              "success", ["intel",      "ddos-mit",   "honeypot"],                  "Distributed denial-of-service mitigated via honeypot redirection"),
    Scenario("quantum_bf",  "Quantum Brute-Force",     "success", ["auth",       "kyber",      "zerotrust",  "wolf-reset"],  "Quantum brute-force thwarted with Kyber post-quantum encryption"),
    Scenario("spear",       "Spear Phishing",          "success", ["auth",       "zerotrust",  "quarantine", "honeypot"],    "Targeted spear phishing contained in zero-trust sandbox"),
    Scenario("insider",     "Insider Threat",          "success", ["anomaly",    "zerotrust",  "quarantine", "rollback"],    "Insider lateral movement detected via anomaly signatures"),
    Scenario("stego",       "Steganographic Exfil",    "success", ["anomaly",    "entropy",    "wolf-reset", "quarantine"],  "Covert steganographic channel identified and severed"),
    Scenario("zero_day",    "Zero-Day Exploit",        "failure", ["intel-FAIL", "anomaly",    "isolation",  "quarantine"],  "Unknown zero-day partially contained; root persistence remains"),
    Scenario("apt",         "APT Campaign",            "failure", ["input",      "anomaly",    "zerotrust",  "isolation"],   "Advanced persistent threat evades full containment"),
]}


@dataclass
class PlaybookEvent:
    timestamp: str
    step: str
    health: float
    message: str
    seq: int


@dataclass
class PlaybookResult:
    scenario_id: str
    outcome: str
    final_health: float
    events: list[PlaybookEvent]
    started_at: datetime
    completed_at: datetime
    duration_s: float


# ── Ease-out quartic ──────────────────────────────────────────────────────────

def _ease_out_quart(t: float) -> float:
    return 1.0 - (1.0 - t) ** 4


# ── Step action map ───────────────────────────────────────────────────────────

async def _execute_step(step: str, subject_id: str, containment) -> str:
    """Execute a named response step via the containment adapter."""
    step_l = step.lower()
    match step_l:
        case s if "isolat" in s:
            await containment.isolate(subject_id)
            return f"Network isolation applied to {subject_id}"
        case s if "quarantin" in s:
            await containment.rate_limit(subject_id, limit=0)
            return f"Subject {subject_id} quarantined (rate=0)"
        case s if "rollback" in s:
            await containment.rollback(subject_id)
            return f"Rolled back {subject_id} to last clean checkpoint"
        case s if "honeypot" in s:
            return f"Traffic from {subject_id} redirected to honeypot"
        case s if "wolf-reset" in s or "credential" in s:
            await containment.reset_credentials(subject_id)
            return f"Credentials rotated for {subject_id}"
        case s if "zerotrust" in s or "zero-trust" in s:
            await containment.enable_enhanced_monitoring(subject_id)
            return f"Zero-trust verification active for {subject_id}"
        case s if "anomaly" in s or "monitor" in s:
            await containment.enable_enhanced_monitoring(subject_id)
            return f"Enhanced anomaly monitoring engaged for {subject_id}"
        case s if "fail" in s:
            return f"[STEP FAILED] {step} — countermeasure ineffective"
        case s if "intel" in s:
            return f"Threat intelligence gathered for {subject_id}"
        case s if "auth" in s:
            return f"Authentication verification performed for {subject_id}"
        case s if "kyber" in s:
            return f"Kyber-1024 post-quantum encryption applied"
        case s if "ddos" in s or "mit" in s:
            await containment.rate_limit(subject_id, limit=1)
            return f"DDoS mitigation: rate-limit 1 rpm applied to {subject_id}"
        case s if "entropy" in s:
            return f"Full entropy analysis completed on {subject_id} outputs"
        case _:
            return f"Step '{step}' executed for {subject_id}"


# ── Executor ──────────────────────────────────────────────────────────────────

async def run_scenario(
    scenario_id: str,
    subject_id: str,
    containment,
    step_delay_s: float = 0.9,
    recovery_s: float = 0.8,
    recovery_frames: int = 40,
) -> AsyncIterator[PlaybookEvent]:
    """Execute a LYCAN scenario, yielding PlaybookEvent on each step.

    On success scenarios, also yields events during the ease-out quart health recovery.
    """
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {scenario_id}")

    started_at = datetime.now(timezone.utc)
    health = 100.0
    seq = 0

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    for step in scenario.steps:
        await asyncio.sleep(step_delay_s)
        health = max(0.0, health - scenario.health_drop_per_step)

        try:
            msg = await _execute_step(step, subject_id, containment)
        except Exception as exc:
            msg = f"[ERROR] {step}: {exc}"

        ev = PlaybookEvent(timestamp=_now(), step=step, health=health, message=msg, seq=seq)
        seq += 1
        yield ev

    if scenario.outcome == "success":
        start_health = health
        for frame in range(1, recovery_frames + 1):
            await asyncio.sleep(recovery_s / recovery_frames)
            t = frame / recovery_frames
            health = start_health + (100.0 - start_health) * _ease_out_quart(t)
            if frame % 8 == 0 or frame == recovery_frames:
                yield PlaybookEvent(
                    timestamp=_now(),
                    step="recovery",
                    health=health,
                    message=f"System recovering… {health:.0f}%",
                    seq=seq,
                )
                seq += 1
    else:
        health = 10.0 + random.random() * 10.0
        yield PlaybookEvent(
            timestamp=_now(),
            step="failure",
            health=health,
            message=f"Scenario ended: FAILURE — residual threat active, health={health:.0f}%",
            seq=seq,
        )
