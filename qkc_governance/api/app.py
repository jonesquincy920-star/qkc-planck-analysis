"""FastAPI application — REST + WebSocket API for the governance platform."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qkc_governance.api.auth import (
    TokenPayload,
    TokenResponse,
    create_token,
    require_operator,
)
from qkc_governance.config import settings
from qkc_governance.governance import GovernanceSystem
from qkc_governance.lycan import SCENARIOS, run_scenario
from qkc_governance.threats.models import (
    AgentObservation,
    FeatureVector,
    ThreatRecord,
    ThreatStatus,
)

log = logging.getLogger(__name__)

# Singleton governance system attached to app state
_gov: GovernanceSystem | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gov
    _gov = GovernanceSystem()
    await _gov.start()
    log.info("GovernanceSystem ready")
    yield
    await _gov.stop()
    log.info("GovernanceSystem shut down")


app = FastAPI(
    title="QKC Governance Platform",
    description="Multi-agent AI threat detection and response system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def gov() -> GovernanceSystem:
    if _gov is None:
        raise HTTPException(status_code=503, detail="Governance system not ready")
    return _gov


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def login(body: LoginRequest):
    """Issue a JWT.  In production, validate against your identity provider."""
    # Placeholder: accept any non-empty credentials and assign operator role.
    # Replace with real IdP validation before deployment.
    if not body.username or not body.password:
        raise HTTPException(status_code=400, detail="username and password required")
    role = "admin" if body.username == "admin" else "operator"
    return create_token(body.username, role)


# ── Observation ingestion ─────────────────────────────────────────────────────

class ObservationRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    request_text: str | None = None
    response_text: str | None = None
    api_endpoint: str | None = None
    token_count: int | None = None
    latency_ms: float | None = None
    resource_accesses: list[str] = []
    error_count: int = 0
    downstream_calls: list[str] = []
    features: dict[str, float] | None = None
    metadata: dict[str, Any] = {}


class ThreatSummary(BaseModel):
    id: str
    subject_id: str
    status: str
    top_type: str
    confidence: float
    is_stego: bool
    claude_available: bool
    created_at: str
    updated_at: str


def _summarise(r: ThreatRecord) -> ThreatSummary:
    t, c = r.top_type()
    return ThreatSummary(
        id=r.id,
        subject_id=r.subject_id,
        status=r.status.value,
        top_type=t.value,
        confidence=round(c, 4),
        is_stego=r.is_stego,
        claude_available=r.claude_analysis is not None,
        created_at=r.created_at.isoformat(),
        updated_at=r.updated_at.isoformat(),
    )


@app.post("/observe", response_model=ThreatSummary, tags=["observations"])
async def submit_observation(
    body: ObservationRequest,
    _: TokenPayload = Depends(require_operator),
    g: GovernanceSystem = Depends(gov),
):
    """Submit a behavioural observation for a monitored AI agent."""
    fv = FeatureVector.from_dict(body.features) if body.features else None
    obs = AgentObservation(
        subject_id=body.subject_id,
        request_text=body.request_text,
        response_text=body.response_text,
        api_endpoint=body.api_endpoint,
        token_count=body.token_count,
        latency_ms=body.latency_ms,
        resource_accesses=body.resource_accesses,
        error_count=body.error_count,
        downstream_calls=body.downstream_calls,
        features=fv,
        metadata=body.metadata,
    )
    record = await g.submit(obs)
    return _summarise(record)


# ── Threats ───────────────────────────────────────────────────────────────────

@app.get("/threats", response_model=list[ThreatSummary], tags=["threats"])
async def list_threats(
    status_filter: str | None = None,
    _: TokenPayload = Depends(require_operator),
    g: GovernanceSystem = Depends(gov),
):
    if status_filter:
        try:
            s = ThreatStatus(status_filter.upper())
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status_filter}")
        records = await g.threats(s)
    else:
        records = await g.threats()
    return [_summarise(r) for r in records]


@app.get("/threats/{threat_id}", tags=["threats"])
async def get_threat(
    threat_id: str,
    _: TokenPayload = Depends(require_operator),
    g: GovernanceSystem = Depends(gov),
):
    record = await g.get_threat(threat_id)
    if record is None:
        raise HTTPException(404, "Threat not found")
    t, c = record.top_type()
    return {
        "id": record.id,
        "subject_id": record.subject_id,
        "status": record.status.value,
        "features": record.features.to_dict(),
        "posterior": {k.value: round(v, 4) for k, v in record.posterior.items()},
        "top_type": t.value,
        "confidence": round(c, 4),
        "effective_confidence": round(record.effective_confidence(), 4),
        "is_stego": record.is_stego,
        "stego_probability": round(record.stego_probability, 4),
        "strength": round(record.strength, 4),
        "true_type": record.true_type.value if record.true_type else None,
        "claude_analysis": {
            "threat_type": record.claude_analysis.threat_type.value,
            "confidence": record.claude_analysis.confidence,
            "reasoning": record.claude_analysis.reasoning,
            "priority": record.claude_analysis.priority.value,
            "action": record.claude_analysis.action.value,
        } if record.claude_analysis else None,
        "policy_violations": [
            {
                "rule_id": v.rule_id,
                "description": v.description,
                "severity": v.severity.value,
                "triggered_at": v.triggered_at.isoformat(),
            }
            for v in record.policy_violations
        ],
        "containment_action": record.containment_action.value if record.containment_action else None,
        "containment_detail": record.containment_detail,
        "located_by": record.located_by,
        "classified_by": record.classified_by,
        "contained_by": record.contained_by,
        "destroyed_by": record.destroyed_by,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


# ── Agents ────────────────────────────────────────────────────────────────────

@app.get("/agents", tags=["agents"])
async def list_agents(
    _: TokenPayload = Depends(require_operator),
    g: GovernanceSystem = Depends(gov),
):
    return [
        {
            "agent_id": a.agent_id,
            "role": a.ROLE,
            "sensors": list(a.SENSORS),
            "active": a.state.active,
            "cycles": a.state.cycles,
            "threats_handled": a.state.threats_handled,
            "errors": a.state.errors,
            "last_active": a.state.last_active.isoformat(),
        }
        for a in g._agents
    ]


# ── Audit trail ───────────────────────────────────────────────────────────────

@app.get("/audit", tags=["audit"])
async def get_audit_log(
    n: int = 100,
    _: TokenPayload = Depends(require_operator),
    g: GovernanceSystem = Depends(gov),
):
    entries = await g.audit.recent(n)
    return [e.to_dict() for e in entries]


@app.get("/audit/verify", tags=["audit"])
async def verify_audit_chain(
    _: TokenPayload = Depends(require_operator),
    g: GovernanceSystem = Depends(gov),
):
    intact = await g.audit.verify_live()
    return {"intact": intact, "entries": len(g.audit)}


# ── LYCAN scenarios ───────────────────────────────────────────────────────────

@app.get("/lycan/scenarios", tags=["lycan"])
async def list_scenarios(_: TokenPayload = Depends(require_operator)):
    return [
        {
            "id": s.id,
            "name": s.name,
            "outcome": s.outcome,
            "steps": s.steps,
            "description": s.description,
            "health_drop_per_step": s.health_drop_per_step,
        }
        for s in SCENARIOS.values()
    ]


@app.websocket("/lycan/run/{scenario_id}")
async def run_lycan_scenario(ws: WebSocket, scenario_id: str):
    """Stream scenario execution events over WebSocket.

    Clients must send a valid JWT as the first message after connecting.
    """
    await ws.accept()

    # Auth handshake
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
        from qkc_governance.api.auth import decode_token
        decode_token(raw)
    except Exception:
        await ws.send_text(json.dumps({"error": "unauthorized"}))
        await ws.close(code=1008)
        return

    if scenario_id not in SCENARIOS:
        await ws.send_text(json.dumps({"error": f"unknown scenario: {scenario_id}"}))
        await ws.close()
        return

    subject_id = f"lycan-subject-{scenario_id}"
    containment = LogOnlyAdapter() if _gov is None else _gov.containment

    try:
        async for event in run_scenario(scenario_id, subject_id, containment):
            await ws.send_text(json.dumps({
                "seq": event.seq,
                "step": event.step,
                "health": round(event.health, 2),
                "message": event.message,
                "timestamp": event.timestamp,
            }))
        await ws.send_text(json.dumps({"done": True}))
    except WebSocketDisconnect:
        pass


# ── Reports ───────────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    scenario_id: str
    outcome: str
    event_log: list[dict[str, str]] = []


@app.post("/report", tags=["reports"])
async def generate_report(
    body: ReportRequest,
    _: TokenPayload = Depends(require_operator),
):
    from qkc_governance.classify.claude_api import generate_incident_report
    sc = SCENARIOS.get(body.scenario_id)
    if sc is None:
        raise HTTPException(400, f"Unknown scenario: {body.scenario_id}")
    text = await generate_incident_report(
        sc.name, sc.description, body.outcome, sc.steps, body.event_log
    )
    return {"report": text}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


def serve(host: str | None = None, port: int | None = None) -> None:
    uvicorn.run(
        "qkc_governance.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
        log_level="info",
    )


from qkc_governance.containment.adapter import LogOnlyAdapter  # noqa: E402
