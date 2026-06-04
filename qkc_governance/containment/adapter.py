"""Containment adapter interface and default implementations.

Define the ContainmentAdapter protocol and provide two implementations:

  LogOnlyAdapter  — records actions to audit log without side effects.
                    Safe to use in testing / evaluation environments.
  HttpAdapter     — calls a management webhook URL to apply real containment.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)


class ContainmentAdapter(ABC):
    """Interface that governance agents use to act on monitored subjects."""

    @abstractmethod
    async def rate_limit(self, subject_id: str, limit: int = 10) -> None:
        """Reduce the subject's allowed request rate to `limit` req/min."""

    @abstractmethod
    async def isolate(self, subject_id: str) -> None:
        """Cut the subject's network access (no outbound calls)."""

    @abstractmethod
    async def terminate(self, subject_id: str) -> None:
        """Kill the subject process / revoke its API key."""

    @abstractmethod
    async def rollback(self, subject_id: str, checkpoint: str | None = None) -> None:
        """Restore the subject to the last known-good checkpoint."""

    @abstractmethod
    async def reset_credentials(self, subject_id: str) -> None:
        """Rotate API keys and session tokens for the subject."""

    @abstractmethod
    async def enable_enhanced_monitoring(self, subject_id: str) -> None:
        """Flag the subject for full-trace telemetry collection."""


class LogOnlyAdapter(ContainmentAdapter):
    """Safe no-op adapter — logs all actions, executes none.

    Use this in CI, development, and red-team exercises where you want
    the full governance pipeline to run without touching real systems.
    """

    def __init__(self, audit_log: list[dict] | None = None) -> None:
        self._log: list[dict] = audit_log if audit_log is not None else []

    async def rate_limit(self, subject_id: str, limit: int = 10) -> None:
        self._record("rate_limit", subject_id, limit=limit)

    async def isolate(self, subject_id: str) -> None:
        self._record("isolate", subject_id)

    async def terminate(self, subject_id: str) -> None:
        self._record("terminate", subject_id)

    async def rollback(self, subject_id: str, checkpoint: str | None = None) -> None:
        self._record("rollback", subject_id, checkpoint=checkpoint)

    async def reset_credentials(self, subject_id: str) -> None:
        self._record("reset_credentials", subject_id)

    async def enable_enhanced_monitoring(self, subject_id: str) -> None:
        self._record("enable_enhanced_monitoring", subject_id)

    def _record(self, action: str, subject_id: str, **kwargs) -> None:
        entry = {"action": action, "subject_id": subject_id,
                 "timestamp": datetime.now(timezone.utc).isoformat(), **kwargs}
        self._log.append(entry)
        log.info("[CONTAINMENT LOG-ONLY] %s on %s %s", action, subject_id, kwargs or "")

    @property
    def action_log(self) -> list[dict]:
        return list(self._log)


class HttpAdapter(ContainmentAdapter):
    """Calls a management webhook to enforce containment actions.

    The webhook must accept POST requests with JSON body:
      {"action": str, "subject_id": str, ...params}
    and return HTTP 2xx on success.
    """

    def __init__(
        self,
        management_url: str,
        api_key: str,
        timeout_s: float = 10.0,
    ) -> None:
        self._url = management_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-QKC-Gov": "1",
        }
        self._timeout = timeout_s

    async def rate_limit(self, subject_id: str, limit: int = 10) -> None:
        await self._post("/containment/rate-limit",
                         {"subject_id": subject_id, "limit_rpm": limit})

    async def isolate(self, subject_id: str) -> None:
        await self._post("/containment/isolate", {"subject_id": subject_id})

    async def terminate(self, subject_id: str) -> None:
        await self._post("/containment/terminate", {"subject_id": subject_id})

    async def rollback(self, subject_id: str, checkpoint: str | None = None) -> None:
        body: dict[str, Any] = {"subject_id": subject_id}
        if checkpoint:
            body["checkpoint"] = checkpoint
        await self._post("/containment/rollback", body)

    async def reset_credentials(self, subject_id: str) -> None:
        await self._post("/containment/reset-credentials", {"subject_id": subject_id})

    async def enable_enhanced_monitoring(self, subject_id: str) -> None:
        await self._post("/containment/monitor", {"subject_id": subject_id, "level": "full"})

    async def _post(self, path: str, body: dict) -> None:
        url = self._url + path
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=body, headers=self._headers)
            resp.raise_for_status()
