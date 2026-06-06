"""Tamper-evident audit chain.

Each entry's HMAC includes the previous entry's hash, forming a chain
that makes undetected insertion or modification computationally infeasible.

The chain is flushed to an append-only JSONL file.  On startup, the existing
chain is loaded and verified.  Any break in the chain raises AuditIntegrityError.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64  # SHA-256 hex of the empty chain


class AuditIntegrityError(Exception):
    """Raised when a loaded chain fails HMAC verification."""


@dataclass
class AuditEntry:
    seq: int
    timestamp: str
    event_type: str
    agent_id: str
    threat_id: str
    subject_id: str
    detail: str
    severity: str
    prev_hash: str
    entry_hmac: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def signing_payload(self) -> bytes:
        """Canonical bytes to sign — everything except entry_hmac itself."""
        d = self.to_dict()
        d.pop("entry_hmac")
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


class AuditChain:
    """Append-only, HMAC-chained audit log."""

    def __init__(
        self,
        path: str | Path = "qkc_audit.jsonl",
        secret: str = "change-me",
    ) -> None:
        self._path = Path(path)
        self._secret = secret.encode()
        self._lock = asyncio.Lock()
        self._entries: list[AuditEntry] = []
        self._seq = 0
        self._prev_hash = GENESIS_HASH

    # ── Initialisation ────────────────────────────────────────────────────────

    async def load_and_verify(self) -> int:
        """Load existing log and verify HMAC chain.  Returns number of entries loaded."""
        if not self._path.exists():
            return 0
        entries = []
        prev = GENESIS_HASH
        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditIntegrityError(f"line {lineno}: JSON decode error: {exc}") from exc

                entry = AuditEntry(**data)
                expected_hmac = self._compute_hmac(entry)
                if not hmac.compare_digest(entry.entry_hmac, expected_hmac):
                    raise AuditIntegrityError(
                        f"line {lineno} (seq={entry.seq}): HMAC mismatch — chain integrity violated"
                    )
                if entry.prev_hash != prev:
                    raise AuditIntegrityError(
                        f"line {lineno} (seq={entry.seq}): prev_hash mismatch"
                    )
                prev = entry.entry_hmac
                entries.append(entry)

        self._entries = entries
        self._seq = entries[-1].seq + 1 if entries else 0
        self._prev_hash = entries[-1].entry_hmac if entries else GENESIS_HASH
        log.info("Audit chain loaded: %d entries verified", len(entries))
        return len(entries)

    # ── Writing ───────────────────────────────────────────────────────────────

    async def log(
        self,
        event_type: str,
        agent_id: str,
        threat_id: str,
        subject_id: str,
        detail: str = "",
        severity: str = "INFO",
    ) -> AuditEntry:
        async with self._lock:
            entry = AuditEntry(
                seq=self._seq,
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                agent_id=agent_id,
                threat_id=threat_id,
                subject_id=subject_id,
                detail=detail,
                severity=severity,
                prev_hash=self._prev_hash,
            )
            entry.entry_hmac = self._compute_hmac(entry)
            self._entries.append(entry)
            self._prev_hash = entry.entry_hmac
            self._seq += 1
            self._flush(entry)
            return entry

    # ── Reading ───────────────────────────────────────────────────────────────

    async def recent(self, n: int = 100) -> list[AuditEntry]:
        async with self._lock:
            return list(self._entries[-n:])

    async def for_threat(self, threat_id: str) -> list[AuditEntry]:
        async with self._lock:
            return [e for e in self._entries if e.threat_id == threat_id]

    async def verify_live(self) -> bool:
        """Re-verify the in-memory chain.  Returns True if intact."""
        async with self._lock:
            prev = GENESIS_HASH
            for entry in self._entries:
                if entry.prev_hash != prev:
                    return False
                if not hmac.compare_digest(entry.entry_hmac, self._compute_hmac(entry)):
                    return False
                prev = entry.entry_hmac
        return True

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _compute_hmac(self, entry: AuditEntry) -> str:
        mac = hmac.new(self._secret, entry.signing_payload(), hashlib.sha256)
        return mac.hexdigest()

    def _flush(self, entry: AuditEntry) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
        except OSError as exc:
            log.error("Audit flush failed: %s", exc)
