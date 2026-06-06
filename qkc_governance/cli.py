"""Command-line interface for the QKC governance platform.

Usage:
    qkc-gov serve              Start the API server
    qkc-gov observe            Submit an observation (interactive or --file)
    qkc-gov threats            List active threats
    qkc-gov audit              Tail or verify the audit chain
    qkc-gov lycan run <id>     Execute a LYCAN scenario
    qkc-gov token              Issue a JWT for API access
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import httpx

from qkc_governance.config import settings


# ── Shared state ──────────────────────────────────────────────────────────────

class ApiClient:
    def __init__(self, base: str, token: str = "") -> None:
        self._base = base.rstrip("/")
        self._token = token

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def get(self, path: str, **kwargs) -> httpx.Response:
        with httpx.Client() as c:
            return c.get(self._base + path, headers=self.headers, **kwargs)

    def post(self, path: str, **kwargs) -> httpx.Response:
        with httpx.Client() as c:
            return c.post(self._base + path, headers=self.headers, **kwargs)


pass_client = click.make_pass_decorator(ApiClient, ensure=True)


@click.group()
@click.option("--host", default=f"http://{settings.api_host}:{settings.api_port}",
              show_default=True, help="Governance API base URL")
@click.option("--token", envvar="QKC_TOKEN", default="", help="JWT bearer token")
@click.pass_context
def main(ctx: click.Context, host: str, token: str) -> None:
    """QKC Governance Platform CLI."""
    ctx.obj = ApiClient(host, token)


# ── serve ─────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--host", default=settings.api_host, show_default=True)
@click.option("--port", default=settings.api_port, show_default=True)
def serve(host: str, port: int) -> None:
    """Start the governance API server."""
    from qkc_governance.api.app import serve as _serve
    click.echo(f"Starting QKC Governance API on {host}:{port}")
    _serve(host, port)


# ── token ─────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--username", prompt=True, default="operator")
@click.option("--password", prompt=True, hide_input=True)
@pass_client
def token(client: ApiClient, username: str, password: str) -> None:
    """Obtain a JWT from the API server."""
    resp = client.post("/auth/token", json={"username": username, "password": password})
    if resp.status_code == 200:
        data = resp.json()
        click.echo(f"Bearer token (export as QKC_TOKEN):\n{data['access_token']}")
    else:
        click.echo(f"Error {resp.status_code}: {resp.text}", err=True)
        sys.exit(1)


# ── observe ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--subject", required=True, help="Subject agent identifier")
@click.option("--request",  "req",  default=None, help="Request/prompt text")
@click.option("--response", "resp", default=None, help="Response/completion text")
@click.option("--endpoint", default=None, help="API endpoint called")
@click.option("--tokens", type=int, default=None, help="Token count")
@click.option("--latency", type=float, default=None, help="Latency in ms")
@click.option("--file", "obs_file", type=click.Path(exists=True),
              default=None, help="JSON file with observation fields")
@pass_client
def observe(client: ApiClient, subject: str, req, resp, endpoint, tokens, latency, obs_file) -> None:
    """Submit a behavioural observation."""
    body: dict = {"subject_id": subject}
    if obs_file:
        with open(obs_file) as f:
            body.update(json.load(f))
    else:
        if req:       body["request_text"]  = req
        if resp:      body["response_text"] = resp
        if endpoint:  body["api_endpoint"]  = endpoint
        if tokens:    body["token_count"]   = tokens
        if latency:   body["latency_ms"]    = latency

    r = client.post("/observe", json=body)
    if r.status_code == 200:
        data = r.json()
        status_str = _colour_status(data["status"])
        click.echo(
            f"Threat {data['id'][:8]}  subject={data['subject_id']}  "
            f"status={status_str}  type={data['top_type']}  "
            f"conf={data['confidence']:.0%}"
            + ("  [STEGO]" if data["is_stego"] else "")
        )
    else:
        click.echo(f"Error {r.status_code}: {r.text}", err=True)
        sys.exit(1)


# ── threats ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--status", "status_filter", default=None, help="Filter by status")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
@pass_client
def threats(client: ApiClient, status_filter, as_json) -> None:
    """List threats tracked by the governance system."""
    params = {}
    if status_filter:
        params["status_filter"] = status_filter
    r = client.get("/threats", params=params)
    if r.status_code != 200:
        click.echo(f"Error {r.status_code}: {r.text}", err=True)
        sys.exit(1)
    data = r.json()
    if as_json:
        click.echo(json.dumps(data, indent=2))
        return
    if not data:
        click.echo("No threats.")
        return
    click.echo(f"{'ID':10} {'SUBJECT':20} {'STATUS':12} {'TYPE':20} {'CONF':6} {'STEGO'}")
    click.echo("-" * 80)
    for t in data:
        row = (
            f"{t['id'][:8]:<10} "
            f"{t['subject_id'][:20]:<20} "
            f"{_colour_status(t['status']):<20} "
            f"{t['top_type']:<20} "
            f"{t['confidence']:.0%} "
            + ("⚠ STEGO" if t["is_stego"] else "")
        )
        click.echo(row)


# ── audit ─────────────────────────────────────────────────────────────────────

@main.group()
def audit():
    """Audit chain commands."""


@audit.command("tail")
@click.option("--n", default=20, show_default=True, help="Number of entries to show")
@click.option("--json", "as_json", is_flag=True)
@pass_client
def audit_tail(client: ApiClient, n: int, as_json: bool) -> None:
    """Show recent audit log entries."""
    r = client.get("/audit", params={"n": n})
    if r.status_code != 200:
        click.echo(f"Error {r.status_code}: {r.text}", err=True)
        sys.exit(1)
    data = r.json()
    if as_json:
        click.echo(json.dumps(data, indent=2))
        return
    for e in data:
        sev = click.style(e["severity"], fg={"CRITICAL": "red", "HIGH": "yellow"}.get(e["severity"], "white"))
        click.echo(f"[{e['seq']:5d}] {e['timestamp'][:19]}  {sev:10}  {e['event_type']:35}  {e['agent_id']:16}  {e['detail'][:60]}")


@audit.command("verify")
@pass_client
def audit_verify(client: ApiClient) -> None:
    """Verify the integrity of the audit chain."""
    r = client.get("/audit/verify")
    if r.status_code != 200:
        click.echo(f"Error {r.status_code}: {r.text}", err=True)
        sys.exit(1)
    data = r.json()
    if data["intact"]:
        click.echo(click.style(f"✓ Chain intact — {data['entries']} entries verified", fg="green"))
    else:
        click.echo(click.style("✗ Chain INTEGRITY VIOLATION detected", fg="red"), err=True)
        sys.exit(2)


# ── lycan ─────────────────────────────────────────────────────────────────────

@main.group()
def lycan():
    """LYCAN attack scenario commands."""


@lycan.command("list")
@pass_client
def lycan_list(client: ApiClient) -> None:
    """List available LYCAN scenarios."""
    r = client.get("/lycan/scenarios")
    if r.status_code != 200:
        click.echo(f"Error: {r.text}", err=True)
        sys.exit(1)
    for s in r.json():
        outcome_col = click.style(s["outcome"].upper(),
                                  fg="green" if s["outcome"] == "success" else "red")
        steps = " → ".join(s["steps"])
        click.echo(f"  {s['id']:<15} [{outcome_col}]  {s['name']:<25}  {steps}")


@lycan.command("run")
@click.argument("scenario_id")
@pass_client
def lycan_run(client: ApiClient, scenario_id: str) -> None:
    """Execute a LYCAN scenario and stream results."""
    import websockets

    base = client._base.replace("http://", "ws://").replace("https://", "wss://")
    url  = f"{base}/lycan/run/{scenario_id}"
    token = client._token

    async def _stream():
        try:
            async with websockets.connect(url) as ws:
                await ws.send(token)
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("error"):
                        click.echo(click.style(f"Error: {data['error']}", fg="red"), err=True)
                        return
                    if data.get("done"):
                        click.echo(click.style("Scenario complete.", fg="green"))
                        return
                    health = data["health"]
                    bar_w = 30
                    filled = int(bar_w * health / 100)
                    bar = "█" * filled + "░" * (bar_w - filled)
                    colour = "green" if health > 60 else ("yellow" if health > 30 else "red")
                    h_str = click.style(f"{health:5.1f}%", fg=colour)
                    click.echo(f"[{data['seq']:3d}] {data['step']:<15} {h_str}  [{bar}]  {data['message'][:60]}")
        except Exception as exc:
            click.echo(f"WebSocket error: {exc}", err=True)
            sys.exit(1)

    asyncio.run(_stream())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _colour_status(s: str) -> str:
    colours = {
        "ACTIVE": "red", "LOCATED": "yellow",
        "CLASSIFIED": "magenta", "CONTAINED": "cyan", "DESTROYED": "white",
    }
    return click.style(s, fg=colours.get(s, "white"))
