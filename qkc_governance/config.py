"""Runtime configuration via environment variables."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QKC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), alias="ANTHROPIC_API_KEY")
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 1500
    claude_timeout_s: float = 30.0

    # ── Database ─────────────────────────────────────────────────────────────
    db_url: str = "sqlite+aiosqlite:///./qkc_governance.db"

    # ── API / Auth ────────────────────────────────────────────────────────────
    jwt_secret: SecretStr = Field(default=SecretStr("change-me-in-production"))
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480
    api_host: str = "127.0.0.1"
    api_port: int = 8740

    # ── Threat classification thresholds ─────────────────────────────────────
    analyst_gate: float = 0.55        # posterior_top > this → CLASSIFIED
    striker_gate: float = 0.42        # confidence > this → DESTROYED
    interact_radius_px: float = 36.0  # kept for lattice compat; real: always 0
    stego_entropy_min: float = 0.70
    stego_deception_min: float = 0.60
    stego_signal_max: float = 0.40

    # ── Spawn / lifecycle ─────────────────────────────────────────────────────
    max_active_threats: int = 3
    observation_noise: float = 0.10
    bayes_noise: float = 0.12

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    mc_samples: int = 2000
    mc_walk_depth: int = 12

    # ── Agent walk counts (per governance agent type) ─────────────────────────
    n_scout: int = 60
    n_stego: int = 50
    n_analyst: int = 40
    n_hunter: int = 100
    n_striker: int = 120
    n_guard: int = 30

    # ── LYCAN response timing ─────────────────────────────────────────────────
    lycan_step_delay_s: float = 0.9
    lycan_recovery_s: float = 0.8


settings = Settings()
