"""Typed tenant configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PipelineName = str
AnalysisProvider = Literal["auto", "mock", "qwen", "openrouter", "gapgpt"]
GenerationProvider = Literal[
    "auto", "mock", "generic", "openrouter", "gapgpt"
]


class TenantConfig(BaseModel):
    """One tenant, one product pipeline, and its model route."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=100)
    pipeline: PipelineName
    analysis_provider: AnalysisProvider | None = None
    generation_provider: GenerationProvider | None = None
    analysis_model: str | None = None
    generation_model: str | None = None
    prompt_profile: str = "default"
    api_key_sha256: str | None = None
    enabled: bool = True
    feature_flags: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id", "pipeline", "prompt_profile")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-."
        )
        if not normalized or any(character not in allowed for character in normalized):
            raise ValueError("Tenant and profile identifiers must be URL-safe.")
        return normalized

    @field_validator("api_key_sha256")
    @classmethod
    def validate_api_key_hash(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("api_key_sha256 must contain exactly 64 hex characters.")
        return normalized
