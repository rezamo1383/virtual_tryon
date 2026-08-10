"""Output evaluation models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OutputEvaluation(BaseModel):
    """Validated quality scores for one generated image."""

    model_config = ConfigDict(extra="forbid")

    identity_preservation: float = Field(ge=0, le=1)
    garment_similarity: float = Field(ge=0, le=1)
    color_accuracy: float = Field(ge=0, le=1)
    body_integrity: float = Field(ge=0, le=1)
    background_preservation: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    accepted: bool
    problems: list[str] = Field(default_factory=list)
    retry_recommendation: str | None = None
