"""Validated input-analysis models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import ALLOWED_GARMENT_CATEGORIES


class PersonAnalysis(BaseModel):
    """Qwen's validated assessment of the source person image."""

    model_config = ConfigDict(extra="forbid")

    person_count: int = Field(ge=0)
    pose: str
    body_visibility: str
    arms_position: str
    image_quality: str
    background_complexity: str
    suitable_for_tryon: bool
    rejection_reason: str | None = None


class GarmentAnalysis(BaseModel):
    """Qwen's validated assessment of the source garment image."""

    model_config = ConfigDict(extra="forbid")

    garment_category: str
    garment_type: str
    sleeve_type: str | None = None
    base_color: str | None = None
    has_logo: bool
    has_pattern: bool
    recommended_tryon_category: str

    @field_validator("garment_category", "recommended_tryon_category")
    @classmethod
    def valid_category(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_GARMENT_CATEGORIES:
            raise ValueError(f"Unsupported garment category: {value}")
        return normalized
