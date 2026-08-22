"""Request models."""

from __future__ import annotations

from pathlib import Path
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


ORIGINAL_GARMENT_COLOR = "original"


class GarmentCategory(str, Enum):
    """Trusted clothing categories accepted by the production endpoint."""

    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"
    DRESS = "dress"
    OUTERWEAR = "outerwear"


class TryOnMode(str, Enum):
    """Generation policy for prepared-product requests."""

    FAST = "fast"
    QUALITY = "quality"


class PreparedTryOnRequest(BaseModel):
    """Production request using a separately prepared product garment."""

    model_config = ConfigDict(extra="forbid")

    person_image: Path
    product_id: str = Field(min_length=1, max_length=128)
    category: GarmentCategory
    tenant_id: str = Field(min_length=1, max_length=100)
    product_title: str | None = Field(default=None, max_length=160)

    @field_validator("product_id", "tenant_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in normalized
        ):
            raise ValueError("identifier contains unsupported characters")
        return normalized

    @field_validator("product_title")
    @classmethod
    def normalize_prepared_product_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None


class GenerationRequest(BaseModel):
    """Domain-neutral two-reference generation request."""

    model_config = ConfigDict(extra="forbid")

    source_image: Path
    reference_image: Path
    job_id: str | None = Field(default=None, exclude=True)
    options: dict[str, Any] = Field(default_factory=dict)


class ClothingOptions(BaseModel):
    """Typed clothing options stored in a domain-neutral request."""

    model_config = ConfigDict(extra="forbid")

    product_title: str | None = Field(default=None, max_length=160)
    colors: list[str] = Field(
        default_factory=lambda: [ORIGINAL_GARMENT_COLOR],
        min_length=1,
        max_length=12,
    )
    candidates_per_color: int = Field(default=2, ge=1, le=8)
    max_retries: int = Field(default=1, ge=0, le=5)
    preserve_face: bool = True
    preserve_pose: bool = True
    preserve_background: bool = True

    @field_validator("product_title")
    @classmethod
    def normalize_product_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None


class WallpaperOptions(BaseModel):
    """Initial wallpaper options accepted by the extensible pipeline."""

    model_config = ConfigDict(extra="forbid")

    preserve_lighting: bool = True
    preserve_room_geometry: bool = True
    candidates_per_job: int = Field(default=1, ge=1, le=4)
    max_retries: int = Field(default=1, ge=0, le=3)
    pattern_scale: float = Field(default=0.18, ge=0.03, le=0.75)


class TryOnRequest(BaseModel):
    """A complete virtual try-on job request."""

    model_config = ConfigDict(extra="forbid")

    person_image: Path
    garment_image: Path
    product_title: str | None = Field(default=None, max_length=160)
    colors: list[str] = Field(
        default_factory=lambda: [ORIGINAL_GARMENT_COLOR],
        min_length=1,
        max_length=12,
    )
    candidates_per_color: int = Field(default=2, ge=1, le=8)
    max_retries: int = Field(default=1, ge=0, le=5)
    preserve_face: bool = True
    preserve_pose: bool = True
    preserve_background: bool = True

    @field_validator("product_title")
    @classmethod
    def normalize_product_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    def to_generation_request(self) -> GenerationRequest:
        """Map the legacy clothing contract to the shared input contract."""

        options = ClothingOptions(
            product_title=self.product_title,
            colors=self.colors,
            candidates_per_color=self.candidates_per_color,
            max_retries=self.max_retries,
            preserve_face=self.preserve_face,
            preserve_pose=self.preserve_pose,
            preserve_background=self.preserve_background,
        )
        return GenerationRequest(
            source_image=self.person_image,
            reference_image=self.garment_image,
            options=options.model_dump(),
        )
