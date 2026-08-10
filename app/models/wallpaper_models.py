"""Typed contracts for wallpaper visualization."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NormalizedPoint(BaseModel):
    """One normalized image coordinate."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class WallAnalysisResult(BaseModel):
    """Vision analysis of the largest wallpaper-suitable wall."""

    model_config = ConfigDict(extra="forbid")

    wall_detected: bool
    confidence: float = Field(ge=0, le=1)
    wall_polygon: list[NormalizedPoint] = Field(default_factory=list, max_length=4)
    wall_count: int = Field(default=0, ge=0)
    occlusions: list[str] = Field(default_factory=list)
    lighting: str = "unknown"
    warnings: list[str] = Field(default_factory=list)


class WallSegmentationResult(BaseModel):
    """Wall mask artifact and geometry."""

    mask_path: Path
    debug_path: Path | None = None
    coverage: float = Field(ge=0, le=1)
    polygon_pixels: list[tuple[int, int]] = Field(min_length=4, max_length=4)
    method: Literal["semantic", "polygon"] = "polygon"
    wall_count: int = Field(default=1, ge=0)
    mean_confidence: float | None = Field(default=None, ge=0, le=1)


class PerspectiveEstimationResult(BaseModel):
    """Perspective transform from a flat texture to the selected wall."""

    transform: list[list[float]]
    destination_quad: list[tuple[int, int]] = Field(min_length=4, max_length=4)
    canvas_size: tuple[int, int]


class TextureRepetitionResult(BaseModel):
    """Perspective-warped wallpaper texture preview."""

    texture_path: Path
    repeat_x: int = Field(ge=1)
    repeat_y: int = Field(ge=1)


class WallpaperOutputEvaluation(BaseModel):
    """Independent quality evaluation of one wallpaper candidate."""

    model_config = ConfigDict(extra="forbid")

    wall_coverage: float = Field(ge=0, le=1)
    pattern_fidelity: float = Field(ge=0, le=1)
    perspective_accuracy: float = Field(ge=0, le=1)
    lighting_preservation: float = Field(ge=0, le=1)
    scene_integrity: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    accepted: bool
    problems: list[str] = Field(default_factory=list)
    retry_recommendation: str | None = None


class WallpaperCandidateResult(BaseModel):
    """One persisted and evaluated wallpaper candidate."""

    path: Path
    attempt: int = Field(ge=0)
    candidate_index: int = Field(ge=1)
    evaluation: WallpaperOutputEvaluation | None = None


class LightingPreservationResult(BaseModel):
    """Lighting-preserved final artifact."""

    image_path: Path


class WallpaperJobResult(BaseModel):
    """Durable result returned by the wallpaper pipeline."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    tenant_id: str
    pipeline: Literal["wallpaper"] = "wallpaper"
    status: Literal[
        "completed",
        "completed_with_failures",
        "rejected",
        "failed",
    ]
    source_image: Path
    reference_image: Path
    completed_stages: list[str] = Field(default_factory=list)
    analysis: WallAnalysisResult | None = None
    segmentation: WallSegmentationResult | None = None
    perspective: PerspectiveEstimationResult | None = None
    texture_preview: Path | None = None
    output: Path | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    accepted: bool | None = None
    retry_count: int = Field(default=0, ge=0)
    candidates_evaluated: int = Field(default=0, ge=0)
    problems: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime
