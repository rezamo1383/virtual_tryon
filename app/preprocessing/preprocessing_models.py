"""Typed runtime and durable results for local image preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class Keypoint(BaseModel):
    """One normalized and pixel-space body landmark."""

    name: str
    x: float
    y: float
    z: float | None = None
    visibility: float = Field(ge=0, le=1)
    pixel_x: int
    pixel_y: int


class BoundingBox(BaseModel):
    """Integer image-space bounding box."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)


class PoseResult(BaseModel):
    """Pose geometry and explainable try-on heuristics."""

    detected_person_count: int = Field(default=0, ge=0)
    keypoints: dict[str, Keypoint] = Field(default_factory=dict)
    shoulder_width: float = Field(default=0, ge=0)
    torso_length: float = Field(default=0, ge=0)
    body_center: tuple[float, float] | None = None
    pose_confidence: float = Field(default=0, ge=0, le=1)
    person_orientation: Literal[
        "frontal", "slightly_turned", "side", "unknown"
    ] = "unknown"
    arms_position: Literal[
        "down", "bent", "crossed", "raised", "partially_hidden", "unknown"
    ] = "unknown"
    upper_body_visible: bool = False
    pose_suitable_for_tryon: bool = False
    warnings: list[str] = Field(default_factory=list)


class PersonPresenceResult(BaseModel):
    """Result of the mandatory local human-presence check."""

    detected: bool
    detected_person_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=0, ge=0, le=1)
    model_name: str


class BackgroundRemovalResult(BaseModel):
    """In-memory transparent cutout and foreground mask."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    mask: Image.Image
    confidence: float | None = Field(default=None, ge=0, le=1)
    model_name: str
    fallback_used: bool = False


class HumanParsingResult(BaseModel):
    """In-memory standardized semantic maps and derived masks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    class_map: np.ndarray
    visualization: Image.Image
    upper_clothes_mask: Image.Image
    arms_mask: Image.Image
    hands_mask: Image.Image
    face_hair_protection_mask: Image.Image
    body_torso_mask: Image.Image
    model_name: str
    degraded_mode: bool = False
    warnings: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Scored validation with actionable diagnostics."""

    accepted: bool
    score: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class PersonPreprocessingResult(BaseModel):
    """Durable person artifacts and local validation."""

    normalized_image_path: Path
    transparent_image_path: Path
    foreground_mask_path: Path
    replace_mask_path: Path
    preserve_mask_path: Path
    pose_debug_path: Path | None = None
    parsing_debug_path: Path | None = None
    pose: PoseResult
    validation: ValidationResult


class GarmentProcessingResult(BaseModel):
    """Durable garment artifacts and geometric metrics."""

    normalized_image_path: Path
    transparent_image_path: Path
    garment_mask_path: Path
    bounding_box: BoundingBox
    dominant_color: str | None = None
    image_dimensions: tuple[int, int]
    alpha_coverage: float = Field(ge=0, le=1)
    symmetry_score: float = Field(ge=0, le=1)
    cropped_edges: list[str] = Field(default_factory=list)
    garment_suitability_score: float = Field(ge=0, le=1)
    foreground_center: tuple[float, float] | None = None
    validation: ValidationResult


class PreprocessingResult(BaseModel):
    """Complete durable local-preprocessing result."""

    person: PersonPreprocessingResult
    garment: GarmentProcessingResult
    device: Literal["cpu", "cuda"]
    degraded_mode: bool
    processing_time_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
