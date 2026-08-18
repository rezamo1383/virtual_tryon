"""Public result model for one prepared product garment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.preprocessing.preprocessing_models import ValidationResult


class GarmentPreparationResult(BaseModel):
    """Result returned by the idempotent product preparation service."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    product_id: str
    cached: bool
    source_sha256: str
    prepared_at: datetime
    normalized_image_path: Path
    transparent_image_path: Path
    garment_mask_path: Path
    validation: ValidationResult
