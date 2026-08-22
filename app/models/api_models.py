"""Stable public HTTP response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from app.models.request_models import GarmentCategory, TryOnMode


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    error: str
    message: str


class TryOnJobResponse(BaseModel):
    """Minimal public acknowledgement for a completed Try-On request."""

    model_config = ConfigDict(extra="forbid")

    job_id: str


class ProductPreparationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    product_id: str
    cached: bool
    prepared_at: datetime


class ProductTryOnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    job_id: str
    product_id: str
    category: GarmentCategory
    mode: TryOnMode
    output_image_url: AnyHttpUrl
    elapsed_ms: int = Field(ge=0)
