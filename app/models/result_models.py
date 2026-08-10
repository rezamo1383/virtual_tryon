"""Pipeline result and candidate metadata models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.analysis_models import GarmentAnalysis, PersonAnalysis
from app.models.evaluation_models import OutputEvaluation
from app.preprocessing.preprocessing_models import PreprocessingResult


class CandidateResult(BaseModel):
    """One candidate and its independently recomputed evaluation."""

    model_config = ConfigDict(extra="forbid")

    color: str
    path: Path
    attempt: int = Field(ge=0)
    candidate_index: int = Field(ge=1)
    evaluation: OutputEvaluation | None = None


class ColorResult(BaseModel):
    """The best result retained for one requested color."""

    model_config = ConfigDict(extra="forbid")

    color: str
    output: Path
    score: float = Field(ge=0, le=1)
    accepted: bool
    retry_count: int = Field(ge=0)
    candidates_evaluated: int = Field(ge=1)
    problems: list[str] = Field(default_factory=list)


class TryOnJobResult(BaseModel):
    """Durable representation of a pipeline job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    tenant_id: str | None = None
    pipeline: Literal["clothing"] | None = None
    status: Literal["completed", "completed_with_failures", "rejected", "failed"]
    person_image: Path
    garment_image: Path
    person_analysis: PersonAnalysis | None = None
    garment_analysis: GarmentAnalysis | None = None
    preprocessing: PreprocessingResult | None = None
    results: list[ColorResult] = Field(default_factory=list)
    rejection_reason: str | None = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime
