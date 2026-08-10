"""Deterministic offline Qwen substitute."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel

from app.clients.qwen_client import QwenClient
from app.models.analysis_models import GarmentAnalysis, PersonAnalysis
from app.models.evaluation_models import OutputEvaluation

ModelT = TypeVar("ModelT", bound=BaseModel)


class MockQwenClient(QwenClient):
    """Return schema-valid, deterministic analyses for local tests."""

    async def analyze_person(self, image_path: Path) -> PersonAnalysis:
        return PersonAnalysis(
            person_count=1,
            pose="front",
            body_visibility="full_body",
            arms_position="relaxed",
            image_quality="good",
            background_complexity="medium",
            suitable_for_tryon=True,
            rejection_reason=None,
        )

    async def analyze_garment(self, image_path: Path) -> GarmentAnalysis:
        return GarmentAnalysis(
            garment_category="upper_body",
            garment_type=image_path.stem or "garment",
            sleeve_type="long",
            base_color=None,
            has_logo=False,
            has_pattern=False,
            recommended_tryon_category="upper_body",
        )

    async def evaluate_output(
        self,
        person_image: Path,
        garment_image: Path,
        output_image: Path,
        requested_color: str,
        product_title: str | None = None,
    ) -> OutputEvaluation:
        digest = hashlib.sha256(
            f"{output_image.name}:{requested_color}".encode("utf-8")
        ).digest()
        variation = (digest[0] / 255.0) * 0.06
        base = 0.86 + variation
        return OutputEvaluation(
            identity_preservation=min(0.98, base + 0.04),
            garment_similarity=min(0.96, base),
            color_accuracy=min(0.97, base + 0.02),
            body_integrity=min(0.98, base + 0.03),
            background_preservation=min(0.99, base + 0.05),
            overall_score=base,
            accepted=True,
            problems=[],
            retry_recommendation=None,
        )

    async def analyze_structured(
        self,
        prompt: str,
        images: list[Path],
        model_type: type[ModelT],
    ) -> ModelT:
        """Return deterministic wallpaper schemas for offline execution."""

        from app.models.wallpaper_models import (
            WallAnalysisResult,
            WallpaperOutputEvaluation,
        )

        if model_type is WallAnalysisResult:
            value = WallAnalysisResult(
                wall_detected=True,
                confidence=0.90,
                wall_polygon=[
                    {"x": 0.05, "y": 0.05},
                    {"x": 0.95, "y": 0.05},
                    {"x": 0.92, "y": 0.78},
                    {"x": 0.08, "y": 0.78},
                ],
                wall_count=1,
                occlusions=[],
                lighting="mixed",
                warnings=["Mock wall analysis used."],
            )
            return cast(ModelT, value)
        if model_type is WallpaperOutputEvaluation:
            value = WallpaperOutputEvaluation(
                wall_coverage=0.91,
                pattern_fidelity=0.92,
                perspective_accuracy=0.90,
                lighting_preservation=0.93,
                scene_integrity=0.96,
                overall_score=0.93,
                accepted=True,
                problems=[],
                retry_recommendation=None,
            )
            return cast(ModelT, value)
        return await super().analyze_structured(
            prompt,
            images,
            model_type,
        )
