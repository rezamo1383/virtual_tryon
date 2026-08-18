"""Regression tests for prepared garments and Fast/Quality policies."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.clients.mock_qwen_client import MockQwenClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.config import Settings
from app.models.request_models import PreparedTryOnRequest
from app.preprocessing.preprocessing_models import (
    BoundingBox,
    GarmentProcessingResult,
    ValidationResult,
)
from app.repositories.prepared_garments import (
    FilesystemPreparedGarmentRepository,
)
from app.services.garment_preparation import GarmentPreparationService
from app.services.pipeline import VirtualTryOnPipeline


class CountingGarmentPreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def preprocess_garment(
        self,
        garment_image: Path,
        job_directory: Path,
    ) -> GarmentProcessingResult:
        self.calls += 1
        root = job_directory / "preprocessing" / "garment"
        root.mkdir(parents=True)
        normalized = root / "normalized.png"
        transparent = root / "transparent_cropped.png"
        mask = root / "garment_mask.png"
        shutil.copy2(garment_image, normalized)
        shutil.copy2(garment_image, transparent)
        shutil.copy2(garment_image, mask)
        return GarmentProcessingResult(
            normalized_image_path=normalized,
            transparent_image_path=transparent,
            garment_mask_path=mask,
            bounding_box=BoundingBox(x=0, y=0, width=512, height=512),
            image_dimensions=(512, 512),
            alpha_coverage=0.5,
            symmetry_score=0.9,
            garment_suitability_score=1.0,
            validation=ValidationResult(accepted=True, score=1.0),
        )


class CountingTryOnClient(TryOnAPIClient):
    supports_text_prompt = True

    def __init__(self) -> None:
        self.calls = 0
        self.categories: list[str] = []
        self.candidate_counts: list[int] = []

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        self.calls += 1
        self.categories.append(category)
        count = int(options.get("candidate_count", 1))
        self.candidate_counts.append(count)
        outputs: list[bytes] = []
        for index in range(count):
            buffer = io.BytesIO()
            Image.new("RGB", (64, 64), (index, 20, 30)).save(
                buffer,
                format="PNG",
            )
            outputs.append(buffer.getvalue())
        return outputs


class CountingQwenClient(MockQwenClient):
    def __init__(self) -> None:
        self.garment_analysis_calls = 0
        self.evaluation_calls = 0

    async def analyze_garment(self, image_path: Path) -> Any:
        self.garment_analysis_calls += 1
        return await super().analyze_garment(image_path)

    async def evaluate_output(self, *args: Any, **kwargs: Any) -> Any:
        self.evaluation_calls += 1
        return await super().evaluate_output(*args, **kwargs)


def _settings(tmp_path: Path, mode: str = "fast") -> Settings:
    return Settings(
        _env_file=None,
        min_image_width=64,
        min_image_height=64,
        output_directory=tmp_path / "outputs",
        prepared_garment_directory=tmp_path / "prepared",
        temp_directory=tmp_path / "temp",
        log_directory=tmp_path / "logs",
        local_preprocessing_enabled=False,
        background_removal_enabled=False,
        min_tryon_suitability_score=0,
        tryon_mode=mode,
        candidates_per_color=2,
        max_generation_retries=0,
    )


@pytest.mark.asyncio
async def test_garment_preparation_is_independent_stored_and_idempotent(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    _, garment = valid_images
    settings = _settings(tmp_path)
    repository = FilesystemPreparedGarmentRepository(
        settings.prepared_garment_directory
    )
    preprocessor = CountingGarmentPreprocessor()
    service = GarmentPreparationService(
        settings=settings,
        repository=repository,
        preprocessor=preprocessor,  # type: ignore[arg-type]
    )

    first = await service.prepare(
        tenant_id="fashion",
        product_id="125",
        garment_image=garment,
    )
    second = await service.prepare(
        tenant_id="fashion",
        product_id="125",
        garment_image=garment,
    )
    forced = await service.prepare(
        tenant_id="fashion",
        product_id="125",
        garment_image=garment,
        force=True,
    )

    assert first.cached is False
    assert second.cached is True
    assert forced.cached is False
    assert preprocessor.calls == 2
    loaded = repository.get("fashion", "125")
    assert loaded.processing.normalized_image_path.is_file()
    metadata = json.loads(
        (
            settings.prepared_garment_directory / "fashion" / "125" / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["processing"]["normalized_image_path"] == "normalized.png"


async def _prepared_pipeline_result(
    settings: Settings,
    person: Path,
    garment: Path,
) -> tuple[Any, CountingTryOnClient, CountingQwenClient]:
    preprocessor = CountingGarmentPreprocessor()
    repository = FilesystemPreparedGarmentRepository(
        settings.prepared_garment_directory
    )
    service = GarmentPreparationService(
        settings=settings,
        repository=repository,
        preprocessor=preprocessor,  # type: ignore[arg-type]
    )
    await service.prepare(
        tenant_id="fashion",
        product_id="125",
        garment_image=garment,
    )
    qwen = CountingQwenClient()
    generator = CountingTryOnClient()
    pipeline = VirtualTryOnPipeline(
        settings=settings,
        qwen_client=qwen,
        tryon_client=generator,
        prepared_garment_repository=repository,
    )
    result = await pipeline.run_prepared(
        PreparedTryOnRequest(
            person_image=person,
            product_id="125",
            category="lower_body",
            tenant_id="fashion",
        )
    )
    await pipeline.aclose()
    return result, generator, qwen


@pytest.mark.asyncio
async def test_fast_mode_uses_trusted_category_and_one_unscored_generation(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    result, generator, qwen = await _prepared_pipeline_result(
        _settings(tmp_path, "fast"),
        person,
        garment,
    )

    assert result.status == "completed"
    assert result.mode.value == "fast"
    assert result.category.value == "lower_body"
    assert generator.calls == 1
    assert generator.candidate_counts == [1]
    assert generator.categories == ["lower_body"]
    assert qwen.garment_analysis_calls == 0
    assert qwen.evaluation_calls == 0
    assert result.results[0].retry_count == 0
    assert result.results[0].candidates_evaluated == 1


@pytest.mark.asyncio
async def test_quality_mode_preserves_multiple_candidates_and_evaluation(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    result, generator, qwen = await _prepared_pipeline_result(
        _settings(tmp_path, "quality"),
        person,
        garment,
    )

    assert result.mode.value == "quality"
    assert generator.calls == 1
    assert generator.candidate_counts == [2]
    assert qwen.garment_analysis_calls == 0
    assert qwen.evaluation_calls == 2
    assert result.results[0].candidates_evaluated == 2
