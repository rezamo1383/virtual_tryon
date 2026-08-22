from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from app.clients.mock_qwen_client import MockQwenClient
from app.clients.mock_tryon_client import MockTryOnClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.config import Settings
from app.core.exceptions import TryOnAPIError
from app.models.analysis_models import PersonAnalysis
from app.models.request_models import TryOnRequest
from app.models.result_models import TryOnJobResult
from app.services.pipeline import VirtualTryOnPipeline


class FailingTryOnClient(TryOnAPIClient):
    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, object],
    ) -> list[bytes]:
        raise TryOnAPIError("simulated provider failure")


class UnsuitableMockQwenClient(MockQwenClient):
    async def analyze_person(self, image_path: Path) -> PersonAnalysis:
        return PersonAnalysis(
            person_count=1,
            pose="leaning",
            body_visibility="partially occluded",
            arms_position="one arm raised, one arm relaxed",
            image_quality="medium",
            background_complexity="high",
            suitable_for_tryon=False,
            rejection_reason="Body is partially occluded and background is complex.",
        )


class PersonAnalysisMustNotRunClient(MockQwenClient):
    async def analyze_person(self, image_path: Path) -> PersonAnalysis:
        raise AssertionError("Person analysis must be disabled")


class EvaluationMustNotRunClient(MockQwenClient):
    async def evaluate_output(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Fast mode must not evaluate generated images")


class CostCountingTryOnClient(MockTryOnClient):
    def __init__(self) -> None:
        self.calls = 0
        self.candidate_counts: list[int] = []

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        self.calls += 1
        self.candidate_counts.append(int(options["candidate_count"]))
        return await super().generate(
            person_image,
            garment_image,
            category,
            options,
        )


@pytest.mark.asyncio
async def test_fast_mode_forces_one_call_without_evaluation_or_retry(
    settings: Settings,
    valid_images: tuple[Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    person, garment = valid_images
    generator = CostCountingTryOnClient()
    pipeline = VirtualTryOnPipeline(
        settings=settings.model_copy(update={"tryon_mode": "fast"}),
        qwen_client=EvaluationMustNotRunClient(),
        tryon_client=generator,
    )

    result = await pipeline.run(
        TryOnRequest(
            person_image=person,
            garment_image=garment,
            candidates_per_color=8,
            max_retries=5,
        )
    )

    assert generator.calls == 1
    assert generator.candidate_counts == [1]
    assert result.results[0].candidates_evaluated == 1
    assert result.results[0].retry_count == 0
    summary = next(
        record
        for record in caplog.records
        if record.message == "tryon_generation_summary"
    )
    assert summary.effective_candidate_count == 1
    assert summary.effective_retry_count == 0
    assert summary.provider_generation_call_count == 1


@pytest.mark.asyncio
async def test_complete_mock_pipeline(
    settings: Settings, valid_images: tuple[Path, Path]
) -> None:
    person, garment = valid_images
    pipeline = VirtualTryOnPipeline(
        settings=settings,
        qwen_client=PersonAnalysisMustNotRunClient(),
        tryon_client=MockTryOnClient(),
    )
    result = await pipeline.run(
        TryOnRequest(
            person_image=person,
            garment_image=garment,
            colors=["red", "#1565C0", "black"],
            candidates_per_color=2,
            max_retries=1,
        )
    )
    job_directory = settings.output_directory / result.job_id
    assert job_directory.is_dir()
    for color in ("red", "blue", "black"):
        assert (job_directory / "variants" / f"{color}.png").is_file()
        assert (job_directory / "candidates" / color / "candidate_01.png").is_file()
        assert (job_directory / "final" / f"{color}.png").is_file()
    data = json.loads((job_directory / "results.json").read_text("utf-8"))
    validated = TryOnJobResult.model_validate(data)
    assert validated.status == "completed"
    assert validated.person_analysis is None
    assert validated.preprocessing is None
    assert not (job_directory / "person_analysis.json").exists()
    assert len(validated.results) == 3
    assert not (settings.temp_directory / result.job_id).exists()


@pytest.mark.asyncio
async def test_original_color_skips_recoloring_and_keeps_product_title(
    settings: Settings,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    pipeline = VirtualTryOnPipeline(
        settings=settings,
        qwen_client=MockQwenClient(),
        tryon_client=MockTryOnClient(),
    )

    def unexpected_recolor(*args: object, **kwargs: object) -> None:
        raise AssertionError("Original-color mode must not recolor the reference.")

    pipeline.colorizer.create_variant = unexpected_recolor  # type: ignore[method-assign]
    result = await pipeline.run(
        TryOnRequest(
            person_image=person,
            garment_image=garment,
            product_title="  تي شرت   مردانه  ",
            candidates_per_color=1,
            max_retries=0,
        )
    )

    job_directory = settings.output_directory / result.job_id
    request_data = json.loads(
        (job_directory / "request.json").read_text(encoding="utf-8")
    )
    assert request_data["colors"] == ["original"]
    assert request_data["product_title"] == "تي شرت مردانه"
    assert (job_directory / "variants" / "original.png").is_file()
    assert (job_directory / "final" / "original.png").is_file()
    assert result.results[0].color == "original"


@pytest.mark.asyncio
async def test_unsuitable_person_continues_when_rejection_is_disabled(
    settings: Settings, valid_images: tuple[Path, Path]
) -> None:
    person, garment = valid_images
    test_settings = settings.model_copy(
        update={
            "person_analysis_enabled": True,
            "reject_unsuitable_person_images": False,
        }
    )
    pipeline = VirtualTryOnPipeline(
        settings=test_settings,
        qwen_client=UnsuitableMockQwenClient(),
        tryon_client=MockTryOnClient(),
    )

    result = await pipeline.run(
        TryOnRequest(
            person_image=person,
            garment_image=garment,
            colors=["red"],
            candidates_per_color=1,
            max_retries=0,
        )
    )

    assert result.status == "completed"
    assert result.person_analysis is not None
    assert result.person_analysis.suitable_for_tryon is False
    assert result.person_analysis.rejection_reason is not None
    assert (test_settings.output_directory / result.job_id / "final" / "red.png").is_file()


@pytest.mark.asyncio
async def test_failed_pipeline_writes_durable_result(
    settings: Settings, valid_images: tuple[Path, Path]
) -> None:
    person, garment = valid_images
    pipeline = VirtualTryOnPipeline(
        settings=settings,
        qwen_client=MockQwenClient(),
        tryon_client=FailingTryOnClient(),
    )
    before = set(settings.output_directory.glob("job_*"))
    with pytest.raises(TryOnAPIError, match="simulated provider failure"):
        await pipeline.run(
            TryOnRequest(
                person_image=person,
                garment_image=garment,
                colors=["red"],
                candidates_per_color=1,
                max_retries=0,
            )
        )
    created = set(settings.output_directory.glob("job_*")) - before
    assert len(created) == 1
    result_data = json.loads(
        (created.pop() / "results.json").read_text(encoding="utf-8")
    )
    assert result_data["status"] == "failed"
    assert result_data["error"] == "simulated provider failure"
