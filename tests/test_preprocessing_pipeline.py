from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from api import create_app
from app.clients.mock_qwen_client import MockQwenClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.config import Settings
from app.models.request_models import TryOnRequest
from app.preprocessing.preprocessing_models import (
    BoundingBox,
    GarmentProcessingResult,
    PersonPreprocessingResult,
    PoseResult,
    PreprocessingResult,
    ValidationResult,
)
from app.services.pipeline import VirtualTryOnPipeline, build_pipeline


class GenerationMustNotRunClient(TryOnAPIClient):
    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, object],
    ) -> list[bytes]:
        raise AssertionError("External generation must not be called")


class RejectingPreprocessor:
    def preprocess(
        self,
        person_image_path: Path,
        garment_image_path: Path,
        job_directory: Path,
        *,
        human_parsing_enabled: bool | None = None,
    ) -> PreprocessingResult:
        artifact = job_directory / "preprocessing" / "placeholder.png"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"x")
        rejected = ValidationResult(
            accepted=False,
            score=0.2,
            rejection_reasons=["Both shoulders must be clearly visible."],
        )
        accepted = ValidationResult(accepted=True, score=0.9)
        return PreprocessingResult(
            person=PersonPreprocessingResult(
                normalized_image_path=artifact,
                transparent_image_path=artifact,
                foreground_mask_path=artifact,
                replace_mask_path=artifact,
                preserve_mask_path=artifact,
                pose=PoseResult(),
                validation=rejected,
            ),
            garment=GarmentProcessingResult(
                normalized_image_path=artifact,
                transparent_image_path=artifact,
                garment_mask_path=artifact,
                bounding_box=BoundingBox(x=0, y=0, width=1, height=1),
                image_dimensions=(1, 1),
                alpha_coverage=1.0,
                symmetry_score=1.0,
                garment_suitability_score=0.9,
                validation=accepted,
            ),
            device="cpu",
            degraded_mode=True,
            processing_time_ms=1,
        )


@pytest.mark.asyncio
async def test_api_generation_is_not_called_when_preprocessing_rejects(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    settings = Settings(
        _env_file=None,
        min_image_width=64,
        min_image_height=64,
        output_directory=tmp_path / "outputs",
        temp_directory=tmp_path / "temp",
        log_directory=tmp_path / "logs",
        local_preprocessing_enabled=True,
        preprocessing_device="cpu",
        delete_temp_files=True,
    )
    pipeline = VirtualTryOnPipeline(
        settings=settings,
        qwen_client=MockQwenClient(),
        tryon_client=GenerationMustNotRunClient(),
        local_preprocessor=RejectingPreprocessor(),  # type: ignore[arg-type]
    )
    result = await pipeline.run(
        TryOnRequest(
            person_image=person,
            garment_image=garment,
            colors=["black"],
            candidates_per_color=1,
            max_retries=0,
        )
    )
    assert result.status == "rejected"
    assert result.preprocessing is not None
    assert "shoulders" in (result.rejection_reason or "").lower()


@pytest.mark.asyncio
async def test_preprocess_endpoint_returns_only_relative_artifact_names(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    settings = Settings(
        _env_file=None,
        min_image_width=64,
        min_image_height=64,
        output_directory=tmp_path / "outputs",
        temp_directory=tmp_path / "temp",
        log_directory=tmp_path / "logs",
        analysis_provider="mock",
        tryon_provider="mock",
        local_preprocessing_enabled=False,
        preprocessing_device="cpu",
    )
    application = create_app(settings)
    pipeline = build_pipeline(settings)
    pipeline.local_preprocessor = RejectingPreprocessor()  # type: ignore[assignment]
    application.state.pipeline = pipeline
    person, garment = valid_images
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with person.open("rb") as person_stream, garment.open("rb") as garment_stream:
            response = await client.post(
                "/v1/preprocess",
                files={
                    "person_image": ("person.jpg", person_stream, "image/jpeg"),
                    "garment_image": (
                        "garment.png",
                        garment_stream,
                        "image/png",
                    ),
                },
                data={"human_parsing_enabled": "false"},
            )
    await pipeline.aclose()
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert all(
        value is None or not Path(value).is_absolute()
        for value in data["artifacts"].values()
    )
