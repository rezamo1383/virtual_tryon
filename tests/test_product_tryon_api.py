"""HTTP contract tests for production prepared-product Try-On."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from api import create_app
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.config import Settings
from app.core.exceptions import TryOnAPIError
from app.core.runtime import build_runtime
from app.models.request_models import GarmentCategory
from app.preprocessing.preprocessing_models import (
    BoundingBox,
    GarmentProcessingResult,
    ValidationResult,
)
from app.repositories.prepared_garments import (
    FilesystemPreparedGarmentRepository,
)
from app.services.pipeline import build_pipeline
from app.utils.json_utils import write_json

API_KEY = "postman-test-key"


class FailingGenerationClient(TryOnAPIClient):
    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        raise TryOnAPIError("provider secret diagnostic")


def _settings(tmp_path: Path) -> Settings:
    tenant_config = tmp_path / "tenants.json"
    tenant_config.write_text(
        json.dumps(
            {
                "tenants": [
                    {
                        "tenant_id": "fashion",
                        "pipeline": "clothing",
                        "analysis_provider": "mock",
                        "generation_provider": "mock",
                        "api_key_sha256": hashlib.sha256(API_KEY.encode()).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return Settings(
        _env_file=None,
        tenant_config_path=tenant_config,
        default_tenant_id="fashion",
        tenant_auth_required=True,
        analysis_provider="mock",
        tryon_provider="mock",
        min_image_width=64,
        min_image_height=64,
        output_directory=tmp_path / "outputs",
        prepared_garment_directory=tmp_path / "prepared",
        temp_directory=tmp_path / "temp",
        log_directory=tmp_path / "logs",
        local_preprocessing_enabled=False,
        background_removal_enabled=False,
        min_tryon_suitability_score=0,
        tryon_mode="fast",
    )


def _store_product(
    settings: Settings,
    garment: Path,
    product_id: str = "125",
) -> None:
    mask = garment.parent / f"{product_id}-mask.png"
    mask.write_bytes(garment.read_bytes())
    processing = GarmentProcessingResult(
        normalized_image_path=garment,
        transparent_image_path=garment,
        garment_mask_path=mask,
        bounding_box=BoundingBox(x=0, y=0, width=512, height=512),
        image_dimensions=(512, 512),
        alpha_coverage=0.5,
        symmetry_score=0.9,
        garment_suitability_score=1.0,
        validation=ValidationResult(accepted=True, score=1.0),
    )
    FilesystemPreparedGarmentRepository(settings.prepared_garment_directory).store(
        tenant_id="fashion",
        product_id=product_id,
        source_sha256="a" * 64,
        processing=processing,
    )


async def _close_runtime(application: Any) -> None:
    runtime = getattr(application.state, "runtime", None)
    if runtime is not None:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_successful_product_tryon_and_generated_image_retrieval(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    settings = _settings(tmp_path)
    _store_product(settings, garment)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with person.open("rb") as source:
            response = await client.post(
                "/api/v1/tryon/products/125",
                headers={"X-API-Key": API_KEY},
                files={"person_image": ("person.jpg", source, "image/jpeg")},
                data={
                    "category": GarmentCategory.UPPER_BODY.value,
                    "product_title": "Men's T-shirt",
                },
            )
        body = response.json()
        image_response = await client.get(
            body["output_image_url"],
            headers={"X-API-Key": API_KEY},
        )

    await _close_runtime(application)
    assert response.status_code == 200
    assert body == {
        "status": "success",
        "job_id": body["job_id"],
        "product_id": "125",
        "category": "upper_body",
        "mode": "fast",
        "output_image_url": (
            f"http://testserver/api/v1/results/{body['job_id']}/image"
        ),
        "elapsed_ms": body["elapsed_ms"],
    }
    assert isinstance(body["elapsed_ms"], int)
    assert "person_image" not in body
    assert "garment_image" not in body
    assert image_response.status_code == 200
    assert image_response.headers["content-type"].startswith("image/png")
    assert image_response.headers["x-content-type-options"] == "nosniff"
    assert image_response.content.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_product_preparation_endpoint_returns_no_filesystem_paths(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    _, garment = valid_images
    settings = _settings(tmp_path)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with garment.open("rb") as source:
            response = await client.post(
                "/api/v1/products/125/garment",
                headers={"X-API-Key": API_KEY},
                files={"garment_image": ("garment.png", source, "image/png")},
                data={"force": "false"},
            )
    await _close_runtime(application)
    assert response.status_code == 200
    assert set(response.json()) == {
        "status",
        "product_id",
        "cached",
        "prepared_at",
    }
    assert response.json()["product_id"] == "125"
    assert str(tmp_path) not in response.text


@pytest.mark.asyncio
async def test_invalid_category_returns_consistent_422(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    settings = _settings(tmp_path)
    _store_product(settings, garment)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with person.open("rb") as source:
            response = await client.post(
                "/api/v1/tryon/products/125",
                headers={"X-API-Key": API_KEY},
                files={"person_image": ("person.jpg", source, "image/jpeg")},
                data={"category": "arbitrary"},
            )
    await _close_runtime(application)
    assert response.status_code == 422
    assert response.json() == {
        "status": "error",
        "error": "validation_error",
        "message": "Invalid or missing field: category.",
    }


@pytest.mark.asyncio
async def test_malformed_multipart_returns_consistent_400(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/tryon/products/125",
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "multipart/form-data; boundary=invalid",
            },
            content=b"this is not valid multipart data",
        )
    await _close_runtime(application)
    assert response.status_code == 400
    assert response.json()["status"] == "error"
    assert response.json()["error"] == "malformed_request"


@pytest.mark.asyncio
async def test_nonexistent_product_returns_404(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    person, _ = valid_images
    settings = _settings(tmp_path)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with person.open("rb") as source:
            response = await client.post(
                "/api/v1/tryon/products/does-not-exist",
                headers={"X-API-Key": API_KEY},
                files={"person_image": ("person.jpg", source, "image/jpeg")},
                data={"category": "upper_body"},
            )
    await _close_runtime(application)
    assert response.status_code == 404
    assert response.json()["error"] == "product_not_found"


@pytest.mark.asyncio
async def test_missing_prepared_garment_artifact_returns_404(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    person, _ = valid_images
    settings = _settings(tmp_path)
    incomplete = settings.prepared_garment_directory / "fashion" / "125"
    incomplete.mkdir(parents=True)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with person.open("rb") as source:
            response = await client.post(
                "/api/v1/tryon/products/125",
                headers={"X-API-Key": API_KEY},
                files={"person_image": ("person.jpg", source, "image/jpeg")},
                data={"category": "upper_body"},
            )
    await _close_runtime(application)
    assert response.status_code == 404
    assert response.json()["error"] == "prepared_garment_not_found"


@pytest.mark.asyncio
async def test_provider_failure_returns_safe_502(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    settings = _settings(tmp_path)
    _store_product(settings, garment)
    application = create_app(settings)
    legacy = build_pipeline(
        settings,
        tryon_client=FailingGenerationClient(),
    )
    application.state.pipeline = legacy
    application.state.runtime = build_runtime(
        settings,
        legacy_pipeline=legacy,
    )
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with person.open("rb") as source:
            response = await client.post(
                "/api/v1/tryon/products/125",
                headers={"X-API-Key": API_KEY},
                files={"person_image": ("person.jpg", source, "image/jpeg")},
                data={"category": "upper_body"},
            )
    await _close_runtime(application)
    assert response.status_code == 502
    assert response.json() == {
        "status": "error",
        "error": "generation_provider_failed",
        "message": "The external image generation provider failed.",
    }
    assert "secret" not in response.text
    assert str(tmp_path) not in response.text


@pytest.mark.asyncio
async def test_invalid_nonexistent_and_traversal_job_ids_are_not_exposed(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    application = create_app(settings)
    malicious_job = "job_20260818_abcdef"
    job_directory = settings.output_directory / malicious_job
    job_directory.mkdir(parents=True)
    write_json(
        job_directory / "results.json",
        {
            "job_id": malicious_job,
            "tenant_id": "fashion",
            "status": "completed",
            "results": [{"output": "../../outside.png"}],
        },
    )
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private")
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        responses = [
            await client.get(
                "/api/v1/results/not-a-job/image",
                headers={"X-API-Key": API_KEY},
            ),
            await client.get(
                "/api/v1/results/job_20260818_000000/image",
                headers={"X-API-Key": API_KEY},
            ),
            await client.get(
                f"/api/v1/results/{malicious_job}/image",
                headers={"X-API-Key": API_KEY},
            ),
            await client.get(
                "/api/v1/results/%2e%2e/image",
                headers={"X-API-Key": API_KEY},
            ),
        ]
    await _close_runtime(application)
    assert all(response.status_code == 404 for response in responses)
    assert all(response.json()["status"] == "error" for response in responses)
    assert all(b"private" not in response.content for response in responses)
