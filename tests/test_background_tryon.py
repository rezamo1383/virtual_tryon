from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from api import create_app
from app.clients.mock_qwen_client import MockQwenClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.config import Settings
from app.core.exceptions import TryOnAPIError
from app.core.runtime import build_runtime
from app.services.pipeline import build_pipeline


def _png_bytes(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return buffer.getvalue()


class BlockingMultiClient(TryOnAPIClient):
    supports_multi_reference = True
    supports_text_prompt = True

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.candidate_counts: list[int] = []

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        return await self.generate_multi(
            person_image,
            [garment_image],
            [category],
            category,
            options,
        )

    async def generate_multi(
        self,
        person_image: Path,
        garment_images: list[Path],
        garment_types: list[str],
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        self.calls += 1
        self.candidate_counts.append(int(options["candidate_count"]))
        self.started.set()
        await self.release.wait()
        return [_png_bytes((1, 2, 3))]


class FailingMultiClient(BlockingMultiClient):
    async def generate_multi(
        self,
        person_image: Path,
        garment_images: list[Path],
        garment_types: list[str],
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        self.calls += 1
        self.candidate_counts.append(int(options["candidate_count"]))
        self.started.set()
        raise TryOnAPIError("provider secret diagnostic")


def _application(settings: Settings, client: TryOnAPIClient):
    application = create_app(settings)
    pipeline = build_pipeline(
        settings,
        qwen_client=MockQwenClient(),
        tryon_client=client,
    )
    runtime = build_runtime(settings, legacy_pipeline=pipeline)
    application.state.pipeline = pipeline
    application.state.runtime = runtime
    return application


async def _post_multi(
    client: httpx.AsyncClient,
    person: Path,
    garment: Path,
) -> httpx.Response:
    return await client.post(
        "/api/v1/tryon",
        files=[
            ("person_image", ("person.jpg", person.read_bytes(), "image/jpeg")),
            (
                "garment_images",
                ("shirt.png", garment.read_bytes(), "image/png"),
            ),
            (
                "garment_images",
                ("pants.png", garment.read_bytes(), "image/png"),
            ),
        ],
        data={
            "garment_types": json.dumps(["T-shirt", "Pants"]),
            "candidates_per_color": "8",
            "max_retries": "5",
        },
    )


@pytest.mark.asyncio
async def test_tryon_returns_202_before_generation_and_cleans_inputs(
    settings: Settings,
    valid_images: tuple[Path, Path],
) -> None:
    provider = BlockingMultiClient()
    application = _application(settings, provider)
    transport = httpx.ASGITransport(app=application)
    person, garment = valid_images

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        response = await _post_multi(http_client, person, garment)
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        await asyncio.wait_for(provider.started.wait(), timeout=2)
        assert provider.release.is_set() is False
        input_directory = settings.temp_directory / "background_uploads" / job_id
        assert input_directory.is_dir()
        status_response = await http_client.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "running"

        provider.release.set()
        await application.state.background_tryon_jobs.wait_all()
        assert provider.calls == 1
        assert provider.candidate_counts == [1]
        assert not input_directory.exists()

        request_data = json.loads(
            (settings.output_directory / job_id / "request.json").read_text("utf-8")
        )
        assert request_data["candidates_per_color"] == 1
        assert request_data["max_retries"] == 0

        completed = await http_client.get(f"/api/v1/jobs/{job_id}")
        assert completed.json()["status"] == "completed"
        image = await http_client.get(f"/api/v1/results/{job_id}/image")
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/png")

    await application.state.runtime.aclose()


@pytest.mark.asyncio
async def test_background_provider_failure_marks_job_failed(
    settings: Settings,
    valid_images: tuple[Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    provider = FailingMultiClient()
    application = _application(settings, provider)
    transport = httpx.ASGITransport(app=application)
    person, garment = valid_images

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        response = await _post_multi(http_client, person, garment)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        await application.state.background_tryon_jobs.wait_all()

        status_response = await http_client.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] == "failed"
        assert "provider secret diagnostic" not in json.dumps(status_data)
        assert "provider secret diagnostic" not in caplog.text
        assert not (
            settings.temp_directory / "background_uploads" / job_id
        ).exists()

    await application.state.runtime.aclose()
