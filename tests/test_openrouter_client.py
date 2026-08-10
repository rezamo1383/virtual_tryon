from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app.clients.openrouter_client import (
    OpenRouterTryOnClient,
    OpenRouterVisionClient,
)
from app.core.exceptions import OpenRouterAPIError
from app.core.config import Settings
from app.services.pipeline import build_pipeline


def _png_bytes(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_openrouter_vision_uses_chat_completions_and_safe_headers(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "person.png"
    image_path.write_bytes(_png_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["http-referer"] == "https://example.test"
        payload = json.loads(request.content)
        assert payload["model"] == "openai/test-vision"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"][0]["type"] == "text"
        assert payload["messages"][0]["content"][1]["type"] == "image_url"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "person_count": 1,
                                    "pose": "front",
                                    "body_visibility": "full_body",
                                    "arms_position": "relaxed",
                                    "image_quality": "good",
                                    "background_complexity": "simple",
                                    "suitable_for_tryon": True,
                                    "rejection_reason": None,
                                }
                            )
                        }
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterVisionClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        model="openai/test-vision",
        http_referer="https://example.test",
        app_name="Test App",
        http_client=http_client,
    )
    result = await client.analyze_person(image_path)
    assert result.suitable_for_tryon is True
    await http_client.aclose()


@pytest.mark.asyncio
async def test_openrouter_image_client_sends_two_references(
    tmp_path: Path,
) -> None:
    person = tmp_path / "person.png"
    garment = tmp_path / "garment.png"
    person.write_bytes(_png_bytes((100, 110, 120)))
    garment.write_bytes(_png_bytes((180, 20, 20)))
    generated = _png_bytes((1, 2, 3))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/images"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "openai/test-image"
        assert payload["n"] == 1
        assert len(payload["input_references"]) == 2
        assert all(
            item["image_url"]["url"].startswith("data:image/png;base64,")
            for item in payload["input_references"]
        )
        return httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": base64.b64encode(generated).decode("ascii")}
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterTryOnClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        model="openai/test-image",
        quality="high",
        http_client=http_client,
    )
    outputs = await client.generate(
        person,
        garment,
        "upper_body",
        {"requested_color": "#C62828", "strict_identity_preservation": True},
    )
    assert len(outputs) == 1
    assert Image.open(io.BytesIO(outputs[0])).size == (64, 64)
    await http_client.aclose()


def test_openrouter_key_is_required() -> None:
    with pytest.raises(OpenRouterAPIError, match="OPENROUTER_API_KEY"):
        OpenRouterVisionClient(
            base_url="https://openrouter.ai/api/v1",
            api_key="",
            model="openai/test-vision",
        )


@pytest.mark.asyncio
async def test_pipeline_selects_openrouter_clients(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        analysis_provider="openrouter",
        tryon_provider="openrouter",
        openrouter_api_key="test-key",
        openrouter_vision_model="openai/test-vision",
        openrouter_image_model="openai/test-image",
        output_directory=tmp_path / "outputs",
        temp_directory=tmp_path / "temp",
        log_directory=tmp_path / "logs",
    )
    pipeline = build_pipeline(settings)
    assert isinstance(pipeline.qwen_client, OpenRouterVisionClient)
    assert isinstance(pipeline.tryon_client, OpenRouterTryOnClient)
    await pipeline.aclose()
