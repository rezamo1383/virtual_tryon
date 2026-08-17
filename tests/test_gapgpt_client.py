from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

import app.clients.gapgpt_client as gapgpt_module
from app.clients.gapgpt_client import (
    GapGPTTryOnClient,
    GapGPTVisionClient,
    _network_error_detail,
)
from app.clients.qwen_client import EVALUATION_PROMPT, GARMENT_PROMPT
from app.clients.tryon_api_client import build_tryon_prompt
from app.core.config import Settings
from app.core.exceptions import GapGPTAPIError, TryOnAPIError
from app.prompts.wallpaper import WallpaperPromptBuilder
from app.services.pipeline import build_pipeline


def _png_bytes(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_prompts_are_compact_and_keep_quality_constraints() -> None:
    prompt = build_tryon_prompt(
        "upper_body",
        {
            "requested_color": "original",
            "product_title": "تي شرت مردانه",
            "strict_identity_preservation": True,
        },
    )

    assert len(GARMENT_PROMPT) <= 200
    assert len(EVALUATION_PROMPT.format(requested_color="#C62828")) <= 260
    assert len(prompt) <= 600
    for requirement in ("identity", "pose", "hands", "garment", "color"):
        assert requirement in prompt
    assert "تي شرت مردانه" in prompt
    assert "other garments" in prompt
    assert "do not recolor" in prompt


@pytest.mark.asyncio
async def test_gapgpt_reuses_privacy_safe_png_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(_png_bytes())
    calls = 0
    original = gapgpt_module.open_image_safe

    def counted_open(path: Path) -> Image.Image:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(gapgpt_module, "open_image_safe", counted_open)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None))
    client = GapGPTTryOnClient(
        base_url="https://api.gapgpt.app/v1",
        api_key="gap-key",
        http_client=http_client,
    )

    first = client._privacy_safe_png(image_path)
    second = client._privacy_safe_png(image_path)

    assert first == second
    assert calls == 1
    await client.aclose()
    await http_client.aclose()


@pytest.mark.asyncio
async def test_gapgpt_vision_uses_openai_compatible_chat(tmp_path: Path) -> None:
    image_path = tmp_path / "person.png"
    image_path.write_bytes(_png_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer gap-key"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-4o"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
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
    client = GapGPTVisionClient(
        base_url="https://api.gapgpt.app/v1",
        api_key="gap-key",
        model="gpt-4o",
        http_client=http_client,
    )
    result = await client.analyze_person(image_path)
    assert result.suitable_for_tryon is True
    await http_client.aclose()


@pytest.mark.asyncio
async def test_gapgpt_image_edit_sends_two_images_and_downloads_url(
    tmp_path: Path,
) -> None:
    person = tmp_path / "person.png"
    garment = tmp_path / "garment.png"
    person.write_bytes(_png_bytes((100, 110, 120)))
    garment.write_bytes(_png_bytes((180, 20, 20)))
    generated = _png_bytes((1, 2, 3))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/images/edits":
            assert request.headers["authorization"] == "Bearer gap-key"
            content_type = request.headers["content-type"]
            assert content_type.startswith("multipart/form-data; boundary=")
            body = request.content
            assert body.count(b'name="image[]"') == 2
            assert b'name="model"' in body
            assert b"gpt-image-2" in body
            assert b'name="prompt"' in body
            assert b'name="temperature"' not in body
            return httpx.Response(
                200,
                json={"data": [{"url": "https://cdn.example.test/generated.png"}]},
            )
        if request.url.host == "cdn.example.test":
            assert "authorization" not in request.headers
            return httpx.Response(200, content=generated, headers={"content-type": "image/png"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GapGPTTryOnClient(
        base_url="https://api.gapgpt.app/v1",
        api_key="gap-key",
        model="gpt-image-2",
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


@pytest.mark.asyncio
async def test_gapgpt_image_edit_accepts_base64_response(tmp_path: Path) -> None:
    person = tmp_path / "person.png"
    garment = tmp_path / "garment.png"
    person.write_bytes(_png_bytes())
    garment.write_bytes(_png_bytes())
    generated = _png_bytes((4, 5, 6))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": base64.b64encode(generated).decode("ascii")}
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GapGPTTryOnClient(
        base_url="https://api.gapgpt.app/v1",
        api_key="gap-key",
        http_client=http_client,
    )
    outputs = await client.generate(person, garment, "upper_body", {})
    assert len(outputs) == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_gapgpt_wallpaper_uses_two_images_and_landscape_size(
    tmp_path: Path,
) -> None:
    room = tmp_path / "room.png"
    wallpaper = tmp_path / "wallpaper.png"
    Image.new("RGB", (96, 64), (220, 220, 220)).save(room)
    wallpaper.write_bytes(_png_bytes((180, 140, 80)))
    generated = _png_bytes((4, 5, 6))

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        assert body.count(b'name="image[]"') == 2
        assert b'filename="room.png"' in body
        assert b'filename="wallpaper-reference.png"' in body
        assert b'filename="wall-mask.png"' not in body
        assert b'filename="placement-guide.png"' not in body
        assert b'1536x1024' in body
        assert b'name="quality"' in body
        assert b'high' in body
        return httpx.Response(
            200,
            json={
                "data": [
                    {"b64_json": base64.b64encode(generated).decode("ascii")}
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GapGPTTryOnClient(
        base_url="https://api.gapgpt.app/v1",
        api_key="gap-key",
        prompt_builder=WallpaperPromptBuilder(),
        http_client=http_client,
    )
    outputs = await client.generate(
        room,
        wallpaper,
        "wallpaper",
        {"generation_quality": "high"},
    )
    assert len(outputs) == 1
    await http_client.aclose()


def test_gapgpt_key_is_required() -> None:
    with pytest.raises(GapGPTAPIError, match="GAPGPT_API_KEY"):
        GapGPTVisionClient(
            base_url="https://api.gapgpt.app/v1",
            api_key="",
            model="gpt-4o",
        )


def test_empty_network_error_keeps_its_type_for_docker_logs() -> None:
    assert _network_error_detail(httpx.ReadError("")) == "ReadError"


@pytest.mark.asyncio
async def test_gapgpt_reports_insufficient_image_balance(tmp_path: Path) -> None:
    person = tmp_path / "person.png"
    garment = tmp_path / "garment.png"
    person.write_bytes(_png_bytes())
    garment.write_bytes(_png_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "error": {
                    "message": "pre-consume quota failed; remaining quota is low"
                }
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GapGPTTryOnClient(
        base_url="https://api.gapgpt.app/v1",
        api_key="gap-key",
        model="gemini-3-pro-image-preview",
        http_client=http_client,
    )
    with pytest.raises(TryOnAPIError, match="balance is insufficient"):
        await client.generate(person, garment, "wallpaper", {})
    await http_client.aclose()


@pytest.mark.asyncio
async def test_pipeline_selects_gapgpt_clients(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        analysis_provider="gapgpt",
        tryon_provider="gapgpt",
        gapgpt_api_key="gap-key",
        output_directory=tmp_path / "outputs",
        temp_directory=tmp_path / "temp",
        log_directory=tmp_path / "logs",
    )
    pipeline = build_pipeline(settings)
    assert isinstance(pipeline.qwen_client, GapGPTVisionClient)
    assert isinstance(pipeline.tryon_client, GapGPTTryOnClient)
    await pipeline.aclose()
