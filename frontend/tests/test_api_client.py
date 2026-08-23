from __future__ import annotations

import json

import httpx
import pytest

from frontend.services.api_client import (
    BackendAPIError,
    BackendClient,
    GarmentUpload,
    UploadedImage,
)


def image(name: str = "image.png") -> UploadedImage:
    return UploadedImage(name, b"image-bytes", "image/png")


def test_health_uses_configured_docker_backend_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "http://virtual-tryon-backend:8000/health"
        )
        return httpx.Response(200, json={"status": "ok"})

    client = BackendClient(
        "http://virtual-tryon-backend:8000",
        transport=httpx.MockTransport(handler),
    )
    assert client.health()["status"] == "ok"
    client.close()


def test_generate_uses_generic_endpoint_and_tenant_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/generate"
        assert request.headers["x-api-key"] == "tenant-key"
        body = request.read()
        assert b'"pattern_scale": 0.18' in body
        assert b"task_type" not in body
        return httpx.Response(
            200,
            json={
                "job_id": "job_20260802_abcdef",
                "status": "completed",
                "output": "final/wallpaper.png",
            },
        )

    client = BackendClient(
        "http://backend",
        transport=httpx.MockTransport(handler),
    )
    result = client.generate(
        source=image("room.png"),
        reference=image("wallpaper.png"),
        options={"pattern_scale": 0.18},
        api_key="tenant-key",
    )
    assert result["status"] == "completed"
    client.close()


def test_tryon_sends_multiple_labeled_garments() -> None:
    post_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal post_count
        post_count += 1
        assert request.url.path == "/api/v1/tryon"
        assert request.headers["x-api-key"] == "tenant-key"
        body = request.read()
        assert body.count(b'name="garment_images"') == 3
        assert "T-shirt".encode() in body
        assert "Pants".encode() in body
        assert "Watch".encode() in body
        assert b'name="candidates_per_color"' in body
        assert b"\r\n2\r\n" in body
        return httpx.Response(
            202,
            json={"job_id": "job_20260802_abcdef"},
        )

    client = BackendClient(
        "http://backend",
        transport=httpx.MockTransport(handler),
    )
    result = client.tryon(
        person=image("person.png"),
        garments=[
            GarmentUpload(image("shirt.png"), "T-shirt"),
            GarmentUpload(image("pants.png"), "Pants"),
            GarmentUpload(image("watch.png"), "Watch"),
        ],
        options={"candidates_per_color": 2},
        api_key="tenant-key",
    )
    assert result == {"job_id": "job_20260802_abcdef"}
    assert post_count == 1
    client.close()


def test_sse_ignores_heartbeats_and_downloads_result_once_with_tenant_key() -> None:
    event_calls = 0
    image_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal event_calls, image_calls
        assert request.headers["x-api-key"] == "tenant-key"
        if request.url.path.endswith("/events"):
            event_calls += 1
            assert request.headers["accept"] == "text/event-stream"
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=(
                    ": keep-alive\n\n"
                    "event: status\n"
                    'data: {"job_id":"job_20260802_abcdef",'
                    '"status":"running"}\n\n'
                    ": keep-alive\n\n"
                    "event: status\n"
                    'data: {"job_id":"job_20260802_abcdef",'
                    '"status":"completed"}\n\n'
                ),
            )
        if request.url.path.endswith("/image"):
            image_calls += 1
            return httpx.Response(
                200,
                headers={"Content-Type": "image/webp"},
                content=b"generated-image",
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    client = BackendClient(
        "http://backend",
        transport=httpx.MockTransport(handler),
    )
    events = list(
        client.job_events(
            "job_20260802_abcdef",
            api_key="tenant-key",
        )
    )
    image_bytes, mime_type = client.result_image(
        "job_20260802_abcdef",
        api_key="tenant-key",
    )

    assert [event["status"] for event in events] == ["running", "completed"]
    assert event_calls == 1
    assert image_calls == 1
    assert image_bytes == b"generated-image"
    assert mime_type == "image/webp"
    client.close()


def test_artifact_download_rejects_path_traversal_locally() -> None:
    client = BackendClient(
        "http://backend",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500)
        ),
    )
    with pytest.raises(BackendAPIError, match="path is invalid"):
        client.artifact(
            "job_20260802_abcdef",
            "../results.json",
            api_key="tenant-key",
        )
    client.close()


def test_provider_errors_are_sanitized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "detail": (
                    "account balance is insufficient; internal request id "
                    "secret-provider-detail"
                )
            },
        )

    client = BackendClient(
        "http://backend",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(
        BackendAPIError,
        match="needs additional credit",
    ) as captured:
        client.generate(
            source=image(),
            reference=image(),
            options=json.loads("{}"),
            api_key="tenant-key",
        )
    assert "secret-provider-detail" not in str(captured.value)
    client.close()


def test_provider_network_error_is_actionable() -> None:
    client = BackendClient(
        "http://backend",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                502,
                json={
                    "detail": (
                        "GapGPT image request failed after retries: ReadError"
                    )
                },
            )
        ),
    )
    with pytest.raises(BackendAPIError, match="provider connection failed"):
        client.generate(
            source=image(),
            reference=image(),
            options={},
            api_key="tenant-key",
        )
    client.close()


def test_output_path_supports_both_products() -> None:
    assert (
        BackendClient.output_path(
            {"results": [{"output": "final/blue.png"}]},
            "clothing",
        )
        == "final/blue.png"
    )
    assert (
        BackendClient.output_path(
            {"output": "final/wallpaper.png"},
            "wallpaper",
        )
        == "final/wallpaper.png"
    )
