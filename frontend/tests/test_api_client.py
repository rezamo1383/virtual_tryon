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
    def handler(request: httpx.Request) -> httpx.Response:
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
            200,
            json={
                "job_id": "job_20260802_abcdef",
                "status": "completed",
                "results": [{"output": "final/original.png"}],
            },
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
    assert result["status"] == "completed"
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
