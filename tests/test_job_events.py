"""SSE contract tests for tenant-owned persisted job state."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from api import create_app
from app.clients.mock_tryon_client import MockTryOnClient
from app.core.config import Settings
from app.core.runtime import build_runtime
from app.services.job_events import JobEventStreamer
from app.services.pipeline import build_pipeline
from app.utils.json_utils import write_json

FASHION_KEY = "fashion-events-key"
OTHER_KEY = "other-events-key"


class FakeRequest:
    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


class CountingTryOnClient(MockTryOnClient):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        self.calls += 1
        return await super().generate(
            person_image,
            garment_image,
            category,
            options,
        )


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
                        "api_key_sha256": hashlib.sha256(
                            FASHION_KEY.encode()
                        ).hexdigest(),
                    },
                    {
                        "tenant_id": "other",
                        "pipeline": "clothing",
                        "analysis_provider": "mock",
                        "generation_provider": "mock",
                        "api_key_sha256": hashlib.sha256(
                            OTHER_KEY.encode()
                        ).hexdigest(),
                    },
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
        use_mock_qwen=True,
        use_mock_tryon=True,
        output_directory=tmp_path / "outputs",
        temp_directory=tmp_path / "temp",
        log_directory=tmp_path / "logs",
        local_preprocessing_enabled=False,
    )


def _write_state(
    settings: Settings,
    job_id: str,
    status: str,
    *,
    tenant_id: str = "fashion",
) -> None:
    write_json(
        settings.output_directory / job_id / "job_state.json",
        {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "pipeline": "clothing",
            "status": status,
            "started_at": "2026-08-22T00:00:00+00:00",
            "completed_at": (
                "2026-08-22T00:01:00+00:00"
                if status
                in {"completed", "completed_with_failures", "failed", "rejected"}
                else None
            ),
        },
    )


def _statuses(body: str) -> list[str]:
    return [
        json.loads(line.removeprefix("data: "))["status"]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _test_streamer(application: Any) -> None:
    application.state.job_event_streamer = JobEventStreamer(
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=1,
    )


async def _close_runtime(application: Any) -> None:
    runtime = getattr(application.state, "runtime", None)
    if runtime is not None:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_sse_emits_current_status_immediately(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job_id = "job_20260822_000001"
    _write_state(settings, job_id, "queued")
    application = create_app(settings)
    _test_streamer(application)

    async def finish_job() -> None:
        await asyncio.sleep(0.04)
        _write_state(settings, job_id, "completed")

    updater = asyncio.create_task(finish_job())
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/jobs/{job_id}/events",
            headers={"X-API-Key": FASHION_KEY},
        )
    await updater
    await _close_runtime(application)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert _statuses(response.text)[0] == "queued"


@pytest.mark.asyncio
async def test_sse_emits_only_status_changes_through_completion(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job_id = "job_20260822_000002"
    _write_state(settings, job_id, "queued")
    application = create_app(settings)
    _test_streamer(application)

    async def advance_job() -> None:
        await asyncio.sleep(0.03)
        _write_state(settings, job_id, "running")
        await asyncio.sleep(0.03)
        _write_state(settings, job_id, "completed")

    updater = asyncio.create_task(advance_job())
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/jobs/{job_id}/events",
            headers={"Authorization": f"Bearer {FASHION_KEY}"},
        )
    await updater
    await _close_runtime(application)

    assert _statuses(response.text) == ["queued", "running", "completed"]
    assert response.text.count("event: status") == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["failed", "rejected"])
async def test_failed_and_rejected_streams_terminate(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    settings = _settings(tmp_path)
    job_id = "job_20260822_000003"
    _write_state(settings, job_id, terminal_status)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/jobs/{job_id}/events",
            headers={"X-API-Key": FASHION_KEY},
        )
    await _close_runtime(application)

    assert _statuses(response.text) == [terminal_status]
    assert application.state.job_event_streamer.active_stream_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    ["completed", "completed_with_failures"],
)
async def test_completed_job_emits_once_and_closes(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    settings = _settings(tmp_path)
    job_id = "job_20260822_000004"
    _write_state(settings, job_id, terminal_status)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/jobs/{job_id}/events",
            headers={"X-API-Key": FASHION_KEY},
        )
    await _close_runtime(application)

    assert _statuses(response.text) == [terminal_status]
    assert response.text.endswith("\n\n")
    assert application.state.job_event_streamer.active_stream_count == 0


@pytest.mark.asyncio
async def test_sse_enforces_tenant_job_ownership(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job_id = "job_20260822_000005"
    _write_state(settings, job_id, "completed", tenant_id="fashion")
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/jobs/{job_id}/events",
            headers={"X-API-Key": OTHER_KEY},
        )
    await _close_runtime(application)

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"
    assert "fashion" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "job_id",
    ["not-a-job", "job_20260822_999999"],
)
async def test_sse_invalid_and_nonexistent_jobs_are_safe_404s(
    tmp_path: Path,
    job_id: str,
) -> None:
    settings = _settings(tmp_path)
    application = create_app(settings)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/jobs/{job_id}/events",
            headers={"X-API-Key": FASHION_KEY},
        )
    await _close_runtime(application)

    assert response.status_code == 404
    assert response.json()["status"] == "error"
    assert response.json()["error"] == "not_found"


@pytest.mark.asyncio
async def test_disconnected_client_leaves_no_active_stream() -> None:
    request = FakeRequest()
    streamer = JobEventStreamer(
        poll_interval_seconds=0.001,
        heartbeat_interval_seconds=1,
    )
    events = streamer.stream(
        request=request,
        job_id="job_20260822_000006",
        initial_status="running",
        load_state=lambda: {"status": "running"},
    )

    assert "\"status\":\"running\"" in await anext(events)
    assert streamer.active_stream_count == 1
    request.disconnected = True
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    assert streamer.active_stream_count == 0


@pytest.mark.asyncio
async def test_idle_stream_heartbeat_is_not_a_status_event() -> None:
    request = FakeRequest()
    streamer = JobEventStreamer(
        poll_interval_seconds=0.001,
        heartbeat_interval_seconds=0.002,
    )
    events = streamer.stream(
        request=request,
        job_id="job_20260822_000007",
        initial_status="running",
        load_state=lambda: {"status": "running"},
    )

    await anext(events)
    heartbeat = await anext(events)
    assert heartbeat == ": keep-alive\n\n"
    assert "event:" not in heartbeat
    await events.aclose()
    assert streamer.active_stream_count == 0


@pytest.mark.asyncio
async def test_sse_does_not_invoke_generation_provider(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job_id = "job_20260822_000008"
    _write_state(settings, job_id, "completed")
    provider = CountingTryOnClient()
    pipeline = build_pipeline(settings, tryon_client=provider)
    runtime = build_runtime(settings, legacy_pipeline=pipeline)
    application = create_app(settings)
    application.state.pipeline = pipeline
    application.state.runtime = runtime
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            f"/api/v1/jobs/{job_id}/events",
            headers={"X-API-Key": FASHION_KEY},
        )
    await runtime.aclose()

    assert response.status_code == 200
    assert provider.calls == 0
