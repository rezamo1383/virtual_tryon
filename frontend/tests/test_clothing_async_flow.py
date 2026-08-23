"""Regression tests for the asynchronous clothing frontend flow."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from frontend.components import results as results_component
from frontend.config.settings import FrontendSettings
from frontend.pages import common
from frontend.services.api_client import GarmentUpload, UploadedImage
from frontend.services.history import HISTORY_KEY

JOB_ID = "job_20260823_abcdef"


class FakeElement:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def progress(self, value: int, **kwargs: Any) -> None:
        self.calls.append({"value": value, **kwargs})


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []
        self.status_elements: list[FakeElement] = []
        self.progress_elements: list[FakeElement] = []

    def status(self, label: str, **kwargs: Any) -> FakeElement:
        element = FakeElement()
        element.calls.append({"label": label, **kwargs})
        self.status_elements.append(element)
        return element

    def progress(self, value: int, **kwargs: Any) -> FakeElement:
        element = FakeElement()
        element.calls.append({"value": value, **kwargs})
        self.progress_elements.append(element)
        return element

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)


class FlowClient:
    def __init__(self, statuses: list[str]) -> None:
        self.statuses = statuses
        self.tryon_calls = 0
        self.event_calls = 0
        self.image_calls = 0
        self.generate_calls = 0
        self.artifact_calls = 0
        self.api_keys: list[str] = []

    def tryon(self, **kwargs: Any) -> dict[str, str]:
        self.tryon_calls += 1
        self.api_keys.append(str(kwargs["api_key"]))
        return {"job_id": JOB_ID}

    def job_events(self, job_id: str, *, api_key: str) -> Iterator[dict[str, str]]:
        self.event_calls += 1
        self.api_keys.append(api_key)
        assert job_id == JOB_ID
        yield from (
            {"job_id": job_id, "status": status} for status in self.statuses
        )

    def result_image(self, job_id: str, *, api_key: str) -> tuple[bytes, str]:
        self.image_calls += 1
        self.api_keys.append(api_key)
        assert job_id == JOB_ID
        return b"final-image", "image/png"

    def generate(self, **kwargs: Any) -> dict[str, str]:
        self.generate_calls += 1
        return {
            "job_id": JOB_ID,
            "status": "completed",
            "output": "final/wallpaper.png",
        }

    def output_path(self, result: dict[str, Any], product: str) -> str | None:
        return str(result.get("output")) if product == "wallpaper" else None

    def artifact(
        self,
        job_id: str,
        artifact_path: str,
        *,
        api_key: str,
    ) -> tuple[bytes, str]:
        self.artifact_calls += 1
        return b"wallpaper-image", "image/png"


def _settings() -> FrontendSettings:
    return FrontendSettings(
        api_base_url="http://backend",
        clothing_api_key="clothing-key",
        wallpaper_api_key="wallpaper-key",
    )


def _source() -> UploadedImage:
    return UploadedImage("person.png", b"person", "image/png")


def _garments() -> list[GarmentUpload]:
    return [
        GarmentUpload(
            UploadedImage("shirt.png", b"shirt", "image/png"),
            "T-shirt",
        )
    ]


def _install_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeStreamlit:
    ui = FakeStreamlit()
    monkeypatch.setattr(common, "st", ui)
    monkeypatch.setattr(
        common,
        "run_with_progress",
        lambda operation, **kwargs: operation(),
    )
    return ui


def test_clothing_job_id_waits_for_sse_then_downloads_and_keeps_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = _install_ui(monkeypatch)
    client = FlowClient(["queued", "running", "completed"])

    record = common.execute_generation(
        product="clothing",
        source=_source(),
        reference=None,
        options={"candidates_per_color": 1, "max_retries": 0},
        client=client,  # type: ignore[arg-type]
        settings=_settings(),
        garments=_garments(),
    )

    assert record is not None
    assert record["job_id"] == JOB_ID
    assert record["output_bytes"] == b"final-image"
    assert record["output_mime"] == "image/png"
    assert record["result"] == {"job_id": JOB_ID, "status": "completed"}
    assert client.tryon_calls == 1
    assert client.event_calls == 1
    assert client.image_calls == 1
    assert client.api_keys == ["clothing-key"] * 3
    assert ui.session_state[HISTORY_KEY][0]["job_id"] == JOB_ID
    assert ui.session_state["clothing_selected_job"] == JOB_ID
    assert ui.session_state["generation_in_progress"] is False
    assert not ui.errors
    assert "Your result is ready." in ui.successes


@pytest.mark.parametrize("terminal_status", ["failed", "rejected"])
def test_failed_or_rejected_clothing_job_does_not_download_image(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    ui = _install_ui(monkeypatch)
    client = FlowClient(["running", terminal_status])

    record = common.execute_generation(
        product="clothing",
        source=_source(),
        reference=None,
        options={},
        client=client,  # type: ignore[arg-type]
        settings=_settings(),
        garments=_garments(),
    )

    assert record is None
    assert client.tryon_calls == 1
    assert client.event_calls == 1
    assert client.image_calls == 0
    assert HISTORY_KEY not in ui.session_state
    assert ui.errors == [
        "Generation could not be completed. Please try another image."
    ]
    assert ui.session_state["generation_in_progress"] is False


def test_wallpaper_keeps_existing_synchronous_artifact_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = _install_ui(monkeypatch)
    client = FlowClient([])
    wallpaper = UploadedImage("wallpaper.png", b"wallpaper", "image/png")

    record = common.execute_generation(
        product="wallpaper",
        source=_source(),
        reference=wallpaper,
        options={"pattern_scale": 0.18},
        client=client,  # type: ignore[arg-type]
        settings=_settings(),
    )

    assert record is not None
    assert record["output_bytes"] == b"wallpaper-image"
    assert client.generate_calls == 1
    assert client.artifact_calls == 1
    assert client.tryon_calls == 0
    assert client.event_calls == 0
    assert client.image_calls == 0
    assert not ui.errors


class DummyTab:
    def __enter__(self) -> DummyTab:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class ResultsUI:
    def __init__(self) -> None:
        self.downloads: list[dict[str, Any]] = []

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tabs(self, labels: list[str]) -> list[DummyTab]:
        return [DummyTab() for _ in labels]

    def write(self, *args: Any, **kwargs: Any) -> None:
        return None

    def download_button(self, *args: Any, **kwargs: Any) -> None:
        self.downloads.append({"args": args, **kwargs})


def test_async_record_still_renders_existing_download_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = ResultsUI()
    monkeypatch.setattr(results_component, "st", ui)
    monkeypatch.setattr(results_component, "zoomable_image", lambda *args: None)
    monkeypatch.setattr(results_component, "comparison_slider", lambda *args, **kwargs: None)
    record = {
        "job_id": JOB_ID,
        "product": "clothing",
        "status": "completed",
        "result": {"job_id": JOB_ID, "status": "completed"},
        "source": _source(),
        "output_bytes": b"final-image",
        "output_mime": "image/png",
        "elapsed_seconds": 2.5,
    }

    results_component.render_completed_job(record)

    assert len(ui.downloads) == 1
    assert ui.downloads[0]["data"] == b"final-image"
    assert ui.downloads[0]["mime"] == "image/png"
    assert ui.downloads[0]["file_name"] == f"clothing-{JOB_ID}.png"
