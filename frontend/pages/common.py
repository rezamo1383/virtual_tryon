"""Shared product-page behavior with no AI or routing logic."""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, cast

import streamlit as st

from frontend.components.progress import run_with_progress
from frontend.config.settings import FrontendSettings
from frontend.services.api_client import (
    BackendAPIError,
    BackendClient,
    GarmentUpload,
    UploadedImage,
)
from frontend.services.history import add_job, find_job


def from_upload(upload: Any | None) -> UploadedImage | None:
    if upload is None:
        return None
    return UploadedImage(
        name=str(upload.name),
        content=upload.getvalue(),
        content_type=str(upload.type or "application/octet-stream"),
    )


def select_example(
    *,
    product: str,
    source_path: Path,
    reference_path: Path,
) -> None:
    st.session_state[f"{product}_example_source"] = UploadedImage.from_path(
        source_path
    )
    st.session_state[f"{product}_example_reference"] = UploadedImage.from_path(
        reference_path
    )
    st.session_state[f"{product}_reset_token"] = (
        st.session_state.get(f"{product}_reset_token", 0) + 1
    )
    st.rerun()


def selected_inputs(
    product: str,
    source_upload: Any | None,
    reference_upload: Any | None,
) -> tuple[UploadedImage | None, UploadedImage | None]:
    source = from_upload(source_upload) or st.session_state.get(
        f"{product}_example_source"
    )
    reference = from_upload(reference_upload) or st.session_state.get(
        f"{product}_example_reference"
    )
    return source, reference


def reset_product(product: str) -> None:
    for suffix in (
        "example_source",
        "example_reference",
        "selected_job",
    ):
        st.session_state.pop(f"{product}_{suffix}", None)
    st.session_state[f"{product}_reset_token"] = (
        st.session_state.get(f"{product}_reset_token", 0) + 1
    )
    st.rerun()


def current_record(product: str) -> dict[str, Any] | None:
    selected = st.session_state.get(f"{product}_selected_job")
    state = cast(MutableMapping[str, Any], st.session_state)
    return find_job(state, selected)


def execute_generation(
    *,
    product: str,
    source: UploadedImage | None,
    reference: UploadedImage | None,
    options: dict[str, Any],
    client: BackendClient,
    settings: FrontendSettings,
    garments: list[GarmentUpload] | None = None,
) -> dict[str, Any] | None:
    if source is None:
        st.warning("Upload a source image before generating.")
        return None
    if product == "clothing" and not garments:
        st.warning("Upload at least one garment or accessory image.")
        return None
    if product != "clothing" and reference is None:
        st.warning("Upload both images before generating.")
        return None
    api_key = settings.api_key_for(product)
    if product == "wallpaper" and not api_key:
        st.error(
            "Wallpaper tenant credentials are not configured for the frontend."
        )
        return None
    started = time.monotonic()
    st.session_state["generation_in_progress"] = True
    try:
        def operation() -> dict[str, Any]:
            if product == "clothing":
                return client.tryon(
                    person=source,
                    garments=garments or [],
                    options=options,
                    api_key=api_key,
                )
            if reference is None:
                raise ValueError("A reference image is required.")
            return client.generate(
                source=source,
                reference=reference,
                options=options,
                api_key=api_key,
            )

        if product == "clothing":
            result = run_with_progress(
                operation,
                completed_label="Submitted",
                completed_text="Submitted · Waiting for generation",
            )
            job_id = str(result.get("job_id", ""))
            if not job_id:
                raise BackendAPIError(
                    "The backend did not return a valid generation job."
                )
            terminal_status = _wait_for_clothing_job(
                client,
                job_id,
                api_key=api_key,
            )
            result = {**result, "status": terminal_status}
            if terminal_status in {"failed", "rejected"}:
                st.error(_safe_result_error(terminal_status))
                return None
            output_bytes, output_mime = client.result_image(
                job_id,
                api_key=api_key,
            )
        else:
            result = run_with_progress(operation)
            if result.get("status") in {"failed", "rejected"}:
                reason = result.get("rejection_reason") or result.get("error")
                st.error(_safe_result_error(reason))
                return None
            job_id = str(result.get("job_id", ""))
            output_path = client.output_path(result, product)
            if not job_id or not output_path:
                st.error("Generation finished without a downloadable result.")
                return None
            output_bytes, output_mime = client.artifact(
                job_id,
                output_path,
                api_key=api_key,
            )
        record = {
            "job_id": job_id,
            "product": product,
            "status": result.get("status", "completed"),
            "result": result,
            "source": source,
            "reference": reference,
            "garments": garments or [],
            "output_bytes": output_bytes,
            "output_mime": output_mime,
            "elapsed_seconds": time.monotonic() - started,
        }
        state = cast(MutableMapping[str, Any], st.session_state)
        add_job(
            state,
            record,
            limit=settings.max_history_items,
        )
        st.session_state[f"{product}_selected_job"] = job_id
        st.success("Your result is ready.")
        return record
    except BackendAPIError as exc:
        st.error(str(exc))
        return None
    finally:
        st.session_state["generation_in_progress"] = False


def _wait_for_clothing_job(
    client: BackendClient,
    job_id: str,
    *,
    api_key: str,
) -> str:
    status_box = st.status("Queued", expanded=True)
    progress = st.progress(12, text="Waiting for generation to start")
    try:
        for event in client.job_events(job_id, api_key=api_key):
            if event.get("job_id") != job_id:
                raise BackendAPIError(
                    "The backend returned an invalid job status event."
                )
            current = str(event.get("status", ""))
            if current == "queued":
                progress.progress(18, text="Queued · Waiting for a worker")
                status_box.update(
                    label="Queued",
                    state="running",
                    expanded=True,
                )
            elif current == "running":
                progress.progress(
                    72,
                    text="Generating · Creating your final outfit",
                )
                status_box.update(
                    label="Generating",
                    state="running",
                    expanded=True,
                )
            elif current in {"completed", "completed_with_failures"}:
                progress.progress(100, text="Completed · Result is ready")
                status_box.update(
                    label="Completed",
                    state="complete",
                    expanded=False,
                )
                return current
            elif current in {"failed", "rejected"}:
                status_box.update(
                    label=("Rejected" if current == "rejected" else "Failed"),
                    state="error",
                    expanded=True,
                )
                return current
    except BackendAPIError:
        status_box.update(
            label="Connection interrupted",
            state="error",
            expanded=True,
        )
        raise
    raise BackendAPIError(
        "The job status stream ended before generation finished."
    )


def _safe_result_error(value: Any) -> str:
    message = str(value or "Generation could not be completed.")
    lowered = message.casefold()
    if "no person was detected" in lowered:
        return "No person was detected in the uploaded person image."
    if "wall" in lowered and "detected" in lowered:
        return "No suitable wall was detected in the room image."
    if "balance" in lowered or "quota" in lowered:
        return "The image generation account needs additional credit."
    return "Generation could not be completed. Please try another image."
