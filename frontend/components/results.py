"""Completed job presentation components."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

import streamlit as st

from frontend.components.media import comparison_slider, zoomable_image
from frontend.services.api_client import UploadedImage


def _duration(result: dict[str, Any], fallback: float) -> float:
    try:
        started = datetime.fromisoformat(str(result["started_at"]))
        completed = datetime.fromisoformat(str(result["completed_at"]))
        return max(0.0, (completed - started).total_seconds())
    except (KeyError, TypeError, ValueError):
        return fallback


def render_completed_job(record: dict[str, Any]) -> None:
    result = record["result"]
    output = record["output_bytes"]
    output_mime = record.get("output_mime", "image/png")
    source: UploadedImage = record["source"]
    job_id = str(record.get("job_id", "—"))
    duration = _duration(result, float(record.get("elapsed_seconds", 0)))
    st.markdown(
        '<span class="status-pill">● Completed</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-row">
          <span class="metric-chip">Job ID <strong>{html.escape(job_id)}</strong></span>
          <span class="metric-chip">Generation time <strong>{duration:.1f}s</strong></span>
          <span class="metric-chip">Result <strong>Full resolution</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Result", "Before / After", "Details"])
    with tabs[0]:
        zoomable_image(output, output_mime)
    with tabs[1]:
        comparison_slider(
            source,
            output,
            output_mime,
            before_label=(
                "Person" if record["product"] == "clothing" else "Room"
            ),
            after_label="Generated",
        )
    with tabs[2]:
        status = result.get("status", "completed")
        score = _score(result, record["product"])
        st.write(f"**Status:** {status}")
        if score is not None:
            st.write(f"**Quality score:** {score:.3f}")
        st.write("Generated securely through the configured tenant backend.")
    extension = _extension(output_mime)
    st.download_button(
        "Download full-resolution result",
        data=output,
        file_name=f"{record['product']}-{job_id}.{extension}",
        mime=output_mime,
        type="primary",
        width="stretch",
    )


def _score(result: dict[str, Any], product: str) -> float | None:
    value: Any = result.get("score")
    if product == "clothing":
        items = result.get("results")
        if isinstance(items, list) and items:
            value = items[0].get("score")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extension(mime_type: str) -> str:
    return {"image/jpeg": "jpg", "image/webp": "webp"}.get(
        mime_type,
        "png",
    )
