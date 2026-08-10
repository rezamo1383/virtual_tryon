"""Non-blocking visual progress around the synchronous backend endpoint."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import Any

import streamlit as st


def run_with_progress(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one HTTP call while keeping meeting-friendly status feedback alive."""

    stages = (
        (0.0, "Uploading", "Securely sending your images", 12),
        (1.0, "Analyzing", "Understanding the scene and reference", 36),
        (3.0, "Generating", "Creating a photorealistic result", 68),
    )
    status = st.status("Waiting", expanded=True)
    progress = st.progress(3, text="Preparing request")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        while not future.done():
            elapsed = time.monotonic() - started
            _, name, detail, base_progress = max(
                (stage for stage in stages if elapsed >= stage[0]),
                key=lambda item: item[0],
            )
            pulse = min(20, int(elapsed) % 21)
            progress.progress(
                min(92, base_progress + pulse),
                text=f"{name} · {detail}",
            )
            status.update(label=name, state="running", expanded=True)
            time.sleep(0.35)
        result = future.result()
    progress.progress(100, text="Completed · Result is ready")
    status.update(label="Completed", state="complete", expanded=False)
    return result
