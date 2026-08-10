"""Wallpaper visualization page."""

from __future__ import annotations

import streamlit as st

from frontend.components.media import preview_image
from frontend.components.results import render_completed_job
from frontend.components.theme import render_hero
from frontend.config.settings import FrontendSettings, PROJECT_ROOT
from frontend.pages.common import (
    current_record,
    execute_generation,
    reset_product,
    select_example,
    selected_inputs,
)
from frontend.services.api_client import BackendClient


def render(client: BackendClient, settings: FrontendSettings) -> None:
    render_hero(
        "Wallpaper visualization",
        "Transform the wall, preserve the room.",
        "Apply a real wallpaper reference to visible walls with scene-aware perspective and lighting.",
    )
    token = st.session_state.get("wallpaper_reset_token", 0)
    controls, workspace = st.columns([0.86, 1.45], gap="large")
    with controls:
        with st.container(border=True):
            st.markdown("### Create a room concept")
            st.caption("PNG, JPG, or WebP · Drag and drop supported")
            room_upload = st.file_uploader(
                "Room image",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"wallpaper_room_{token}",
            )
            wallpaper_upload = st.file_uploader(
                "Wallpaper image",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"wallpaper_reference_{token}",
            )
            source, reference = selected_inputs(
                "wallpaper",
                room_upload,
                wallpaper_upload,
            )
            scale = st.slider(
                "Pattern scale",
                min_value=0.03,
                max_value=0.75,
                value=0.18,
                step=0.01,
                help="Lower values create smaller, more frequent repeats.",
            )
            generate = st.button(
                "Generate visualization",
                type="primary",
                width="stretch",
                disabled=st.session_state.get(
                    "generation_in_progress",
                    False,
                ),
            )
            if st.button("Reset inputs", width="stretch"):
                reset_product("wallpaper")
            if generate:
                execute_generation(
                    product="wallpaper",
                    source=source,
                    reference=reference,
                    options={
                        "pattern_scale": scale,
                        "preserve_lighting": True,
                        "candidates_per_job": 1,
                        "max_retries": 0,
                    },
                    client=client,
                    settings=settings,
                )
    with workspace:
        with st.container(border=True):
            st.markdown("### Preview & result")
            first, second = st.columns(2)
            with first:
                preview_image(source, "Room preview")
            with second:
                preview_image(reference, "Wallpaper preview")
            record = current_record("wallpaper")
            if record:
                st.divider()
                render_completed_job(record)
            else:
                st.markdown(
                    '<div class="empty-preview">Your transformed room will appear here.</div>',
                    unsafe_allow_html=True,
                )
    _examples()


def _examples() -> None:
    with st.expander("Example images", expanded=False):
        st.caption("Load a ready-to-present room and wallpaper pair.")
        room = PROJECT_ROOT / "inputs" / "rooms" / "room1.png"
        wallpaper = (
            PROJECT_ROOT / "inputs" / "wallpapers" / "wallpaper.png"
        )
        if room.is_file() and wallpaper.is_file():
            left, right = st.columns(2)
            left.image(str(room), caption="Room", width="stretch")
            right.image(
                str(wallpaper),
                caption="Wallpaper",
                width="stretch",
            )
            if st.button("Use wallpaper example", width="stretch"):
                select_example(
                    product="wallpaper",
                    source_path=room,
                    reference_path=wallpaper,
                )
        else:
            st.info("Add example images to the backend inputs directory.")
