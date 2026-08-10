"""Navigation, product identity, backend health, and recent jobs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from frontend.config.settings import FrontendSettings
from frontend.services.history import HISTORY_KEY, clear_history, initialize_history


def render_sidebar(
    settings: FrontendSettings,
    *,
    backend_is_online: Callable[[], bool],
) -> str:
    initialize_history(st.session_state)
    with st.sidebar:
        initials = "".join(
            word[0] for word in settings.company_name.split()[:2]
        ).upper()
        st.markdown(
            f"""
            <div class="sidebar-brand">
              <div class="logo-mark">{initials or 'AI'}</div>
              <div><strong>{settings.company_name}</strong><span>AI Visualization Platform</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
        _navigation_button("⌂  Home", "home")
        _navigation_button("◈  Clothing", "clothing")
        _navigation_button("▦  Wallpaper", "wallpaper")
        st.divider()
        theme = st.radio(
            "Appearance",
            options=["Light", "Dark"],
            horizontal=True,
            key="theme",
        )
        online = backend_is_online()
        status_color = "#26C281" if online else "#F59E5B"
        status_text = "Backend online" if online else "Backend offline"
        st.markdown(
            f"<span style='color:{status_color};font-size:.8rem;font-weight:700'>● {status_text}</span>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown('<div class="section-label">Recent jobs</div>', unsafe_allow_html=True)
        jobs: list[dict[str, Any]] = st.session_state[HISTORY_KEY]
        if not jobs:
            st.caption("Completed generations will appear here.")
        for record in jobs[:8]:
            product = str(record.get("product", "clothing"))
            icon = "◈" if product == "clothing" else "▦"
            job_id = str(record.get("job_id", "job"))
            if st.button(
                f"{icon}  {job_id[-10:]}",
                key=f"history_{job_id}",
                width="stretch",
            ):
                st.session_state["page"] = product
                st.session_state[f"{product}_selected_job"] = job_id
                st.rerun()
        if jobs and st.button(
            "Clear history",
            key="clear_history",
            width="stretch",
        ):
            clear_history(st.session_state)
            st.session_state.pop("clothing_selected_job", None)
            st.session_state.pop("wallpaper_selected_job", None)
            st.rerun()
    return theme


def _navigation_button(label: str, page: str) -> None:
    if st.button(
        label,
        key=f"nav_{page}",
        width="stretch",
        type="primary" if st.session_state.get("page") == page else "secondary",
    ):
        st.session_state["page"] = page
        st.rerun()
