"""Streamlit entry point for the client-facing demonstration application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import streamlit as st


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.sidebar import render_sidebar  # noqa: E402
from frontend.components.theme import apply_theme  # noqa: E402
from frontend.config.settings import load_frontend_settings  # noqa: E402
from frontend.pages import clothing, home, wallpaper  # noqa: E402
from frontend.services.api_client import BackendAPIError, BackendClient  # noqa: E402


st.set_page_config(
    page_title="Vision Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "A secure demonstration interface for the AI generation platform.",
    },
)


def _secrets() -> dict[str, Any]:
    try:
        return dict(st.secrets)
    except (FileNotFoundError, TypeError):
        return {}


settings = load_frontend_settings(_secrets())
st.session_state.setdefault("page", "home")
st.session_state.setdefault("theme", "Light")
st.session_state.setdefault("generation_in_progress", False)


@st.cache_resource
def _client(base_url: str, timeout: float) -> BackendClient:
    return BackendClient(base_url, timeout_seconds=timeout)


def _backend_online(base_url: str, timeout: float) -> bool:
    probe = BackendClient(base_url, timeout_seconds=min(timeout, 4.0))
    try:
        return probe.health().get("status") == "ok"
    except BackendAPIError as exc:
        parsed = urlsplit(base_url)
        LOGGER.warning(
            "backend_health_check_failed",
            extra={
                "backend_host": parsed.hostname or "invalid",
                "backend_port": parsed.port,
                "error": str(exc),
                "cause": type(exc.__cause__).__name__ if exc.__cause__ else None,
            },
        )
        return False
    finally:
        probe.close()


client = _client(settings.api_base_url, settings.request_timeout_seconds)
apply_theme(st.session_state["theme"])
theme = render_sidebar(
    settings,
    backend_is_online=lambda: _backend_online(
        settings.api_base_url,
        settings.request_timeout_seconds,
    ),
)
if theme != st.session_state["theme"]:
    st.session_state["theme"] = theme
    st.rerun()

page = st.session_state["page"]
if page == "clothing":
    clothing.render(client, settings)
elif page == "wallpaper":
    wallpaper.render(client, settings)
else:
    home.render()

st.markdown(
    f'<div class="app-footer">{settings.company_name} · Secure tenant-aware generation</div>',
    unsafe_allow_html=True,
)
