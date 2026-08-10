"""Environment-backed settings for the Streamlit client."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRONTEND_ROOT.parent


@dataclass(frozen=True)
class FrontendSettings:
    """Runtime settings that never contain backend business configuration."""

    api_base_url: str = "http://127.0.0.1:8000"
    clothing_api_key: str = ""
    wallpaper_api_key: str = ""
    request_timeout_seconds: float = 600.0
    company_name: str = "Vision Studio"
    max_history_items: int = 12

    def api_key_for(self, product: str) -> str:
        return (
            self.clothing_api_key
            if product == "clothing"
            else self.wallpaper_api_key
        )


def _value(
    name: str,
    secrets: Mapping[str, Any] | None,
    default: str,
) -> str:
    if value := os.getenv(name):
        return value.strip()
    if secrets is not None:
        value = secrets.get(name)
        if value is not None:
            return str(value).strip()
    return default


def load_frontend_settings(
    secrets: Mapping[str, Any] | None = None,
) -> FrontendSettings:
    """Load deployment values from environment first, then Streamlit secrets."""

    timeout = _value("API_TIMEOUT_SECONDS", secrets, "600")
    history_size = _value("MAX_HISTORY_ITEMS", secrets, "12")
    return FrontendSettings(
        api_base_url=_value(
            "API_BASE_URL",
            secrets,
            "http://127.0.0.1:8000",
        ).rstrip("/"),
        clothing_api_key=_value("CLOTHING_API_KEY", secrets, ""),
        wallpaper_api_key=_value("WALLPAPER_API_KEY", secrets, ""),
        request_timeout_seconds=max(30.0, float(timeout)),
        company_name=_value("COMPANY_NAME", secrets, "Vision Studio"),
        max_history_items=max(1, int(history_size)),
    )
