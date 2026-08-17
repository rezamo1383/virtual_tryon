"""Environment-backed settings for the Streamlit client."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


FRONTEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRONTEND_ROOT.parent
DEVELOPMENT_API_BASE_URL = "http://127.0.0.1:8000"


class FrontendConfigurationError(ValueError):
    """Raised when required frontend runtime configuration is missing."""


@dataclass(frozen=True)
class FrontendSettings:
    """Runtime settings that never contain backend business configuration."""

    api_base_url: str
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


def _api_base_url(secrets: Mapping[str, Any] | None) -> str:
    value = _value("API_BASE_URL", secrets, "").rstrip("/")
    app_env = _value("APP_ENV", secrets, "development").casefold()
    if not value and app_env == "development":
        value = DEVELOPMENT_API_BASE_URL
    if not value:
        raise FrontendConfigurationError(
            "API_BASE_URL is required outside local development. Set it as an "
            "environment variable or in Streamlit secrets."
        )
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise FrontendConfigurationError(
            "API_BASE_URL must be an absolute HTTP or HTTPS URL."
        )
    try:
        parsed.port
    except ValueError as exc:
        raise FrontendConfigurationError(
            "API_BASE_URL contains an invalid port."
        ) from exc
    return value


def load_frontend_settings(
    secrets: Mapping[str, Any] | None = None,
) -> FrontendSettings:
    """Load deployment values from environment first, then Streamlit secrets."""

    timeout = _value("API_TIMEOUT_SECONDS", secrets, "600")
    history_size = _value("MAX_HISTORY_ITEMS", secrets, "12")
    return FrontendSettings(
        api_base_url=_api_base_url(secrets),
        clothing_api_key=_value("CLOTHING_API_KEY", secrets, ""),
        wallpaper_api_key=_value("WALLPAPER_API_KEY", secrets, ""),
        request_timeout_seconds=max(30.0, float(timeout)),
        company_name=_value("COMPANY_NAME", secrets, "Vision Studio"),
        max_history_items=max(1, int(history_size)),
    )
