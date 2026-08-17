from __future__ import annotations

import pytest

from frontend.config.settings import (
    FrontendConfigurationError,
    load_frontend_settings,
)


def test_environment_has_priority_over_streamlit_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_BASE_URL", "https://api.example.com/")
    monkeypatch.setenv("API_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("MAX_HISTORY_ITEMS", "7")
    monkeypatch.setenv("CLOTHING_API_KEY", "environment-clothing-key")
    monkeypatch.setenv("WALLPAPER_API_KEY", "environment-wallpaper-key")
    monkeypatch.setenv("COMPANY_NAME", "Environment Studio")

    settings = load_frontend_settings(
        {
            "API_BASE_URL": "https://secret-api.example.com",
            "API_TIMEOUT_SECONDS": 90,
            "MAX_HISTORY_ITEMS": 3,
            "CLOTHING_API_KEY": "secret-clothing-key",
            "WALLPAPER_API_KEY": "secret-wallpaper-key",
            "COMPANY_NAME": "Secrets Studio",
        }
    )

    assert settings.api_base_url == "https://api.example.com"
    assert settings.request_timeout_seconds == 45
    assert settings.max_history_items == 7
    assert settings.clothing_api_key == "environment-clothing-key"
    assert settings.wallpaper_api_key == "environment-wallpaper-key"
    assert settings.company_name == "Environment Studio"


def test_streamlit_secrets_are_used_when_environment_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "API_BASE_URL",
        "API_TIMEOUT_SECONDS",
        "MAX_HISTORY_ITEMS",
        "CLOTHING_API_KEY",
        "WALLPAPER_API_KEY",
        "COMPANY_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_frontend_settings(
        {
            "API_BASE_URL": "http://development-backend:8000/",
            "API_TIMEOUT_SECONDS": 120,
            "MAX_HISTORY_ITEMS": 5,
            "CLOTHING_API_KEY": "clothing-key",
            "WALLPAPER_API_KEY": "wallpaper-key",
            "COMPANY_NAME": "Secrets Studio",
        }
    )

    assert settings.api_base_url == "http://development-backend:8000"
    assert settings.request_timeout_seconds == 120
    assert settings.max_history_items == 5
    assert settings.clothing_api_key == "clothing-key"
    assert settings.wallpaper_api_key == "wallpaper-key"
    assert settings.company_name == "Secrets Studio"


def test_local_development_uses_loopback_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("APP_ENV", "development")

    settings = load_frontend_settings({})

    assert settings.api_base_url == "http://127.0.0.1:8000"


def test_api_base_url_is_required_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("APP_ENV", "production")

    with pytest.raises(
        FrontendConfigurationError,
        match="API_BASE_URL is required",
    ):
        load_frontend_settings({})


def test_api_base_url_must_be_absolute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_BASE_URL", "backend:8000")

    with pytest.raises(
        FrontendConfigurationError,
        match="absolute HTTP or HTTPS URL",
    ):
        load_frontend_settings({})
