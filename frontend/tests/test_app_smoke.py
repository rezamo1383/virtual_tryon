from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_and_product_pages_render_without_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_BASE_URL", "http://test-backend:8000")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception

    clothing_button = next(
        button
        for button in app.button
        if button.label.startswith("Open Clothing")
    )
    app = clothing_button.click().run()
    assert not app.exception
    assert [item.label for item in app.get("file_uploader")] == [
        "Person image",
        "Garment image 1",
    ]
    assert [item.label for item in app.selectbox].count(
        "What should be transferred from this image?"
    ) == 1
    add_button = next(
        button for button in app.button if button.label.startswith("+ Add")
    )
    app = add_button.click().run()
    assert [item.label for item in app.get("file_uploader")] == [
        "Person image",
        "Garment image 1",
        "Garment image 2",
    ]
    assert [item.label for item in app.selectbox].count(
        "What should be transferred from this image?"
    ) == 2
    assert not app.get("color_picker")

    wallpaper_button = next(
        button for button in app.button if "Wallpaper" in button.label
    )
    app = wallpaper_button.click().run()
    assert not app.exception
    assert [item.label for item in app.get("file_uploader")] == [
        "Room image",
        "Wallpaper image",
    ]
    assert [item.label for item in app.slider] == ["Pattern scale"]
