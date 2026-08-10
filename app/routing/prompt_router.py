"""Resolve tenant prompt profiles without embedding prompts in services."""

from __future__ import annotations

from app.core.exceptions import PipelineRoutingError
from app.prompts.base import PromptBuilder
from app.prompts.clothing import ClothingPromptBuilder
from app.prompts.wallpaper import WallpaperPromptBuilder

class PromptRouter:
    """Registry-backed prompt-profile resolver."""

    def __init__(self) -> None:
        self._builders: dict[tuple[str, str], PromptBuilder] = {
            ("clothing", "default"): ClothingPromptBuilder(),
            ("wallpaper", "default"): WallpaperPromptBuilder(),
        }

    def register(
        self,
        pipeline: str,
        profile: str,
        builder: PromptBuilder,
    ) -> None:
        """Register or replace a prompt profile during application setup."""

        self._builders[(pipeline, profile)] = builder

    def resolve(
        self,
        pipeline: str,
        profile: str,
    ) -> PromptBuilder:
        """Return the exact builder selected by tenant configuration."""

        builder = self._builders.get((pipeline, profile))
        if builder is None:
            raise PipelineRoutingError(
                f"Prompt profile '{profile}' is not configured for "
                f"pipeline '{pipeline}'."
            )
        return builder
