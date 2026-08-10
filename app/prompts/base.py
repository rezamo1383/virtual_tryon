"""Prompt-builder contract shared by routing infrastructure."""

from __future__ import annotations

from typing import Any, Protocol


class PromptBuilder(Protocol):
    """Marker contract implemented by domain-specific prompt builders."""

    @property
    def domain(self) -> str:
        """Return the pipeline domain served by this builder."""


class GenerationPromptBuilder(PromptBuilder, Protocol):
    """Prompt contract required by two-reference generation providers."""

    def generation(
        self,
        category: str,
        options: dict[str, Any],
    ) -> str:
        """Build one stateless image-generation instruction."""
