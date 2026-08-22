"""Central generation behavior for prepared-product try-on requests."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.models.request_models import ClothingOptions, TryOnMode, TryOnRequest


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    """Resolved candidate, retry, and evaluation behavior."""

    mode: TryOnMode
    candidates: int
    max_retries: int
    evaluate_outputs: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> GenerationPolicy:
        mode = TryOnMode(settings.tryon_mode)
        if mode is TryOnMode.FAST:
            return cls(
                mode=mode,
                candidates=1,
                max_retries=0,
                evaluate_outputs=False,
            )
        return cls(
            mode=mode,
            candidates=settings.candidates_per_color,
            max_retries=settings.max_generation_retries,
            evaluate_outputs=True,
        )

    def apply_to_options(self, options: ClothingOptions) -> ClothingOptions:
        """Force the cost ceiling in Fast mode and preserve Quality inputs."""

        if self.mode is not TryOnMode.FAST:
            return options
        return options.model_copy(
            update={"candidates_per_color": 1, "max_retries": 0}
        )

    def apply_to_request(self, request: TryOnRequest) -> TryOnRequest:
        """Force the same Fast policy on the legacy single-garment request."""

        if self.mode is not TryOnMode.FAST:
            return request
        return request.model_copy(
            update={"candidates_per_color": 1, "max_retries": 0}
        )
