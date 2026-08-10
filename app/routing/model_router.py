"""Resolve tenant model/provider selections into effective runtime settings."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.tenant.models import TenantConfig


@dataclass(frozen=True)
class ModelRoute:
    """Resolved provider/model route for one tenant pipeline."""

    settings: Settings
    analysis_provider: str
    generation_provider: str
    analysis_model: str
    generation_model: str


class ModelRouter:
    """Apply tenant model choices without mutating global settings."""

    def __init__(self, settings: Settings) -> None:
        self._base = settings

    def resolve(self, tenant: TenantConfig) -> ModelRoute:
        """Return an isolated Settings copy configured for one tenant."""

        analysis_provider = (
            tenant.analysis_provider or self._base.analysis_provider
        )
        generation_provider = (
            tenant.generation_provider or self._base.tryon_provider
        )
        updates: dict[str, object] = {
            "analysis_provider": analysis_provider,
            "tryon_provider": generation_provider,
        }
        if tenant.analysis_model:
            if analysis_provider == "openrouter":
                updates["openrouter_vision_model"] = tenant.analysis_model
            elif analysis_provider == "gapgpt":
                updates["gapgpt_vision_model"] = tenant.analysis_model
            elif analysis_provider == "qwen":
                updates["qwen_model"] = tenant.analysis_model
        if tenant.generation_model:
            if generation_provider == "openrouter":
                updates["openrouter_image_model"] = tenant.generation_model
            elif generation_provider == "gapgpt":
                updates["gapgpt_image_model"] = tenant.generation_model
            elif generation_provider == "generic":
                updates["tryon_model"] = tenant.generation_model
        effective = self._base.model_copy(update=updates)
        return ModelRoute(
            settings=effective,
            analysis_provider=analysis_provider,
            generation_provider=generation_provider,
            analysis_model=self._analysis_model(
                effective,
                analysis_provider,
            ),
            generation_model=self._generation_model(
                effective,
                generation_provider,
            ),
        )

    @staticmethod
    def _analysis_model(settings: Settings, provider: str) -> str:
        return {
            "openrouter": settings.openrouter_vision_model,
            "gapgpt": settings.gapgpt_vision_model,
            "qwen": settings.qwen_model,
            "mock": "mock-vision",
        }.get(provider, "")

    @staticmethod
    def _generation_model(settings: Settings, provider: str) -> str:
        return {
            "openrouter": settings.openrouter_image_model,
            "gapgpt": settings.gapgpt_image_model,
            "generic": settings.tryon_model,
            "mock": "mock-generation",
        }.get(provider, "")
