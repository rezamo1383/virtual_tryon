"""Domain pipeline factories registered with the infrastructure router."""

from __future__ import annotations

from app.core.exceptions import PipelineRoutingError
from app.pipelines.base import BasePipeline
from app.pipelines.clothing import ClothingPipeline
from app.pipelines.wallpaper import WallpaperPipeline
from app.prompts.base import PromptBuilder
from app.prompts.clothing import ClothingPromptBuilder
from app.prompts.wallpaper import WallpaperPromptBuilder
from app.providers.factory import build_provider_bundle
from app.routing.model_router import ModelRoute
from app.services.pipeline import build_pipeline
from app.tenant.models import TenantConfig


def build_clothing_pipeline(
    tenant: TenantConfig,
    route: ModelRoute,
    builder: PromptBuilder,
) -> BasePipeline:
    """Build the existing clothing engine behind its shared adapter."""

    if not isinstance(builder, ClothingPromptBuilder):
        raise PipelineRoutingError(
            "Clothing tenant selected an incompatible prompt builder."
        )
    engine = build_pipeline(
        route.settings,
        prompt_builder=builder,
    )
    return ClothingPipeline(
        tenant_id=tenant.tenant_id,
        engine=engine,
    )


def build_wallpaper_pipeline(
    tenant: TenantConfig,
    route: ModelRoute,
    builder: PromptBuilder,
) -> BasePipeline:
    """Build the operational wallpaper pipeline and tenant providers."""

    if not isinstance(builder, WallpaperPromptBuilder):
        raise PipelineRoutingError(
            "Wallpaper tenant selected an incompatible prompt builder."
        )
    providers = build_provider_bundle(
        route.settings,
        prompt_builder=builder,
    )
    return WallpaperPipeline(
        tenant_id=tenant.tenant_id,
        settings=route.settings,
        analysis_client=providers.analysis,
        generation_client=providers.generation,
        prompt_builder=builder,
    )
