"""Composition root for shared multi-domain infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.pipelines.base import BasePipeline
from app.pipelines.clothing import ClothingPipeline
from app.pipelines.factory import (
    build_clothing_pipeline,
    build_wallpaper_pipeline,
)
from app.routing.model_router import ModelRouter
from app.routing.prompt_router import PromptRouter
from app.routing.task_router import TaskRouter
from app.services.pipeline import VirtualTryOnPipeline
from app.tenant.resolver import TenantResolver
from app.tenant.store import TenantConfigStore


@dataclass
class PlatformRuntime:
    """Application-scoped services shared by FastAPI and CLI."""

    settings: Settings
    tenant_store: TenantConfigStore
    tenant_resolver: TenantResolver
    model_router: ModelRouter
    prompt_router: PromptRouter
    task_router: TaskRouter

    async def aclose(self) -> None:
        await self.task_router.aclose()


def build_runtime(
    settings: Settings,
    *,
    legacy_pipeline: VirtualTryOnPipeline | None = None,
) -> PlatformRuntime:
    """Build tenant and routing infrastructure around optional legacy engine."""

    store = TenantConfigStore(
        config_path=settings.tenant_config_path,
        default_tenant_id=settings.default_tenant_id,
        fallback_analysis_provider=settings.analysis_provider,
        fallback_generation_provider=settings.tryon_provider,
    )
    resolver = TenantResolver(
        store,
        default_tenant_id=settings.default_tenant_id,
        authentication_required=settings.tenant_auth_required,
    )
    model_router = ModelRouter(settings)
    prompt_router = PromptRouter()
    task_router = TaskRouter(
        settings=settings,
        model_router=model_router,
        prompt_router=prompt_router,
    )
    task_router.register_factory("clothing", build_clothing_pipeline)
    task_router.register_factory("wallpaper", build_wallpaper_pipeline)
    if legacy_pipeline is not None:
        default = store.get(settings.default_tenant_id)
        if default is not None and default.pipeline == "clothing":
            pipeline: BasePipeline = ClothingPipeline(
                tenant_id=default.tenant_id,
                engine=legacy_pipeline,
            )
            task_router.register(default.tenant_id, pipeline)
    return PlatformRuntime(
        settings=settings,
        tenant_store=store,
        tenant_resolver=resolver,
        model_router=model_router,
        prompt_router=prompt_router,
        task_router=task_router,
    )
