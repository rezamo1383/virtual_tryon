"""Tenant-aware dispatch to domain pipelines."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import PipelineRoutingError
from app.models.request_models import GenerationRequest
from app.pipelines.base import BasePipeline
from app.preprocessing.preprocessing_models import PreprocessingResult
from app.prompts.base import PromptBuilder
from app.routing.model_router import ModelRoute, ModelRouter
from app.routing.prompt_router import PromptRouter
from app.tenant.models import TenantConfig

PipelineFactory = Callable[
    [TenantConfig, ModelRoute, PromptBuilder],
    BasePipeline,
]


class TaskRouter:
    """Dispatch requests using tenant configuration, never client task input."""

    def __init__(
        self,
        *,
        settings: Settings,
        model_router: ModelRouter,
        prompt_router: PromptRouter,
    ) -> None:
        self._settings = settings
        self._model_router = model_router
        self._prompt_router = prompt_router
        self._pipelines: dict[str, BasePipeline] = {}
        self._factories: dict[str, PipelineFactory] = {}
        self._lock = asyncio.Lock()

    def register_factory(
        self,
        pipeline_name: str,
        factory: PipelineFactory,
    ) -> None:
        """Register a domain factory without changing router infrastructure."""

        self._factories[pipeline_name] = factory

    def register(
        self,
        tenant_id: str,
        pipeline: BasePipeline,
    ) -> None:
        """Register an already-built pipeline, primarily for compatibility."""

        self._pipelines[tenant_id] = pipeline

    async def dispatch(
        self,
        tenant: TenantConfig,
        request: GenerationRequest,
    ) -> BaseModel:
        """Resolve and run the pipeline owned by the authenticated tenant."""

        pipeline = await self.get_pipeline(tenant)
        return await pipeline.run(request)

    async def preprocess(
        self,
        tenant: TenantConfig,
        request: GenerationRequest,
        job_directory: Path,
        *,
        human_parsing_enabled: bool | None = None,
    ) -> PreprocessingResult:
        """Invoke an optional preprocessing capability through the router."""

        pipeline = await self.get_pipeline(tenant)
        operation = getattr(pipeline, "preprocess", None)
        if operation is None:
            raise PipelineRoutingError(
                f"Pipeline '{tenant.pipeline}' does not expose preprocessing."
            )
        result = await operation(
            request,
            job_directory,
            human_parsing_enabled=human_parsing_enabled,
        )
        if not isinstance(result, PreprocessingResult):
            raise PipelineRoutingError(
                "Pipeline returned an invalid preprocessing result."
            )
        return result

    async def get_pipeline(self, tenant: TenantConfig) -> BasePipeline:
        """Return one lazily constructed, tenant-scoped pipeline singleton."""

        existing = self._pipelines.get(tenant.tenant_id)
        if existing is not None:
            if existing.pipeline_name != tenant.pipeline:
                raise PipelineRoutingError(
                    "Registered pipeline conflicts with tenant configuration."
                )
            return existing
        async with self._lock:
            existing = self._pipelines.get(tenant.tenant_id)
            if existing is not None:
                return existing
            route = self._model_router.resolve(tenant)
            pipeline = self._build_pipeline(tenant, route)
            self._pipelines[tenant.tenant_id] = pipeline
            return pipeline

    def _build_pipeline(
        self,
        tenant: TenantConfig,
        route: ModelRoute,
    ) -> BasePipeline:
        builder = self._prompt_router.resolve(
            tenant.pipeline,
            tenant.prompt_profile,
        )
        factory = self._factories.get(tenant.pipeline)
        if factory is None:
            raise PipelineRoutingError(
                f"No pipeline factory is registered for '{tenant.pipeline}'."
            )
        return factory(tenant, route, builder)

    async def aclose(self) -> None:
        """Close every unique cached pipeline."""

        unique = {id(item): item for item in self._pipelines.values()}
        if unique:
            await asyncio.gather(
                *(pipeline.aclose() for pipeline in unique.values())
            )
        self._pipelines.clear()
