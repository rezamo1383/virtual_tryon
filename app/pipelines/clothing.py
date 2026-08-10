"""Adapter exposing the existing try-on engine as a shared pipeline."""

from __future__ import annotations

from pathlib import Path

from app.models.request_models import (
    ClothingOptions,
    GenerationRequest,
    TryOnRequest,
)
from app.models.result_models import TryOnJobResult
from app.pipelines.base import BasePipeline
from app.preprocessing.preprocessing_models import PreprocessingResult
from app.services.pipeline import VirtualTryOnPipeline
from app.validators.clothing import ClothingValidator


class ClothingPipeline(BasePipeline):
    """Map shared inputs to the unchanged VirtualTryOnPipeline engine."""

    pipeline_name = "clothing"

    def __init__(
        self,
        *,
        tenant_id: str,
        engine: VirtualTryOnPipeline,
    ) -> None:
        self.tenant_id = tenant_id
        self.engine = engine
        self._validator = ClothingValidator(engine.settings)

    async def run(self, request: GenerationRequest) -> TryOnJobResult:
        options = ClothingOptions.model_validate(request.options)
        legacy = self._validator.validate(
            TryOnRequest(
                person_image=request.source_image,
                garment_image=request.reference_image,
                **options.model_dump(),
            )
        )
        result = await self.engine.run(legacy)
        routed = result.model_copy(
            update={
                "tenant_id": self.tenant_id,
                "pipeline": self.pipeline_name,
            }
        )
        job_directory = self.engine.settings.output_directory / result.job_id
        self.engine.output_manager.write_result(job_directory, routed)
        return routed

    async def preprocess(
        self,
        request: GenerationRequest,
        job_directory: Path,
        *,
        human_parsing_enabled: bool | None = None,
    ) -> PreprocessingResult:
        """Expose clothing-only local preprocessing through router capability."""

        options = ClothingOptions.model_validate(request.options)
        legacy = self._validator.validate(
            TryOnRequest(
                person_image=request.source_image,
                garment_image=request.reference_image,
                **options.model_dump(),
            )
        )
        return await self.engine.preprocess_inputs(
            legacy.person_image,
            legacy.garment_image,
            job_directory,
            human_parsing_enabled=human_parsing_enabled,
        )

    async def aclose(self) -> None:
        await self.engine.aclose()
