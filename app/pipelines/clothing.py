"""Adapter exposing the existing try-on engine as a shared pipeline."""

from __future__ import annotations

from pathlib import Path

from app.models.request_models import (
    ClothingOptions,
    GenerationRequest,
    PreparedTryOnRequest,
    TryOnRequest,
)
from app.models.prepared_garment_models import GarmentPreparationResult
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
        result = await self.engine.run(legacy, job_id=request.job_id)
        routed = result.model_copy(
            update={
                "tenant_id": self.tenant_id,
                "pipeline": self.pipeline_name,
            }
        )
        job_directory = self.engine.settings.output_directory / result.job_id
        self.engine.output_manager.write_result(job_directory, routed)
        return routed

    async def run_multi(
        self,
        *,
        person_image: Path,
        garment_images: list[Path],
        garment_types: list[str],
        options: ClothingOptions,
        job_id: str | None = None,
    ) -> TryOnJobResult:
        """Run a complete outfit through one multi-reference generation call."""

        result = await self.engine.run_multi(
            person_image=person_image,
            garment_images=garment_images,
            garment_types=garment_types,
            options=options,
            job_id=job_id,
        )
        routed = result.model_copy(
            update={
                "tenant_id": self.tenant_id,
                "pipeline": self.pipeline_name,
            }
        )
        job_directory = self.engine.settings.output_directory / result.job_id
        self.engine.output_manager.write_result(job_directory, routed)
        return routed

    async def run_prepared(
        self,
        request: PreparedTryOnRequest,
    ) -> TryOnJobResult:
        """Run the tenant's optimized prepared-product path."""

        if request.tenant_id != self.tenant_id:
            raise ValueError("Prepared request tenant does not match pipeline.")
        return await self.engine.run_prepared(request)

    async def prepare_garment(
        self,
        *,
        product_id: str,
        garment_image: Path,
        force: bool = False,
    ) -> GarmentPreparationResult:
        """Prepare one tenant-owned product independently."""

        return await self.engine.prepare_garment(
            tenant_id=self.tenant_id,
            product_id=product_id,
            garment_image=garment_image,
            force=force,
        )

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
