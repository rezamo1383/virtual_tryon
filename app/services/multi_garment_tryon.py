"""Shared sequential orchestration for labeled garment references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import InputValidationError, PipelineRoutingError
from app.core.runtime import PlatformRuntime
from app.models.request_models import ClothingOptions, GenerationRequest
from app.models.result_models import TryOnJobResult
from app.tenant.models import TenantConfig
from app.utils.file_utils import ensure_within


@dataclass(frozen=True)
class LabeledGarment:
    """A garment image and the single item selected from it."""

    image: Path
    garment_type: str


@dataclass(frozen=True)
class MultiGarmentTryOnResult:
    """Final result plus metadata for all executed stages."""

    result: TryOnJobResult
    applied_items: tuple[str, ...]
    stage_job_ids: tuple[str, ...]


async def run_multi_garment_tryon(
    *,
    runtime: PlatformRuntime,
    tenant: TenantConfig,
    person_image: Path,
    garments: list[LabeledGarment],
    options: ClothingOptions,
) -> MultiGarmentTryOnResult:
    """Apply each labeled item to the output of the preceding stage."""

    if tenant.pipeline != "clothing":
        raise PipelineRoutingError(
            "Multi-garment try-on requires a clothing tenant."
        )
    if not garments:
        raise InputValidationError("At least one garment image is required.")
    if len(garments) > 8:
        raise InputValidationError(
            "A maximum of 8 garment images is supported."
        )
    if len(garments) > 1 and not (
        len(options.colors) == 1
        and options.colors[0].casefold() == "original"
    ):
        raise InputValidationError(
            "Multi-garment try-on currently preserves original colors only."
        )

    source_path = person_image
    stage_job_ids: list[str] = []
    applied_items: list[str] = []
    result: TryOnJobResult | None = None
    for index, garment in enumerate(garments):
        garment_type = " ".join(garment.garment_type.split())
        if not garment_type:
            raise InputValidationError("Garment types cannot be empty.")
        if len(garment_type) > 80:
            raise InputValidationError(
                "Each garment type must be 80 characters or fewer."
            )
        final_stage = index == len(garments) - 1
        stage_options = options.model_copy(
            update={
                "product_title": garment_type,
                "candidates_per_color": (
                    options.candidates_per_color if final_stage else 1
                ),
                "max_retries": options.max_retries if final_stage else 0,
            }
        )
        dispatched = await runtime.task_router.dispatch(
            tenant,
            GenerationRequest(
                source_image=source_path,
                reference_image=garment.image,
                options=stage_options.model_dump(),
            ),
        )
        if not isinstance(dispatched, TryOnJobResult):
            raise PipelineRoutingError(
                "The clothing tenant returned an incompatible result."
            )
        result = dispatched
        stage_job_ids.append(result.job_id)
        applied_items.append(garment_type)
        if result.status in {"failed", "rejected"}:
            break
        if not result.results:
            raise PipelineRoutingError(
                "A try-on stage completed without an output image."
            )
        source_path = ensure_within(
            runtime.settings.output_directory
            / result.job_id
            / result.results[0].output,
            runtime.settings.output_directory,
        )

    if result is None:
        raise PipelineRoutingError("No try-on stage was executed.")
    return MultiGarmentTryOnResult(
        result=result,
        applied_items=tuple(applied_items),
        stage_job_ids=tuple(stage_job_ids),
    )
