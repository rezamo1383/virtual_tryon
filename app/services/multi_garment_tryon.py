"""Single-call orchestration for labeled garment references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import InputValidationError, PipelineRoutingError
from app.core.runtime import PlatformRuntime
from app.models.request_models import ClothingOptions, GenerationRequest
from app.models.result_models import TryOnJobResult
from app.tenant.models import TenantConfig


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
    job_id: str | None = None,
) -> MultiGarmentTryOnResult:
    """Apply all labeled items together in one generation call."""

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

    applied_items: list[str] = []
    for garment in garments:
        garment_type = " ".join(garment.garment_type.split())
        if not garment_type:
            raise InputValidationError("Garment types cannot be empty.")
        if len(garment_type) > 80:
            raise InputValidationError(
                "Each garment type must be 80 characters or fewer."
            )
        applied_items.append(garment_type)

    if len(garments) == 1:
        dispatched = await runtime.task_router.dispatch(
            tenant,
            GenerationRequest(
                source_image=person_image,
                reference_image=garments[0].image,
                job_id=job_id,
                options=options.model_copy(
                    update={"product_title": applied_items[0]}
                ).model_dump(),
            ),
        )
    else:
        pipeline = await runtime.task_router.get_pipeline(tenant)
        operation = getattr(pipeline, "run_multi", None)
        if operation is None:
            raise PipelineRoutingError(
                "The clothing pipeline does not support single-call multi-garment generation."
            )
        dispatched = await operation(
            person_image=person_image,
            garment_images=[item.image for item in garments],
            garment_types=applied_items,
            options=options.model_copy(
                update={"product_title": ", ".join(applied_items)}
            ),
            job_id=job_id,
        )
    if not isinstance(dispatched, TryOnJobResult):
        raise PipelineRoutingError(
            "The clothing tenant returned an incompatible result."
        )
    return MultiGarmentTryOnResult(
        result=dispatched,
        applied_items=tuple(applied_items),
        stage_job_ids=(dispatched.job_id,),
    )
