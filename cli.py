"""Typer CLI for the tenant-aware visual generation platform."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.exceptions import AIPlatformError
from app.core.logging_config import configure_logging
from app.core.runtime import build_runtime
from app.models.request_models import (
    ClothingOptions,
    GenerationRequest,
    TryOnRequest,
    WallpaperOptions,
)
from app.models.result_models import TryOnJobResult
from app.models.wallpaper_models import WallpaperJobResult
from app.preprocessing.image_preprocessor import LocalImagePreprocessor
from app.services.input_validator import InputValidator
from app.services.multi_garment_tryon import (
    LabeledGarment,
    MultiGarmentTryOnResult,
    run_multi_garment_tryon,
)
from app.services.pipeline import build_pipeline

app = typer.Typer(
    name="ai-generation",
    help="Tenant-aware multi-domain visual generation CLI.",
    no_args_is_help=True,
)


def expand_color_arguments(arguments: list[str]) -> list[str]:
    """Allow both ``--colors red blue`` and repeated ``--colors red`` forms."""

    expanded: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value != "--colors":
            expanded.append(value)
            index += 1
            continue
        expanded.append(value)
        index += 1
        if index >= len(arguments) or arguments[index].startswith("--"):
            continue
        expanded.append(arguments[index])
        index += 1
        while index < len(arguments) and not arguments[index].startswith("--"):
            expanded.extend(["--colors", arguments[index]])
            index += 1
    return expanded


def _settings() -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_directory)
    return settings


def _request_from_options(
    *,
    person: Path | None,
    garment: Path | None,
    product_title: str | None,
    colors: list[str] | None,
    candidates_per_color: int | None,
    max_retries: int | None,
    request_json: Path | None,
    settings: Settings,
) -> TryOnRequest:
    if request_json:
        try:
            data = json.loads(request_json.read_text(encoding="utf-8"))
            return TryOnRequest.model_validate(data)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise typer.BadParameter(f"Invalid request JSON: {exc}") from exc
    if person is None or garment is None:
        raise typer.BadParameter(
            "--person and --garment are required."
        )
    return TryOnRequest(
        person_image=person,
        garment_image=garment,
        product_title=product_title,
        colors=colors or ["original"],
        candidates_per_color=(
            candidates_per_color
            if candidates_per_color is not None
            else settings.candidates_per_color
        ),
        max_retries=(
            max_retries
            if max_retries is not None
            else settings.max_generation_retries
        ),
    )


def labeled_garments_from_options(
    garments: list[Path] | None,
    garment_types: list[str] | None,
    *,
    product_title: str | None = None,
) -> list[LabeledGarment]:
    """Pair repeated garment paths with their selected item labels."""

    paths = garments or []
    if not paths:
        raise typer.BadParameter("At least one --garment is required.")
    if len(paths) > 8:
        raise typer.BadParameter("A maximum of 8 --garment options is supported.")
    labels = garment_types or []
    if not labels and len(paths) == 1:
        labels = [product_title or "garment"]
    if len(labels) != len(paths):
        raise typer.BadParameter(
            "Repeat --garment-type exactly once for each --garment."
        )
    normalized = [" ".join(label.split()) for label in labels]
    if any(not label for label in normalized):
        raise typer.BadParameter("Garment types cannot be empty.")
    if any(len(label) > 80 for label in normalized):
        raise typer.BadParameter(
            "Each --garment-type must be 80 characters or fewer."
        )
    return [
        LabeledGarment(path, label)
        for path, label in zip(paths, normalized, strict=True)
    ]


def _clothing_options(
    *,
    colors: list[str] | None,
    candidates_per_color: int | None,
    max_retries: int | None,
    settings: Settings,
) -> ClothingOptions:
    return ClothingOptions(
        colors=colors or ["original"],
        candidates_per_color=(
            candidates_per_color
            if candidates_per_color is not None
            else settings.candidates_per_color
        ),
        max_retries=(
            max_retries
            if max_retries is not None
            else settings.max_generation_retries
        ),
    )


async def _run_pipeline(
    request: TryOnRequest,
    settings: Settings,
    *,
    tenant_id: str | None = None,
) -> None:
    """Backward-compatible clothing execution through the Task Router."""

    runtime = build_runtime(settings)
    try:
        tenant = runtime.tenant_resolver.resolve_for_cli(
            tenant_id=tenant_id,
            pipeline="clothing",
        )
        result = await runtime.task_router.dispatch(
            tenant,
            request.to_generation_request(),
        )
    finally:
        await runtime.aclose()
    if not isinstance(result, TryOnJobResult):
        raise RuntimeError("Clothing pipeline returned an invalid result.")
    _print_tryon_result(result, settings)


async def _run_multi_pipeline(
    *,
    person: Path,
    garments: list[LabeledGarment],
    options: ClothingOptions,
    settings: Settings,
    tenant_id: str | None = None,
) -> MultiGarmentTryOnResult:
    runtime = build_runtime(settings)
    try:
        tenant = runtime.tenant_resolver.resolve_for_cli(
            tenant_id=tenant_id,
            pipeline="clothing",
        )
        multi_result = await run_multi_garment_tryon(
            runtime=runtime,
            tenant=tenant,
            person_image=person,
            garments=garments,
            options=options,
        )
    finally:
        await runtime.aclose()
    _print_tryon_result(multi_result.result, settings)
    typer.echo("Applied items: " + " -> ".join(multi_result.applied_items))
    typer.echo("Stage jobs: " + ", ".join(multi_result.stage_job_ids))
    return multi_result


def _print_tryon_result(
    result: TryOnJobResult,
    settings: Settings,
) -> None:
    typer.secho(f"Job ID: {result.job_id}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Tenant: {result.tenant_id or settings.default_tenant_id}")
    typer.echo("Pipeline: clothing")
    typer.echo(f"Status: {result.status}")
    typer.echo(
        f"Results: {(settings.output_directory / result.job_id / 'results.json').resolve()}"
    )
    for item in result.results:
        marker = "accepted" if item.accepted else "best-effort"
        typer.echo(
            f"  {item.color}: {item.output} | score={item.score:.3f} | {marker}"
        )


async def _run_wallpaper(
    request: GenerationRequest,
    settings: Settings,
    *,
    tenant_id: str | None,
) -> None:
    runtime = build_runtime(settings)
    try:
        tenant = runtime.tenant_resolver.resolve_for_cli(
            tenant_id=tenant_id,
            pipeline="wallpaper",
        )
        result = await runtime.task_router.dispatch(tenant, request)
    finally:
        await runtime.aclose()
    if not isinstance(result, WallpaperJobResult):
        raise RuntimeError("Wallpaper pipeline returned an invalid result.")
    typer.secho(f"Job ID: {result.job_id}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Tenant: {result.tenant_id}")
    typer.echo("Pipeline: wallpaper")
    typer.echo(f"Status: {result.status}")
    if result.output:
        typer.echo(
            "Output: "
            + str(
                (
                    settings.output_directory
                    / result.job_id
                    / result.output
                ).resolve()
            )
        )
    if result.score is not None:
        typer.echo(f"Score: {result.score:.3f}")
    typer.echo(f"Accepted: {result.accepted}")
    if result.rejection_reason:
        typer.echo(f"Rejection reason: {result.rejection_reason}")


@app.command()
def run(
    person: Path | None = typer.Option(None, "--person", exists=False),
    garment: list[Path] | None = typer.Option(
        None,
        "--garment",
        exists=False,
        help="Garment image; repeat for a multi-item look.",
    ),
    garment_type: list[str] | None = typer.Option(
        None,
        "--garment-type",
        help="Item selected from each garment image; repeat in matching order.",
    ),
    product_title: str | None = typer.Option(
        None,
        "--product-title",
        help="Backward-compatible label for a single garment.",
    ),
    colors: list[str] | None = typer.Option(None, "--colors"),
    candidates_per_color: int | None = typer.Option(
        None, "--candidates-per-color", min=1, max=8
    ),
    max_retries: int | None = typer.Option(None, "--max-retries", min=0, max=5),
    request_json: Path | None = typer.Option(None, "--request-json"),
    tenant: str | None = typer.Option(None, "--tenant"),
) -> None:
    """Backward-compatible alias for the clothing pipeline."""

    settings = _settings()
    try:
        if request_json is not None:
            request = _request_from_options(
                person=person,
                garment=garment[0] if garment else None,
                product_title=product_title,
                colors=colors,
                candidates_per_color=candidates_per_color,
                max_retries=max_retries,
                request_json=request_json,
                settings=settings,
            )
            asyncio.run(
                _run_pipeline(
                    request,
                    settings,
                    tenant_id=tenant,
                )
            )
        else:
            if person is None:
                raise typer.BadParameter("--person is required.")
            labeled_garments = labeled_garments_from_options(
                garment,
                garment_type,
                product_title=product_title,
            )
            asyncio.run(
                _run_multi_pipeline(
                    person=person,
                    garments=labeled_garments,
                    options=_clothing_options(
                        colors=colors,
                        candidates_per_color=candidates_per_color,
                        max_retries=max_retries,
                        settings=settings,
                    ),
                    settings=settings,
                    tenant_id=tenant,
                )
            )
    except (AIPlatformError, ValidationError, OSError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def clothing(
    person: Path = typer.Option(..., "--person"),
    garment: list[Path] | None = typer.Option(
        None,
        "--garment",
        help="Garment image; repeat for a multi-item look.",
    ),
    garment_type: list[str] | None = typer.Option(
        None,
        "--garment-type",
        help="Item selected from each garment image; repeat in matching order.",
    ),
    product_title: str | None = typer.Option(
        None,
        "--product-title",
        help="Backward-compatible label for a single garment.",
    ),
    colors: list[str] | None = typer.Option(None, "--colors", hidden=True),
    candidates_per_color: int | None = typer.Option(
        None,
        "--candidates-per-color",
        min=1,
        max=8,
    ),
    max_retries: int | None = typer.Option(
        None,
        "--max-retries",
        min=0,
        max=5,
    ),
    tenant: str | None = typer.Option(None, "--tenant"),
) -> None:
    """Run the clothing product through tenant-aware routing."""

    settings = _settings()
    try:
        labeled_garments = labeled_garments_from_options(
            garment,
            garment_type,
            product_title=product_title,
        )
        asyncio.run(
            _run_multi_pipeline(
                person=person,
                garments=labeled_garments,
                options=_clothing_options(
                    colors=colors,
                    candidates_per_color=candidates_per_color,
                    max_retries=max_retries,
                    settings=settings,
                ),
                settings=settings,
                tenant_id=tenant,
            )
        )
    except (AIPlatformError, ValidationError, OSError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def wallpaper(
    room: Path = typer.Option(..., "--room"),
    wallpaper_image: Path = typer.Option(..., "--wallpaper"),
    tenant: str | None = typer.Option(None, "--tenant"),
    preserve_lighting: bool = typer.Option(
        True,
        "--preserve-lighting/--no-preserve-lighting",
    ),
    candidates_per_job: int = typer.Option(
        1,
        "--candidates-per-job",
        min=1,
        max=4,
    ),
    max_retries: int = typer.Option(
        1,
        "--max-retries",
        min=0,
        max=3,
    ),
    pattern_scale: float = typer.Option(
        0.18,
        "--pattern-scale",
        min=0.03,
        max=0.75,
    ),
) -> None:
    """Generate a wallpaper visualization through the Task Router."""

    settings = _settings()
    request = GenerationRequest(
        source_image=room,
        reference_image=wallpaper_image,
        options=WallpaperOptions(
            preserve_lighting=preserve_lighting,
            candidates_per_job=candidates_per_job,
            max_retries=max_retries,
            pattern_scale=pattern_scale,
        ).model_dump(),
    )
    try:
        asyncio.run(
            _run_wallpaper(
                request,
                settings,
                tenant_id=tenant,
            )
        )
    except (AIPlatformError, ValidationError, OSError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def validate(
    person: Path = typer.Option(..., "--person"),
    garment: Path = typer.Option(..., "--garment"),
) -> None:
    """Validate person and garment images without calling either API."""

    settings = _settings()
    request = TryOnRequest(
        person_image=person,
        garment_image=garment,
        candidates_per_color=1,
    )
    try:
        validated = InputValidator(settings).validate(request)
    except AIPlatformError as exc:
        typer.secho(f"Invalid: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho("Inputs are valid.", fg=typer.colors.GREEN)
    typer.echo(f"Person: {validated.person_image}")
    typer.echo(f"Garment: {validated.garment_image}")


@app.command()
def preprocess(
    person: Path = typer.Option(..., "--person"),
    garment: Path = typer.Option(..., "--garment"),
    output: Path = typer.Option(..., "--output"),
    disable_human_parsing: bool = typer.Option(
        False,
        "--disable-human-parsing",
    ),
) -> None:
    """Run local preprocessing without any external API call."""

    settings = _settings()
    try:
        request = InputValidator(settings).validate(
            TryOnRequest(
                person_image=person,
                garment_image=garment,
                candidates_per_color=1,
            )
        )
        output_directory = output.resolve(strict=False)
        output_directory.mkdir(parents=True, exist_ok=True)
        result = LocalImagePreprocessor(settings).preprocess(
            request.person_image,
            request.garment_image,
            output_directory,
            human_parsing_enabled=not disable_human_parsing,
        )
    except (AIPlatformError, OSError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    accepted = (
        result.person.validation.accepted
        and result.garment.validation.accepted
    )
    score = min(
        result.person.validation.score,
        result.garment.validation.score,
    )
    warnings = list(
        dict.fromkeys(
            [
                *result.warnings,
                *result.person.validation.warnings,
                *result.garment.validation.warnings,
            ]
        )
    )
    reasons = [
        *result.person.validation.rejection_reasons,
        *result.garment.validation.rejection_reasons,
    ]
    typer.echo(f"Selected device: {result.device}")
    typer.echo(f"Suitability score: {score:.3f}")
    typer.echo(f"Accepted: {str(accepted).lower()}")
    typer.echo(f"Degraded mode: {str(result.degraded_mode).lower()}")
    typer.echo("Warnings: " + (", ".join(warnings) if warnings else "none"))
    typer.echo(
        "Rejection reasons: " + (", ".join(reasons) if reasons else "none")
    )
    typer.echo("Artifacts:")
    for path in (
        result.person.normalized_image_path,
        result.person.foreground_mask_path,
        result.person.replace_mask_path,
        result.person.preserve_mask_path,
        result.garment.normalized_image_path,
        result.garment.garment_mask_path,
    ):
        typer.echo(f"  {path}")
    typer.echo(f"Processing time: {result.processing_time_ms} ms")


async def _analyze(person: Path, garment: Path, settings: Settings) -> None:
    validator = InputValidator(settings)
    request = validator.validate(
        TryOnRequest(
            person_image=person,
            garment_image=garment,
            candidates_per_color=1,
        )
    )
    pipeline = build_pipeline(settings)
    try:
        if settings.person_analysis_enabled:
            person_result, garment_result = await asyncio.gather(
                pipeline.person_analyzer.analyze(request.person_image),
                pipeline.garment_analyzer.analyze(request.garment_image),
            )
        else:
            person_result = None
            garment_result = await pipeline.garment_analyzer.analyze(
                request.garment_image
            )
    finally:
        await pipeline.aclose()
    typer.echo("Person analysis:")
    if person_result is None:
        typer.echo("Skipped (PERSON_ANALYSIS_ENABLED=false)")
    else:
        typer.echo(person_result.model_dump_json(indent=2))
    typer.echo("Garment analysis:")
    typer.echo(garment_result.model_dump_json(indent=2))


@app.command()
def analyze(
    person: Path = typer.Option(..., "--person"),
    garment: Path = typer.Option(..., "--garment"),
) -> None:
    """Analyze valid inputs without generating a try-on image."""

    settings = _settings()
    try:
        asyncio.run(_analyze(person, garment, settings))
    except AIPlatformError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("config-check")
def config_check() -> None:
    """Show provider configuration without revealing API keys."""

    settings = _settings()
    runtime = build_runtime(settings)
    typer.echo(f"Default tenant: {settings.default_tenant_id}")
    typer.echo(
        "Tenant authentication: "
        + ("required" if settings.tenant_auth_required else "compatibility mode")
    )
    typer.echo(
        "Configured tenants: "
        + ", ".join(
            f"{tenant.tenant_id}={tenant.pipeline}"
            + (
                f"[{tenant.generation_model}]"
                if tenant.generation_model
                else ""
            )
            for tenant in runtime.tenant_store.all()
        )
    )
    analysis = settings.analysis_provider
    generation = settings.tryon_provider
    if analysis == "auto":
        analysis = "mock" if settings.use_mock_qwen else "qwen"
    if generation == "auto":
        generation = "mock" if settings.use_mock_tryon else "generic"
    typer.echo(f"Analysis provider: {analysis}")
    typer.echo(f"Try-on provider: {generation}")
    typer.echo(
        "Wallpaper segmentation: "
        + settings.wallpaper_segmentation_backend
        + (
            " ("
            + settings.wallpaper_segmentation_runtime
            + ": "
            + (
                settings.wallpaper_segmentation_onnx_filename
                if settings.wallpaper_segmentation_runtime == "onnx"
                else settings.wallpaper_segmentation_model
            )
            + ")"
            if settings.wallpaper_segmentation_backend == "semantic"
            else ""
        )
    )
    typer.echo(
        "Person analysis: "
        + ("enabled" if settings.person_analysis_enabled else "disabled")
    )
    typer.echo(
        "Local preprocessing: "
        + ("enabled" if settings.local_preprocessing_enabled else "disabled")
    )
    typer.echo(
        "Preprocessing warm-up: "
        + ("enabled" if settings.preprocessing_warmup_enabled else "disabled")
    )
    typer.echo(
        "Preprocessing debug images: "
        + ("enabled" if settings.save_preprocessing_debug_images else "disabled")
    )
    if "openrouter" in {analysis, generation}:
        typer.echo(f"OpenRouter base URL: {settings.openrouter_api_base_url}")
        typer.echo(
            "OpenRouter API key: "
            + ("configured" if settings.openrouter_api_key else "MISSING")
        )
        typer.echo(
            "OpenRouter vision model: "
            + (settings.openrouter_vision_model or "MISSING")
        )
        typer.echo(
            "OpenRouter image model: "
            + (settings.openrouter_image_model or "MISSING")
        )
        required_missing = not settings.openrouter_api_key or (
            analysis == "openrouter" and not settings.openrouter_vision_model
        ) or (
            generation == "openrouter" and not settings.openrouter_image_model
        )
        if required_missing:
            raise typer.Exit(code=1)
    if "gapgpt" in {analysis, generation}:
        typer.echo(f"GapGPT base URL: {settings.gapgpt_api_base_url}")
        typer.echo(
            "GapGPT API key: "
            + ("configured" if settings.gapgpt_api_key else "MISSING")
        )
        typer.echo(
            "GapGPT vision model: "
            + (settings.gapgpt_vision_model or "MISSING")
        )
        typer.echo(
            "GapGPT image model: "
            + (settings.gapgpt_image_model or "MISSING")
        )
        typer.echo(
            "GapGPT image endpoint: "
            + settings.gapgpt_image_edit_endpoint
        )
        required_missing = not settings.gapgpt_api_key or (
            analysis == "gapgpt" and not settings.gapgpt_vision_model
        ) or (
            generation == "gapgpt" and not settings.gapgpt_image_model
        )
        if required_missing:
            raise typer.Exit(code=1)


if __name__ == "__main__":
    import sys

    app(args=expand_color_arguments(sys.argv[1:]))
