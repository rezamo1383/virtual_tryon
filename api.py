"""FastAPI transport for the tenant-aware visual generation platform."""

from __future__ import annotations

import json
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AIPlatformError,
    InputValidationError,
    PipelineRoutingError,
    TenantAuthenticationError,
)
from app.core.logging_config import configure_logging
from app.core.runtime import PlatformRuntime, build_runtime
from app.models.request_models import ClothingOptions, GenerationRequest
from app.preprocessing.preprocessing_exceptions import PreprocessingError
from app.preprocessing.preprocessing_models import PreprocessingResult
from app.services.multi_garment_tryon import (
    LabeledGarment,
    run_multi_garment_tryon,
)
from app.services.pipeline import VirtualTryOnPipeline, build_pipeline
from app.tenant.models import TenantConfig
from app.utils.file_utils import ensure_within, secure_temp_name
from app.utils.hashing import create_job_id
from app.utils.json_utils import read_json

JOB_ID_RE = re.compile(r"^job_\d{8}_[0-9a-f]{6}$")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory with application-scoped tenant routing."""

    effective_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(instance: FastAPI) -> AsyncIterator[None]:
        configure_logging(
            effective_settings.log_level,
            effective_settings.log_directory,
        )
        effective_settings.temp_directory.mkdir(parents=True, exist_ok=True)
        effective_settings.output_directory.mkdir(parents=True, exist_ok=True)
        legacy_pipeline = build_pipeline(effective_settings)
        await legacy_pipeline.warmup()
        runtime = build_runtime(
            effective_settings,
            legacy_pipeline=legacy_pipeline,
        )
        instance.state.pipeline = legacy_pipeline
        instance.state.runtime = runtime
        yield
        await runtime.aclose()

    application = FastAPI(
        title="Multi-Domain AI Generation API",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.state.settings = effective_settings

    @application.exception_handler(AIPlatformError)
    async def platform_error_handler(
        request: Request,
        exc: AIPlatformError,
    ) -> JSONResponse:
        if isinstance(exc, TenantAuthenticationError):
            status_code = 401
        elif isinstance(
            exc,
            (InputValidationError, PreprocessingError, PipelineRoutingError),
        ):
            status_code = 422
        else:
            status_code = 502
        return JSONResponse(
            status_code=status_code,
            content={"detail": str(exc)},
        )

    @application.get("/health")
    async def health(request: Request) -> dict[str, object]:
        current: Settings = request.app.state.settings
        runtime = _runtime_for_request(request)
        return {
            "status": "ok",
            "platform": "multi-domain",
            "tenant_count": len(runtime.tenant_store.all()),
            "default_tenant_id": current.default_tenant_id,
            "tenant_auth_required": current.tenant_auth_required,
            "mock_qwen": current.use_mock_qwen,
            "mock_tryon": current.use_mock_tryon,
            "analysis_provider": current.analysis_provider,
            "tryon_provider": current.tryon_provider,
            "person_analysis_enabled": current.person_analysis_enabled,
            "local_preprocessing_enabled": (
                current.local_preprocessing_enabled
            ),
            "reject_unsuitable_person_images": (
                current.reject_unsuitable_person_images
            ),
        }

    @application.post("/api/v1/generate")
    async def generate(
        request: Request,
        source_image: UploadFile = File(...),
        reference_image: UploadFile = File(...),
        options: str = Form("{}"),
    ) -> dict[str, object]:
        """Authenticate a tenant and dispatch a domain-neutral request."""

        current: Settings = request.app.state.settings
        runtime = _runtime_for_request(request)
        tenant = _resolve_tenant(request, runtime)
        upload_directory = _create_upload_directory(current)
        try:
            source_path = await _save_upload(
                source_image,
                upload_directory,
                current.max_image_size_mb,
            )
            reference_path = await _save_upload(
                reference_image,
                upload_directory,
                current.max_image_size_mb,
            )
            result = await runtime.task_router.dispatch(
                tenant,
                GenerationRequest(
                    source_image=source_path,
                    reference_image=reference_path,
                    options=_parse_options(options),
                ),
            )
            return _model_response(result)
        finally:
            shutil.rmtree(upload_directory, ignore_errors=True)

    @application.post("/api/v1/tryon")
    async def tryon(
        request: Request,
        person_image: UploadFile = File(...),
        garment_image: UploadFile | None = File(None),
        garment_images: list[UploadFile] | None = File(None),
        garment_types: str = Form("[]"),
        product_title: str | None = Form(None),
        colors: list[str] | None = Form(None),
        candidates_per_color: int = Form(2, ge=1, le=8),
        max_retries: int = Form(1, ge=0, le=5),
        preserve_face: bool = Form(True),
        preserve_pose: bool = Form(True),
        preserve_background: bool = Form(True),
    ) -> dict[str, object]:
        """Apply one or more labeled garment references to a person."""

        current: Settings = request.app.state.settings
        runtime = _runtime_for_request(request)
        tenant = _resolve_tenant(request, runtime)
        if tenant.pipeline != "clothing":
            raise PipelineRoutingError(
                "The try-on endpoint requires a clothing tenant."
            )
        uploads = list(garment_images or [])
        if garment_image is not None:
            if uploads:
                raise InputValidationError(
                    "Send garment_image or garment_images, not both."
                )
            uploads.append(garment_image)
        if not uploads:
            raise InputValidationError(
                "At least one garment image is required."
            )
        if len(uploads) > 8:
            raise InputValidationError(
                "A maximum of 8 garment images is supported."
            )
        labels = _parse_garment_types(
            garment_types,
            count=len(uploads),
            fallback=product_title,
        )
        parsed_colors = _parse_colors(colors) if colors else ["original"]
        if len(uploads) > 1 and not (
            len(parsed_colors) == 1
            and parsed_colors[0].casefold() == "original"
        ):
            raise InputValidationError(
                "Multi-garment try-on currently preserves original colors only."
            )

        upload_directory = _create_upload_directory(current)
        try:
            person_path = await _save_upload(
                person_image,
                upload_directory,
                current.max_image_size_mb,
            )
            garment_paths = [
                await _save_upload(
                    upload,
                    upload_directory,
                    current.max_image_size_mb,
                )
                for upload in uploads
            ]
            multi_result = await run_multi_garment_tryon(
                runtime=runtime,
                tenant=tenant,
                person_image=person_path,
                garments=[
                    LabeledGarment(path, label)
                    for path, label in zip(
                        garment_paths,
                        labels,
                        strict=True,
                    )
                ],
                options=ClothingOptions(
                    colors=parsed_colors,
                    candidates_per_color=candidates_per_color,
                    max_retries=max_retries,
                    preserve_face=preserve_face,
                    preserve_pose=preserve_pose,
                    preserve_background=preserve_background,
                ),
            )
            response = _model_response(multi_result.result)
            response["applied_items"] = list(multi_result.applied_items)
            response["stage_job_ids"] = list(multi_result.stage_job_ids)
            return response
        finally:
            shutil.rmtree(upload_directory, ignore_errors=True)

    @application.post("/v1/preprocess")
    async def preprocess_endpoint(
        request: Request,
        person_image: UploadFile = File(...),
        garment_image: UploadFile = File(...),
        human_parsing_enabled: bool = Form(True),
    ) -> dict[str, object]:
        """Route local preprocessing without exposing absolute paths."""

        current: Settings = request.app.state.settings
        runtime = _runtime_for_request(request)
        tenant = _resolve_tenant(request, runtime)
        upload_directory = _create_upload_directory(current)
        job_id = create_job_id()
        job_directory = ensure_within(
            current.output_directory / job_id,
            current.output_directory,
        )
        job_directory.mkdir(parents=True, exist_ok=False)
        try:
            person_path = await _save_upload(
                person_image,
                upload_directory,
                current.max_image_size_mb,
            )
            garment_path = await _save_upload(
                garment_image,
                upload_directory,
                current.max_image_size_mb,
            )
            result = await runtime.task_router.preprocess(
                tenant,
                GenerationRequest(
                    source_image=person_path,
                    reference_image=garment_path,
                    options=ClothingOptions(candidates_per_color=1).model_dump(),
                ),
                job_directory,
                human_parsing_enabled=human_parsing_enabled,
            )
            accepted = (
                result.person.validation.accepted
                and result.garment.validation.accepted
            )
            return _preprocessing_api_response(
                job_id,
                result,
                job_directory,
                accepted=accepted,
            )
        finally:
            shutil.rmtree(upload_directory, ignore_errors=True)

    @application.get("/api/v1/jobs/{job_id}")
    async def job_status(
        job_id: str,
        request: Request,
    ) -> dict[str, object]:
        runtime = _runtime_for_request(request)
        tenant = _resolve_tenant(request, runtime)
        data = _load_job(
            job_id,
            request.app.state.settings,
            tenant=tenant,
        )
        return {
            "job_id": data.get("job_id"),
            "tenant_id": data.get("tenant_id"),
            "pipeline": data.get("pipeline"),
            "status": data.get("status"),
            "started_at": data.get("started_at"),
            "completed_at": data.get("completed_at"),
        }

    @application.get("/api/v1/jobs/{job_id}/results")
    async def job_results(
        job_id: str,
        request: Request,
    ) -> dict[str, object]:
        runtime = _runtime_for_request(request)
        tenant = _resolve_tenant(request, runtime)
        return _load_job(
            job_id,
            request.app.state.settings,
            tenant=tenant,
        )

    @application.get("/api/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
    async def job_artifact(
        job_id: str,
        artifact_path: str,
        request: Request,
    ) -> FileResponse:
        """Return only a tenant-owned final image referenced by job results."""

        runtime = _runtime_for_request(request)
        tenant = _resolve_tenant(request, runtime)
        current: Settings = request.app.state.settings
        data = _load_job(job_id, current, tenant=tenant)
        normalized = PurePosixPath(artifact_path.replace("\\", "/"))
        requested = normalized.as_posix()
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or requested not in _final_artifact_paths(data)
        ):
            raise HTTPException(status_code=404, detail="Artifact not found")
        job_directory = ensure_within(
            current.output_directory / job_id,
            current.output_directory,
        )
        try:
            target = ensure_within(job_directory / Path(*normalized.parts), job_directory)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail="Artifact not found",
            ) from exc
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        media_type = media_types.get(target.suffix.lower())
        if media_type is None or not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(
            target,
            media_type=media_type,
            filename=target.name,
            headers={"Cache-Control": "private, max-age=300"},
        )

    return application


def _runtime_for_request(request: Request) -> PlatformRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if isinstance(runtime, PlatformRuntime):
        return runtime
    settings: Settings = request.app.state.settings
    legacy = getattr(request.app.state, "pipeline", None)
    runtime = build_runtime(
        settings,
        legacy_pipeline=(
            legacy if isinstance(legacy, VirtualTryOnPipeline) else None
        ),
    )
    request.app.state.runtime = runtime
    return runtime


def _resolve_tenant(
    request: Request,
    runtime: PlatformRuntime,
) -> TenantConfig:
    return runtime.tenant_resolver.resolve(_extract_api_key(request))


def _extract_api_key(request: Request) -> str | None:
    direct = request.headers.get("x-api-key", "").strip()
    authorization = request.headers.get("authorization", "").strip()
    bearer = ""
    if authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    if direct and bearer and direct != bearer:
        raise TenantAuthenticationError(
            "Conflicting tenant API credentials."
        )
    return direct or bearer or None


def _create_upload_directory(settings: Settings) -> Path:
    directory = settings.temp_directory / f"upload_{secure_temp_name('')}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


async def _save_upload(
    upload: UploadFile,
    directory: Path,
    maximum_mb: int,
) -> Path:
    extension = Path(upload.filename or "").suffix.lower()
    path = directory / secure_temp_name(extension or ".bin")
    maximum = maximum_mb * 1024 * 1024
    total = 0
    with path.open("wb") as destination:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > maximum:
                raise InputValidationError(
                    f"Uploaded file exceeds the {maximum_mb} MB limit."
                )
            destination.write(chunk)
    await upload.close()
    return path


def _parse_options(value: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            "options must be a valid JSON object."
        ) from exc
    if not isinstance(decoded, dict):
        raise InputValidationError("options must be a JSON object.")
    if "task_type" in decoded:
        raise InputValidationError(
            "task_type is tenant-controlled and must not be sent."
        )
    return decoded


def _parse_garment_types(
    value: str,
    *,
    count: int,
    fallback: str | None,
) -> list[str]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            "garment_types must be a valid JSON array."
        ) from exc
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) for item in decoded
    ):
        raise InputValidationError(
            "garment_types must be a JSON array of strings."
        )
    if not decoded and count == 1:
        decoded = [fallback or "garment"]
    if len(decoded) != count:
        raise InputValidationError(
            "Each garment image must have exactly one garment type."
        )
    normalized = [" ".join(item.split()) for item in decoded]
    if any(not item for item in normalized):
        raise InputValidationError("Garment types cannot be empty.")
    if any(len(item) > 80 for item in normalized):
        raise InputValidationError(
            "Each garment type must be 80 characters or fewer."
        )
    return normalized


def _parse_colors(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise InputValidationError(
                    "Invalid JSON colors form field."
                ) from exc
            if not isinstance(decoded, list) or not all(
                isinstance(item, str) for item in decoded
            ):
                raise InputValidationError(
                    "colors JSON must be an array of strings."
                )
            parsed.extend(decoded)
        else:
            parsed.extend(item for item in stripped.split(",") if item)
    if not parsed:
        raise InputValidationError("At least one color is required.")
    return parsed


def _load_job(
    job_id: str,
    settings: Settings,
    *,
    tenant: TenantConfig,
) -> dict[str, object]:
    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    job_directory = ensure_within(
        settings.output_directory / job_id,
        settings.output_directory,
    )
    result_path = job_directory / "results.json"
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="Job not found")
    data = read_json(result_path)
    owner = data.get("tenant_id")
    if owner is None and settings.tenant_auth_required:
        raise HTTPException(status_code=404, detail="Job not found")
    if owner is not None and owner != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return data


def _model_response(result: BaseModel) -> dict[str, object]:
    return result.model_dump(mode="json")


def _final_artifact_paths(data: dict[str, object]) -> set[str]:
    """Collect normalized final outputs explicitly referenced by a job."""

    paths: set[str] = set()
    output = data.get("output")
    if isinstance(output, str) and output:
        paths.add(PurePosixPath(output.replace("\\", "/")).as_posix())
    results = data.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            value = item.get("output")
            if isinstance(value, str) and value:
                paths.add(
                    PurePosixPath(value.replace("\\", "/")).as_posix()
                )
    return paths


def _preprocessing_api_response(
    job_id: str,
    result: PreprocessingResult,
    job_directory: Path,
    *,
    accepted: bool,
) -> dict[str, object]:
    """Build a client-safe response containing relative artifact names only."""

    artifact_paths = {
        "person_normalized": result.person.normalized_image_path,
        "person_transparent": result.person.transparent_image_path,
        "person_foreground_mask": result.person.foreground_mask_path,
        "replace_mask": result.person.replace_mask_path,
        "preserve_mask": result.person.preserve_mask_path,
        "pose_debug": result.person.pose_debug_path,
        "parsing_debug": result.person.parsing_debug_path,
        "garment_normalized": result.garment.normalized_image_path,
        "garment_transparent": result.garment.transparent_image_path,
        "garment_mask": result.garment.garment_mask_path,
    }
    artifacts: dict[str, str | None] = {}
    for name, path in artifact_paths.items():
        if path is None:
            artifacts[name] = None
            continue
        safe = ensure_within(path, job_directory)
        artifacts[name] = safe.relative_to(
            job_directory.resolve()
        ).as_posix()
    return {
        "job_id": job_id,
        "status": "accepted" if accepted else "rejected",
        "device": result.device,
        "degraded_mode": result.degraded_mode,
        "validation": {
            "person": result.person.validation.model_dump(mode="json"),
            "garment": result.garment.validation.model_dump(mode="json"),
        },
        "pose": result.person.pose.model_dump(mode="json"),
        "artifacts": artifacts,
        "processing_time_ms": result.processing_time_ms,
        "warnings": result.warnings,
    }


app = create_app()
