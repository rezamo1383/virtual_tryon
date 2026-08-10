from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from PIL import Image

from api import _parse_options, create_app
from app.core.config import Settings
from app.core.exceptions import InputValidationError, TenantAuthenticationError
from app.core.runtime import build_runtime
from app.models.request_models import (
    ClothingOptions,
    GenerationRequest,
    WallpaperOptions,
)
from app.models.result_models import TryOnJobResult
from app.models.wallpaper_models import WallAnalysisResult, WallpaperJobResult
from app.pipelines.wallpaper import WallpaperPipeline
from app.routing.model_router import ModelRouter
from app.routing.prompt_router import PromptRouter
from app.services.wallpaper_processing import (
    OpenCVLightingPreserver,
    SemanticWallSegmentationEngine,
    WallpaperReferencePreprocessor,
)
from app.tenant.models import TenantConfig
from app.tenant.resolver import TenantResolver
from app.tenant.store import TenantConfigStore


def _platform_settings(tmp_path: Path, **updates: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "min_image_width": 64,
        "min_image_height": 64,
        "output_directory": tmp_path / "outputs",
        "temp_directory": tmp_path / "temp",
        "log_directory": tmp_path / "logs",
        "model_cache_directory": tmp_path / "models",
        "tenant_config_path": tmp_path / "missing-tenants.json",
        "analysis_provider": "mock",
        "tryon_provider": "mock",
        "use_mock_qwen": True,
        "use_mock_tryon": True,
        "local_preprocessing_enabled": False,
        "wallpaper_segmentation_backend": "polygon",
    }
    values.update(updates)
    return Settings(**values)


def test_tenant_resolver_maps_api_key_without_task_type(
    tmp_path: Path,
) -> None:
    api_key = "tenant-secret"
    config = tmp_path / "tenants.json"
    config.write_text(
        json.dumps(
            {
                "tenants": [
                    {
                        "tenant_id": "fashion",
                        "pipeline": "clothing",
                        "api_key_sha256": hashlib.sha256(
                            api_key.encode()
                        ).hexdigest(),
                    },
                    {
                        "tenant_id": "interiors",
                        "pipeline": "wallpaper",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    store = TenantConfigStore(
        config_path=config,
        default_tenant_id="fashion",
        fallback_analysis_provider="mock",
        fallback_generation_provider="mock",
    )
    resolver = TenantResolver(
        store,
        default_tenant_id="fashion",
        authentication_required=True,
    )
    resolved = resolver.resolve(api_key)
    assert resolved.tenant_id == "fashion"
    assert resolved.pipeline == "clothing"
    with pytest.raises(TenantAuthenticationError):
        resolver.resolve("wrong")
    with pytest.raises(TenantAuthenticationError):
        resolver.resolve(None)


def test_fallback_tenants_support_wallpaper_company_alias(
    tmp_path: Path,
) -> None:
    store = TenantConfigStore(
        config_path=tmp_path / "missing-tenants.json",
        default_tenant_id="legacy-clothing",
        fallback_analysis_provider="mock",
        fallback_generation_provider="mock",
    )
    resolver = TenantResolver(
        store,
        default_tenant_id="legacy-clothing",
        authentication_required=False,
    )

    tenant = resolver.resolve_for_cli(
        tenant_id="wallpaper_company",
        pipeline="wallpaper",
    )

    assert tenant.pipeline == "wallpaper"
    assert store.get("wallpaper-demo") is not None


def test_model_and_prompt_routes_are_tenant_scoped(tmp_path: Path) -> None:
    settings = _platform_settings(tmp_path)
    tenant = TenantConfig(
        tenant_id="fashion",
        pipeline="clothing",
        analysis_provider="openrouter",
        generation_provider="gapgpt",
        analysis_model="vision-per-tenant",
        generation_model="image-per-tenant",
    )
    route = ModelRouter(settings).resolve(tenant)
    assert route.analysis_provider == "openrouter"
    assert route.generation_provider == "gapgpt"
    assert route.settings.openrouter_vision_model == "vision-per-tenant"
    assert route.settings.gapgpt_image_model == "image-per-tenant"
    assert PromptRouter().resolve("clothing", "default").domain == "clothing"
    assert PromptRouter().resolve("wallpaper", "default").domain == "wallpaper"


def test_wall_mask_removes_an_island_enclosed_by_a_larger_wall(
    tmp_path: Path,
) -> None:
    settings = _platform_settings(
        tmp_path,
        wallpaper_min_wall_region_coverage=0.001,
    )
    engine = SemanticWallSegmentationEngine(settings)
    raw = np.zeros((100, 100), dtype=np.uint8)
    raw[10:90, 10:90] = 255
    raw[30:70, 30:70] = 0
    raw[45:55, 45:55] = 255

    filtered, count = engine._filter_regions(raw)

    assert count == 1
    assert filtered[50, 50] == 0
    assert filtered[15, 15] == 255


def test_wallpaper_reference_preprocessor_removes_promotional_footer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reference.png"
    image = Image.new("RGB", (100, 100), (220, 210, 190))
    values = np.asarray(image).copy()
    values[92:] = (80, 10, 120)
    Image.fromarray(values).save(source)

    output = WallpaperReferencePreprocessor._prepare_sync(
        source,
        tmp_path / "artifacts",
    )

    with Image.open(output) as cleaned:
        assert cleaned.size == (100, 92)


def test_wallpaper_candidate_registration_corrects_small_frame_shift() -> None:
    room = np.zeros((120, 160, 3), dtype=np.uint8)
    room[15:45, 20:70] = (230, 180, 60)
    room[65:105, 85:145] = (40, 180, 220)
    transform = np.float32([[1, 0, 5], [0, 1, 7]])
    shifted = cv2.warpAffine(room, transform, (160, 120))
    wall_mask = np.zeros((120, 160), dtype=np.uint8)

    aligned = OpenCVLightingPreserver._register_to_room(
        shifted,
        room,
        wall_mask,
    )

    before = np.mean(np.abs(shifted.astype(float) - room.astype(float)))
    after = np.mean(np.abs(aligned.astype(float) - room.astype(float)))
    assert after < before * 0.25


def test_api_options_cannot_override_tenant_pipeline() -> None:
    with pytest.raises(InputValidationError, match="task_type"):
        _parse_options('{"task_type":"wallpaper"}')


@pytest.mark.asyncio
async def test_task_router_preserves_clothing_pipeline(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    settings = _platform_settings(tmp_path)
    runtime = build_runtime(settings)
    person, garment = valid_images
    try:
        tenant = runtime.tenant_resolver.resolve(None)
        result = await runtime.task_router.dispatch(
            tenant,
            GenerationRequest(
                source_image=person,
                reference_image=garment,
                options=ClothingOptions(
                    colors=["red"],
                    candidates_per_color=1,
                    max_retries=0,
                ).model_dump(),
            ),
        )
    finally:
        await runtime.aclose()
    assert isinstance(result, TryOnJobResult)
    assert result.status == "completed"
    assert result.tenant_id == "legacy-clothing"
    assert result.pipeline == "clothing"


@pytest.mark.asyncio
async def test_wallpaper_pipeline_generates_and_preserves_scene(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    settings = _platform_settings(tmp_path)
    runtime = build_runtime(settings)
    room, wallpaper = valid_images
    try:
        tenant = runtime.tenant_resolver.resolve_for_cli(
            tenant_id="wallpaper-demo",
            pipeline="wallpaper",
        )
        result = await runtime.task_router.dispatch(
            tenant,
            GenerationRequest(
                source_image=room,
                reference_image=wallpaper,
                options=WallpaperOptions().model_dump(),
            ),
        )
    finally:
        await runtime.aclose()
    assert isinstance(result, WallpaperJobResult)
    assert result.status == "completed"
    assert result.pipeline == "wallpaper"
    assert result.accepted is True
    assert result.output is not None
    job_directory = settings.output_directory / result.job_id
    final_path = job_directory / result.output
    assert final_path.is_file()
    assert (job_directory / "wallpaper" / "wall_mask.png").is_file()
    assert (
        job_directory / "wallpaper" / "wallpaper_reference_clean.png"
    ).is_file()
    assert (job_directory / "wallpaper" / "texture_perspective.png").is_file()
    assert (job_directory / "candidate_metadata.json").is_file()
    assert result.completed_stages == [
        "wall_analysis",
        "reference_preprocessing",
        "wall_segmentation",
        "perspective_estimation",
        "texture_repetition",
        "wallpaper_generation",
        "output_evaluation",
        "lighting_preservation",
    ]
    with (
        Image.open(room) as original_image,
        Image.open(final_path) as final_image,
        Image.open(job_directory / "wallpaper" / "wall_mask.png") as mask_image,
    ):
        original = np.asarray(original_image.convert("RGB"))
        final = np.asarray(final_image.convert("RGB"))
        mask = np.asarray(mask_image.convert("L"))
        np.testing.assert_array_equal(final[mask == 0], original[mask == 0])


@pytest.mark.asyncio
async def test_wallpaper_pipeline_rejects_when_no_wall_is_detected(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    class NoWallAnalyzer:
        async def analyze(self, room_image: Path) -> WallAnalysisResult:
            return WallAnalysisResult(
                wall_detected=False,
                confidence=0.95,
                wall_count=0,
            )

    settings = _platform_settings(tmp_path)
    runtime = build_runtime(settings)
    room, wallpaper = valid_images
    try:
        tenant = runtime.tenant_resolver.resolve_for_cli(
            tenant_id="wallpaper-demo",
            pipeline="wallpaper",
        )
        pipeline = await runtime.task_router.get_pipeline(tenant)
        assert isinstance(pipeline, WallpaperPipeline)
        pipeline.wall_analyzer = NoWallAnalyzer()
        result = await runtime.task_router.dispatch(
            tenant,
            GenerationRequest(
                source_image=room,
                reference_image=wallpaper,
                options=WallpaperOptions().model_dump(),
            ),
        )
    finally:
        await runtime.aclose()
    assert isinstance(result, WallpaperJobResult)
    assert result.status == "rejected"
    assert result.output is None
    assert "wall" in (result.rejection_reason or "").lower()
    assert not (
        settings.output_directory / result.job_id / "candidates"
    ).exists()


@pytest.mark.asyncio
async def test_generic_api_resolves_tenant_from_api_key(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    api_key = "fashion-api-key"
    config = tmp_path / "tenants.json"
    config.write_text(
        json.dumps(
            {
                "tenants": [
                    {
                        "tenant_id": "fashion",
                        "pipeline": "clothing",
                        "analysis_provider": "mock",
                        "generation_provider": "mock",
                        "api_key_sha256": hashlib.sha256(
                            api_key.encode()
                        ).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = _platform_settings(
        tmp_path,
        tenant_config_path=config,
        default_tenant_id="fashion",
        tenant_auth_required=True,
    )
    application = create_app(settings)
    person, garment = valid_images
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with person.open("rb") as source, garment.open("rb") as reference:
            response = await client.post(
                "/api/v1/generate",
                headers={"X-API-Key": api_key},
                files={
                    "source_image": ("source.jpg", source, "image/jpeg"),
                    "reference_image": (
                        "reference.png",
                        reference,
                        "image/png",
                    ),
                },
                data={
                    "options": json.dumps(
                        {
                            "product_title": "تي شرت مردانه",
                            "candidates_per_color": 1,
                            "max_retries": 0,
                        }
                    )
                },
            )
    await application.state.runtime.aclose()
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "fashion"
    assert data["pipeline"] == "clothing"
    assert data["status"] == "completed"
    assert data["results"][0]["color"] == "original"


@pytest.mark.asyncio
async def test_tryon_api_applies_each_labeled_garment_in_order(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    settings = _platform_settings(tmp_path)
    application = create_app(settings)
    person, first_garment = valid_images
    second_garment = tmp_path / "watch.png"
    Image.new("RGBA", (256, 256), (80, 60, 30, 255)).save(second_garment)
    transport = httpx.ASGITransport(app=application)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/tryon",
            files=[
                (
                    "person_image",
                    ("person.jpg", person.read_bytes(), "image/jpeg"),
                ),
                (
                    "garment_images",
                    ("shirt.png", first_garment.read_bytes(), "image/png"),
                ),
                (
                    "garment_images",
                    ("watch.png", second_garment.read_bytes(), "image/png"),
                ),
            ],
            data={
                "garment_types": json.dumps(["T-shirt", "Watch"]),
                "candidates_per_color": "2",
                "max_retries": "0",
            },
        )
    await application.state.runtime.aclose()

    assert response.status_code == 200
    data = response.json()
    assert data["applied_items"] == ["T-shirt", "Watch"]
    assert len(data["stage_job_ids"]) == 2
    assert data["job_id"] == data["stage_job_ids"][-1]
    first_request = json.loads(
        (
            settings.output_directory
            / data["stage_job_ids"][0]
            / "request.json"
        ).read_text(encoding="utf-8")
    )
    final_request = json.loads(
        (
            settings.output_directory
            / data["stage_job_ids"][1]
            / "request.json"
        ).read_text(encoding="utf-8")
    )
    assert first_request["product_title"] == "T-shirt"
    assert first_request["candidates_per_color"] == 1
    assert final_request["product_title"] == "Watch"
    assert final_request["candidates_per_color"] == 2
    assert data["stage_job_ids"][0] in final_request["person_image"]


@pytest.mark.asyncio
async def test_generic_api_runs_wallpaper_pipeline_from_tenant_key(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    api_key = "wallpaper-api-key"
    config = tmp_path / "tenants.json"
    config.write_text(
        json.dumps(
            {
                "tenants": [
                    {
                        "tenant_id": "interiors",
                        "pipeline": "wallpaper",
                        "analysis_provider": "mock",
                        "generation_provider": "mock",
                        "api_key_sha256": hashlib.sha256(
                            api_key.encode()
                        ).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = _platform_settings(
        tmp_path,
        tenant_config_path=config,
        default_tenant_id="interiors",
        tenant_auth_required=True,
    )
    application = create_app(settings)
    room, wallpaper = valid_images
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        with room.open("rb") as source, wallpaper.open("rb") as reference:
            response = await client.post(
                "/api/v1/generate",
                headers={"X-API-Key": api_key},
                files={
                    "source_image": ("room.jpg", source, "image/jpeg"),
                    "reference_image": (
                        "wallpaper.png",
                        reference,
                        "image/png",
                    ),
                },
                data={
                    "options": WallpaperOptions(
                        candidates_per_job=1,
                        max_retries=0,
                    ).model_dump_json()
                },
            )
        data = response.json()
        artifact_path = str(data["output"]).replace("\\", "/")
        artifact_response = await client.get(
            f"/api/v1/jobs/{data['job_id']}/artifacts/{artifact_path}",
            headers={"X-API-Key": api_key},
        )
        private_file_response = await client.get(
            f"/api/v1/jobs/{data['job_id']}/artifacts/results.json",
            headers={"X-API-Key": api_key},
        )
    await application.state.runtime.aclose()
    assert response.status_code == 200
    assert data["tenant_id"] == "interiors"
    assert data["pipeline"] == "wallpaper"
    assert data["status"] == "completed"
    assert data["accepted"] is True
    assert (
        settings.output_directory
        / data["job_id"]
        / Path(data["output"])
    ).is_file()
    assert artifact_response.status_code == 200
    assert artifact_response.headers["content-type"].startswith("image/")
    assert artifact_response.content
    assert private_file_response.status_code == 404
