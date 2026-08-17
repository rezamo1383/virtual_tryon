"""Environment-backed application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"
    tenant_config_path: Path = Path("config/tenants.json")
    default_tenant_id: str = "legacy-clothing"
    tenant_auth_required: bool = False
    max_image_size_mb: int = Field(10, ge=1, le=100)
    min_image_width: int = Field(512, ge=32)
    min_image_height: int = Field(512, ge=32)
    max_image_dimension: int = Field(12000, ge=512)
    min_aspect_ratio: float = Field(0.2, gt=0)
    max_aspect_ratio: float = Field(5.0, gt=0)

    qwen_api_base_url: str = ""
    qwen_api_key: str = ""
    qwen_model: str = ""
    qwen_timeout_seconds: float = Field(60, gt=0)
    qwen_validation_retries: int = Field(2, ge=0, le=5)
    use_mock_qwen: bool = True

    analysis_provider: Literal[
        "auto", "mock", "qwen", "openrouter", "gapgpt"
    ] = "auto"
    tryon_provider: Literal[
        "auto", "mock", "generic", "openrouter", "gapgpt"
    ] = "auto"

    openrouter_api_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    openrouter_vision_model: str = ""
    openrouter_image_model: str = ""
    openrouter_timeout_seconds: float = Field(180, gt=0)
    openrouter_http_referer: str = ""
    openrouter_app_name: str = "Virtual Try-On"
    openrouter_image_quality: Literal["auto", "low", "medium", "high"] = "high"
    openrouter_image_size: str = ""

    gapgpt_api_base_url: str = "https://api.gapgpt.app/v1"
    gapgpt_api_key: str = ""
    gapgpt_vision_model: str = "gpt-4o"
    gapgpt_image_model: str = "gpt-image-2"
    gapgpt_timeout_seconds: float = Field(180, gt=0)
    gapgpt_image_edit_endpoint: str = "/images/edits"
    gapgpt_image_field_name: Literal["image", "image[]"] = "image[]"
    gapgpt_image_quality: Literal["auto", "low", "medium", "high"] = "medium"
    gapgpt_image_size: str = "1024x1536"
    gapgpt_wallpaper_image_size: Literal[
        "auto", "1024x1024", "1024x1536", "1536x1024"
    ] = "auto"

    tryon_api_base_url: str = ""
    tryon_api_key: str = ""
    tryon_model: str = ""
    tryon_timeout_seconds: float = Field(180, gt=0)
    use_mock_tryon: bool = True
    tryon_provider_supports_mask: bool = False

    candidates_per_color: int = Field(2, ge=1, le=8)
    max_generation_retries: int = Field(1, ge=0, le=5)
    min_acceptance_score: float = Field(0.80, ge=0, le=1)
    wallpaper_min_acceptance_score: float = Field(0.75, ge=0, le=1)
    wallpaper_mask_feather_radius: int = Field(1, ge=0, le=50)
    wallpaper_segmentation_backend: Literal["semantic", "polygon"] = "semantic"
    wallpaper_segmentation_runtime: Literal["onnx", "torch"] = "onnx"
    wallpaper_segmentation_model: str = (
        "nvidia/segformer-b2-finetuned-ade-512-512"
    )
    wallpaper_segmentation_onnx_url: str = (
        "https://huggingface.co/Xenova/"
        "segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model.onnx"
    )
    wallpaper_segmentation_onnx_filename: str = "segformer-b2-ade512.onnx"
    wallpaper_segmentation_onnx_sha256: str = (
        "819c15e6af8c4de3359c1de7ab0a17d0dde495df1d16f8908a7163f8038e0fa0"
    )
    wallpaper_segmentation_device: Literal["auto", "cpu", "cuda"] = "auto"
    wallpaper_wall_confidence_threshold: float = Field(0.25, ge=0, le=1)
    wallpaper_min_wall_region_coverage: float = Field(0.001, ge=0, le=0.1)
    wallpaper_min_wall_component_ratio: float = Field(0.01, ge=0, le=0.25)
    person_analysis_enabled: bool = False
    reject_unsuitable_person_images: bool = True

    local_preprocessing_enabled: bool = True
    person_presence_check_enabled: bool = True
    background_removal_enabled: bool = True
    pose_estimation_enabled: bool = True
    human_parsing_enabled: bool = True
    human_parsing_required: bool = False
    preprocessing_device: Literal["auto", "cpu", "cuda"] = "auto"

    person_target_width: int = Field(768, ge=64, le=4096)
    person_target_height: int = Field(1024, ge=64, le=4096)
    garment_target_width: int = Field(768, ge=64, le=4096)
    garment_target_height: int = Field(1024, ge=64, le=4096)
    preprocessing_inference_max_dimension: int = Field(1600, ge=256, le=4096)

    pose_min_detection_confidence: float = Field(0.5, ge=0, le=1)
    pose_min_tracking_confidence: float = Field(0.5, ge=0, le=1)
    min_shoulder_visibility: float = Field(0.55, ge=0, le=1)
    min_tryon_suitability_score: float = Field(0.70, ge=0, le=1)

    mask_morphology_kernel: int = Field(5, ge=1, le=51)
    mask_dilation_kernel: int = Field(9, ge=1, le=101)
    mask_dilation_iterations: int = Field(1, ge=0, le=10)
    mask_feather_radius: int = Field(5, ge=0, le=50)

    save_preprocessing_debug_images: bool = True
    preprocessing_warmup_enabled: bool = False
    preprocessing_fail_open: bool = False
    preprocessing_timeout_seconds: float = Field(120, gt=0, le=1800)
    preprocessing_max_concurrency: int = Field(1, ge=1, le=16)

    model_cache_directory: Path = Path("models")
    local_model_offline_mode: bool = False
    model_download_timeout_seconds: float = Field(120, gt=0, le=1800)
    background_removal_person_model: str = "u2net_human_seg"
    background_removal_garment_model: str = "u2netp"
    human_parsing_model_url: str = (
        "https://huggingface.co/pirocheto/schp-atr-18/resolve/main/"
        "onnx/schp-atr-18-int8-static.onnx"
    )
    human_parsing_model_filename: str = "schp-atr-18-int8-static.onnx"
    human_parsing_model_sha256: str = ""

    temp_directory: Path = Path("temp")
    output_directory: Path = Path("outputs")
    log_directory: Path = Path("logs")
    delete_temp_files: bool = True

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("openrouter_api_base_url", "gapgpt_api_base_url")
    @classmethod
    def validate_provider_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith("https://"):
            raise ValueError("Provider API base URLs must use HTTPS.")
        return normalized

    @field_validator("gapgpt_image_edit_endpoint")
    @classmethod
    def validate_gapgpt_image_endpoint(cls, value: str) -> str:
        normalized = "/" + value.strip().lstrip("/")
        if "://" in normalized or ".." in normalized:
            raise ValueError("GAPGPT_IMAGE_EDIT_ENDPOINT must be a relative API path.")
        return normalized

    @field_validator(
        "mask_morphology_kernel",
        "mask_dilation_kernel",
    )
    @classmethod
    def odd_mask_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("Mask kernel sizes must be odd.")
        return value

    @field_validator(
        "human_parsing_model_filename",
        "wallpaper_segmentation_onnx_filename",
    )
    @classmethod
    def safe_model_filename(cls, value: str) -> str:
        filename = Path(value.strip()).name
        if not filename or filename != value.strip():
            raise ValueError(
                "Local model filenames must be plain filenames."
            )
        return filename

    @field_validator("human_parsing_model_sha256")
    @classmethod
    def valid_optional_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized and (
            len(normalized) != 64
            or any(character not in "0123456789abcdef" for character in normalized)
        ):
            raise ValueError(
                "HUMAN_PARSING_MODEL_SHA256 must be empty or 64 hex characters."
            )
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings singleton."""

    return Settings()
