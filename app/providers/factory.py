"""Construct provider clients from an already-resolved model route."""

from __future__ import annotations

from dataclasses import dataclass

from app.clients.gapgpt_client import GapGPTTryOnClient, GapGPTVisionClient
from app.clients.mock_qwen_client import MockQwenClient
from app.clients.mock_tryon_client import MockTryOnClient
from app.clients.mock_wallpaper_client import MockWallpaperClient
from app.clients.openrouter_client import (
    OpenRouterTryOnClient,
    OpenRouterVisionClient,
)
from app.clients.qwen_client import OpenAICompatibleQwenClient, QwenClient
from app.clients.tryon_api_client import GenericRESTTryOnClient, TryOnAPIClient
from app.core.config import Settings
from app.prompts.base import GenerationPromptBuilder
from app.prompts.clothing import ClothingPromptBuilder


@dataclass(frozen=True)
class ProviderBundle:
    """Analysis and generation providers owned by one pipeline instance."""

    analysis: QwenClient
    generation: TryOnAPIClient


def build_provider_bundle(
    settings: Settings,
    *,
    prompt_builder: GenerationPromptBuilder | None = None,
    analysis_client: QwenClient | None = None,
    generation_client: TryOnAPIClient | None = None,
) -> ProviderBundle:
    """Build tenant-scoped providers while preserving legacy settings."""

    prompts = prompt_builder or ClothingPromptBuilder()
    clothing_prompts = (
        prompts if isinstance(prompts, ClothingPromptBuilder) else None
    )
    analysis_provider = settings.analysis_provider
    if analysis_provider == "auto":
        analysis_provider = "mock" if settings.use_mock_qwen else "qwen"
    generation_provider = settings.tryon_provider
    if generation_provider == "auto":
        generation_provider = (
            "mock" if settings.use_mock_tryon else "generic"
        )

    analysis = analysis_client
    if analysis is None:
        if analysis_provider == "mock":
            analysis = MockQwenClient()
        elif analysis_provider == "gapgpt":
            analysis = GapGPTVisionClient(
                base_url=settings.gapgpt_api_base_url,
                api_key=settings.gapgpt_api_key,
                model=settings.gapgpt_vision_model,
                timeout_seconds=settings.gapgpt_timeout_seconds,
                validation_retries=settings.qwen_validation_retries,
                prompt_builder=clothing_prompts,
            )
        elif analysis_provider == "openrouter":
            analysis = OpenRouterVisionClient(
                base_url=settings.openrouter_api_base_url,
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_vision_model,
                timeout_seconds=settings.openrouter_timeout_seconds,
                validation_retries=settings.qwen_validation_retries,
                http_referer=settings.openrouter_http_referer,
                app_name=settings.openrouter_app_name,
                prompt_builder=clothing_prompts,
            )
        else:
            analysis = OpenAICompatibleQwenClient(
                base_url=settings.qwen_api_base_url,
                api_key=settings.qwen_api_key,
                model=settings.qwen_model,
                timeout_seconds=settings.qwen_timeout_seconds,
                validation_retries=settings.qwen_validation_retries,
                prompt_builder=clothing_prompts,
            )

    generation = generation_client
    if generation is None:
        if generation_provider == "mock":
            generation = (
                MockWallpaperClient()
                if prompts.domain == "wallpaper"
                else MockTryOnClient()
            )
        elif generation_provider == "gapgpt":
            generation = GapGPTTryOnClient(
                base_url=settings.gapgpt_api_base_url,
                api_key=settings.gapgpt_api_key,
                model=settings.gapgpt_image_model,
                timeout_seconds=settings.gapgpt_timeout_seconds,
                edit_endpoint=settings.gapgpt_image_edit_endpoint,
                image_field_name=settings.gapgpt_image_field_name,
                quality=settings.gapgpt_image_quality,
                size=settings.gapgpt_image_size,
                wallpaper_size=settings.gapgpt_wallpaper_image_size,
                prompt_builder=prompts,
            )
        elif generation_provider == "openrouter":
            generation = OpenRouterTryOnClient(
                base_url=settings.openrouter_api_base_url,
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_image_model,
                timeout_seconds=settings.openrouter_timeout_seconds,
                quality=settings.openrouter_image_quality,
                size=settings.openrouter_image_size,
                http_referer=settings.openrouter_http_referer,
                app_name=settings.openrouter_app_name,
                prompt_builder=prompts,
            )
        else:
            generation = GenericRESTTryOnClient(
                base_url=settings.tryon_api_base_url,
                api_key=settings.tryon_api_key,
                model=settings.tryon_model,
                timeout_seconds=settings.tryon_timeout_seconds,
                supports_mask=settings.tryon_provider_supports_mask,
            )
    return ProviderBundle(
        analysis=analysis,
        generation=generation,
    )
