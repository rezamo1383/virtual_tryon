"""External-service client abstractions and implementations."""

from app.clients.qwen_client import OpenAICompatibleQwenClient, QwenClient
from app.clients.mock_wallpaper_client import MockWallpaperClient
from app.clients.openrouter_client import (
    OpenRouterTryOnClient,
    OpenRouterVisionClient,
)
from app.clients.gapgpt_client import GapGPTTryOnClient, GapGPTVisionClient
from app.clients.tryon_api_client import GenericRESTTryOnClient, TryOnAPIClient

__all__ = [
    "GenericRESTTryOnClient",
    "GapGPTTryOnClient",
    "GapGPTVisionClient",
    "OpenAICompatibleQwenClient",
    "OpenRouterTryOnClient",
    "OpenRouterVisionClient",
    "QwenClient",
    "MockWallpaperClient",
    "TryOnAPIClient",
]
