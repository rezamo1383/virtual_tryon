"""OpenRouter adapters for vision analysis and image-to-image generation."""

from __future__ import annotations

import base64
import binascii
import logging
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.clients.qwen_client import OpenAICompatibleQwenClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.exceptions import OpenRouterAPIError, TryOnAPIError
from app.prompts.base import GenerationPromptBuilder
from app.prompts.clothing import ClothingPromptBuilder
from app.utils.image_utils import (
    decode_image_bytes,
    image_to_png_data_url,
)

LOGGER = logging.getLogger(__name__)


def openrouter_headers(
    *,
    api_key: str,
    http_referer: str = "",
    app_name: str = "",
) -> dict[str, str]:
    """Build safe OpenRouter headers without ever logging credentials."""

    headers = {"Authorization": f"Bearer {api_key}"}
    if http_referer.strip():
        headers["HTTP-Referer"] = http_referer.strip()
    if app_name.strip():
        headers["X-OpenRouter-Title"] = app_name.strip()
    return headers


class OpenRouterVisionClient(OpenAICompatibleQwenClient):
    """Use an OpenRouter vision model through its Chat Completions API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60,
        validation_retries: int = 2,
        http_referer: str = "",
        app_name: str = "",
        http_client: httpx.AsyncClient | None = None,
        prompt_builder: ClothingPromptBuilder | None = None,
    ) -> None:
        if not api_key:
            raise OpenRouterAPIError(
                "OPENROUTER_API_KEY is required when OpenRouter analysis is enabled."
            )
        if not model:
            raise OpenRouterAPIError(
                "OPENROUTER_VISION_MODEL is required when OpenRouter analysis is enabled."
            )
        headers = openrouter_headers(
            api_key=api_key,
            http_referer=http_referer,
            app_name=app_name,
        )
        # The parent adds Authorization itself, so only attribution headers are passed.
        attribution_headers = {
            key: value for key, value in headers.items() if key != "Authorization"
        }
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            validation_retries=validation_retries,
            request_headers=attribution_headers,
            provider_label="OpenRouter",
            error_class=OpenRouterAPIError,
            http_client=http_client,
            prompt_builder=prompt_builder,
        )


class _TransientOpenRouterImageError(Exception):
    """Internal marker used only for retryable OpenRouter image failures."""


class OpenRouterTryOnClient(TryOnAPIClient):
    """Generate try-on candidates using OpenRouter's dedicated Image API."""

    supports_text_prompt = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180,
        quality: str = "high",
        size: str = "",
        http_referer: str = "",
        app_name: str = "",
        http_client: httpx.AsyncClient | None = None,
        prompt_builder: GenerationPromptBuilder | None = None,
    ) -> None:
        if not api_key:
            raise OpenRouterAPIError(
                "OPENROUTER_API_KEY is required when OpenRouter image generation is enabled."
            )
        if not model:
            raise OpenRouterAPIError(
                "OPENROUTER_IMAGE_MODEL is required when OpenRouter image generation is enabled."
            )
        self._endpoint = (
            base_url.rstrip("/")
            if base_url.rstrip("/").endswith("/images")
            else f"{base_url.rstrip('/')}/images"
        )
        self._model = model
        self._quality = quality
        self._size = size.strip()
        self._headers = {
            **openrouter_headers(
                api_key=api_key,
                http_referer=http_referer,
                app_name=app_name,
            ),
            "Content-Type": "application/json",
        }
        self._owns_client = http_client is None
        self._prompt_builder = prompt_builder or ClothingPromptBuilder()
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds)
        )

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        """Generate one candidate per call for broad provider compatibility."""

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": self._prompt_builder.generation(category, options),
            "n": 1,
            "quality": self._quality,
            "output_format": "png",
            "input_references": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_png_data_url(person_image)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_png_data_url(garment_image)},
                },
            ],
        }
        if self._size:
            payload["size"] = self._size
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=15),
                retry=retry_if_exception_type(
                    (
                        _TransientOpenRouterImageError,
                        httpx.TimeoutException,
                        httpx.NetworkError,
                    )
                ),
                reraise=True,
            ):
                with attempt:
                    return await self._post(payload)
        except TryOnAPIError:
            raise
        except Exception as exc:
            raise TryOnAPIError(
                f"OpenRouter image request failed after retries: {exc}"
            ) from exc
        raise TryOnAPIError("OpenRouter image request failed without a response.")

    async def _post(self, payload: dict[str, Any]) -> list[bytes]:
        response = await self._client.post(
            self._endpoint,
            json=payload,
            headers=self._headers,
        )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            LOGGER.warning(
                "openrouter_image_temporary_error",
                extra={"status_code": response.status_code},
            )
            raise _TransientOpenRouterImageError(
                f"Temporary OpenRouter HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            raise TryOnAPIError(
                "OpenRouter image API returned HTTP "
                f"{response.status_code}. Verify the image model supports input references."
            )
        try:
            body = response.json()
            items = body["data"]
            if not isinstance(items, list) or not items:
                raise ValueError("empty data")
            decoded: list[bytes] = []
            for item in items:
                encoded = item.get("b64_json") if isinstance(item, dict) else None
                if not isinstance(encoded, str) or not encoded:
                    continue
                decoded.append(
                    decode_image_bytes(base64.b64decode(encoded, validate=True))
                )
            if not decoded:
                raise ValueError("no decodable image")
            return decoded
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise TryOnAPIError(
                "OpenRouter image API returned an invalid image response."
            ) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
