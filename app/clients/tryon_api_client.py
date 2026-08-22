"""Provider-neutral virtual try-on client and Generic REST adapter."""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.exceptions import TryOnAPIError
from app.prompts.clothing import ClothingPromptBuilder
from app.utils.image_utils import decode_image_bytes

LOGGER = logging.getLogger(__name__)


def build_tryon_prompt(category: str, options: dict[str, Any]) -> str:
    """Backward-compatible facade for the clothing prompt builder."""

    return ClothingPromptBuilder().generation(category, options)


class TryOnAPIClient(ABC):
    """Provider-neutral image generation interface."""

    supports_mask: bool = False
    supports_text_prompt: bool = False
    supports_multi_reference: bool = False

    @abstractmethod
    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        """Generate one or more image candidates."""

    async def generate_multi(
        self,
        person_image: Path,
        garment_images: list[Path],
        garment_types: list[str],
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        """Generate from one person and multiple independent references."""

        if len(garment_images) == 1 and len(garment_types) == 1:
            return await self.generate(
                person_image,
                garment_images[0],
                category,
                {**options, "garment_types": garment_types},
            )
        raise TryOnAPIError(
            "The configured generation provider does not support multiple "
            "garment references in one request."
        )

    async def aclose(self) -> None:
        """Release owned network resources, if any."""


class _TransientTryOnError(Exception):
    """Internal retry marker for temporary provider failures."""


class GenericRESTAdapter:
    """Isolate the configurable REST request and response wire format."""

    def build_multipart(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        model: str,
        options: dict[str, Any],
    ) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str]]:
        """Build the generic multipart form expected by the initial adapter."""

        mutable_options = dict(options)
        files: dict[str, tuple[str, bytes, str]] = {
            "person_image": (
                "person.png",
                person_image.read_bytes(),
                "application/octet-stream",
            ),
            "garment_image": (
                "garment.png",
                garment_image.read_bytes(),
                "application/octet-stream",
            ),
        }
        replace_mask_path = mutable_options.pop("replace_mask_path", None)
        if replace_mask_path:
            mask_path = Path(str(replace_mask_path))
            files["replace_mask"] = (
                "replace_mask.png",
                mask_path.read_bytes(),
                "image/png",
            )
        data = {
            "category": category,
            "model": model,
            **{key: str(value).lower() if isinstance(value, bool) else str(value)
               for key, value in mutable_options.items()},
        }
        return files, data

    def parse_response(self, response: httpx.Response) -> list[bytes]:
        """Parse raw-image or JSON/base64 generic provider responses."""

        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("image/"):
            return [decode_image_bytes(response.content)]
        try:
            body = response.json()
        except ValueError as exc:
            raise TryOnAPIError("Try-on API returned neither image nor JSON") from exc
        raw_items = body.get("images") or body.get("outputs") or body.get("data")
        if not isinstance(raw_items, list) or not raw_items:
            raise TryOnAPIError("Try-on API response contains no images")
        decoded: list[bytes] = []
        for item in raw_items:
            value = item
            if isinstance(item, dict):
                value = item.get("b64_json") or item.get("base64") or item.get("image")
            if not isinstance(value, str):
                continue
            if value.startswith("data:"):
                value = value.split(",", 1)[-1]
            try:
                decoded.append(decode_image_bytes(base64.b64decode(value, validate=True)))
            except Exception as exc:
                raise TryOnAPIError("Try-on API returned invalid base64 image data") from exc
        if not decoded:
            raise TryOnAPIError("Try-on API response contains no decodable images")
        return decoded


class GenericRESTTryOnClient(TryOnAPIClient):
    """Async multipart client for a configurable generic try-on endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180,
        supports_mask: bool = False,
        adapter: GenericRESTAdapter | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise TryOnAPIError(
                "Try-on API is enabled but TRYON_API_BASE_URL is missing."
            )
        self._base_url = base_url
        self._model = model
        self.supports_mask = supports_mask
        self._adapter = adapter or GenericRESTAdapter()
        self._owns_client = http_client is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds), headers=headers
        )

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        files, data = self._adapter.build_multipart(
            person_image, garment_image, category, self._model, options
        )
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=15),
                retry=retry_if_exception_type(
                    (_TransientTryOnError, httpx.TimeoutException, httpx.NetworkError)
                ),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.post(
                        self._base_url, files=files, data=data
                    )
                    if response.status_code == 429:
                        LOGGER.warning("tryon_rate_limited", extra={"status_code": 429})
                        raise _TransientTryOnError("Try-on provider rate limit")
                    if 500 <= response.status_code < 600:
                        LOGGER.warning(
                            "tryon_temporary_http_error",
                            extra={"status_code": response.status_code},
                        )
                        raise _TransientTryOnError(
                            f"Temporary try-on HTTP {response.status_code}"
                        )
                    if response.status_code >= 400:
                        raise TryOnAPIError(
                            f"Try-on API returned HTTP {response.status_code}"
                        )
                    return self._adapter.parse_response(response)
        except TryOnAPIError:
            raise
        except Exception as exc:
            raise TryOnAPIError(f"Try-on request failed after retries: {exc}") from exc
        raise TryOnAPIError("Try-on request failed without a response")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
