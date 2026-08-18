"""GapGPT adapters for OpenAI-compatible vision and image editing APIs."""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import logging
import socket
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.clients.qwen_client import OpenAICompatibleQwenClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.exceptions import GapGPTAPIError, TryOnAPIError
from app.prompts.base import GenerationPromptBuilder
from app.prompts.clothing import ClothingPromptBuilder
from app.utils.image_utils import decode_image_bytes, open_image_safe

LOGGER = logging.getLogger(__name__)


def _provider_transport() -> httpx.AsyncHTTPTransport:
    """Keep long image-generation connections alive across Docker NAT."""

    socket_options: list[tuple[int, int, int]] = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    for option_name, value in (
        ("TCP_KEEPIDLE", 10),
        ("TCP_KEEPINTVL", 5),
        ("TCP_KEEPCNT", 3),
    ):
        option = getattr(socket, option_name, None)
        if option is not None:
            socket_options.append((socket.IPPROTO_TCP, option, value))
    return httpx.AsyncHTTPTransport(
        retries=1,
        socket_options=socket_options,
    )


def _network_error_detail(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class GapGPTVisionClient(OpenAICompatibleQwenClient):
    """Analyze and evaluate images through GapGPT Chat Completions."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120,
        validation_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        prompt_builder: ClothingPromptBuilder | None = None,
    ) -> None:
        if not api_key:
            raise GapGPTAPIError(
                "GAPGPT_API_KEY is required when GapGPT analysis is enabled."
            )
        if not model:
            raise GapGPTAPIError(
                "GAPGPT_VISION_MODEL is required when GapGPT analysis is enabled."
            )
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            validation_retries=validation_retries,
            provider_label="GapGPT",
            error_class=GapGPTAPIError,
            http_client=http_client,
            prompt_builder=prompt_builder,
        )


class _TransientGapGPTImageError(Exception):
    """Internal marker for retryable GapGPT image failures."""


class GapGPTTryOnClient(TryOnAPIClient):
    """Send person and garment references to GapGPT's image-edit API."""

    supports_text_prompt = True
    _PNG_CACHE_MAX_ENTRIES = 8
    _PNG_CACHE_MAX_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "gpt-image-1.5",
        timeout_seconds: float = 180,
        edit_endpoint: str = "/images/edits",
        image_field_name: str = "image[]",
        quality: str = "medium",
        size: str = "1024x1536",
        wallpaper_size: str = "auto",
        http_client: httpx.AsyncClient | None = None,
        prompt_builder: GenerationPromptBuilder | None = None,
    ) -> None:
        if not api_key:
            raise GapGPTAPIError(
                "GAPGPT_API_KEY is required when GapGPT image generation is enabled."
            )
        if not model:
            raise GapGPTAPIError(
                "GAPGPT_IMAGE_MODEL is required when GapGPT image generation is enabled."
            )
        if image_field_name not in {"image", "image[]"}:
            raise GapGPTAPIError(
                "GAPGPT_IMAGE_FIELD_NAME must be either 'image' or 'image[]'."
            )
        endpoint = "/" + edit_endpoint.strip().lstrip("/")
        self._endpoint = f"{base_url.rstrip('/')}{endpoint}"
        self._api_key = api_key
        self._model = model
        self._field_name = image_field_name
        self._quality = quality
        self._size = size.strip()
        self._wallpaper_size = wallpaper_size.strip()
        self._owns_client = http_client is None
        self._prompt_builder = prompt_builder or ClothingPromptBuilder()
        self._png_cache: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
        self._png_cache_size = 0
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            transport=_provider_transport(),
            trust_env=False,
        )

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        """Generate one candidate while preserving both supplied references."""

        source_name = "room.png" if category == "wallpaper" else "person.png"
        reference_name = (
            "wallpaper-reference.png" if category == "wallpaper" else "garment.png"
        )
        files = [
            (
                self._field_name,
                (source_name, self._privacy_safe_png(person_image), "image/png"),
            ),
            (
                self._field_name,
                (
                    reference_name,
                    self._privacy_safe_png(garment_image),
                    "image/png",
                ),
            ),
        ]
        quality = str(options.get("generation_quality", self._quality))
        if quality not in {"auto", "low", "medium", "high"}:
            quality = self._quality
        data = {
            "model": self._model,
            "prompt": self._prompt_builder.generation(category, options),
            "n": "1",
            "quality": quality,
            "output_format": "png",
        }
        size = self._resolved_size(person_image, category)
        if size:
            data["size"] = size

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=15),
                retry=retry_if_exception_type(
                    (
                        _TransientGapGPTImageError,
                        httpx.TimeoutException,
                        httpx.NetworkError,
                    )
                ),
                reraise=True,
            ):
                with attempt:
                    return await self._post(files, data)
        except TryOnAPIError:
            raise
        except Exception as exc:
            detail = _network_error_detail(exc)
            LOGGER.error(
                "gapgpt_image_network_failed",
                extra={"error_type": type(exc).__name__, "error": detail},
            )
            raise TryOnAPIError(
                f"GapGPT image request failed after retries: {detail}"
            ) from exc
        raise TryOnAPIError("GapGPT image request failed without a response.")

    def _resolved_size(self, source_image: Path, category: str) -> str:
        if category != "wallpaper":
            return self._size
        if self._wallpaper_size != "auto":
            return self._wallpaper_size
        width, height = open_image_safe(source_image).size
        ratio = width / max(1, height)
        if ratio > 1.15:
            return "1536x1024"
        if ratio < 0.87:
            return "1024x1536"
        return "1024x1024"

    async def _post(
        self,
        files: list[tuple[str, tuple[str, bytes, str]]],
        data: dict[str, str],
    ) -> list[bytes]:
        response = await self._client.post(
            self._endpoint,
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            LOGGER.warning(
                "gapgpt_image_temporary_error",
                extra={"status_code": response.status_code},
            )
            raise _TransientGapGPTImageError(
                f"Temporary GapGPT HTTP {response.status_code}"
            )
        if response.status_code == 404:
            raise TryOnAPIError(
                "GapGPT returned HTTP 404 for the configured image-edit endpoint "
                f"({self._endpoint}). Confirm that your API plan exposes "
                f"/v1/images/edits for model '{self._model}'."
            )
        if response.status_code in {402, 403}:
            detail = self._safe_error_message(response)
            if "quota" in detail.casefold() or "balance" in detail.casefold():
                raise TryOnAPIError(
                    "GapGPT account balance is insufficient for image model "
                    f"'{self._model}': {detail}"
                )
        if response.status_code >= 400:
            detail = self._safe_error_message(response)
            suffix = f": {detail}" if detail else ""
            raise TryOnAPIError(
                f"GapGPT image API returned HTTP {response.status_code}{suffix}"
            )
        return await self._parse_images(response)

    async def _parse_images(self, response: httpx.Response) -> list[bytes]:
        try:
            body = response.json()
            items = body["data"]
            if not isinstance(items, list) or not items:
                raise ValueError("empty data")
        except (KeyError, TypeError, ValueError) as exc:
            raise TryOnAPIError(
                "GapGPT image API returned an invalid response envelope."
            ) from exc

        decoded: list[bytes] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            encoded = item.get("b64_json")
            if isinstance(encoded, str) and encoded:
                try:
                    decoded.append(
                        decode_image_bytes(base64.b64decode(encoded, validate=True))
                    )
                    continue
                except (ValueError, binascii.Error) as exc:
                    raise TryOnAPIError(
                        "GapGPT returned invalid base64 image data."
                    ) from exc
            url = item.get("url")
            if isinstance(url, str) and url:
                decoded.append(await self._download_image(url))
        if not decoded:
            raise TryOnAPIError(
                "GapGPT image response contains neither b64_json nor a file URL."
            )
        return decoded

    async def _download_image(self, url: str) -> bytes:
        current = url
        for _ in range(4):
            self._validate_public_https_url(current)
            response = await self._client.get(current)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise TryOnAPIError(
                        "GapGPT image download returned an empty redirect."
                    )
                current = urljoin(current, location)
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise _TransientGapGPTImageError(
                    f"Temporary GapGPT image download HTTP {response.status_code}"
                )
            if response.status_code >= 400:
                raise TryOnAPIError(
                    f"GapGPT generated-image URL returned HTTP {response.status_code}."
                )
            try:
                return decode_image_bytes(response.content)
            except Exception as exc:
                raise TryOnAPIError(
                    "GapGPT generated-image URL did not return a valid image."
                ) from exc
        raise TryOnAPIError("GapGPT image URL redirected too many times.")

    @staticmethod
    def _validate_public_https_url(url: str) -> None:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not hostname:
            raise TryOnAPIError("GapGPT returned an unsafe generated-image URL.")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise TryOnAPIError("GapGPT returned a local generated-image URL.")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return
        if not address.is_global:
            raise TryOnAPIError("GapGPT returned a non-public generated-image URL.")

    def _privacy_safe_png(self, path: Path) -> bytes:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        cached = self._png_cache.get(key)
        if cached is not None:
            self._png_cache.move_to_end(key)
            return cached
        image = open_image_safe(path)
        normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        buffer = io.BytesIO()
        normalized.save(buffer, format="PNG", optimize=True)
        encoded = buffer.getvalue()
        self._png_cache[key] = encoded
        self._png_cache_size += len(encoded)
        while (
            len(self._png_cache) > self._PNG_CACHE_MAX_ENTRIES
            or self._png_cache_size > self._PNG_CACHE_MAX_BYTES
        ) and len(self._png_cache) > 1:
            _, removed = self._png_cache.popitem(last=False)
            self._png_cache_size -= len(removed)
        return encoded

    @staticmethod
    def _safe_error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return ""
        if not isinstance(body, dict):
            return ""
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        else:
            message = error
        if not isinstance(message, str):
            return ""
        return " ".join(message.split())[:300]

    async def aclose(self) -> None:
        self._png_cache.clear()
        self._png_cache_size = 0
        if self._owns_client:
            await self._client.aclose()
