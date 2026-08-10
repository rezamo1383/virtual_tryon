"""Qwen vision-language client with strict response validation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.exceptions import QwenAPIError, VirtualTryOnError
from app.models.analysis_models import GarmentAnalysis, PersonAnalysis
from app.models.evaluation_models import OutputEvaluation
from app.prompts.clothing import (
    ClothingPromptBuilder,
    GARMENT_ANALYSIS_PROMPT,
    OUTPUT_EVALUATION_PROMPT,
    PERSON_ANALYSIS_PROMPT,
)
from app.utils.image_utils import image_to_data_url
from app.utils.json_utils import extract_first_json_object

LOGGER = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)

PERSON_PROMPT = PERSON_ANALYSIS_PROMPT
GARMENT_PROMPT = GARMENT_ANALYSIS_PROMPT
EVALUATION_PROMPT = OUTPUT_EVALUATION_PROMPT


class QwenClient(ABC):
    """Provider-neutral interface used by the pipeline."""

    @abstractmethod
    async def analyze_person(self, image_path: Path) -> PersonAnalysis:
        """Analyze whether a person image is fit for try-on."""

    @abstractmethod
    async def analyze_garment(self, image_path: Path) -> GarmentAnalysis:
        """Classify and describe a garment image."""

    @abstractmethod
    async def evaluate_output(
        self,
        person_image: Path,
        garment_image: Path,
        output_image: Path,
        requested_color: str,
        product_title: str | None = None,
    ) -> OutputEvaluation:
        """Evaluate one generated try-on candidate."""

    async def analyze_structured(
        self,
        prompt: str,
        images: list[Path],
        model_type: type[ModelT],
    ) -> ModelT:
        """Analyze arbitrary domain images into a validated model."""

        raise QwenAPIError(
            f"{type(self).__name__} does not support structured domain analysis."
        )

    async def aclose(self) -> None:
        """Release owned network resources, if any."""


class _TransientQwenError(Exception):
    """Internal retry marker for temporary provider failures."""


class OpenAICompatibleQwenClient(QwenClient):
    """Qwen client for OpenAI-compatible chat-completion APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60,
        validation_retries: int = 2,
        request_headers: dict[str, str] | None = None,
        provider_label: str = "Qwen",
        error_class: type[VirtualTryOnError] = QwenAPIError,
        http_client: httpx.AsyncClient | None = None,
        prompt_builder: ClothingPromptBuilder | None = None,
    ) -> None:
        if not base_url or not api_key or not model:
            raise error_class(
                f"{provider_label} is enabled but its API URL, key, or model is missing."
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._provider_label = provider_label
        self._error_class = error_class
        self._validation_retries = validation_retries
        self._request_headers = {
            "Authorization": f"Bearer {api_key}",
            **(request_headers or {}),
        }
        self._prompt_builder = prompt_builder or ClothingPromptBuilder()
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def analyze_person(self, image_path: Path) -> PersonAnalysis:
        return await self._request_model(
            self._prompt_builder.person_analysis(),
            [image_path],
            PersonAnalysis,
        )

    async def analyze_garment(self, image_path: Path) -> GarmentAnalysis:
        return await self._request_model(
            self._prompt_builder.reference_analysis(),
            [image_path],
            GarmentAnalysis,
        )

    async def evaluate_output(
        self,
        person_image: Path,
        garment_image: Path,
        output_image: Path,
        requested_color: str,
        product_title: str | None = None,
    ) -> OutputEvaluation:
        return await self._request_model(
            self._prompt_builder.output_evaluation(
                requested_color,
                product_title,
            ),
            [person_image, garment_image, output_image],
            OutputEvaluation,
        )

    async def analyze_structured(
        self,
        prompt: str,
        images: list[Path],
        model_type: type[ModelT],
    ) -> ModelT:
        return await self._request_model(prompt, images, model_type)

    async def _request_model(
        self, prompt: str, images: list[Path], model_type: type[ModelT]
    ) -> ModelT:
        last_error: Exception | None = None
        validation_prompt = prompt
        for validation_attempt in range(self._validation_retries + 1):
            try:
                content = await self._post_with_transport_retry(validation_prompt, images)
                data = extract_first_json_object(content)
                return model_type.model_validate(data)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                LOGGER.warning(
                    "qwen_response_validation_failed",
                    extra={
                        "model_type": model_type.__name__,
                        "validation_attempt": validation_attempt + 1,
                    },
                )
                validation_prompt = (
                    prompt + "\nSchema mismatch. Return one JSON object with all keys/types."
                )
        raise self._error_class(
            f"{self._provider_label} returned invalid {model_type.__name__} JSON after "
            f"{self._validation_retries + 1} attempts: {last_error}"
        )

    async def _post_with_transport_retry(
        self, prompt: str, images: list[Path]
    ) -> str:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=8),
                retry=retry_if_exception_type(
                    (_TransientQwenError, httpx.TimeoutException, httpx.NetworkError)
                ),
                reraise=True,
            ):
                with attempt:
                    return await self._post(prompt, images)
        except VirtualTryOnError:
            raise
        except Exception as exc:
            raise self._error_class(
                f"{self._provider_label} request failed after retries: {exc}"
            ) from exc
        raise self._error_class(
            f"{self._provider_label} request failed without a response"
        )

    async def _post(self, prompt: str, images: list[Path]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image)},
            }
            for image in images
        )
        payload = {
            "model": self._model,
            # Exactly one stateless user turn: no system prompt or prior history.
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        endpoint = (
            self._base_url
            if self._base_url.endswith("/chat/completions")
            else f"{self._base_url}/chat/completions"
        )
        try:
            response = await self._client.post(
                endpoint, json=payload, headers=self._request_headers
            )
        except (httpx.TimeoutException, httpx.NetworkError):
            raise
        if response.status_code == 429 or 500 <= response.status_code < 600:
            LOGGER.warning(
                "qwen_temporary_http_error", extra={"status_code": response.status_code}
            )
            raise _TransientQwenError(f"Temporary Qwen HTTP {response.status_code}")
        if response.status_code >= 400:
            detail = self._safe_error_message(response)
            suffix = f": {detail}" if detail else ""
            raise self._error_class(
                f"{self._provider_label} API returned HTTP "
                f"{response.status_code}{suffix}"
            )
        try:
            body = response.json()
            result = body["choices"][0]["message"]["content"]
            if isinstance(result, list):
                result = "".join(
                    part.get("text", "") for part in result if isinstance(part, dict)
                )
            if not isinstance(result, str) or not result.strip():
                raise ValueError("empty message content")
            return result
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise self._error_class(
                f"{self._provider_label} API returned an invalid response envelope"
            ) from exc

    @staticmethod
    def _safe_error_message(response: httpx.Response) -> str:
        """Extract a bounded provider message without logging request content."""

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
        if self._owns_client:
            await self._client.aclose()
