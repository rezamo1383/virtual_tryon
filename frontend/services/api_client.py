"""Typed HTTP client for the existing tenant-aware FastAPI backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Iterable, Iterator
from typing import Any
from urllib.parse import quote

import httpx


@dataclass(frozen=True)
class UploadedImage:
    """Transport-safe image owned by the frontend session."""

    name: str
    content: bytes
    content_type: str

    @classmethod
    def from_path(cls, path: Path) -> UploadedImage:
        suffix = path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",  
        }.get(suffix, "application/octet-stream")
        return cls(path.name, path.read_bytes(), media_type)


@dataclass(frozen=True)
class GarmentUpload:
    """One garment reference and the item selected from that image."""

    image: UploadedImage
    garment_type: str


class BackendAPIError(RuntimeError):
    """A safe, user-facing backend communication error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BackendClient:
    """Call only the public API; never invoke providers or pipelines directly."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 600,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def health(self) -> dict[str, Any]:
        return self._json_request("GET", "/health")

    def generate(
        self,
        *,
        source: UploadedImage,
        reference: UploadedImage,
        options: dict[str, Any],
        api_key: str,
    ) -> dict[str, Any]:
        files = {
            "source_image": (
                source.name,
                source.content,
                source.content_type,
            ),
            "reference_image": (
                reference.name,
                reference.content,
                reference.content_type,
            ),
        }
        return self._json_request(
            "POST",
            "/api/v1/generate",
            headers=self._headers(api_key),
            files=files,
            data={"options": json.dumps(options)},
        )

    def tryon(
        self,
        *,
        person: UploadedImage,
        garments: list[GarmentUpload],
        options: dict[str, Any],
        api_key: str,
    ) -> dict[str, Any]:
        if not garments:
            raise ValueError("At least one garment upload is required.")
        files: list[tuple[str, tuple[str, bytes, str]]] = [
            (
                "person_image",
                (person.name, person.content, person.content_type),
            ),
            *[
                (
                    "garment_images",
                    (
                        garment.image.name,
                        garment.image.content,
                        garment.image.content_type,
                    ),
                )
                for garment in garments
            ],
        ]
        data = {
            "garment_types": json.dumps(
                [garment.garment_type for garment in garments],
                ensure_ascii=False,
            ),
            "candidates_per_color": str(
                options.get("candidates_per_color", 1)
            ),
            "max_retries": str(options.get("max_retries", 0)),
            "preserve_face": _form_bool(options.get("preserve_face", True)),
            "preserve_pose": _form_bool(options.get("preserve_pose", True)),
            "preserve_background": _form_bool(
                options.get("preserve_background", True)
            ),
        }
        return self._json_request(
            "POST",
            "/api/v1/tryon",
            headers=self._headers(api_key),
            files=files,
            data=data,
        )

    def job_status(self, job_id: str, *, api_key: str) -> dict[str, Any]:
        return self._json_request(
            "GET",
            f"/api/v1/jobs/{quote(job_id, safe='')}",
            headers=self._headers(api_key),
        )

    def job_results(self, job_id: str, *, api_key: str) -> dict[str, Any]:
        return self._json_request(
            "GET",
            f"/api/v1/jobs/{quote(job_id, safe='')}/results",
            headers=self._headers(api_key),
        )

    def job_events(
        self,
        job_id: str,
        *,
        api_key: str,
    ) -> Iterator[dict[str, Any]]:
        """Yield status events from the tenant-owned SSE job stream."""

        endpoint = f"/api/v1/jobs/{quote(job_id, safe='')}/events"
        headers = {
            **self._headers(api_key),
            "Accept": "text/event-stream",
        }
        try:
            with self._client.stream(
                "GET",
                endpoint,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise self._response_error(response)
                content_type = response.headers.get("content-type", "")
                if not content_type.casefold().startswith("text/event-stream"):
                    raise BackendAPIError(
                        "The backend returned an invalid job event stream."
                    )
                yield from _parse_status_events(response.iter_lines())
        except BackendAPIError:
            raise
        except httpx.TimeoutException as exc:
            raise BackendAPIError(
                "Waiting for image generation timed out. Please try again."
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendAPIError(
                "The backend went offline while waiting for generation."
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendAPIError(
                "The job status stream was interrupted. Please try again."
            ) from exc

    def result_image(
        self,
        job_id: str,
        *,
        api_key: str,
    ) -> tuple[bytes, str]:
        """Download the final image from the dedicated public result endpoint."""

        endpoint = f"/api/v1/results/{quote(job_id, safe='')}/image"
        try:
            response = self._client.get(
                endpoint,
                headers=self._headers(api_key),
            )
        except httpx.TimeoutException as exc:
            raise BackendAPIError(
                "The result download timed out. Please try again."
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendAPIError(
                "The result could not be downloaded from the backend."
            ) from exc
        if response.status_code >= 400:
            raise self._response_error(response)
        return (
            response.content,
            response.headers.get("content-type", "image/png").split(";")[0],
        )

    def artifact(
        self,
        job_id: str,
        artifact_path: str,
        *,
        api_key: str,
    ) -> tuple[bytes, str]:
        normalized = PurePosixPath(artifact_path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise BackendAPIError("The generated artifact path is invalid.")
        endpoint = (
            f"/api/v1/jobs/{quote(job_id, safe='')}/artifacts/"
            f"{quote(normalized.as_posix(), safe='/')}"
        )
        try:
            response = self._client.get(
                endpoint,
                headers=self._headers(api_key),
            )
        except httpx.TimeoutException as exc:
            raise BackendAPIError(
                "The result download timed out. Please try again."
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendAPIError(
                "The result could not be downloaded from the backend."
            ) from exc
        if response.status_code >= 400:
            raise self._response_error(response)
        return (
            response.content,
            response.headers.get("content-type", "image/png").split(";")[0],
        )

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def output_path(result: dict[str, Any], product: str) -> str | None:
        if product == "wallpaper":
            value = result.get("output")
            return str(value) if value else None
        color_results = result.get("results")
        if not isinstance(color_results, list) or not color_results:
            return None
        value = color_results[0].get("output")
        return str(value) if value else None

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {"X-API-Key": api_key} if api_key else {}

    def _json_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise BackendAPIError(
                "Generation took too long. Please try again."
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendAPIError(
                "The backend is offline. Start FastAPI and try again."
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendAPIError(
                "The backend could not be reached. Please try again."
            ) from exc
        if response.status_code >= 400:
            raise self._response_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendAPIError(
                "The backend returned an invalid response."
            ) from exc
        if not isinstance(payload, dict):
            raise BackendAPIError("The backend returned an invalid response.")
        return payload

    @staticmethod
    def _response_error(response: httpx.Response) -> BackendAPIError:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = str(payload.get("detail", ""))
        except ValueError:
            pass
        lowered = detail.casefold()
        if response.status_code == 401:
            message = "This product is not configured with a valid tenant key."
        elif "no person was detected" in lowered:
            message = "No person was detected in the uploaded person image."
        elif "wall" in lowered and "detected" in lowered:
            message = "No suitable wall was detected in the room image."
        elif "balance" in lowered or "quota" in lowered:
            message = "The image generation account needs additional credit."
        elif response.status_code == 413:
            message = "One of the uploaded images is too large."
        elif response.status_code == 422 and detail:
            message = detail[:240]
        elif response.status_code >= 500 and (
            "image request failed after retries" in lowered
            or "network" in lowered
            or "connect" in lowered
        ):
            message = (
                "The backend reached the image provider, but the provider "
                "connection failed. Check the backend logs and retry."
            )
        elif response.status_code >= 500:
            message = "The generation service is temporarily unavailable."
        else:
            message = "The request could not be completed. Please try again."
        return BackendAPIError(message, status_code=response.status_code)


def _form_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _parse_status_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    event_type = ""
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            event = _decode_status_event(event_type, data_lines)
            if event is not None:
                yield event
            event_type = ""
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_type = value
        elif field == "data":
            data_lines.append(value)

    event = _decode_status_event(event_type, data_lines)
    if event is not None:
        yield event


def _decode_status_event(
    event_type: str,
    data_lines: list[str],
) -> dict[str, Any] | None:
    if event_type != "status" or not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise BackendAPIError(
            "The backend returned an invalid job status event."
        ) from exc
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("job_id"), str)
        or not isinstance(payload.get("status"), str)
    ):
        raise BackendAPIError(
            "The backend returned an invalid job status event."
        )
    return payload
