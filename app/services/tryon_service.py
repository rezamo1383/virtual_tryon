"""Candidate generation and durable storage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.clients.tryon_api_client import TryOnAPIClient
from app.core.exceptions import TryOnAPIError
from app.models.result_models import CandidateResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateBatch:
    """Saved candidates plus the exact number of provider invocations."""

    candidates: list[CandidateResult]
    provider_call_count: int


@dataclass
class ProviderCallCounter:
    """Job-scoped counter that also records failed provider invocations."""

    count: int = 0


class TryOnService:
    """Generate a requested count of candidates using an injected provider."""

    def __init__(self, client: TryOnAPIClient) -> None:
        self._client = client

    async def generate_candidates(
        self,
        *,
        person_image: Path,
        garment_image: Path,
        category: str,
        color: str,
        output_directory: Path,
        count: int,
        attempt: int,
        start_index: int,
        options: dict[str, Any],
        job_id: str | None = None,
        call_counter: ProviderCallCounter | None = None,
    ) -> CandidateBatch:
        output_directory.mkdir(parents=True, exist_ok=True)
        collected: list[bytes] = []
        calls = 0
        while len(collected) < count and calls < count:
            call_options = {
                **options,
                "candidate_count": count - len(collected),
                "requested_color": color,
            }
            calls += 1
            if call_counter is not None:
                call_counter.count += 1
            LOGGER.info(
                "provider_generation_call",
                extra={
                    "job_id": job_id,
                    "provider_generation_call_count": (
                        call_counter.count if call_counter is not None else calls
                    ),
                    "multi_garment": False,
                },
            )
            generated = await self._client.generate(
                person_image, garment_image, category, call_options
            )
            if not generated:
                raise TryOnAPIError("Try-on provider returned an empty candidate list.")
            collected.extend(generated[: count - len(collected)])
        if len(collected) < count:
            raise TryOnAPIError(
                f"Try-on provider returned {len(collected)} of {count} candidates."
            )

        results: list[CandidateResult] = []
        for offset, image_bytes in enumerate(collected):
            candidate_index = start_index + offset
            path = output_directory / f"candidate_{candidate_index:02d}.png"
            path.write_bytes(image_bytes)
            results.append(
                CandidateResult(
                    color=color,
                    path=path,
                    attempt=attempt,
                    candidate_index=candidate_index,
                )
            )
            LOGGER.info(
                "candidate_saved",
                extra={
                    "color": color,
                    "attempt": attempt,
                    "candidate_index": candidate_index,
                },
            )
        return CandidateBatch(results, calls)

    async def generate_multi_candidates(
        self,
        *,
        person_image: Path,
        garment_images: list[Path],
        garment_types: list[str],
        category: str,
        color: str,
        output_directory: Path,
        count: int,
        options: dict[str, Any],
        job_id: str | None = None,
        call_counter: ProviderCallCounter | None = None,
    ) -> CandidateBatch:
        """Generate a complete outfit using exactly one provider request."""

        if not self._client.supports_multi_reference:
            raise TryOnAPIError(
                "The configured generation provider cannot process multiple "
                "garment references in one request."
            )
        output_directory.mkdir(parents=True, exist_ok=True)
        if call_counter is not None:
            call_counter.count += 1
        LOGGER.info(
            "provider_generation_call",
            extra={
                "job_id": job_id,
                "provider_generation_call_count": (
                    call_counter.count if call_counter is not None else 1
                ),
                "multi_garment": True,
            },
        )
        generated = await self._client.generate_multi(
            person_image,
            garment_images,
            garment_types,
            category,
            {
                **options,
                "candidate_count": count,
                "requested_color": color,
            },
        )
        if len(generated) < count:
            raise TryOnAPIError(
                f"Try-on provider returned {len(generated)} of {count} candidates "
                "from the single multi-reference request."
            )
        results: list[CandidateResult] = []
        for offset, image_bytes in enumerate(generated[:count]):
            candidate_index = offset + 1
            path = output_directory / f"candidate_{candidate_index:02d}.png"
            path.write_bytes(image_bytes)
            results.append(
                CandidateResult(
                    color=color,
                    path=path,
                    attempt=0,
                    candidate_index=candidate_index,
                )
            )
            LOGGER.info(
                "multi_garment_candidate_saved",
                extra={
                    "color": color,
                    "candidate_index": candidate_index,
                    "garment_count": len(garment_images),
                    "provider_calls": 1,
                },
            )
        return CandidateBatch(results, 1)
