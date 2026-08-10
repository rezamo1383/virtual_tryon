"""Candidate generation and durable storage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.clients.tryon_api_client import TryOnAPIClient
from app.core.exceptions import TryOnAPIError
from app.models.result_models import CandidateResult

LOGGER = logging.getLogger(__name__)


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
    ) -> list[CandidateResult]:
        output_directory.mkdir(parents=True, exist_ok=True)
        collected: list[bytes] = []
        calls = 0
        while len(collected) < count and calls < count:
            call_options = {
                **options,
                "candidate_count": count - len(collected),
                "requested_color": color,
            }
            generated = await self._client.generate(
                person_image, garment_image, category, call_options
            )
            if not generated:
                raise TryOnAPIError("Try-on provider returned an empty candidate list.")
            collected.extend(generated[: count - len(collected)])
            calls += 1
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
        return results
