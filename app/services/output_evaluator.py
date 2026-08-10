"""Candidate evaluation with application-owned score calculation."""

from __future__ import annotations

from pathlib import Path

from app.clients.qwen_client import QwenClient
from app.core.constants import SCORE_WEIGHTS
from app.core.exceptions import OutputEvaluationError
from app.models.evaluation_models import OutputEvaluation


def calculate_weighted_score(evaluation: OutputEvaluation) -> float:
    """Recompute and clamp the policy-owned quality score."""

    score = sum(
        getattr(evaluation, field) * weight for field, weight in SCORE_WEIGHTS.items()
    )
    return max(0.0, min(1.0, round(score, 6)))


class OutputEvaluator:
    """Evaluate candidates and disregard any LLM-provided overall calculation."""

    def __init__(self, client: QwenClient, minimum_score: float) -> None:
        self._client = client
        self._minimum_score = minimum_score

    async def evaluate(
        self,
        person_image: Path,
        garment_image: Path,
        output_image: Path,
        requested_color: str,
        product_title: str | None = None,
    ) -> OutputEvaluation:
        try:
            evaluation = await self._client.evaluate_output(
                person_image,
                garment_image,
                output_image,
                requested_color,
                product_title,
            )
            score = calculate_weighted_score(evaluation)
            accepted = score >= self._minimum_score and evaluation.accepted
            return evaluation.model_copy(
                update={"overall_score": score, "accepted": accepted}
            )
        except Exception as exc:
            raise OutputEvaluationError(
                f"Could not evaluate candidate '{output_image.name}': {exc}"
            ) from exc
