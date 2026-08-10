"""Best-candidate selection policy."""

from __future__ import annotations

from app.models.result_models import CandidateResult


class ResultSelector:
    """Choose the highest independently scored candidate."""

    def __init__(self, minimum_score: float) -> None:
        self._minimum_score = minimum_score

    def select_best(self, candidates: list[CandidateResult]) -> CandidateResult:
        """Return the best evaluated candidate."""

        evaluated = [item for item in candidates if item.evaluation is not None]
        if not evaluated:
            raise ValueError("No evaluated candidates are available.")
        return max(
            evaluated,
            key=lambda item: (
                item.evaluation.overall_score if item.evaluation else -1,
                -item.attempt,
                -item.candidate_index,
            ),
        )

    def is_accepted(self, candidate: CandidateResult) -> bool:
        """Apply the configured score threshold and evaluation verdict."""

        evaluation = candidate.evaluation
        return bool(
            evaluation
            and evaluation.accepted
            and evaluation.overall_score >= self._minimum_score
        )
