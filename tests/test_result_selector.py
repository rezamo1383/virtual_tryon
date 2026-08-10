from __future__ import annotations

from pathlib import Path

import pytest

from app.models.evaluation_models import OutputEvaluation
from app.models.result_models import CandidateResult
from app.services.output_evaluator import calculate_weighted_score
from app.services.result_selector import ResultSelector


def evaluation(score: float, *, accepted: bool = True) -> OutputEvaluation:
    return OutputEvaluation(
        identity_preservation=score,
        garment_similarity=score,
        color_accuracy=score,
        body_integrity=score,
        background_preservation=score,
        overall_score=score,
        accepted=accepted,
        problems=[],
    )


def candidate(index: int, score: float) -> CandidateResult:
    return CandidateResult(
        color="#000000",
        path=Path(f"candidate_{index}.png"),
        attempt=0,
        candidate_index=index,
        evaluation=evaluation(score),
    )


def test_selects_highest_score() -> None:
    selected = ResultSelector(0.8).select_best(
        [candidate(1, 0.81), candidate(2, 0.94)]
    )
    assert selected.candidate_index == 2


def test_rejects_below_threshold() -> None:
    selected = candidate(1, 0.79)
    assert ResultSelector(0.8).is_accepted(selected) is False


def test_weighted_score() -> None:
    item = OutputEvaluation(
        identity_preservation=1.0,
        garment_similarity=0.8,
        color_accuracy=0.6,
        body_integrity=0.4,
        background_preservation=0.2,
        overall_score=0.0,
        accepted=True,
        problems=[],
    )
    assert calculate_weighted_score(item) == pytest.approx(0.70)


def test_weighted_score_is_bounded() -> None:
    assert 0 <= calculate_weighted_score(evaluation(1.0)) <= 1
