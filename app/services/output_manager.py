"""Job directory layout and output persistence."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.models.result_models import CandidateResult
from app.utils.json_utils import write_json


class OutputManager:
    """Own creation and persistence of the public job artifact tree."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def create_job_directory(self, job_id: str) -> Path:
        path = self.output_root / job_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    @staticmethod
    def write_request(
        job_directory: Path,
        request: BaseModel | dict[str, Any],
    ) -> None:
        """Persist a request from any registered domain."""

        write_json(job_directory / "request.json", request)

    @staticmethod
    def copy_artifact(source: Path, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    @staticmethod
    def write_candidate_metadata(
        job_directory: Path, candidates: list[CandidateResult]
    ) -> None:
        write_json(
            job_directory / "candidate_metadata.json",
            {"candidates": [item.model_dump(mode="json") for item in candidates]},
        )

    @staticmethod
    def write_metadata(
        job_directory: Path,
        filename: str,
        value: BaseModel | dict[str, Any],
    ) -> None:
        """Persist domain metadata under a safe fixed filename."""

        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("Metadata filename must be a plain .json name.")
        write_json(job_directory / filename, value)

    @staticmethod
    def write_result(job_directory: Path, result: BaseModel) -> None:
        """Persist a result from any registered domain."""

        write_json(job_directory / "results.json", result)
