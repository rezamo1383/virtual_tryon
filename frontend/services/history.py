"""Session-local recent job management."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, MutableMapping


HISTORY_KEY = "recent_jobs"


def initialize_history(state: MutableMapping[str, Any]) -> None:
    state.setdefault(HISTORY_KEY, [])


def add_job(
    state: MutableMapping[str, Any],
    record: dict[str, Any],
    *,
    limit: int,
) -> None:
    initialize_history(state)
    enriched = {
        "created_at": datetime.now(UTC).isoformat(),
        **record,
    }
    state[HISTORY_KEY] = [
        enriched,
        *[
            item
            for item in state[HISTORY_KEY]
            if item.get("job_id") != enriched.get("job_id")
        ],
    ][:limit]


def clear_history(state: MutableMapping[str, Any]) -> None:
    state[HISTORY_KEY] = []


def find_job(
    state: MutableMapping[str, Any],
    job_id: str | None,
) -> dict[str, Any] | None:
    if not job_id:
        return None
    initialize_history(state)
    return next(
        (
            item
            for item in state[HISTORY_KEY]
            if item.get("job_id") == job_id
        ),
        None,
    )
