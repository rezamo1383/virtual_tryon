"""Server-Sent Events for persisted job lifecycle changes."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from itertools import count
from typing import Protocol

LOGGER = logging.getLogger(__name__)

TERMINAL_JOB_STATUSES = frozenset(
    {
        "completed",
        "completed_with_failures",
        "failed",
        "rejected",
    }
)


class DisconnectAwareRequest(Protocol):
    """Small request surface needed by the event generator."""

    async def is_disconnected(self) -> bool: ...


class JobEventStreamer:
    """Stream changes from the existing on-disk job state."""

    def __init__(
        self,
        *,
        poll_interval_seconds: float = 0.5,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        if poll_interval_seconds <= 0 or heartbeat_interval_seconds <= 0:
            raise ValueError("SSE intervals must be positive.")
        self.poll_interval_seconds = poll_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._tokens = count(1)
        self._active_streams: set[int] = set()

    @property
    def active_stream_count(self) -> int:
        """Expose active generator count for health diagnostics and tests."""

        return len(self._active_streams)

    async def stream(
        self,
        *,
        request: DisconnectAwareRequest,
        job_id: str,
        initial_status: str,
        load_state: Callable[[], dict[str, object]],
    ) -> AsyncIterator[str]:
        """Emit the current status, changes, and idle heartbeats."""

        token = next(self._tokens)
        self._active_streams.add(token)
        last_status = initial_status
        loop = asyncio.get_running_loop()
        last_emission = loop.time()
        try:
            yield _status_event(job_id, initial_status)
            if initial_status in TERMINAL_JOB_STATUSES:
                return

            while True:
                await asyncio.sleep(self.poll_interval_seconds)
                if await request.is_disconnected():
                    LOGGER.info(
                        "job_event_stream_disconnected",
                        extra={"job_id": job_id},
                    )
                    return

                try:
                    state = await asyncio.to_thread(load_state)
                except (OSError, ValueError):
                    LOGGER.warning(
                        "job_event_state_unavailable",
                        extra={"job_id": job_id},
                    )
                    return
                current_status = state.get("status")
                if not isinstance(current_status, str) or not current_status:
                    LOGGER.warning(
                        "job_event_status_invalid",
                        extra={"job_id": job_id},
                    )
                    return

                now = loop.time()
                if current_status != last_status:
                    yield _status_event(job_id, current_status)
                    last_status = current_status
                    last_emission = now
                    if current_status in TERMINAL_JOB_STATUSES:
                        return
                elif now - last_emission >= self.heartbeat_interval_seconds:
                    yield ": keep-alive\n\n"
                    last_emission = now
        finally:
            self._active_streams.discard(token)


def _status_event(job_id: str, status: str) -> str:
    payload = json.dumps(
        {"job_id": job_id, "status": status},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: status\ndata: {payload}\n\n"
