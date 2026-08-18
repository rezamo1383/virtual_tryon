"""Lightweight structured timing helpers."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator


def log_stage_timing(
    logger: logging.Logger,
    *,
    pipeline: str,
    stage: str,
    started: float,
    status: str = "completed",
    **context: Any,
) -> int:
    """Log one stage duration and return its rounded milliseconds."""

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "pipeline_stage_timing",
        extra={
            "pipeline": pipeline,
            "stage": stage,
            "status": status,
            "elapsed_ms": elapsed_ms,
            **context,
        },
    )
    return elapsed_ms


@contextmanager
def timed_stage(
    logger: logging.Logger,
    *,
    pipeline: str,
    stage: str,
    **context: Any,
) -> Iterator[None]:
    """Log elapsed time for a synchronous or asynchronous code block."""

    started = time.perf_counter()
    status = "completed"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        log_stage_timing(
            logger,
            pipeline=pipeline,
            stage=stage,
            started=started,
            status=status,
            **context,
        )
