"""Hashing and identifier helpers."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime


def create_job_id() -> str:
    """Create a sortable, non-identifying job identifier."""

    date = datetime.now(UTC).strftime("%Y%m%d")
    return f"job_{date}_{secrets.token_hex(3)}"


def short_hash(value: str) -> str:
    """Return a short stable SHA-256 identifier."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
