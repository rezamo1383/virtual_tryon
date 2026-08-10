"""Safe path and filesystem helpers."""

from __future__ import annotations

import re
import secrets
import shutil
from pathlib import Path


def safe_slug(value: str, fallback: str = "item") -> str:
    """Convert untrusted text into a short safe filename component."""

    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return (cleaned or fallback)[:64]


def secure_temp_name(suffix: str = ".png") -> str:
    """Return an unpredictable filename, preserving only a controlled suffix."""

    return f"{secrets.token_hex(16)}{suffix.lower()}"


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve a path and ensure it stays within a trusted root."""

    resolved = path.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Path escapes trusted directory: {path}")
    return resolved


def remove_tree(path: Path, allowed_root: Path) -> None:
    """Remove a job temp directory only after containment verification."""

    resolved = ensure_within(path, allowed_root)
    if resolved.exists() and resolved != allowed_root.resolve():
        shutil.rmtree(resolved)
