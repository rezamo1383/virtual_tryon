"""Quality-retry policy."""

from __future__ import annotations

from typing import Any


class RetryManager:
    """Bound output-quality retries and strengthen preservation options."""

    STRICT_OPTIONS = {
        "preserve_face": True,
        "preserve_pose": True,
        "preserve_background": True,
        "strict_identity_preservation": True,
        "only_replace_garment_region": True,
    }

    @staticmethod
    def should_retry(accepted: bool, retry_count: int, max_retries: int) -> bool:
        return not accepted and retry_count < max_retries

    @classmethod
    def options_for_attempt(
        cls,
        base_options: dict[str, Any],
        attempt: int,
        *,
        strict_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = dict(base_options)
        if attempt > 0:
            options.update(strict_options or cls.STRICT_OPTIONS)
        options["quality_retry_attempt"] = attempt
        return options
