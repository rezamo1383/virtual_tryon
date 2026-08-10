"""Atomic, cache-aware local model acquisition."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.preprocessing.preprocessing_exceptions import ModelUnavailableError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Download metadata for one optional local model."""

    filename: str
    url: str
    sha256: str = ""
    approximate_size_mb: int | None = None


class ModelManager:
    """Reuse cached weights and download atomically when permitted."""

    def __init__(
        self,
        cache_directory: Path,
        *,
        offline: bool,
        timeout_seconds: float,
    ) -> None:
        self.cache_directory = cache_directory
        self.offline = offline
        self.timeout_seconds = timeout_seconds

    def ensure(self, spec: ModelSpec) -> Path:
        """Return a verified cached model, downloading it at most once."""

        self.cache_directory.mkdir(parents=True, exist_ok=True)
        target = (self.cache_directory / spec.filename).resolve(strict=False)
        root = self.cache_directory.resolve(strict=False)
        if root not in target.parents:
            raise ModelUnavailableError("Model path escapes the model cache.")
        if target.is_file():
            self._verify_checksum(target, spec.sha256)
            return target
        if self.offline:
            raise ModelUnavailableError(
                f"Local model '{spec.filename}' is missing in offline mode."
            )
        if not spec.url.startswith("https://"):
            raise ModelUnavailableError(
                f"Model '{spec.filename}' has no safe HTTPS download URL."
            )
        temporary = target.with_suffix(target.suffix + ".download")
        try:
            request = urllib.request.Request(
                spec.url,
                headers={"User-Agent": "virtual-tryon-preprocessor/1.0"},
            )
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                with temporary.open("wb") as destination:
                    while chunk := response.read(1024 * 1024):
                        destination.write(chunk)
            self._verify_checksum(temporary, spec.sha256)
            temporary.replace(target)
            LOGGER.info(
                "local_model_downloaded",
                extra={
                    "stage": "model_download",
                    "model_name": spec.filename,
                    "output_size": target.stat().st_size,
                },
            )
            return target
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not download local model '{spec.filename}': {exc}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _verify_checksum(path: Path, expected: str) -> None:
        if not expected:
            return
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ModelUnavailableError(
                f"Checksum mismatch for local model '{path.name}'."
            )
