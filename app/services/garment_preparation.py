"""Idempotent, offline preparation of reusable product garments."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import InputValidationError
from app.models.prepared_garment_models import GarmentPreparationResult
from app.preprocessing.image_preprocessor import LocalImagePreprocessor
from app.repositories.prepared_garments import (
    PreparedGarment,
    PreparedGarmentRepository,
)
from app.services.input_validator import InputValidator
from app.utils.file_utils import remove_tree, secure_temp_name
from app.utils.timing import log_stage_timing

import logging

LOGGER = logging.getLogger(__name__)


class GarmentPreparationService:
    """Validate, preprocess, and store a product garment once."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: PreparedGarmentRepository,
        preprocessor: LocalImagePreprocessor,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.preprocessor = preprocessor
        self.validator = InputValidator(settings)
        self._lock = asyncio.Lock()

    async def prepare(
        self,
        *,
        tenant_id: str,
        product_id: str,
        garment_image: Path,
        force: bool = False,
    ) -> GarmentPreparationResult:
        """Prepare idempotently, reusing artifacts for unchanged source bytes."""

        started = time.perf_counter()
        validated = self.validator.validate_image(garment_image, role="garment")
        source_sha256 = await asyncio.to_thread(_sha256_file, validated)
        async with self._lock:
            existing = self.repository.find(tenant_id, product_id)
            if (
                existing is not None
                and existing.source_sha256 == source_sha256
                and not force
            ):
                log_stage_timing(
                    LOGGER,
                    pipeline="clothing_product_preparation",
                    stage="total",
                    started=started,
                    cached=True,
                )
                return _result(existing, cached=True)

            work = self.settings.temp_directory / f"prepare_{secure_temp_name('')}"
            work.mkdir(parents=True, exist_ok=False)
            try:
                operation = asyncio.to_thread(
                    self.preprocessor.preprocess_garment,
                    validated,
                    work,
                )
                processing = await asyncio.wait_for(
                    operation,
                    timeout=self.settings.preprocessing_timeout_seconds,
                )
                if not processing.validation.accepted:
                    reasons = "; ".join(processing.validation.rejection_reasons)
                    raise InputValidationError(
                        "Garment preparation was rejected: "
                        + (
                            reasons
                            or "Garment suitability score is below the configured threshold."
                        )
                    )
                prepared = self.repository.store(
                    tenant_id=tenant_id,
                    product_id=product_id,
                    source_sha256=source_sha256,
                    processing=processing,
                )
                log_stage_timing(
                    LOGGER,
                    pipeline="clothing_product_preparation",
                    stage="total",
                    started=started,
                    cached=False,
                )
                return _result(prepared, cached=False)
            finally:
                remove_tree(work, self.settings.temp_directory)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result(
    prepared: PreparedGarment,
    *,
    cached: bool,
) -> GarmentPreparationResult:
    return GarmentPreparationResult(
        tenant_id=prepared.tenant_id,
        product_id=prepared.product_id,
        cached=cached,
        source_sha256=prepared.source_sha256,
        prepared_at=prepared.prepared_at,
        normalized_image_path=prepared.processing.normalized_image_path,
        transparent_image_path=prepared.processing.transparent_image_path,
        garment_mask_path=prepared.processing.garment_mask_path,
        validation=prepared.processing.validation,
    )
