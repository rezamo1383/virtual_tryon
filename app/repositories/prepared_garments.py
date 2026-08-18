"""Tenant-isolated storage for preprocessed product garments."""

from __future__ import annotations

import re
import secrets
import shutil
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import (
    InputValidationError,
    PreparedGarmentNotFoundError,
    ProductNotFoundError,
)
from app.preprocessing.preprocessing_models import GarmentProcessingResult
from app.utils.file_utils import ensure_within
from app.utils.json_utils import read_json, write_json

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ARTIFACT_NAMES = {
    "normalized_image_path": "normalized.png",
    "transparent_image_path": "transparent_cropped.png",
    "garment_mask_path": "garment_mask.png",
}


class PreparedGarment(BaseModel):
    """Complete durable prepared garment resolved to safe local paths."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    product_id: str
    source_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    prepared_at: datetime
    processing: GarmentProcessingResult


class PreparedGarmentRepository(ABC):
    """Replaceable repository contract for prepared product garments."""

    @abstractmethod
    def find(self, tenant_id: str, product_id: str) -> PreparedGarment | None:
        """Return a prepared product or ``None`` when the product is unknown."""

    def get(self, tenant_id: str, product_id: str) -> PreparedGarment:
        """Return a complete prepared garment with clear not-found errors."""

        prepared = self.find(tenant_id, product_id)
        if prepared is None:
            raise ProductNotFoundError("Product was not found for this tenant.")
        return prepared

    @abstractmethod
    def store(
        self,
        *,
        tenant_id: str,
        product_id: str,
        source_sha256: str,
        processing: GarmentProcessingResult,
    ) -> PreparedGarment:
        """Atomically replace one product's prepared artifacts."""


class FilesystemPreparedGarmentRepository(PreparedGarmentRepository):
    """Filesystem implementation with strict identifiers and atomic writes."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def find(self, tenant_id: str, product_id: str) -> PreparedGarment | None:
        directory = self._product_directory(tenant_id, product_id)
        if not directory.exists():
            return None
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise PreparedGarmentNotFoundError(
                "Prepared garment metadata is missing for this product."
            )
        try:
            metadata = read_json(metadata_path)
            if metadata.get("schema_version") != 1:
                raise ValueError("unsupported schema")
            if (
                metadata.get("tenant_id") != tenant_id
                or metadata.get("product_id") != product_id
            ):
                raise ValueError("ownership mismatch")
            processing_data = metadata.get("processing")
            if not isinstance(processing_data, dict):
                raise ValueError("processing metadata missing")
            resolved_processing = dict(processing_data)
            for field, filename in _ARTIFACT_NAMES.items():
                stored = processing_data.get(field)
                if stored != filename:
                    raise ValueError("unsafe artifact metadata")
                artifact = ensure_within(directory / filename, directory)
                if not artifact.is_file():
                    raise ValueError("artifact missing")
                resolved_processing[field] = artifact
            return PreparedGarment(
                tenant_id=tenant_id,
                product_id=product_id,
                source_sha256=str(metadata["source_sha256"]),
                prepared_at=metadata["prepared_at"],
                processing=GarmentProcessingResult.model_validate(resolved_processing),
            )
        except PreparedGarmentNotFoundError:
            raise
        except Exception as exc:
            raise PreparedGarmentNotFoundError(
                "Prepared garment artifacts are incomplete or invalid."
            ) from exc

    def store(
        self,
        *,
        tenant_id: str,
        product_id: str,
        source_sha256: str,
        processing: GarmentProcessingResult,
    ) -> PreparedGarment:
        target = self._product_directory(tenant_id, product_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = ensure_within(
            target.parent / f".{product_id}.{secrets.token_hex(8)}.staging",
            target.parent,
        )
        backup = ensure_within(
            target.parent / f".{product_id}.{secrets.token_hex(8)}.backup",
            target.parent,
        )
        staging.mkdir(parents=False, exist_ok=False)
        prepared_at = datetime.now(UTC)
        try:
            metadata_processing = processing.model_dump(mode="json")
            for field, filename in _ARTIFACT_NAMES.items():
                source = Path(getattr(processing, field)).resolve(strict=True)
                if not source.is_file():
                    raise InputValidationError(
                        "Prepared garment source artifact is missing."
                    )
                shutil.copy2(source, staging / filename)
                metadata_processing[field] = filename
            write_json(
                staging / "metadata.json",
                {
                    "schema_version": 1,
                    "tenant_id": tenant_id,
                    "product_id": product_id,
                    "source_sha256": source_sha256,
                    "prepared_at": prepared_at.isoformat(),
                    "processing": metadata_processing,
                },
            )
            if target.exists():
                target.replace(backup)
            staging.replace(target)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        prepared = self.find(tenant_id, product_id)
        if prepared is None:  # pragma: no cover - guarded by atomic store
            raise PreparedGarmentNotFoundError(
                "Prepared garment could not be loaded after storage."
            )
        return prepared

    def _product_directory(self, tenant_id: str, product_id: str) -> Path:
        self._validate_identifier(tenant_id, "tenant_id", 100)
        self._validate_identifier(product_id, "product_id", 128)
        return ensure_within(self.root / tenant_id / product_id, self.root)

    @staticmethod
    def _validate_identifier(value: str, field: str, maximum: int) -> None:
        if len(value) > maximum or not _IDENTIFIER_RE.fullmatch(value):
            raise InputValidationError(f"Invalid {field}.")
