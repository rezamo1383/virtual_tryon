"""Common validation reusable by every visual generation domain."""

from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import InputValidationError
from app.models.request_models import GenerationRequest
from app.services.input_validator import InputValidator


class CommonImagePairValidator:
    """Validate two distinct images using the existing hardened validator."""

    def __init__(self, settings: Settings) -> None:
        self._images = InputValidator(settings)

    def validate(
        self,
        request: GenerationRequest,
        *,
        source_role: str,
        reference_role: str,
    ) -> GenerationRequest:
        source = self._images.validate_image(
            request.source_image,
            role=source_role,
        )
        reference = self._images.validate_image(
            request.reference_image,
            role=reference_role,
        )
        if source == reference:
            raise InputValidationError(
                f"{source_role} and {reference_role} must be distinct files."
            )
        return request.model_copy(
            update={
                "source_image": source,
                "reference_image": reference,
            }
        )
