"""Clothing-domain validation adapter."""

from __future__ import annotations

from app.core.config import Settings
from app.models.request_models import TryOnRequest
from app.services.input_validator import InputValidator


class ClothingValidator:
    """Preserve the complete existing clothing validation contract."""

    def __init__(self, settings: Settings) -> None:
        self._legacy = InputValidator(settings)

    def validate(self, request: TryOnRequest) -> TryOnRequest:
        return self._legacy.validate(request)
