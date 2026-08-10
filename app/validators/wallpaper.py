"""Wallpaper-domain validation."""

from __future__ import annotations

from app.core.config import Settings
from app.models.request_models import GenerationRequest, WallpaperOptions
from app.validators.common import CommonImagePairValidator


class WallpaperValidator:
    """Validate room and wallpaper references independently of FastAPI."""

    def __init__(self, settings: Settings) -> None:
        self._common = CommonImagePairValidator(settings)

    def validate(self, request: GenerationRequest) -> GenerationRequest:
        WallpaperOptions.model_validate(request.options)
        return self._common.validate(
            request,
            source_role="room",
            reference_role="wallpaper",
        )
