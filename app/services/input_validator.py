"""Request and image validation at the trust boundary."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.core.constants import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_IMAGE_MIME_TYPES
from app.core.exceptions import InputValidationError
from app.models.request_models import ORIGINAL_GARMENT_COLOR, TryOnRequest
from app.utils.file_utils import secure_temp_name
from app.utils.image_utils import normalize_color

PIL_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class InputValidator:
    """Validate filesystem inputs before any model or provider is called."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def validate(self, request: TryOnRequest) -> TryOnRequest:
        """Validate the complete request and return canonical absolute paths."""

        person = self.validate_image(request.person_image, role="person")
        garment = self.validate_image(request.garment_image, role="garment")
        if person == garment:
            raise InputValidationError(
                "Person image and garment image must be two distinct files."
            )
        for color in request.colors:
            if color.casefold() == ORIGINAL_GARMENT_COLOR:
                continue
            try:
                normalize_color(color)
            except ValueError as exc:
                raise InputValidationError(str(exc)) from exc
        if request.candidates_per_color < 1 or request.candidates_per_color > 8:
            raise InputValidationError("candidates_per_color must be between 1 and 8.")
        return request.model_copy(
            update={"person_image": person, "garment_image": garment}
        )

    def validate_image(self, path: Path, *, role: str) -> Path:
        """Verify path safety, size, signature, dimensions, and decodability."""

        if ".." in path.parts:
            raise InputValidationError(f"{role} image path traversal is not allowed.")
        try:
            resolved = path.expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise InputValidationError(f"{role} image does not exist: {path}") from exc
        if not resolved.is_file():
            raise InputValidationError(f"{role} image is not a regular file: {path}")
        extension = resolved.suffix.lower()
        if extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise InputValidationError(
                f"{role} image extension '{extension}' is not supported."
            )
        maximum_bytes = self._settings.max_image_size_mb * 1024 * 1024
        if resolved.stat().st_size > maximum_bytes:
            raise InputValidationError(
                f"{role} image exceeds {self._settings.max_image_size_mb} MB."
            )
        try:
            with Image.open(resolved) as image:
                image.verify()
                actual_format = image.format
            with Image.open(resolved) as image:
                image.load()
                width, height = image.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InputValidationError(
                f"{role} image is corrupt or undecodable."
            ) from exc
        actual_mime = PIL_MIME_BY_FORMAT.get(actual_format or "")
        if actual_mime not in ALLOWED_IMAGE_MIME_TYPES:
            raise InputValidationError(
                f"{role} image has unsupported content type '{actual_mime}'."
            )
        expected_formats = {
            ".jpg": {"JPEG"},
            ".jpeg": {"JPEG"},
            ".png": {"PNG"},
            ".webp": {"WEBP"},
        }
        if actual_format not in expected_formats[extension]:
            raise InputValidationError(
                f"{role} file extension does not match its actual image type."
            )
        if (
            width < self._settings.min_image_width
            or height < self._settings.min_image_height
        ):
            raise InputValidationError(
                f"{role} image is too small: {width}x{height}; minimum is "
                f"{self._settings.min_image_width}x{self._settings.min_image_height}."
            )
        if max(width, height) > self._settings.max_image_dimension:
            raise InputValidationError(
                f"{role} image exceeds maximum dimension "
                f"{self._settings.max_image_dimension}."
            )
        ratio = width / height
        if not (
            self._settings.min_aspect_ratio
            <= ratio
            <= self._settings.max_aspect_ratio
        ):
            raise InputValidationError(
                f"{role} image aspect ratio {ratio:.2f} is outside the accepted range."
            )
        return resolved

    @staticmethod
    def create_safe_temp_filename(suffix: str = ".png") -> str:
        """Create a random filename that does not trust input names."""

        if suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            suffix = ".png"
        return secure_temp_name(suffix)
