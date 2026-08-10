"""Expected local-preprocessing failures."""

from app.core.exceptions import VirtualTryOnError


class PreprocessingError(VirtualTryOnError):
    """Base exception for local preprocessing failures."""


class PersonNotDetectedError(PreprocessingError):
    """Raised when the person input contains no detectable human."""


class BackgroundRemovalError(PreprocessingError):
    """Raised when foreground extraction has no safe fallback."""


class PoseEstimationError(PreprocessingError):
    """Raised when pose estimation fails unexpectedly."""


class HumanParsingError(PreprocessingError):
    """Raised when semantic human parsing is unavailable or invalid."""


class ModelUnavailableError(PreprocessingError):
    """Raised when a required local model cannot be obtained."""


class PreprocessingPathError(PreprocessingError):
    """Raised when an artifact path escapes its job directory."""
