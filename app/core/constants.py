"""Application-wide constants."""

from __future__ import annotations

from typing import Final

ALLOWED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp"}
)
ALLOWED_IMAGE_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
ALLOWED_GARMENT_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"upper_body", "lower_body", "dress", "outerwear"}
)
COMMON_COLORS: Final[dict[str, str]] = {
    "red": "#C62828",
    "blue": "#1565C0",
    "black": "#000000",
    "white": "#FFFFFF",
    "green": "#2E7D32",
    "yellow": "#F9A825",
    "orange": "#EF6C00",
    "purple": "#6A1B9A",
    "pink": "#D81B60",
    "gray": "#757575",
    "grey": "#757575",
    "navy": "#0D1B3E",
    "brown": "#6D4C41",
    "beige": "#D7CCC8",
}
SCORE_WEIGHTS: Final[dict[str, float]] = {
    "identity_preservation": 0.30,
    "garment_similarity": 0.25,
    "color_accuracy": 0.20,
    "body_integrity": 0.15,
    "background_preservation": 0.10,
}
