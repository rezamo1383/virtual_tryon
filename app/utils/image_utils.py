"""Image loading, normalization, color, and base64 helpers."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.constants import COMMON_COLORS

HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def normalize_color(value: str) -> tuple[str, tuple[int, int, int]]:
    """Resolve a color name or six-digit hex string to canonical hex and RGB."""

    cleaned = value.strip().lower()
    hex_value = COMMON_COLORS.get(cleaned, value.strip())
    if not HEX_COLOR_RE.fullmatch(hex_value):
        raise ValueError(f"Invalid color '{value}'. Use a supported name or #RRGGBB.")
    canonical = hex_value.upper()
    return canonical, tuple(int(canonical[index : index + 2], 16) for index in (1, 3, 5))


def open_image_safe(path: Path) -> Image.Image:
    """Fully decode an image and apply EXIF orientation."""

    with Image.open(path) as source:
        source.verify()
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).copy()


def save_privacy_safe_png(image: Image.Image, path: Path) -> Path:
    """Save a PNG without EXIF or source metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    clean = Image.fromarray(np.asarray(image.convert("RGBA")))
    clean.save(path, format="PNG", optimize=True)
    return path


def image_to_data_url(path: Path) -> str:
    """Encode an image as a data URL without logging its content."""

    image = open_image_safe(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_to_png_data_url(path: Path) -> str:
    """Encode an image as a lossless PNG data URL for image-to-image providers."""

    image = open_image_safe(path)
    if "A" in image.getbands():
        normalized = image.convert("RGBA")
    else:
        normalized = image.convert("RGB")
    buffer = io.BytesIO()
    normalized.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def decode_image_bytes(payload: bytes) -> bytes:
    """Validate that bytes are a decodable image and normalize them to PNG."""

    with Image.open(io.BytesIO(payload)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Clean a mask with morphology and retain meaningful components."""

    binary = np.where(mask > 32, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        minimum_area = max(64, int(binary.size * 0.0005))
        retained = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
                retained[labels == label] = 255
        binary = retained
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary, contours, -1, 255, thickness=cv2.FILLED)
    return binary
