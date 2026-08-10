"""Metadata-free, aspect-ratio-preserving image normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometry needed to apply identical normalization to related masks."""

    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    target_size: tuple[int, int]
    offset: tuple[int, int]


def normalize_orientation(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and detach from source metadata."""

    return ImageOps.exif_transpose(image).copy()


def downscale_for_inference(
    image: Image.Image,
    max_dimension: int,
) -> Image.Image:
    """Bound inference memory without upscaling small inputs."""

    normalized = normalize_orientation(image)
    if max(normalized.size) <= max_dimension:
        return normalized
    scale = max_dimension / max(normalized.size)
    size = (
        max(1, round(normalized.width * scale)),
        max(1, round(normalized.height * scale)),
    )
    return normalized.resize(size, Image.Resampling.LANCZOS)


def letterbox_image(
    image: Image.Image,
    target_size: tuple[int, int],
    *,
    output_mode: str,
    allow_upscale: bool = False,
) -> tuple[Image.Image, LetterboxTransform]:
    """Resize without distortion and center on a fixed-size canvas."""

    source = normalize_orientation(image).convert(output_mode)
    target_width, target_height = target_size
    scale = min(target_width / source.width, target_height / source.height)
    if not allow_upscale:
        scale = min(1.0, scale)
    resized_size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )
    resized = source.resize(resized_size, Image.Resampling.LANCZOS)
    offset = (
        (target_width - resized.width) // 2,
        (target_height - resized.height) // 2,
    )
    fill = (0, 0, 0, 0) if output_mode == "RGBA" else (255, 255, 255)
    canvas = Image.new(output_mode, target_size, fill)
    if output_mode == "RGBA":
        canvas.alpha_composite(resized, offset)
    else:
        canvas.paste(resized, offset)
    return canvas, LetterboxTransform(
        source_size=source.size,
        resized_size=resized_size,
        target_size=target_size,
        offset=offset,
    )


def letterbox_mask(
    mask: Image.Image,
    transform: LetterboxTransform,
    *,
    feathered: bool = False,
) -> Image.Image:
    """Apply image normalization geometry to a grayscale mask."""

    source = mask.convert("L")
    if source.size != transform.source_size:
        source = source.resize(transform.source_size, Image.Resampling.NEAREST)
    resample = Image.Resampling.BILINEAR if feathered else Image.Resampling.NEAREST
    resized = source.resize(transform.resized_size, resample)
    canvas = Image.new("L", transform.target_size, 0)
    canvas.paste(resized, transform.offset)
    return canvas


def save_clean_image(
    image: Image.Image,
    path: Path,
    *,
    output_format: str = "PNG",
    jpeg_quality: int = 95,
) -> Path:
    """Save pixels only, without EXIF or source metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGBA" if "A" in image.getbands() and output_format == "PNG" else "RGB"
    clean = Image.fromarray(np.asarray(image.convert(mode)))
    options: dict[str, object] = {"optimize": True}
    if output_format == "JPEG":
        options["quality"] = jpeg_quality
    clean.save(path, format=output_format, **options)
    return path
