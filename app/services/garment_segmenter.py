"""Garment foreground segmentation with layered fallbacks."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.exceptions import ImageProcessingError
from app.utils.image_utils import clean_binary_mask, save_privacy_safe_png

LOGGER = logging.getLogger(__name__)


class GarmentSegmenter:
    """Create a normalized RGBA garment and a cleaned, feathered mask."""

    def segment(
        self,
        image_path: Path,
        output_directory: Path,
    ) -> tuple[Path, Path]:
        """Segment the garment and write normalized image and mask."""

        output_directory.mkdir(parents=True, exist_ok=True)
        normalized_path = output_directory / "garment_normalized.png"
        mask_path = output_directory / "garment_mask.png"
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGBA")
            array = np.asarray(image)
            alpha = array[:, :, 3]
            if alpha.min() < 250 and np.count_nonzero(alpha > 16) < alpha.size * 0.995:
                raw_mask = alpha
                LOGGER.info("garment_segmentation_alpha_mask")
            else:
                rembg_result = self._try_rembg(image)
                if rembg_result is not None:
                    image, raw_mask = rembg_result
                    array = np.asarray(image)
                    LOGGER.info("garment_segmentation_rembg")
                else:
                    raw_mask = self._background_color_fallback(array[:, :, :3])
                    LOGGER.info("garment_segmentation_color_fallback")
            cleaned = clean_binary_mask(raw_mask)
            feathered = cv2.GaussianBlur(cleaned, (5, 5), sigmaX=1.0)
            rgba = np.asarray(image).copy()
            rgba[:, :, 3] = feathered
            save_privacy_safe_png(Image.fromarray(rgba), normalized_path)
            Image.fromarray(feathered).save(mask_path, format="PNG", optimize=True)
            return normalized_path, mask_path
        except ImageProcessingError:
            raise
        except Exception as exc:
            raise ImageProcessingError(f"Garment segmentation failed: {exc}") from exc

    @staticmethod
    def _try_rembg(image: Image.Image) -> tuple[Image.Image, np.ndarray] | None:
        try:
            from rembg import remove  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            result = remove(buffer.getvalue())
            output = Image.open(io.BytesIO(result)).convert("RGBA")
            alpha = np.asarray(output)[:, :, 3]
            if np.count_nonzero(alpha > 16) < alpha.size * 0.01:
                return None
            return output, alpha
        except Exception:
            LOGGER.warning("rembg_failed_using_fallback", exc_info=False)
            return None

    @staticmethod
    def _background_color_fallback(rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        patch = max(3, min(height, width) // 20)
        corners = np.concatenate(
            [
                rgb[:patch, :patch].reshape(-1, 3),
                rgb[:patch, -patch:].reshape(-1, 3),
                rgb[-patch:, :patch].reshape(-1, 3),
                rgb[-patch:, -patch:].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(corners, axis=0)
        lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(
            np.uint8([[background]]), cv2.COLOR_RGB2LAB
        ).astype(np.float32)[0, 0]
        distance = np.linalg.norm(lab - bg_lab, axis=2)
        threshold = max(12.0, float(np.percentile(distance, 35)))
        return np.where(distance > threshold, 255, 0).astype(np.uint8)
