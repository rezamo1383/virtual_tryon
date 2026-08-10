"""Texture-preserving garment recoloring."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.exceptions import ImageProcessingError
from app.utils.image_utils import normalize_color


class GarmentColorizer:
    """Recolor masked pixels while retaining luminance, folds, and highlights."""

    def create_variant(
        self,
        image_path: Path,
        mask_path: Path,
        target_color: str,
        output_path: Path,
    ) -> Path:
        """Create one RGBA garment color variant."""

        try:
            canonical, rgb_target = normalize_color(target_color)
            with Image.open(image_path) as source:
                original = np.asarray(ImageOps.exif_transpose(source).convert("RGBA"))
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L"), dtype=np.float32) / 255.0
            if mask.shape != original.shape[:2]:
                raise ImageProcessingError("Garment mask dimensions do not match image.")

            rgb = original[:, :, :3].copy()
            recolored = self._recolor_rgb(rgb, canonical, rgb_target)
            alpha = mask[:, :, None]
            blended = np.clip(
                rgb.astype(np.float32) * (1.0 - alpha)
                + recolored.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)
            # Pixels completely outside the mask remain byte-identical.
            blended[mask <= 0] = rgb[mask <= 0]
            output = np.dstack([blended, original[:, :, 3]])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(output).save(output_path, format="PNG", optimize=True)
            return output_path
        except ImageProcessingError:
            raise
        except Exception as exc:
            raise ImageProcessingError(f"Garment colorization failed: {exc}") from exc

    @staticmethod
    def _recolor_rgb(
        rgb: np.ndarray, canonical: str, target_rgb: tuple[int, int, int]
    ) -> np.ndarray:
        source_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        luminance = source_lab[:, :, 0].astype(np.float32) / 255.0
        local_detail = luminance - cv2.GaussianBlur(luminance, (0, 0), 5)

        if canonical == "#FFFFFF":
            value = np.clip(0.70 + luminance * 0.28 + local_detail * 0.9, 0, 1)
            saturation = np.clip(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
                * 0.12,
                0,
                30,
            )
            hsv = np.zeros_like(rgb)
            hsv[:, :, 1] = saturation.astype(np.uint8)
            hsv[:, :, 2] = (value * 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if canonical == "#000000":
            value = np.clip(0.055 + luminance * 0.25 + local_detail * 0.8, 0.03, 0.34)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = (hsv[:, :, 1].astype(np.float32) * 0.25).astype(np.uint8)
            hsv[:, :, 2] = (value * 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        target_pixel = np.uint8([[target_rgb]])
        target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2HSV)[0, 0]
        source_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv = source_hsv.copy()
        hsv[:, :, 0] = target_hsv[0]
        source_saturation = source_hsv[:, :, 1].astype(np.float32)
        target_saturation = float(target_hsv[1])
        hsv[:, :, 1] = np.clip(
            target_saturation * 0.78 + source_saturation * 0.22, 0, 255
        ).astype(np.uint8)
        target_value = float(target_hsv[2]) / 255.0
        value = np.clip(
            luminance * (0.58 + target_value * 0.42) + local_detail * 0.65,
            0.03,
            1.0,
        )
        hsv[:, :, 2] = (value * 255).astype(np.uint8)
        hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        target_lab = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2LAB)[0, 0]
        lab = cv2.cvtColor(hsv_rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = source_lab[:, :, 0]
        lab[:, :, 1] = np.clip(
            lab[:, :, 1].astype(np.float32) * 0.65 + float(target_lab[1]) * 0.35,
            0,
            255,
        ).astype(np.uint8)
        lab[:, :, 2] = np.clip(
            lab[:, :, 2].astype(np.float32) * 0.65 + float(target_lab[2]) * 0.35,
            0,
            255,
        ).astype(np.uint8)
        lab_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return cv2.addWeighted(hsv_rgb, 0.65, lab_rgb, 0.35, 0)
