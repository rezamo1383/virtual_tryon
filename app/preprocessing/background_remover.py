"""Local person and garment background removal with cached rembg sessions."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

from app.preprocessing.preprocessing_exceptions import BackgroundRemovalError
from app.preprocessing.preprocessing_models import BackgroundRemovalResult
from app.utils.image_utils import clean_binary_mask

LOGGER = logging.getLogger(__name__)


class BackgroundRemover:
    """Prefer rembg while retaining a deterministic local fallback."""

    _sessions: dict[tuple[str, str], Any] = {}
    _session_lock = threading.Lock()

    def __init__(
        self,
        *,
        enabled: bool,
        device: Literal["cpu", "cuda"],
        model_cache_directory: Path,
        person_model: str,
        garment_model: str,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.model_cache_directory = model_cache_directory
        self.person_model = person_model
        self.garment_model = garment_model

    def remove_background(
        self,
        image: Image.Image,
        subject_type: Literal["person", "garment"],
    ) -> BackgroundRemovalResult:
        """Return an RGBA cutout, grayscale mask, and execution metadata."""

        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        if alpha.min() < 250 and np.count_nonzero(alpha > 16) < alpha.size * 0.995:
            mask = clean_binary_mask(alpha)
            return self._result(
                rgba,
                mask,
                model_name="source_alpha",
                fallback_used=False,
            )
        if not self.enabled:
            return self._result(
                rgba,
                alpha,
                model_name="disabled",
                fallback_used=False,
            )
        try:
            return self._remove_with_rembg(rgba, subject_type)
        except Exception as exc:
            LOGGER.warning(
                "background_removal_fallback",
                extra={
                    "stage": f"{subject_type}_background_removal",
                    "device": self.device,
                    "fallback_used": True,
                    "warning": type(exc).__name__,
                },
            )
        try:
            mask = self._color_foreground_mask(np.asarray(rgba)[:, :, :3])
            return self._result(
                rgba,
                mask,
                model_name="local_color_fallback",
                fallback_used=True,
            )
        except Exception as exc:
            raise BackgroundRemovalError(
                f"{subject_type.title()} background removal failed: {exc}"
            ) from exc

    def _remove_with_rembg(
        self,
        image: Image.Image,
        subject_type: Literal["person", "garment"],
    ) -> BackgroundRemovalResult:
        try:
            from rembg import remove
        except ImportError as exc:
            raise BackgroundRemovalError(
                "rembg is not installed; install preprocessing dependencies."
            ) from exc
        model_name = (
            self.person_model if subject_type == "person" else self.garment_model
        )
        session = self._get_session(model_name)
        output = remove(image, session=session, force_return_bytes=False)
        if not isinstance(output, Image.Image):
            raise BackgroundRemovalError("rembg returned an unexpected result.")
        output = output.convert("RGBA")
        mask = clean_binary_mask(np.asarray(output.getchannel("A")))
        coverage = float(np.count_nonzero(mask > 16) / mask.size)
        if not 0.005 < coverage < 0.995:
            raise BackgroundRemovalError("rembg produced an implausible mask.")
        return self._result(
            output,
            mask,
            model_name=model_name,
            fallback_used=False,
        )

    def _get_session(self, model_name: str) -> Any:
        key = (model_name, self.device)
        with self._session_lock:
            if key in self._sessions:
                return self._sessions[key]
            try:
                from rembg import new_session
            except ImportError as exc:
                raise BackgroundRemovalError("rembg is not installed.") from exc
            self.model_cache_directory.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(
                "U2NET_HOME",
                str(self.model_cache_directory.resolve(strict=False)),
            )
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            session = new_session(model_name, providers=providers)
            self._sessions[key] = session
            return session

    @staticmethod
    def _result(
        image: Image.Image,
        mask_array: np.ndarray,
        *,
        model_name: str,
        fallback_used: bool,
    ) -> BackgroundRemovalResult:
        mask_array = clean_binary_mask(mask_array)
        rgba = np.asarray(image.convert("RGBA")).copy()
        rgba[:, :, 3] = mask_array
        coverage = float(np.count_nonzero(mask_array) / mask_array.size)
        confidence = max(0.0, min(1.0, 1.0 - abs(coverage - 0.45)))
        return BackgroundRemovalResult(
            image=Image.fromarray(rgba),
            mask=Image.fromarray(mask_array),
            confidence=confidence,
            model_name=model_name,
            fallback_used=fallback_used,
        )

    @staticmethod
    def _color_foreground_mask(rgb: np.ndarray) -> np.ndarray:
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
        lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(
            np.float32
        )
        bg_lab = cv2.cvtColor(
            np.uint8([[background]]), cv2.COLOR_RGB2LAB
        ).astype(np.float32)[0, 0]
        distance = np.linalg.norm(lab - bg_lab, axis=2)
        threshold = max(10.0, float(np.percentile(distance, 40)))
        mask = np.where(distance > threshold, 255, 0).astype(np.uint8)
        return clean_binary_mask(mask)
