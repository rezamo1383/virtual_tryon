"""Lightweight local human-presence detection without external APIs."""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
from PIL import Image

from app.preprocessing.preprocessing_models import (
    HumanParsingResult,
    PersonPresenceResult,
    PoseResult,
)

LOGGER = logging.getLogger(__name__)

_MIN_HOG_PERSON_HEIGHT_RATIO = 0.30
# A real person can occupy a small part of an environmental photo. Require
# only a small global footprint here and rely on multiple semantic regions to
# distinguish it from isolated parsing artifacts.
_MIN_SEMANTIC_HUMAN_COVERAGE = 0.003
_MIN_SEMANTIC_REGION_COUNT = 2


class PersonPresenceDetector:
    """Detect a person using an existing pose, HOG body, or Haar face."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(
        self,
        image: Image.Image,
        pose: PoseResult | None = None,
    ) -> PersonPresenceResult:
        """Return positive when any local detector finds a human."""

        if pose is not None and pose.detected_person_count > 0:
            return PersonPresenceResult(
                detected=True,
                detected_person_count=pose.detected_person_count,
                confidence=pose.pose_confidence,
                model_name="mediapipe_pose",
            )

        bgr = cv2.cvtColor(
            np.asarray(image.convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )
        bgr = self._downscale(bgr, max_dimension=1200)
        with self._lock:
            boxes, weights = self._hog.detectMultiScale(
                bgr,
                hitThreshold=0.0,
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.05,
            )
        image_height = bgr.shape[0]
        plausible = [
            (box, weight)
            for box, weight in zip(boxes, weights, strict=False)
            if float(box[3]) / max(1, image_height)
            >= _MIN_HOG_PERSON_HEIGHT_RATIO
        ]
        if plausible:
            confidence = max(
                (float(weight) for _, weight in plausible),
                default=0.0,
            )
            return PersonPresenceResult(
                detected=True,
                detected_person_count=len(plausible),
                confidence=max(0.0, min(1.0, confidence)),
                model_name="opencv_hog_person",
            )

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        minimum_face_size = max(40, round(min(gray.shape) * 0.07))
        with self._lock:
            faces = (
                ()
                if self._face.empty()
                else self._face.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(minimum_face_size, minimum_face_size),
                )
            )
        if len(faces):
            return PersonPresenceResult(
                detected=True,
                detected_person_count=len(faces),
                confidence=0.75,
                model_name="opencv_haar_face",
            )

        return PersonPresenceResult(
            detected=False,
            detected_person_count=0,
            confidence=0.0,
            model_name="opencv_hog_and_haar",
        )

    @staticmethod
    def has_semantic_human_evidence(parsing: HumanParsingResult) -> bool:
        """Reject low-coverage, single-region parsing false positives."""

        if parsing.degraded_mode:
            return True
        class_map = np.asarray(parsing.class_map)
        if class_map.size == 0:
            return False
        foreground = class_map != 0
        coverage = float(np.count_nonzero(foreground) / class_map.size)
        region_count = int(np.count_nonzero(np.unique(class_map) != 0))
        return (
            coverage >= _MIN_SEMANTIC_HUMAN_COVERAGE
            and region_count >= _MIN_SEMANTIC_REGION_COUNT
        )

    @staticmethod
    def _downscale(
        image: np.ndarray,
        *,
        max_dimension: int,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(width, height)
        if longest <= max_dimension:
            return image
        scale = max_dimension / longest
        return cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
