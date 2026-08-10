"""MediaPipe pose extraction and explainable upper-body heuristics."""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.preprocessing.preprocessing_models import Keypoint, PoseResult

LOGGER = logging.getLogger(__name__)

LANDMARK_INDICES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}
SKELETON_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)


def _distance(first: Keypoint, second: Keypoint) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _visible(
    keypoints: dict[str, Keypoint],
    name: str,
    threshold: float = 0.35,
) -> bool:
    return name in keypoints and keypoints[name].visibility >= threshold


def classify_orientation(
    keypoints: dict[str, Keypoint],
) -> str:
    """Classify orientation from shoulder width and depth asymmetry."""

    if not all(
        _visible(keypoints, name)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    ):
        return "unknown"
    left = keypoints["left_shoulder"]
    right = keypoints["right_shoulder"]
    width = _distance(left, right)
    torso = (
        _distance(left, keypoints["left_hip"])
        + _distance(right, keypoints["right_hip"])
    ) / 2
    if torso <= 0:
        return "unknown"
    width_ratio = width / torso
    depth_delta = abs((left.z or 0.0) - (right.z or 0.0))
    if width_ratio < 0.32 or depth_delta > 0.28:
        return "side"
    if width_ratio < 0.55 or depth_delta > 0.14:
        return "slightly_turned"
    return "frontal"


def _segments_intersect(
    first_start: Keypoint,
    first_end: Keypoint,
    second_start: Keypoint,
    second_end: Keypoint,
) -> bool:
    def orientation(
        a: Keypoint,
        b: Keypoint,
        c: Keypoint,
    ) -> float:
        return (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)

    return (
        orientation(first_start, first_end, second_start)
        * orientation(first_start, first_end, second_end)
        < 0
        and orientation(second_start, second_end, first_start)
        * orientation(second_start, second_end, first_end)
        < 0
    )


def classify_arms_position(
    keypoints: dict[str, Keypoint],
) -> str:
    """Classify arms using visibility, torso crossing, and relative height."""

    required = (
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    )
    if not all(_visible(keypoints, name) for name in required):
        return "partially_hidden"
    ls = keypoints["left_shoulder"]
    rs = keypoints["right_shoulder"]
    le = keypoints["left_elbow"]
    re = keypoints["right_elbow"]
    lw = keypoints["left_wrist"]
    rw = keypoints["right_wrist"]
    center_x = (ls.x + rs.x) / 2
    shoulders_y = (ls.y + rs.y) / 2
    hips = [keypoints[name] for name in ("left_hip", "right_hip") if name in keypoints]
    torso_bottom = (
        sum(point.y for point in hips) / len(hips)
        if hips
        else shoulders_y + 0.35
    )
    wrists_on_torso = all(
        shoulders_y - 0.08 <= point.y <= torso_bottom + 0.08
        for point in (lw, rw)
    )
    crossed_sides = lw.x > center_x and rw.x < center_x
    crossed_lines = _segments_intersect(le, lw, re, rw)
    if wrists_on_torso and (crossed_sides or crossed_lines):
        return "crossed"
    if lw.y < ls.y or rw.y < rs.y:
        return "raised"
    if lw.y > le.y > ls.y and rw.y > re.y > rs.y:
        return "down"
    return "bent"


class MediaPipePoseEstimator:
    """Load one CPU MediaPipe Pose instance and serialize access to it."""

    _instances: dict[tuple[float, float], Any] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        *,
        enabled: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        min_shoulder_visibility: float,
    ) -> None:
        self.enabled = enabled
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_shoulder_visibility = min_shoulder_visibility

    def estimate(self, image: Image.Image) -> tuple[PoseResult, Image.Image]:
        """Estimate one pose or return an explicit degraded result."""

        debug = image.convert("RGB").copy()
        if not self.enabled:
            return (
                PoseResult(warnings=["Pose estimation is disabled."]),
                debug,
            )
        try:
            estimator = self._get_estimator()
            rgb = np.asarray(image.convert("RGB"))
            with self._lock:
                raw = estimator.process(rgb)
            landmarks = getattr(raw, "pose_landmarks", None)
            if landmarks is None:
                return (
                    PoseResult(warnings=["No person pose was detected."]),
                    debug,
                )
            keypoints = self._extract_keypoints(
                landmarks.landmark,
                image.width,
                image.height,
            )
            result = self._build_result(keypoints)
            self._draw_debug(debug, result)
            return result, debug
        except Exception as exc:
            LOGGER.warning(
                "pose_estimation_fallback",
                extra={
                    "stage": "pose_estimation",
                    "fallback_used": True,
                    "warning": type(exc).__name__,
                },
            )
            return (
                PoseResult(
                    warnings=[
                        "MediaPipe pose unavailable; pose validation is degraded."
                    ]
                ),
                debug,
            )

    def _get_estimator(self) -> Any:
        key = (
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        if key in self._instances:
            return self._instances[key]
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed") from exc
        estimator = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._instances[key] = estimator
        return estimator

    @staticmethod
    def _extract_keypoints(
        landmarks: list[Any],
        width: int,
        height: int,
    ) -> dict[str, Keypoint]:
        result: dict[str, Keypoint] = {}
        for name, index in LANDMARK_INDICES.items():
            point = landmarks[index]
            result[name] = Keypoint(
                name=name,
                x=float(point.x),
                y=float(point.y),
                z=float(point.z),
                visibility=max(0.0, min(1.0, float(point.visibility))),
                pixel_x=round(float(point.x) * width),
                pixel_y=round(float(point.y) * height),
            )
        return result

    def _build_result(self, keypoints: dict[str, Keypoint]) -> PoseResult:
        shoulders_visible = all(
            _visible(keypoints, name, self.min_shoulder_visibility)
            for name in ("left_shoulder", "right_shoulder")
        )
        hips_visible = all(
            _visible(keypoints, name, 0.35)
            for name in ("left_hip", "right_hip")
        )
        if shoulders_visible:
            shoulder_width = _distance(
                keypoints["left_shoulder"],
                keypoints["right_shoulder"],
            )
        else:
            shoulder_width = 0.0
        if shoulders_visible and hips_visible:
            shoulder_center = (
                (
                    keypoints["left_shoulder"].x
                    + keypoints["right_shoulder"].x
                )
                / 2,
                (
                    keypoints["left_shoulder"].y
                    + keypoints["right_shoulder"].y
                )
                / 2,
            )
            hip_center = (
                (keypoints["left_hip"].x + keypoints["right_hip"].x) / 2,
                (keypoints["left_hip"].y + keypoints["right_hip"].y) / 2,
            )
            torso_length = math.dist(shoulder_center, hip_center)
            body_center = (
                (shoulder_center[0] + hip_center[0]) / 2,
                (shoulder_center[1] + hip_center[1]) / 2,
            )
        else:
            torso_length = 0.0
            body_center = None
        confidence = sum(
            point.visibility for point in keypoints.values()
        ) / max(1, len(keypoints))
        orientation = classify_orientation(keypoints)
        arms = classify_arms_position(keypoints)
        suitable = (
            shoulders_visible
            and orientation != "side"
            and arms not in {"crossed", "partially_hidden"}
        )
        warnings: list[str] = []
        if arms == "crossed":
            warnings.append("Arms appear to cross the torso.")
        elif arms == "partially_hidden":
            warnings.append("One or more arm landmarks are hidden.")
        if orientation == "side":
            warnings.append("Side-facing pose reduces try-on reliability.")
        return PoseResult(
            detected_person_count=1,
            keypoints=keypoints,
            shoulder_width=shoulder_width,
            torso_length=torso_length,
            body_center=body_center,
            pose_confidence=confidence,
            person_orientation=orientation,
            arms_position=arms,
            upper_body_visible=shoulders_visible and hips_visible,
            pose_suitable_for_tryon=suitable,
            warnings=warnings,
        )

    @staticmethod
    def _draw_debug(image: Image.Image, pose: PoseResult) -> None:
        draw = ImageDraw.Draw(image)
        for start_name, end_name in SKELETON_EDGES:
            if start_name in pose.keypoints and end_name in pose.keypoints:
                start = pose.keypoints[start_name]
                end = pose.keypoints[end_name]
                draw.line(
                    (start.pixel_x, start.pixel_y, end.pixel_x, end.pixel_y),
                    fill=(0, 220, 255),
                    width=max(2, image.width // 256),
                )
        radius = max(3, image.width // 180)
        for point in pose.keypoints.values():
            draw.ellipse(
                (
                    point.pixel_x - radius,
                    point.pixel_y - radius,
                    point.pixel_x + radius,
                    point.pixel_y + radius,
                ),
                fill=(255, 80, 40),
            )
