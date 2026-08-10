"""Current-clothing replacement and identity-protection mask construction."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.preprocessing.preprocessing_models import HumanParsingResult, PoseResult


def _pose_torso_mask(
    size: tuple[int, int],
    pose: PoseResult,
) -> np.ndarray:
    required = (
        "left_shoulder",
        "right_shoulder",
        "right_hip",
        "left_hip",
    )
    if not all(name in pose.keypoints for name in required):
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [
            (
                pose.keypoints[name].pixel_x,
                pose.keypoints[name].pixel_y,
            )
            for name in required
        ],
        fill=255,
    )
    return np.asarray(image)


def _pose_hands_mask(
    size: tuple[int, int],
    pose: PoseResult,
) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    radius = max(7, round(pose.shoulder_width * size[0] * 0.08))
    for name in ("left_wrist", "right_wrist"):
        point = pose.keypoints.get(name)
        if point is not None:
            draw.ellipse(
                (
                    point.pixel_x - radius,
                    point.pixel_y - radius,
                    point.pixel_x + radius,
                    point.pixel_y + radius,
                ),
                fill=255,
            )
    return np.asarray(image)


def build_clothing_masks(
    *,
    foreground_mask: Image.Image,
    parsing: HumanParsingResult,
    pose: PoseResult,
    morphology_kernel: int,
    dilation_kernel: int,
    dilation_iterations: int,
    feather_radius: int,
) -> tuple[Image.Image, Image.Image]:
    """Build replace/preserve masks while preventing identity-region overlap."""

    foreground = np.asarray(foreground_mask.convert("L"))
    size = foreground_mask.size
    clothes = np.asarray(
        parsing.upper_clothes_mask.convert("L").resize(
            size,
            Image.Resampling.NEAREST,
        )
    )
    torso = np.asarray(
        parsing.body_torso_mask.convert("L").resize(
            size,
            Image.Resampling.NEAREST,
        )
    )
    if np.count_nonzero(clothes) < clothes.size * 0.005:
        clothes = _pose_torso_mask(size, pose)
    replace = np.maximum(clothes, torso)
    replace = cv2.bitwise_and(replace, foreground)

    identity = np.asarray(
        parsing.face_hair_protection_mask.convert("L").resize(
            size,
            Image.Resampling.NEAREST,
        )
    )
    hands = np.asarray(
        parsing.hands_mask.convert("L").resize(
            size,
            Image.Resampling.NEAREST,
        )
    )
    hands = np.maximum(hands, _pose_hands_mask(size, pose))
    arms = np.asarray(
        parsing.arms_mask.convert("L").resize(
            size,
            Image.Resampling.NEAREST,
        )
    )
    preserve = np.maximum(np.maximum(identity, hands), arms)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morphology_kernel, morphology_kernel),
    )
    replace = cv2.morphologyEx(replace, cv2.MORPH_CLOSE, kernel)
    replace = cv2.morphologyEx(replace, cv2.MORPH_OPEN, kernel)
    if dilation_iterations:
        dilation = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_kernel, dilation_kernel),
        )
        replace = cv2.dilate(
            replace,
            dilation,
            iterations=dilation_iterations,
        )
    preserve = cv2.dilate(preserve, kernel, iterations=1)
    replace[preserve > 0] = 0
    if feather_radius:
        kernel_size = feather_radius * 2 + 1
        replace = cv2.GaussianBlur(replace, (kernel_size, kernel_size), 0)
        replace[preserve > 0] = 0
    return Image.fromarray(replace), Image.fromarray(preserve)
