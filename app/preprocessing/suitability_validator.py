"""Local suitability scoring for person and garment inputs."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.preprocessing.preprocessing_models import (
    HumanParsingResult,
    PoseResult,
    ValidationResult,
)


def _image_quality_metrics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "blur_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_brightness": float(gray.mean()),
        "dark_fraction": float(np.count_nonzero(gray < 20) / gray.size),
        "bright_fraction": float(np.count_nonzero(gray > 245) / gray.size),
    }


def validate_person(
    image: Image.Image,
    pose: PoseResult,
    parsing: HumanParsingResult,
    *,
    min_shoulder_visibility: float,
    min_score: float,
    pose_required: bool = True,
) -> ValidationResult:
    """Score person visibility, pose, exposure, and torso occlusion."""

    metrics = _image_quality_metrics(image)
    metrics.update(
        {
            "person_count": pose.detected_person_count,
            "pose_confidence": pose.pose_confidence,
            "shoulder_width": pose.shoulder_width,
            "torso_length": pose.torso_length,
        }
    )
    warnings = list(pose.warnings)
    reasons: list[str] = []
    score = 1.0
    if pose_required and pose.detected_person_count != 1:
        score -= 0.40
        reasons.append("Exactly one detectable person is required.")
    shoulder_visibility = [
        pose.keypoints[name].visibility
        for name in ("left_shoulder", "right_shoulder")
        if name in pose.keypoints
    ]
    minimum_visibility = min(shoulder_visibility, default=0.0)
    metrics["minimum_shoulder_visibility"] = minimum_visibility
    if pose_required and minimum_visibility < min_shoulder_visibility:
        score -= 0.25
        reasons.append("Both shoulders must be clearly visible.")
    if pose_required and not pose.upper_body_visible:
        score -= 0.15
        reasons.append("The upper body is not sufficiently visible.")
    nose = pose.keypoints.get("nose")
    if nose is not None and (
        nose.x < 0.03 or nose.x > 0.97 or nose.y < 0.02 or nose.y > 0.75
    ):
        score -= 0.10
        warnings.append("The face appears close to or outside the image crop.")
    if pose_required and pose.person_orientation == "side":
        score -= 0.15
        warnings.append("A side-facing person may reduce garment alignment.")
    if pose_required and pose.arms_position == "crossed":
        score -= 0.12
        warnings.append("Crossed arms cover part of the current clothing.")
    elif pose_required and pose.arms_position == "partially_hidden":
        score -= 0.10
        warnings.append("Hidden arm landmarks reduce mask accuracy.")
    if metrics["blur_variance"] < 35:
        score -= 0.12
        warnings.append("Person image appears blurry.")
    if metrics["dark_fraction"] > 0.45:
        score -= 0.12
        warnings.append("Person image is very dark.")
    if metrics["bright_fraction"] > 0.45:
        score -= 0.12
        warnings.append("Person image is overexposed.")
    hands = np.asarray(parsing.hands_mask.convert("L")) > 0
    clothes = np.asarray(parsing.upper_clothes_mask.convert("L")) > 0
    overlap = float(np.count_nonzero(hands & clothes) / max(1, np.count_nonzero(clothes)))
    metrics["hands_over_clothing_fraction"] = overlap
    if overlap > 0.25:
        score -= 0.10
        warnings.append("Hands cover a large portion of the clothing region.")
    score = max(0.0, min(1.0, score))
    return ValidationResult(
        accepted=score >= min_score and not reasons,
        score=score,
        warnings=list(dict.fromkeys(warnings)),
        rejection_reasons=reasons,
        metrics=metrics,
    )


def validate_garment(
    image: Image.Image,
    mask: Image.Image,
    *,
    cropped_edges: list[str],
    component_count: int,
    min_score: float,
    background_mask_required: bool = True,
) -> ValidationResult:
    """Score garment foreground size, isolation, and boundary clipping."""

    mask_array = np.asarray(mask.convert("L"))
    coverage = float(np.count_nonzero(mask_array > 16) / mask_array.size)
    metrics = {
        "foreground_fraction": coverage,
        "component_count": component_count,
        "width": image.width,
        "height": image.height,
        "cropped_edges": cropped_edges,
    }
    warnings: list[str] = []
    reasons: list[str] = []
    score = 1.0
    if coverage < 0.03:
        score -= 0.65
        reasons.append("Garment foreground is too small.")
    elif coverage < 0.10:
        score -= 0.20
        warnings.append("Garment occupies a small part of the image.")
    if background_mask_required and coverage > 0.98:
        score -= 0.35
        reasons.append("A removable garment background was not detected.")
    if cropped_edges:
        score -= min(0.25, len(cropped_edges) * 0.07)
        warnings.append(
            "Garment foreground touches image edge(s): "
            + ", ".join(cropped_edges)
        )
    if component_count == 0:
        score = 0.0
        reasons.append("No garment component was detected.")
    elif component_count > 2:
        score -= 0.20
        warnings.append("Multiple foreground objects may be present.")
    score = max(0.0, min(1.0, score))
    return ValidationResult(
        accepted=score >= min_score and not reasons,
        score=score,
        warnings=warnings,
        rejection_reasons=list(dict.fromkeys(reasons)),
        metrics=metrics,
    )
