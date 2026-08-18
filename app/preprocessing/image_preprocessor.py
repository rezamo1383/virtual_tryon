"""Orchestrate all local preprocessing before external API calls."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.core.config import Settings
from app.preprocessing.background_remover import BackgroundRemover
from app.preprocessing.device_manager import get_compute_device
from app.preprocessing.human_parser import (
    HeuristicHumanParser,
    HumanParser,
    OnnxAtrHumanParser,
)
from app.preprocessing.image_normalizer import (
    downscale_for_inference,
    letterbox_image,
    letterbox_mask,
    save_clean_image,
)
from app.preprocessing.mask_processor import build_clothing_masks
from app.preprocessing.model_manager import ModelManager, ModelSpec
from app.preprocessing.person_detector import PersonPresenceDetector
from app.preprocessing.pose_estimator import MediaPipePoseEstimator
from app.preprocessing.preprocessing_exceptions import (
    HumanParsingError,
    PersonNotDetectedError,
    PreprocessingError,
    PreprocessingPathError,
)
from app.preprocessing.preprocessing_models import (
    BoundingBox,
    GarmentProcessingResult,
    HumanParsingResult,
    PersonPreprocessingResult,
    PreprocessingResult,
)
from app.preprocessing.suitability_validator import (
    validate_garment,
    validate_person,
)
from app.utils.file_utils import ensure_within
from app.utils.image_utils import open_image_safe
from app.utils.json_utils import write_json

LOGGER = logging.getLogger(__name__)


class LocalImagePreprocessor:
    """CPU-first synchronous preprocessing service with injectable models."""

    def __init__(
        self,
        settings: Settings,
        *,
        background_remover: BackgroundRemover | None = None,
        pose_estimator: MediaPipePoseEstimator | None = None,
        human_parser: HumanParser | None = None,
        person_detector: PersonPresenceDetector | None = None,
    ) -> None:
        self.settings = settings
        self._log_context = threading.local()
        self.device = get_compute_device(settings.preprocessing_device)
        self.background_remover = background_remover or BackgroundRemover(
            enabled=settings.background_removal_enabled,
            device=self.device,
            model_cache_directory=settings.model_cache_directory / "rembg",
            person_model=settings.background_removal_person_model,
            garment_model=settings.background_removal_garment_model,
        )
        self.pose_estimator = pose_estimator or MediaPipePoseEstimator(
            enabled=settings.pose_estimation_enabled,
            min_detection_confidence=settings.pose_min_detection_confidence,
            min_tracking_confidence=settings.pose_min_tracking_confidence,
            min_shoulder_visibility=settings.min_shoulder_visibility,
        )
        self.person_detector = person_detector or PersonPresenceDetector()
        manager = ModelManager(
            settings.model_cache_directory / "human_parsing",
            offline=settings.local_model_offline_mode,
            timeout_seconds=settings.model_download_timeout_seconds,
        )
        self.human_parser = human_parser or OnnxAtrHumanParser(
            model_manager=manager,
            model_spec=ModelSpec(
                filename=settings.human_parsing_model_filename,
                url=settings.human_parsing_model_url,
                sha256=settings.human_parsing_model_sha256,
                approximate_size_mb=66,
            ),
            device=self.device,
        )

    def warmup(self) -> int:
        """Initialize the configured parser before the first user request."""

        if not self.settings.human_parsing_enabled:
            return 0
        started = time.perf_counter()
        self.human_parser.parse(Image.new("RGB", (512, 512), (127, 127, 127)))
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "human_parsing_warmup_completed",
            extra={
                "stage": "human_parsing_warmup",
                "device": self.device,
                "elapsed_ms": elapsed_ms,
            },
        )
        return elapsed_ms

    def preprocess(
        self,
        person_image_path: Path,
        garment_image_path: Path,
        job_directory: Path,
        *,
        human_parsing_enabled: bool | None = None,
    ) -> PreprocessingResult:
        """Run local processing and persist only artifacts under the job."""

        started = time.perf_counter()
        job_root = job_directory.resolve(strict=False)
        self._log_context.job_id = job_root.name
        artifact_root = self._safe_artifact_path(
            job_root,
            Path("preprocessing"),
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        person = self._load_for_inference(person_image_path, "person")
        garment = self._load_for_inference(garment_image_path, "garment")

        person_bg = self._timed(
            "person_background_removal",
            lambda: self.background_remover.remove_background(person, "person"),
            input_size=person.size,
        )
        garment_bg = self._timed(
            "garment_background_removal",
            lambda: self.background_remover.remove_background(
                garment,
                "garment",
            ),
            input_size=garment.size,
        )
        pose, pose_debug = self._timed(
            "pose_estimation",
            lambda: self.pose_estimator.estimate(person),
            input_size=person.size,
        )
        presence = None
        if self.settings.person_presence_check_enabled:
            presence = self._timed(
                "person_presence_check",
                lambda: self.person_detector.detect(person, pose),
                input_size=person.size,
            )
            if not presence.detected:
                LOGGER.warning(
                    "person_not_detected",
                    extra={
                        "stage": "person_presence_check",
                        "job_id": job_root.name,
                        "device": "cpu",
                        "model_name": presence.model_name,
                        "accepted": False,
                    },
                )
                raise PersonNotDetectedError(
                    "No person was detected in the person image."
                )

        parsing_enabled = (
            self.settings.human_parsing_enabled
            if human_parsing_enabled is None
            else human_parsing_enabled
        )
        parsing, parsing_warning = self._parse_human(
            person,
            pose,
            person_bg.mask,
            enabled=parsing_enabled,
        )
        if (
            presence is not None
            and presence.detected
            and not self.person_detector.has_semantic_human_evidence(parsing)
        ):
            LOGGER.warning(
                "person_not_detected",
                extra={
                    "stage": "human_parsing_confirmation",
                    "job_id": job_root.name,
                    "device": self.device,
                    "model_name": parsing.model_name,
                    "accepted": False,
                },
            )
            raise PersonNotDetectedError("No person was detected in the person image.")
        replace_mask, preserve_mask = build_clothing_masks(
            foreground_mask=person_bg.mask,
            parsing=parsing,
            pose=pose,
            morphology_kernel=self.settings.mask_morphology_kernel,
            dilation_kernel=self.settings.mask_dilation_kernel,
            dilation_iterations=self.settings.mask_dilation_iterations,
            feather_radius=self.settings.mask_feather_radius,
        )

        person_validation = validate_person(
            person,
            pose,
            parsing,
            min_shoulder_visibility=self.settings.min_shoulder_visibility,
            min_score=self.settings.min_tryon_suitability_score,
            pose_required=self.settings.pose_estimation_enabled,
        )
        garment_metrics = self._garment_metrics(
            garment_bg.image,
            garment_bg.mask,
        )
        garment_validation = validate_garment(
            garment,
            garment_bg.mask,
            cropped_edges=garment_metrics["cropped_edges"],
            component_count=garment_metrics["component_count"],
            min_score=self.settings.min_tryon_suitability_score,
            background_mask_required=(self.settings.background_removal_enabled),
        )

        person_result = self._save_person_artifacts(
            artifact_root,
            person,
            person_bg.image,
            person_bg.mask,
            replace_mask,
            preserve_mask,
            pose_debug,
            parsing,
            pose,
            person_validation,
        )
        garment_result = self._save_garment_artifacts(
            artifact_root,
            garment_bg.image,
            garment_bg.mask,
            garment_validation,
            garment_metrics,
        )
        warnings = [
            *person_validation.warnings,
            *garment_validation.warnings,
            *parsing.warnings,
        ]
        if person_bg.fallback_used:
            warnings.append("Person background removal used a local fallback.")
        if garment_bg.fallback_used:
            warnings.append("Garment background removal used a local fallback.")
        if parsing_warning:
            warnings.append(parsing_warning)
        result = PreprocessingResult(
            person=person_result,
            garment=garment_result,
            device=self.device,
            degraded_mode=bool(
                parsing.degraded_mode
                or person_bg.fallback_used
                or garment_bg.fallback_used
                or pose.detected_person_count != 1
            ),
            processing_time_ms=round((time.perf_counter() - started) * 1000),
            warnings=list(dict.fromkeys(warnings)),
        )
        write_json(artifact_root / "preprocessing.json", result)
        LOGGER.info(
            "local_preprocessing_completed",
            extra={
                "stage": "preprocessing_complete",
                "job_id": job_root.name,
                "device": self.device,
                "elapsed_ms": result.processing_time_ms,
                "accepted": (
                    person_validation.accepted and garment_validation.accepted
                ),
                "degraded_mode": result.degraded_mode,
            },
        )
        return result

    def preprocess_person(
        self,
        person_image_path: Path,
        job_directory: Path,
        *,
        human_parsing_enabled: bool | None = None,
    ) -> PersonPreprocessingResult:
        """Preprocess only the request-specific person image."""

        job_root = job_directory.resolve(strict=False)
        self._log_context.job_id = job_root.name
        artifact_root = self._safe_artifact_path(job_root, Path("preprocessing"))
        artifact_root.mkdir(parents=True, exist_ok=True)
        person = self._load_for_inference(person_image_path, "person")
        person_bg = self._timed(
            "person_background_removal",
            lambda: self.background_remover.remove_background(person, "person"),
            input_size=person.size,
        )
        pose, pose_debug = self._timed(
            "pose_estimation",
            lambda: self.pose_estimator.estimate(person),
            input_size=person.size,
        )
        presence = None
        if self.settings.person_presence_check_enabled:
            presence = self._timed(
                "person_presence_check",
                lambda: self.person_detector.detect(person, pose),
                input_size=person.size,
            )
            if not presence.detected:
                raise PersonNotDetectedError(
                    "No person was detected in the person image."
                )
        parsing_enabled = (
            self.settings.human_parsing_enabled
            if human_parsing_enabled is None
            else human_parsing_enabled
        )
        parsing, _ = self._parse_human(
            person,
            pose,
            person_bg.mask,
            enabled=parsing_enabled,
        )
        if (
            presence is not None
            and presence.detected
            and not self.person_detector.has_semantic_human_evidence(parsing)
        ):
            raise PersonNotDetectedError("No person was detected in the person image.")
        replace_mask, preserve_mask = build_clothing_masks(
            foreground_mask=person_bg.mask,
            parsing=parsing,
            pose=pose,
            morphology_kernel=self.settings.mask_morphology_kernel,
            dilation_kernel=self.settings.mask_dilation_kernel,
            dilation_iterations=self.settings.mask_dilation_iterations,
            feather_radius=self.settings.mask_feather_radius,
        )
        validation = validate_person(
            person,
            pose,
            parsing,
            min_shoulder_visibility=self.settings.min_shoulder_visibility,
            min_score=self.settings.min_tryon_suitability_score,
            pose_required=self.settings.pose_estimation_enabled,
        )
        return self._save_person_artifacts(
            artifact_root,
            person,
            person_bg.image,
            person_bg.mask,
            replace_mask,
            preserve_mask,
            pose_debug,
            parsing,
            pose,
            validation,
        )

    def preprocess_garment(
        self,
        garment_image_path: Path,
        job_directory: Path,
    ) -> GarmentProcessingResult:
        """Preprocess only a reusable product garment image."""

        job_root = job_directory.resolve(strict=False)
        self._log_context.job_id = job_root.name
        artifact_root = self._safe_artifact_path(job_root, Path("preprocessing"))
        artifact_root.mkdir(parents=True, exist_ok=True)
        garment = self._load_for_inference(garment_image_path, "garment")
        garment_bg = self._timed(
            "garment_background_removal",
            lambda: self.background_remover.remove_background(
                garment,
                "garment",
            ),
            input_size=garment.size,
        )
        metrics = self._garment_metrics(garment_bg.image, garment_bg.mask)
        validation = validate_garment(
            garment,
            garment_bg.mask,
            cropped_edges=metrics["cropped_edges"],
            component_count=metrics["component_count"],
            min_score=self.settings.min_tryon_suitability_score,
            background_mask_required=self.settings.background_removal_enabled,
        )
        return self._save_garment_artifacts(
            artifact_root,
            garment_bg.image,
            garment_bg.mask,
            validation,
            metrics,
        )

    def _load_for_inference(self, path: Path, subject: str) -> Image.Image:
        try:
            image = open_image_safe(path)
            if image.width < 2 or image.height < 2:
                raise ValueError("image dimensions are invalid")
            return downscale_for_inference(
                image,
                self.settings.preprocessing_inference_max_dimension,
            )
        except Exception as exc:
            raise PreprocessingError(
                f"Could not load {subject} image for preprocessing: {exc}"
            ) from exc

    def _parse_human(
        self,
        image: Image.Image,
        pose: Any,
        foreground_mask: Image.Image,
        *,
        enabled: bool,
    ) -> tuple[HumanParsingResult, str | None]:
        if enabled:
            try:
                return (
                    self._timed(
                        "human_parsing",
                        lambda: self.human_parser.parse(image),
                        input_size=image.size,
                    ),
                    None,
                )
            except Exception as exc:
                if self.settings.human_parsing_required:
                    if isinstance(exc, HumanParsingError):
                        raise
                    raise HumanParsingError(
                        f"Required human parsing failed: {exc}"
                    ) from exc
                warning = "Human parsing unavailable; pose/foreground fallback used."
        else:
            warning = "Human parsing is disabled; pose/foreground fallback used."
        fallback = HeuristicHumanParser(
            pose=pose,
            foreground_mask=foreground_mask,
            warning=warning,
        ).parse(image)
        LOGGER.warning(
            "human_parsing_fallback",
            extra={
                "stage": "human_parsing",
                "job_id": getattr(self._log_context, "job_id", "unknown"),
                "device": self.device,
                "fallback_used": True,
                "degraded_mode": True,
                "warning": warning,
            },
        )
        return fallback, warning

    def _save_person_artifacts(
        self,
        root: Path,
        person: Image.Image,
        transparent: Image.Image,
        foreground: Image.Image,
        replace: Image.Image,
        preserve: Image.Image,
        pose_debug: Image.Image,
        parsing: HumanParsingResult,
        pose: Any,
        validation: Any,
    ) -> PersonPreprocessingResult:
        directory = root / "person"
        target = (
            self.settings.person_target_width,
            self.settings.person_target_height,
        )
        normalized, transform = letterbox_image(
            person,
            target,
            output_mode="RGB",
        )
        transparent_normalized, _ = letterbox_image(
            transparent,
            target,
            output_mode="RGBA",
        )
        foreground_normalized = letterbox_mask(foreground, transform)
        replace_normalized = letterbox_mask(
            replace,
            transform,
            feathered=True,
        )
        preserve_normalized = letterbox_mask(preserve, transform)
        paths = {
            "normalized": directory / "normalized.png",
            "transparent": directory / "transparent.png",
            "foreground": directory / "foreground_mask.png",
            "replace": directory / "replace_mask.png",
            "preserve": directory / "preserve_mask.png",
        }
        save_clean_image(normalized, paths["normalized"])
        save_clean_image(transparent_normalized, paths["transparent"])
        save_clean_image(foreground_normalized, paths["foreground"])
        save_clean_image(replace_normalized, paths["replace"])
        save_clean_image(preserve_normalized, paths["preserve"])
        pose_path: Path | None = None
        parsing_path: Path | None = None
        if self.settings.save_preprocessing_debug_images:
            pose_path = directory / "pose_debug.png"
            parsing_path = directory / "parsing_debug.png"
            save_clean_image(pose_debug, pose_path)
            save_clean_image(parsing.visualization, parsing_path)
        return PersonPreprocessingResult(
            normalized_image_path=paths["normalized"].resolve(),
            transparent_image_path=paths["transparent"].resolve(),
            foreground_mask_path=paths["foreground"].resolve(),
            replace_mask_path=paths["replace"].resolve(),
            preserve_mask_path=paths["preserve"].resolve(),
            pose_debug_path=pose_path.resolve() if pose_path else None,
            parsing_debug_path=parsing_path.resolve() if parsing_path else None,
            pose=pose,
            validation=validation,
        )

    def _save_garment_artifacts(
        self,
        root: Path,
        transparent: Image.Image,
        mask: Image.Image,
        validation: Any,
        metrics: dict[str, Any],
    ) -> GarmentProcessingResult:
        directory = root / "garment"
        bbox: BoundingBox = metrics["bounding_box"]
        if bbox.width and bbox.height:
            box = (
                bbox.x,
                bbox.y,
                bbox.x + bbox.width,
                bbox.y + bbox.height,
            )
            cropped = transparent.crop(box)
            cropped_mask = mask.crop(box)
        else:
            cropped = transparent
            cropped_mask = mask
        target = (
            self.settings.garment_target_width,
            self.settings.garment_target_height,
        )
        normalized, transform = letterbox_image(
            cropped,
            target,
            output_mode="RGBA",
        )
        normalized_mask = letterbox_mask(cropped_mask, transform)
        normalized_array = np.asarray(normalized).copy()
        normalized_array[:, :, 3] = np.asarray(normalized_mask)
        normalized = Image.fromarray(normalized_array)
        transparent_path = directory / "transparent_cropped.png"
        normalized_path = directory / "normalized.png"
        mask_path = directory / "garment_mask.png"
        save_clean_image(cropped, transparent_path)
        save_clean_image(normalized, normalized_path)
        save_clean_image(normalized_mask, mask_path)
        return GarmentProcessingResult(
            normalized_image_path=normalized_path.resolve(),
            transparent_image_path=transparent_path.resolve(),
            garment_mask_path=mask_path.resolve(),
            bounding_box=bbox,
            dominant_color=metrics["dominant_color"],
            image_dimensions=transparent.size,
            alpha_coverage=metrics["alpha_coverage"],
            symmetry_score=metrics["symmetry_score"],
            cropped_edges=metrics["cropped_edges"],
            garment_suitability_score=validation.score,
            foreground_center=metrics["foreground_center"],
            validation=validation,
        )

    @staticmethod
    def _garment_metrics(
        image: Image.Image,
        mask: Image.Image,
    ) -> dict[str, Any]:
        array = np.asarray(mask.convert("L"))
        binary = np.where(array > 16, 255, 0).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            8,
        )
        components = [
            index
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= binary.size * 0.005
        ]
        if components:
            largest = max(
                components,
                key=lambda index: stats[index, cv2.CC_STAT_AREA],
            )
            x, y, width, height, _ = stats[largest]
            center = centroids[largest]
            bbox = BoundingBox(
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
            )
            foreground_center = (
                float(center[0] / binary.shape[1]),
                float(center[1] / binary.shape[0]),
            )
        else:
            bbox = BoundingBox(x=0, y=0, width=0, height=0)
            foreground_center = None
        edges: list[str] = []
        if np.any(binary[0, :]):
            edges.append("top")
        if np.any(binary[-1, :]):
            edges.append("bottom")
        if np.any(binary[:, 0]):
            edges.append("left")
        if np.any(binary[:, -1]):
            edges.append("right")
        alpha_coverage = float(np.count_nonzero(binary) / binary.size)
        rgb = np.asarray(image.convert("RGB"))
        foreground_pixels = rgb[binary > 0]
        if foreground_pixels.size:
            dominant = np.median(foreground_pixels, axis=0).astype(np.uint8)
            dominant_color = "#{:02X}{:02X}{:02X}".format(*dominant)
        else:
            dominant_color = None
        flipped = np.fliplr(binary)
        symmetry = 1.0 - float(
            np.mean(np.abs(binary.astype(np.float32) - flipped)) / 255.0
        )
        return {
            "bounding_box": bbox,
            "foreground_center": foreground_center,
            "cropped_edges": edges,
            "component_count": len(components),
            "alpha_coverage": alpha_coverage,
            "symmetry_score": max(0.0, min(1.0, symmetry)),
            "dominant_color": dominant_color,
        }

    def _safe_artifact_path(self, root: Path, relative: Path) -> Path:
        try:
            return ensure_within(root / relative, root)
        except ValueError as exc:
            raise PreprocessingPathError(str(exc)) from exc

    def _timed(
        self,
        stage: str,
        operation: Any,
        *,
        input_size: tuple[int, int],
    ) -> Any:
        started = time.perf_counter()
        result = operation()
        output_image = getattr(result, "image", None)
        if isinstance(result, tuple):
            output_image = next(
                (item for item in result if isinstance(item, Image.Image)),
                None,
            )
        output_size = (
            output_image.size if isinstance(output_image, Image.Image) else input_size
        )
        model_name = getattr(result, "model_name", None)
        LOGGER.info(
            "local_preprocessing_stage",
            extra={
                "stage": stage,
                "job_id": getattr(self._log_context, "job_id", "unknown"),
                "device": self.device,
                "input_size": input_size,
                "output_size": output_size,
                "model_name": model_name,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return result
