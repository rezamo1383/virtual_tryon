"""Concrete wallpaper analysis, geometry, generation, and evaluation stages."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.clients.qwen_client import QwenClient
from app.clients.tryon_api_client import TryOnAPIClient
from app.core.exceptions import InputValidationError, TryOnAPIError
from app.core.config import Settings
from app.models.wallpaper_models import (
    LightingPreservationResult,
    PerspectiveEstimationResult,
    TextureRepetitionResult,
    WallAnalysisResult,
    WallSegmentationResult,
    WallpaperCandidateResult,
    WallpaperOutputEvaluation,
)
from app.prompts.wallpaper import WallpaperPromptBuilder
from app.preprocessing.model_manager import ModelManager, ModelSpec
from app.utils.image_utils import (
    decode_image_bytes,
    open_image_safe,
    save_privacy_safe_png,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SemanticWallPrediction:
    """One reusable semantic wall prediction in source-image coordinates."""

    mask: np.ndarray
    confidence: float
    wall_count: int
    polygon_pixels: list[tuple[int, int]]


class SemanticWallSegmentationEngine:
    """Detect visible wall pixels locally with an ADE20K semantic model."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.wallpaper_segmentation_model
        self._runtime = settings.wallpaper_segmentation_runtime
        self._cache_directory = (
            settings.model_cache_directory / "wall_segmentation"
        )
        self._onnx_spec = ModelSpec(
            filename=settings.wallpaper_segmentation_onnx_filename,
            url=settings.wallpaper_segmentation_onnx_url,
            sha256=settings.wallpaper_segmentation_onnx_sha256,
            approximate_size_mb=110,
        )
        self._model_manager = ModelManager(
            self._cache_directory,
            offline=settings.local_model_offline_mode,
            timeout_seconds=settings.model_download_timeout_seconds,
        )
        self._offline = settings.local_model_offline_mode
        self._requested_device = settings.wallpaper_segmentation_device
        self._confidence_threshold = (
            settings.wallpaper_wall_confidence_threshold
        )
        self._min_region_coverage = (
            settings.wallpaper_min_wall_region_coverage
        )
        self._min_component_ratio = (
            settings.wallpaper_min_wall_component_ratio
        )
        self._processor: Any | None = None
        self._model: Any | None = None
        self._onnx_session: Any | None = None
        self._torch: Any | None = None
        self._device = "cpu"
        self._lock = threading.RLock()
        self._cached_key: tuple[Path, int, int] | None = None
        self._cached_prediction: _SemanticWallPrediction | None = None

    def predict(self, room_image: Path) -> _SemanticWallPrediction:
        """Return a precise visible-wall mask, caching sequential pipeline use."""

        resolved = room_image.resolve(strict=True)
        stat = resolved.stat()
        key = (resolved, stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key == self._cached_key and self._cached_prediction is not None:
                return self._cached_prediction
            self._ensure_model()
            prediction = self._predict_uncached(resolved)
            self._cached_key = key
            self._cached_prediction = prediction
            return prediction

    def _ensure_model(self) -> None:
        if self._model is not None or self._onnx_session is not None:
            return
        if self._runtime == "onnx":
            self._ensure_onnx_model()
            return
        self._ensure_torch_model()

    def _ensure_onnx_model(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise InputValidationError(
                "Semantic wall segmentation requires onnxruntime. "
                "Install requirements-preprocessing.txt."
            ) from exc
        try:
            model_path = self._model_manager.ensure(self._onnx_spec)
            options = ort.SessionOptions()
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise InputValidationError(
                "ONNX wall segmentation model could not be loaded."
            ) from exc
        self._onnx_session = session
        self._wall_ids = [0]
        LOGGER.info(
            "wall_segmentation_model_loaded",
            extra={
                "stage": "wall_segmentation",
                "model": self._onnx_spec.filename,
                "device": "cpu",
                "runtime": "onnx",
            },
        )

    def _ensure_torch_model(self) -> None:
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                AutoModelForSemanticSegmentation,
            )
        except ImportError as exc:
            raise InputValidationError(
                "Semantic wall segmentation requires torch and transformers. "
                "Install requirements-preprocessing.txt or select the polygon backend."
            ) from exc

        self._cache_directory.mkdir(parents=True, exist_ok=True)
        load_options: dict[str, object] = {
            "cache_dir": str(self._cache_directory.resolve()),
            "trust_remote_code": False,
        }
        try:
            try:
                processor = AutoImageProcessor.from_pretrained(
                    self._model_name,
                    **load_options,
                    local_files_only=True,
                )
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    self._model_name,
                    **load_options,
                    local_files_only=True,
                    use_safetensors=False,
                )
            except Exception:
                if self._offline:
                    raise
                processor = AutoImageProcessor.from_pretrained(
                    self._model_name,
                    **load_options,
                    local_files_only=False,
                )
                model = AutoModelForSemanticSegmentation.from_pretrained(
                    self._model_name,
                    **load_options,
                    local_files_only=False,
                    use_safetensors=False,
                )
        except Exception as exc:
            mode = "offline cache" if self._offline else "model registry"
            raise InputValidationError(
                "Wall segmentation model could not be loaded from the "
                f"{mode}: {self._model_name}."
            ) from exc

        if self._requested_device == "cuda" and not torch.cuda.is_available():
            raise InputValidationError(
                "Wallpaper segmentation requested CUDA, but CUDA is unavailable."
            )
        self._device = (
            "cuda"
            if self._requested_device == "cuda"
            or (
                self._requested_device == "auto"
                and torch.cuda.is_available()
            )
            else "cpu"
        )
        model.to(self._device)
        model.eval()
        wall_ids = [
            int(label_id)
            for label_id, label in model.config.id2label.items()
            if str(label).strip().casefold() == "wall"
        ]
        if not wall_ids:
            raise InputValidationError(
                "Configured semantic model does not expose an ADE wall class."
            )
        self._torch = torch
        self._processor = processor
        self._model = model
        self._wall_ids = wall_ids
        LOGGER.info(
            "wall_segmentation_model_loaded",
            extra={
                "stage": "wall_segmentation",
                "model": self._model_name,
                "device": self._device,
                "runtime": "torch",
            },
        )

    def _predict_uncached(self, room_image: Path) -> _SemanticWallPrediction:
        if self._onnx_session is not None:
            raw_mask, probability_map = self._predict_onnx(room_image)
        else:
            raw_mask, probability_map = self._predict_torch(room_image)
        mask, wall_count = self._filter_regions(raw_mask)
        wall_values = probability_map[mask > 0]
        confidence = float(wall_values.mean()) if wall_values.size else 0.0
        polygon = self._representative_quad(mask)
        return _SemanticWallPrediction(
            mask=mask,
            confidence=confidence,
            wall_count=wall_count,
            polygon_pixels=polygon,
        )

    def _predict_torch(
        self,
        room_image: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        if torch is None or self._processor is None or self._model is None:
            raise RuntimeError("Wall segmentation model was not initialized.")
        image = open_image_safe(room_image).convert("RGB")
        width, height = image.size
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            logits = self._model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits, dim=1)[0]
            semantic_labels = probabilities.argmax(dim=0)
            wall_probability = probabilities[self._wall_ids].amax(dim=0)
            wall_ids_tensor = torch.tensor(
                self._wall_ids,
                device=semantic_labels.device,
            )
            is_wall = torch.isin(semantic_labels, wall_ids_tensor)
            is_wall &= wall_probability >= self._confidence_threshold
            raw_mask = is_wall.to(torch.uint8).cpu().numpy() * 255
            probability_map = wall_probability.cpu().numpy()
        return raw_mask, probability_map

    def _predict_onnx(
        self,
        room_image: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        session = self._onnx_session
        if session is None:
            raise RuntimeError("ONNX wall segmentation model was not initialized.")
        image = open_image_safe(room_image).convert("RGB")
        width, height = image.size
        resized = image.resize((512, 512), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        standard_deviation = np.asarray(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        pixels = (pixels - mean) / standard_deviation
        pixel_values = np.transpose(pixels, (2, 0, 1))[None, ...]
        input_name = session.get_inputs()[0].name
        logits = np.asarray(
            session.run(None, {input_name: pixel_values})[0],
            dtype=np.float32,
        )[0]
        shifted = logits - logits.max(axis=0, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(
            axis=0,
            keepdims=True,
        )
        labels = self._resize_argmax(logits, (512, 512))
        wall_probability = probabilities[self._wall_ids].max(axis=0)
        wall_at_model_size = np.isin(labels, self._wall_ids).astype(np.uint8)
        probability_at_model_size = cv2.resize(
            wall_probability,
            (512, 512),
            interpolation=cv2.INTER_LINEAR,
        )
        wall_at_model_size &= (
            probability_at_model_size >= self._confidence_threshold
        ).astype(np.uint8)
        raw_mask = cv2.resize(
            wall_at_model_size,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) * 255
        probability_map = cv2.resize(
            probability_at_model_size,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        return raw_mask, probability_map

    @staticmethod
    def _resize_argmax(
        logits: np.ndarray,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Match bilinear logit upsampling without allocating a huge tensor."""

        width, height = size
        maximum = np.full((height, width), -np.inf, dtype=np.float32)
        labels = np.zeros((height, width), dtype=np.uint8)
        for label_id, channel in enumerate(logits):
            resized = cv2.resize(
                channel,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            update = resized > maximum
            maximum[update] = resized[update]
            labels[update] = label_id
        return labels

    def _filter_regions(self, mask: np.ndarray) -> tuple[np.ndarray, int]:
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        largest_area = (
            int(stats[1:, cv2.CC_STAT_AREA].max())
            if count > 1
            else 0
        )
        minimum = max(
            64,
            round(mask.shape[0] * mask.shape[1] * self._min_region_coverage),
            round(largest_area * self._min_component_ratio),
        )
        candidates: list[tuple[int, int, np.ndarray]] = []
        for label_id in range(1, count):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < minimum:
                continue
            component = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            candidates.append(
                (label_id, area, max(contours, key=cv2.contourArea))
            )

        filtered = np.zeros_like(mask)
        kept = 0
        for label_id, area, _ in candidates:
            center = (
                float(stats[label_id, cv2.CC_STAT_LEFT])
                + float(stats[label_id, cv2.CC_STAT_WIDTH]) / 2,
                float(stats[label_id, cv2.CC_STAT_TOP])
                + float(stats[label_id, cv2.CC_STAT_HEIGHT]) / 2,
            )
            enclosed = any(
                other_area > area
                and cv2.pointPolygonTest(other_contour, center, False) >= 0
                for other_id, other_area, other_contour in candidates
                if other_id != label_id
            )
            if enclosed:
                continue
            filtered[labels == label_id] = 255
            kept += 1
        return filtered, kept

    @staticmethod
    def _representative_quad(mask: np.ndarray) -> list[tuple[int, int]]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(largest)
        return [
            (x, y),
            (x + width - 1, y),
            (x + width - 1, y + height - 1),
            (x, y + height - 1),
        ]


class SemanticWallAnalyzer:
    """Derive wall availability and geometry from local pixel segmentation."""

    def __init__(self, engine: SemanticWallSegmentationEngine) -> None:
        self._engine = engine

    async def analyze(self, room_image: Path) -> WallAnalysisResult:
        prediction = await asyncio.to_thread(self._engine.predict, room_image)
        width, height = open_image_safe(room_image).size
        polygon = [
            {
                "x": x / max(1, width - 1),
                "y": y / max(1, height - 1),
            }
            for x, y in prediction.polygon_pixels
        ]
        return WallAnalysisResult(
            wall_detected=bool(prediction.wall_count and polygon),
            confidence=prediction.confidence,
            wall_polygon=polygon,
            wall_count=prediction.wall_count,
            occlusions=[
                "Visible foreground objects are excluded by semantic segmentation."
            ],
            lighting="preserved from source image",
            warnings=[],
        )


class SemanticWallSegmenter:
    """Persist a local semantic mask that excludes visible non-wall objects."""

    def __init__(self, engine: SemanticWallSegmentationEngine) -> None:
        self._engine = engine

    async def segment(
        self,
        room_image: Path,
        analysis: WallAnalysisResult,
        output_directory: Path,
    ) -> WallSegmentationResult:
        del analysis
        return await asyncio.to_thread(
            self._segment_sync,
            room_image,
            output_directory,
        )

    def _segment_sync(
        self,
        room_image: Path,
        output_directory: Path,
    ) -> WallSegmentationResult:
        prediction = self._engine.predict(room_image)
        if prediction.wall_count == 0 or not prediction.polygon_pixels:
            raise InputValidationError(
                "No wallpaper-suitable wall pixels were detected."
            )
        image = open_image_safe(room_image).convert("RGB")
        width, height = image.size
        output_directory.mkdir(parents=True, exist_ok=True)
        mask_path = output_directory / "wall_mask.png"
        Image.fromarray(prediction.mask).save(mask_path, format="PNG")

        rgb = np.asarray(image).copy()
        overlay = rgb.copy()
        overlay[prediction.mask > 0] = (0, 210, 150)
        debug = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)
        contours, _ = cv2.findContours(
            prediction.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            debug,
            contours,
            -1,
            (255, 70, 30),
            max(2, width // 400),
        )
        debug_path = output_directory / "wall_debug.png"
        save_privacy_safe_png(Image.fromarray(debug), debug_path)
        coverage = float(np.count_nonzero(prediction.mask)) / max(
            1,
            width * height,
        )
        return WallSegmentationResult(
            mask_path=mask_path.resolve(),
            debug_path=debug_path.resolve(),
            coverage=coverage,
            polygon_pixels=prediction.polygon_pixels,
            method="semantic",
            wall_count=prediction.wall_count,
            mean_confidence=prediction.confidence,
        )


class VisionWallAnalyzer:
    """Analyze the main visible wall with the tenant vision provider."""

    def __init__(
        self,
        client: QwenClient,
        prompts: WallpaperPromptBuilder,
    ) -> None:
        self._client = client
        self._prompts = prompts

    async def analyze(self, room_image: Path) -> WallAnalysisResult:
        result = await self._client.analyze_structured(
            self._prompts.wall_analysis(),
            [room_image],
            WallAnalysisResult,
        )
        if result.wall_detected and len(result.wall_polygon) != 4:
            raise InputValidationError(
                "Wall analysis did not return a valid four-point polygon."
            )
        return result


class PolygonWallSegmenter:
    """Convert normalized wall geometry into a reusable local mask."""

    async def segment(
        self,
        room_image: Path,
        analysis: WallAnalysisResult,
        output_directory: Path,
    ) -> WallSegmentationResult:
        return await asyncio.to_thread(
            self._segment_sync,
            room_image,
            analysis,
            output_directory,
        )

    @staticmethod
    def _segment_sync(
        room_image: Path,
        analysis: WallAnalysisResult,
        output_directory: Path,
    ) -> WallSegmentationResult:
        image = open_image_safe(room_image).convert("RGB")
        width, height = image.size
        points = [
            (
                min(width - 1, max(0, round(point.x * (width - 1)))),
                min(height - 1, max(0, round(point.y * (height - 1)))),
            )
            for point in analysis.wall_polygon
        ]
        if len(points) != 4:
            raise InputValidationError(
                "No valid wall polygon is available for segmentation."
            )
        polygon = np.asarray(points, dtype=np.int32)
        area = abs(float(cv2.contourArea(polygon)))
        coverage = area / max(1, width * height)
        if coverage < 0.04:
            raise InputValidationError(
                "The detected wall is too small for wallpaper visualization."
            )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        output_directory.mkdir(parents=True, exist_ok=True)
        mask_path = output_directory / "wall_mask.png"
        Image.fromarray(mask).save(mask_path, format="PNG")

        debug = image.copy()
        draw = ImageDraw.Draw(debug)
        draw.line([*points, points[0]], fill=(0, 255, 180), width=max(3, width // 240))
        debug_path = output_directory / "wall_debug.png"
        save_privacy_safe_png(debug, debug_path)
        return WallSegmentationResult(
            mask_path=mask_path.resolve(),
            debug_path=debug_path.resolve(),
            coverage=coverage,
            polygon_pixels=points,
            method="polygon",
            wall_count=1,
        )


class WallpaperReferencePreprocessor:
    """Normalize a pattern sample and remove an obvious promotional footer."""

    async def prepare(
        self,
        wallpaper_image: Path,
        output_directory: Path,
    ) -> Path:
        return await asyncio.to_thread(
            self._prepare_sync,
            wallpaper_image,
            output_directory,
        )

    @staticmethod
    def _prepare_sync(
        wallpaper_image: Path,
        output_directory: Path,
    ) -> Path:
        image = open_image_safe(wallpaper_image).convert("RGB")
        values = np.asarray(image, dtype=np.float32)
        height = values.shape[0]
        row_difference = np.mean(
            np.abs(values[1:] - values[:-1]),
            axis=(1, 2),
        )
        baseline_end = max(2, round(height * 0.65))
        baseline = float(np.median(row_difference[:baseline_end]))
        search_start = max(1, round(height * 0.70))
        search_end = min(height - 1, round(height * 0.98))
        footer_start: int | None = None
        if search_end > search_start:
            region = row_difference[search_start:search_end]
            boundary = search_start + int(np.argmax(region))
            score = float(row_difference[boundary])
            if score >= max(24.0, baseline * 2.0):
                footer_start = boundary + 1
        if footer_start is not None:
            image = image.crop((0, 0, image.width, footer_start))
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / "wallpaper_reference_clean.png"
        save_privacy_safe_png(image, path)
        return path.resolve()


class OpenCVPerspectiveEstimator:
    """Build a homography from a flat texture canvas to the selected wall."""

    async def estimate(
        self,
        room_image: Path,
        segmentation: WallSegmentationResult,
    ) -> PerspectiveEstimationResult:
        return await asyncio.to_thread(
            self._estimate_sync,
            room_image,
            segmentation,
        )

    @staticmethod
    def _estimate_sync(
        room_image: Path,
        segmentation: WallSegmentationResult,
    ) -> PerspectiveEstimationResult:
        width, height = open_image_safe(room_image).size
        source = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        destination = np.float32(segmentation.polygon_pixels)
        transform = cv2.getPerspectiveTransform(source, destination)
        if not np.all(np.isfinite(transform)):
            raise InputValidationError("Wall perspective could not be estimated.")
        return PerspectiveEstimationResult(
            transform=transform.tolist(),
            destination_quad=segmentation.polygon_pixels,
            canvas_size=(width, height),
        )


class OpenCVTextureRepeater:
    """Tile the wallpaper sample and warp it onto the wall perspective."""

    async def repeat(
        self,
        wallpaper_image: Path,
        perspective: PerspectiveEstimationResult,
        output_directory: Path,
        *,
        pattern_scale: float,
    ) -> TextureRepetitionResult:
        return await asyncio.to_thread(
            self._repeat_sync,
            wallpaper_image,
            perspective,
            output_directory,
            pattern_scale,
        )

    @staticmethod
    def _repeat_sync(
        wallpaper_image: Path,
        perspective: PerspectiveEstimationResult,
        output_directory: Path,
        pattern_scale: float,
    ) -> TextureRepetitionResult:
        reference = open_image_safe(wallpaper_image).convert("RGBA")
        alpha_box = reference.getchannel("A").getbbox()
        if alpha_box:
            reference = reference.crop(alpha_box)
        width, height = perspective.canvas_size
        tile_width = max(32, round(width * pattern_scale))
        tile_height = max(
            32,
            round(tile_width * reference.height / reference.width),
        )
        tile = np.asarray(
            reference.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        )
        repeat_x = width // tile_width + 2
        repeat_y = height // tile_height + 2
        flat = np.tile(tile, (repeat_y, repeat_x, 1))[:height, :width]
        transform = np.asarray(perspective.transform, dtype=np.float32)
        warped = cv2.warpPerspective(
            flat,
            transform,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / "texture_perspective.png"
        Image.fromarray(warped).save(path, format="PNG")
        return TextureRepetitionResult(
            texture_path=path.resolve(),
            repeat_x=repeat_x,
            repeat_y=repeat_y,
        )


class WallpaperGenerationService:
    """Generate and persist a bounded number of wallpaper candidates."""

    def __init__(self, client: TryOnAPIClient) -> None:
        self._client = client

    async def generate_candidates(
        self,
        *,
        room_image: Path,
        wallpaper_image: Path,
        segmentation: WallSegmentationResult,
        output_directory: Path,
        count: int,
        attempt: int,
        start_index: int,
        options: dict[str, Any],
    ) -> list[WallpaperCandidateResult]:
        output_directory.mkdir(parents=True, exist_ok=True)
        provider_options = dict(options)
        if self._client.supports_mask:
            provider_options["replace_mask_path"] = str(
                segmentation.mask_path
            )
        collected: list[bytes] = []
        calls = 0
        while len(collected) < count and calls < count:
            generated = await self._client.generate(
                room_image,
                wallpaper_image,
                "wallpaper",
                {
                    **provider_options,
                    "candidate_count": count - len(collected),
                },
            )
            if not generated:
                raise TryOnAPIError(
                    "Wallpaper provider returned an empty candidate list."
                )
            collected.extend(generated[: count - len(collected)])
            calls += 1
        if len(collected) < count:
            raise TryOnAPIError(
                f"Wallpaper provider returned {len(collected)} of {count} candidates."
            )
        candidates: list[WallpaperCandidateResult] = []
        for offset, payload in enumerate(collected):
            index = start_index + offset
            path = output_directory / f"candidate_{index:02d}.png"
            path.write_bytes(decode_image_bytes(payload))
            candidates.append(
                WallpaperCandidateResult(
                    path=path.resolve(),
                    attempt=attempt,
                    candidate_index=index,
                )
            )
        return candidates


class WallpaperOutputEvaluator:
    """Evaluate candidates with the tenant vision model and local weighting."""

    def __init__(
        self,
        client: QwenClient,
        prompts: WallpaperPromptBuilder,
    ) -> None:
        self._client = client
        self._prompts = prompts

    async def evaluate(
        self,
        room_image: Path,
        wallpaper_image: Path,
        output_image: Path,
    ) -> WallpaperOutputEvaluation:
        model_result = await self._client.analyze_structured(
            self._prompts.output_evaluation(),
            [room_image, wallpaper_image, output_image],
            WallpaperOutputEvaluation,
        )
        score = (
            model_result.wall_coverage * 0.20
            + model_result.pattern_fidelity * 0.25
            + model_result.perspective_accuracy * 0.20
            + model_result.lighting_preservation * 0.15
            + model_result.scene_integrity * 0.20
        )
        return model_result.model_copy(update={"overall_score": score})


class OpenCVLightingPreserver:
    """Restore original scene outside the wall and retain wall illumination."""

    def __init__(self, feather_radius: int) -> None:
        self._feather_radius = feather_radius

    async def preserve(
        self,
        room_image: Path,
        generated_image: Path,
        segmentation: WallSegmentationResult,
        output_directory: Path,
        *,
        preserve_lighting: bool,
    ) -> LightingPreservationResult:
        return await asyncio.to_thread(
            self._preserve_sync,
            room_image,
            generated_image,
            segmentation,
            output_directory,
            preserve_lighting,
        )

    def _preserve_sync(
        self,
        room_image: Path,
        generated_image: Path,
        segmentation: WallSegmentationResult,
        output_directory: Path,
        preserve_lighting: bool,
    ) -> LightingPreservationResult:
        room = np.asarray(open_image_safe(room_image).convert("RGB"))
        height, width = room.shape[:2]
        generated = self._fit_to_size(
            open_image_safe(generated_image).convert("RGB"),
            (width, height),
        )
        mask = np.asarray(open_image_safe(segmentation.mask_path).convert("L"))
        generated = self._register_to_room(generated, room, mask)
        candidate = generated.astype(np.float32)
        if preserve_lighting:
            room_gray = cv2.cvtColor(room, cv2.COLOR_RGB2GRAY).astype(np.float32)
            illumination = cv2.GaussianBlur(room_gray, (0, 0), sigmaX=21)
            wall_values = illumination[mask > 0]
            baseline = float(np.mean(wall_values)) if wall_values.size else 128.0
            ratio = np.clip(illumination / max(1.0, baseline), 0.55, 1.35)
            correction = 1.0 + (ratio - 1.0) * 0.25
            candidate = np.clip(candidate * correction[:, :, None], 0, 255)
        alpha = self._inside_feather_alpha(
            mask,
            self._feather_radius,
        )[:, :, None]
        output = room.astype(np.float32) * (1.0 - alpha) + candidate * alpha
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / "wallpaper.png"
        save_privacy_safe_png(
            Image.fromarray(output.astype(np.uint8)),
            path,
        )
        LOGGER.info(
            "wallpaper_lighting_preserved",
            extra={"stage": "lighting_preservation", "output_size": [width, height]},
        )
        return LightingPreservationResult(image_path=path.resolve())

    @staticmethod
    def _fit_to_size(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
        """Preserve normalized composition coordinates for mask registration."""

        if image.size == size:
            return np.asarray(image)
        return np.asarray(
            image.resize(size, Image.Resampling.LANCZOS)
        )

    @staticmethod
    def _register_to_room(
        generated: np.ndarray,
        room: np.ndarray,
        wall_mask: np.ndarray,
    ) -> np.ndarray:
        """Align model output to original objects before applying the wall mask."""

        valid = (wall_mask == 0).astype(np.uint8) * 255
        if np.count_nonzero(valid) < valid.size * 0.15:
            return generated
        template = cv2.cvtColor(room, cv2.COLOR_RGB2GRAY).astype(np.float32)
        candidate = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY).astype(np.float32)
        template /= 255.0
        candidate /= 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            150,
            1e-5,
        )
        try:
            correlation, warp = cv2.findTransformECC(
                template,
                candidate,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                inputMask=valid,
                gaussFiltSize=5,
            )
        except cv2.error:
            return generated
        if not np.isfinite(correlation) or correlation < 0.35:
            return generated
        height, width = room.shape[:2]
        aligned = cv2.warpAffine(
            generated,
            warp,
            (width, height),
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        LOGGER.info(
            "wallpaper_candidate_registered",
            extra={
                "stage": "lighting_preservation",
                "registration_score": round(float(correlation), 4),
            },
        )
        return aligned

    @staticmethod
    def _inside_feather_alpha(mask: np.ndarray, radius: int) -> np.ndarray:
        """Anti-alias only inside the mask so no wallpaper leaks outside it."""

        binary = (mask > 0).astype(np.uint8)
        if radius <= 0:
            return binary.astype(np.float32)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        return np.clip(distance / float(radius), 0.0, 1.0).astype(np.float32)
