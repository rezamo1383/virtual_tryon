"""Replaceable ONNX human parsing with a pose/foreground fallback."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.preprocessing.model_manager import ModelManager, ModelSpec
from app.preprocessing.preprocessing_exceptions import HumanParsingError
from app.preprocessing.preprocessing_models import HumanParsingResult, PoseResult

LOGGER = logging.getLogger(__name__)

# Standard project labels. The selected ATR model lacks explicit hands/torso;
# those regions are augmented from pose in the mask processor and fallback.
BACKGROUND = 0
HAIR = 1
FACE = 2
NECK = 3
TORSO = 4
UPPER_CLOTHES = 5
COAT_JACKET = 6
LEFT_ARM = 7
RIGHT_ARM = 8
LEFT_HAND = 9
RIGHT_HAND = 10
SCARF = 11

STANDARD_COLORS = np.array(
    [
        (0, 0, 0),
        (90, 45, 120),
        (255, 190, 160),
        (210, 150, 130),
        (70, 160, 210),
        (220, 60, 80),
        (160, 40, 60),
        (255, 140, 100),
        (255, 170, 100),
        (255, 220, 180),
        (255, 220, 180),
        (240, 190, 40),
    ],
    dtype=np.uint8,
)


class HumanParser(Protocol):
    """Provider-neutral semantic parser interface."""

    def parse(self, image: Image.Image) -> HumanParsingResult:
        """Parse a person image into standardized semantic regions."""


def _mask(class_map: np.ndarray, labels: tuple[int, ...]) -> Image.Image:
    value = np.isin(class_map, labels).astype(np.uint8) * 255
    return Image.fromarray(value)


def _visualize(class_map: np.ndarray) -> Image.Image:
    bounded = np.clip(class_map, 0, len(STANDARD_COLORS) - 1)
    return Image.fromarray(STANDARD_COLORS[bounded])


def _build_result(
    class_map: np.ndarray,
    *,
    model_name: str,
    degraded_mode: bool,
    warnings: list[str] | None = None,
) -> HumanParsingResult:
    return HumanParsingResult(
        class_map=class_map.astype(np.uint8),
        visualization=_visualize(class_map),
        upper_clothes_mask=_mask(
            class_map,
            (UPPER_CLOTHES, COAT_JACKET, SCARF),
        ),
        arms_mask=_mask(class_map, (LEFT_ARM, RIGHT_ARM)),
        hands_mask=_mask(class_map, (LEFT_HAND, RIGHT_HAND)),
        face_hair_protection_mask=_mask(
            class_map,
            (HAIR, FACE, NECK, LEFT_HAND, RIGHT_HAND),
        ),
        body_torso_mask=_mask(class_map, (TORSO, UPPER_CLOTHES, COAT_JACKET)),
        model_name=model_name,
        degraded_mode=degraded_mode,
        warnings=warnings or [],
    )


class OnnxAtrHumanParser:
    """CPU-first SCHP ATR-18 INT8 parser with shared ONNX sessions."""

    _sessions: dict[tuple[str, str], Any] = {}
    _session_lock = threading.Lock()

    def __init__(
        self,
        *,
        model_manager: ModelManager,
        model_spec: ModelSpec,
        device: str,
    ) -> None:
        self.model_manager = model_manager
        self.model_spec = model_spec
        self.device = device

    def parse(self, image: Image.Image) -> HumanParsingResult:
        """Run 512px SCHP inference and map ATR classes to project classes."""

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise HumanParsingError(
                "onnxruntime is not installed for human parsing."
            ) from exc
        model_path = self.model_manager.ensure(self.model_spec)
        session = self._get_session(ort, model_path)
        rgb = image.convert("RGB")
        resized = rgb.resize((512, 512), Image.Resampling.BILINEAR)
        values = np.asarray(resized, dtype=np.float32) / 255.0
        values = (values - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        tensor = np.transpose(values, (2, 0, 1))[None].astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})
        logits = next(
            (
                output
                for output in outputs
                if isinstance(output, np.ndarray)
                and output.ndim == 4
                and output.shape[1] == 18
            ),
            None,
        )
        if logits is None:
            raise HumanParsingError(
                "Human parsing model returned no 18-class logits."
            )
        atr_map = np.argmax(logits, axis=1)[0].astype(np.uint8)
        atr_map = cv2.resize(
            atr_map,
            rgb.size,
            interpolation=cv2.INTER_NEAREST,
        )
        standard = np.zeros_like(atr_map)
        mapping = {
            2: HAIR,
            11: FACE,
            4: UPPER_CLOTHES,
            7: COAT_JACKET,
            14: LEFT_ARM,
            15: RIGHT_ARM,
            17: SCARF,
        }
        for source, target in mapping.items():
            standard[atr_map == source] = target
        return _build_result(
            standard,
            model_name=self.model_spec.filename,
            degraded_mode=False,
        )

    def _get_session(self, ort: Any, model_path: Path) -> Any:
        providers = (
            ("CUDAExecutionProvider", "CPUExecutionProvider")
            if self.device == "cuda"
            else ("CPUExecutionProvider",)
        )
        available = set(ort.get_available_providers())
        selected = tuple(provider for provider in providers if provider in available)
        if not selected:
            selected = ("CPUExecutionProvider",)
        key = (str(model_path.resolve()), ",".join(selected))
        with self._session_lock:
            if key in self._sessions:
                return self._sessions[key]
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, (os_cpu_count() or 2)))
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=list(selected),
            )
            self._sessions[key] = session
            return session


def os_cpu_count() -> int | None:
    """Small seam for deterministic tests."""

    import os

    return os.cpu_count()


class HeuristicHumanParser:
    """Degraded semantic masks derived only from pose and foreground pixels."""

    def __init__(
        self,
        *,
        pose: PoseResult,
        foreground_mask: Image.Image,
        warning: str,
    ) -> None:
        self.pose = pose
        self.foreground_mask = foreground_mask
        self.warning = warning

    def parse(self, image: Image.Image) -> HumanParsingResult:
        width, height = image.size
        foreground = np.asarray(
            self.foreground_mask.convert("L").resize(
                image.size,
                Image.Resampling.NEAREST,
            )
        )
        class_map = np.zeros((height, width), dtype=np.uint8)
        points = self.pose.keypoints
        required = {
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        }
        if required.issubset(points):
            torso = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(torso)
            polygon = [
                (
                    points["left_shoulder"].pixel_x,
                    points["left_shoulder"].pixel_y,
                ),
                (
                    points["right_shoulder"].pixel_x,
                    points["right_shoulder"].pixel_y,
                ),
                (points["right_hip"].pixel_x, points["right_hip"].pixel_y),
                (points["left_hip"].pixel_x, points["left_hip"].pixel_y),
            ]
            draw.polygon(polygon, fill=255)
            torso_array = np.asarray(torso)
            class_map[(torso_array > 0) & (foreground > 0)] = UPPER_CLOTHES
        self._draw_limbs(class_map, LEFT_ARM, "left", width, height)
        self._draw_limbs(class_map, RIGHT_ARM, "right", width, height)
        self._draw_face(class_map, width, height)
        return _build_result(
            class_map,
            model_name="pose_foreground_fallback",
            degraded_mode=True,
            warnings=[self.warning],
        )

    def _draw_limbs(
        self,
        class_map: np.ndarray,
        arm_label: int,
        side: str,
        width: int,
        height: int,
    ) -> None:
        names = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
        if not all(name in self.pose.keypoints for name in names):
            return
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        points = [
            (
                self.pose.keypoints[name].pixel_x,
                self.pose.keypoints[name].pixel_y,
            )
            for name in names
        ]
        line_width = max(8, round(self.pose.shoulder_width * width * 0.14))
        draw.line(points, fill=255, width=line_width, joint="curve")
        arm = np.asarray(canvas) > 0
        class_map[arm] = arm_label
        wrist = points[-1]
        radius = max(5, line_width // 2)
        y_grid, x_grid = np.ogrid[:height, :width]
        hand = (
            (x_grid - wrist[0]) ** 2 + (y_grid - wrist[1]) ** 2
            <= radius**2
        )
        class_map[hand] = LEFT_HAND if side == "left" else RIGHT_HAND

    def _draw_face(
        self,
        class_map: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        nose = self.pose.keypoints.get("nose")
        if nose is None:
            return
        shoulders = [
            self.pose.keypoints.get("left_shoulder"),
            self.pose.keypoints.get("right_shoulder"),
        ]
        radius = (
            max(8, round(self.pose.shoulder_width * width * 0.24))
            if all(shoulders)
            else max(8, min(width, height) // 14)
        )
        y_grid, x_grid = np.ogrid[:height, :width]
        face = (
            (x_grid - nose.pixel_x) ** 2 + (y_grid - nose.pixel_y) ** 2
            <= radius**2
        )
        class_map[face] = FACE
