from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.preprocessing.background_remover import BackgroundRemover
from app.preprocessing.device_manager import get_compute_device
from app.preprocessing.human_parser import (
    FACE,
    LEFT_HAND,
    UPPER_CLOTHES,
)
from app.preprocessing.image_normalizer import (
    letterbox_image,
    normalize_orientation,
)
from app.preprocessing.image_preprocessor import LocalImagePreprocessor
from app.preprocessing.mask_processor import build_clothing_masks
from app.preprocessing.person_detector import PersonPresenceDetector
from app.preprocessing.preprocessing_exceptions import (
    PersonNotDetectedError,
    PreprocessingError,
    PreprocessingPathError,
)
from app.preprocessing.preprocessing_models import (
    BackgroundRemovalResult,
    HumanParsingResult,
    Keypoint,
    PoseResult,
)
from app.preprocessing.pose_estimator import classify_arms_position
from app.preprocessing.suitability_validator import (
    validate_garment,
    validate_person,
)


def _keypoint(name: str, x: float, y: float, visibility: float = 0.95) -> Keypoint:
    return Keypoint(
        name=name,
        x=x,
        y=y,
        z=0.0,
        visibility=visibility,
        pixel_x=round(x * 100),
        pixel_y=round(y * 120),
    )


def _pose(*, shoulders: bool = True) -> PoseResult:
    points = {
        "nose": _keypoint("nose", 0.5, 0.15),
        "left_hip": _keypoint("left_hip", 0.38, 0.70),
        "right_hip": _keypoint("right_hip", 0.62, 0.70),
        "left_elbow": _keypoint("left_elbow", 0.28, 0.46),
        "right_elbow": _keypoint("right_elbow", 0.72, 0.46),
        "left_wrist": _keypoint("left_wrist", 0.25, 0.68),
        "right_wrist": _keypoint("right_wrist", 0.75, 0.68),
    }
    if shoulders:
        points.update(
            {
                "left_shoulder": _keypoint("left_shoulder", 0.35, 0.32),
                "right_shoulder": _keypoint("right_shoulder", 0.65, 0.32),
            }
        )
    return PoseResult(
        detected_person_count=1,
        keypoints=points,
        shoulder_width=0.3 if shoulders else 0,
        torso_length=0.38 if shoulders else 0,
        body_center=(0.5, 0.51) if shoulders else None,
        pose_confidence=0.95,
        person_orientation="frontal",
        arms_position="down",
        upper_body_visible=shoulders,
        pose_suitable_for_tryon=shoulders,
    )


def _parsing(size: tuple[int, int] = (100, 120)) -> HumanParsingResult:
    class_map = np.zeros((size[1], size[0]), dtype=np.uint8)
    class_map[35:90, 25:75] = UPPER_CLOTHES
    class_map[8:30, 40:60] = FACE
    class_map[62:74, 20:32] = LEFT_HAND
    return HumanParsingResult(
        class_map=class_map,
        visualization=Image.new("RGB", size),
        upper_clothes_mask=Image.fromarray(
            np.where(class_map == UPPER_CLOTHES, 255, 0).astype(np.uint8)
        ),
        arms_mask=Image.new("L", size),
        hands_mask=Image.fromarray(
            np.where(class_map == LEFT_HAND, 255, 0).astype(np.uint8)
        ),
        face_hair_protection_mask=Image.fromarray(
            np.where(
                np.isin(class_map, (FACE, LEFT_HAND)),
                255,
                0,
            ).astype(np.uint8)
        ),
        body_torso_mask=Image.fromarray(
            np.where(class_map == UPPER_CLOTHES, 255, 0).astype(np.uint8)
        ),
        model_name="fake",
    )


def test_cuda_failure_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
        ),
        tensor=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("GPU lost")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert get_compute_device("auto") == "cpu"


def test_invalid_image_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jpg"
    invalid.write_text("not an image", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        preprocessing_device="cpu",
        model_cache_directory=tmp_path / "models",
    )
    preprocessor = LocalImagePreprocessor(settings)
    with pytest.raises(PreprocessingError, match="Could not load person"):
        preprocessor.preprocess(invalid, invalid, tmp_path / "job")


@pytest.mark.parametrize("filename", ["7.jpg", "8.jpg"])
def test_non_person_scene_is_rejected_even_with_zero_threshold(
    tmp_path: Path,
    filename: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    person = project_root / "inputs" / "persons" / filename
    if not person.is_file():
        person = tmp_path / filename
        Image.new("RGB", (640, 640), (15, 20, 35)).save(
            person,
            format="JPEG",
        )
    garment = project_root / "inputs" / "garments" / "gray_sweatshirt.png"
    settings = Settings(
        _env_file=None,
        preprocessing_device="cpu",
        model_cache_directory=tmp_path / "models",
        person_presence_check_enabled=True,
        pose_estimation_enabled=False,
        background_removal_enabled=False,
        human_parsing_enabled=False,
        min_tryon_suitability_score=0,
    )
    preprocessor = LocalImagePreprocessor(settings)
    with pytest.raises(PersonNotDetectedError, match="No person was detected"):
        preprocessor.preprocess(person, garment, tmp_path / "job")


@pytest.mark.parametrize("filename", ["3.jpg", "9.jpg"])
def test_local_person_detector_accepts_real_person(filename: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    with Image.open(project_root / "inputs" / "persons" / filename) as image:
        result = PersonPresenceDetector().detect(image)
    assert result.detected
    assert result.detected_person_count >= 1


def test_semantic_confirmation_accepts_small_multi_region_person() -> None:
    class_map = np.zeros((100, 100), dtype=np.uint8)
    class_map[10:12, 10:25] = UPPER_CLOTHES
    class_map[12:14, 10:20] = FACE
    parsing = _parsing((100, 100)).model_copy(update={"class_map": class_map})
    assert PersonPresenceDetector.has_semantic_human_evidence(parsing)


def test_semantic_confirmation_rejects_single_region_false_positive() -> None:
    class_map = np.zeros((100, 100), dtype=np.uint8)
    class_map[10:30, 10:20] = FACE
    parsing = _parsing((100, 100)).model_copy(update={"class_map": class_map})
    assert not PersonPresenceDetector.has_semantic_human_evidence(parsing)


def test_exif_orientation_is_corrected() -> None:
    image = Image.new("RGB", (40, 20), "red")
    image.getexif()[274] = 6
    assert normalize_orientation(image).size == (20, 40)


def test_resize_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (400, 200))
    normalized, transform = letterbox_image(
        image,
        (300, 300),
        output_mode="RGB",
        allow_upscale=False,
    )
    assert normalized.size == (300, 300)
    assert transform.resized_size == (300, 150)
    assert transform.offset == (0, 75)


def test_background_mask_dimensions_match_image(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 80), "white")
    values = np.asarray(image).copy()
    values[15:70, 25:75] = (30, 60, 100)
    remover = BackgroundRemover(
        enabled=False,
        device="cpu",
        model_cache_directory=tmp_path,
        person_model="u2netp",
        garment_model="u2netp",
    )
    result = remover.remove_background(
        Image.fromarray(values),
        "person",
    )
    assert result.image.size == image.size
    assert result.mask.size == image.size


def test_pose_result_schema_is_valid() -> None:
    result = PoseResult.model_validate(_pose().model_dump())
    assert result.pose_suitable_for_tryon
    assert result.keypoints["left_shoulder"].visibility == pytest.approx(0.95)


def test_crossed_arms_heuristic_is_explainable() -> None:
    pose = _pose()
    points = dict(pose.keypoints)
    points["left_wrist"] = _keypoint("left_wrist", 0.65, 0.48)
    points["right_wrist"] = _keypoint("right_wrist", 0.35, 0.48)
    assert classify_arms_position(points) == "crossed"


def test_missing_shoulders_reduce_suitability_score() -> None:
    image = Image.new("RGB", (100, 120), (120, 130, 140))
    good = validate_person(
        image,
        _pose(shoulders=True),
        _parsing(),
        min_shoulder_visibility=0.55,
        min_score=0.70,
    )
    bad = validate_person(
        image,
        _pose(shoulders=False),
        _parsing(),
        min_shoulder_visibility=0.55,
        min_score=0.70,
    )
    assert bad.score < good.score
    assert not bad.accepted


def test_preserve_mask_protects_face_and_hands() -> None:
    parsing = _parsing()
    foreground = Image.new("L", (100, 120), 255)
    replace, preserve = build_clothing_masks(
        foreground_mask=foreground,
        parsing=parsing,
        pose=_pose(),
        morphology_kernel=3,
        dilation_kernel=3,
        dilation_iterations=1,
        feather_radius=1,
    )
    replace_values = np.asarray(replace)
    preserve_values = np.asarray(preserve)
    assert np.count_nonzero(preserve_values[8:30, 40:60]) > 0
    assert np.count_nonzero(preserve_values[62:74, 20:32]) > 0
    assert np.count_nonzero(replace_values[preserve_values > 0]) == 0


def test_garment_touching_edge_creates_warning() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, :50] = 255
    result = validate_garment(
        Image.new("RGB", (100, 100)),
        Image.fromarray(mask),
        cropped_edges=["left"],
        component_count=1,
        min_score=0.5,
    )
    assert any("left" in warning for warning in result.warnings)


def test_rembg_model_session_is_loaded_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    fake_module = SimpleNamespace(
        new_session=lambda name, providers: calls.append(name) or object()
    )
    monkeypatch.setitem(sys.modules, "rembg", fake_module)
    BackgroundRemover._sessions.clear()
    remover = BackgroundRemover(
        enabled=True,
        device="cpu",
        model_cache_directory=tmp_path,
        person_model="u2netp",
        garment_model="u2netp",
    )
    assert remover._get_session("u2netp") is remover._get_session("u2netp")
    assert calls == ["u2netp"]


def test_human_parsing_failure_activates_fallback(
    tmp_path: Path,
    valid_images: tuple[Path, Path],
) -> None:
    class FakeBackgroundRemover:
        def remove_background(
            self,
            image: Image.Image,
            subject_type: str,
        ) -> BackgroundRemovalResult:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            mask[
                image.height // 10 : image.height * 9 // 10,
                image.width // 10 : image.width * 9 // 10,
            ] = 255
            rgba = np.asarray(image.convert("RGBA")).copy()
            rgba[:, :, 3] = mask
            return BackgroundRemovalResult(
                image=Image.fromarray(rgba),
                mask=Image.fromarray(mask),
                model_name=f"fake_{subject_type}",
            )

    class FakePoseEstimator:
        def estimate(self, image: Image.Image) -> tuple[PoseResult, Image.Image]:
            return _pose(), image.convert("RGB")

    class FailingParser:
        def parse(self, image: Image.Image) -> HumanParsingResult:
            raise RuntimeError("simulated parser failure")

    person, garment = valid_images
    settings = Settings(
        _env_file=None,
        preprocessing_device="cpu",
        model_cache_directory=tmp_path / "models",
        output_directory=tmp_path / "outputs",
        person_target_width=128,
        person_target_height=160,
        garment_target_width=128,
        garment_target_height=160,
        min_tryon_suitability_score=0.5,
        human_parsing_enabled=True,
        human_parsing_required=False,
    )
    result = LocalImagePreprocessor(
        settings,
        background_remover=FakeBackgroundRemover(),  # type: ignore[arg-type]
        pose_estimator=FakePoseEstimator(),  # type: ignore[arg-type]
        human_parser=FailingParser(),
    ).preprocess(person, garment, tmp_path / "job")
    assert result.degraded_mode
    assert any("fallback" in warning.lower() for warning in result.warnings)
    assert result.person.parsing_debug_path is not None


def test_human_parser_warmup_primes_injected_parser(tmp_path: Path) -> None:
    parsed_sizes: list[tuple[int, int]] = []

    class RecordingParser:
        def parse(self, image: Image.Image) -> HumanParsingResult:
            parsed_sizes.append(image.size)
            return _parsing(image.size)

    settings = Settings(
        _env_file=None,
        preprocessing_device="cpu",
        model_cache_directory=tmp_path / "models",
        human_parsing_enabled=True,
    )
    preprocessor = LocalImagePreprocessor(
        settings,
        human_parser=RecordingParser(),
    )

    assert preprocessor.warmup() >= 0
    assert parsed_sizes == [(512, 512)]


def test_human_parser_warmup_skips_when_disabled(tmp_path: Path) -> None:
    class UnexpectedParser:
        def parse(self, image: Image.Image) -> HumanParsingResult:
            raise AssertionError("disabled parser must not be called")

    settings = Settings(
        _env_file=None,
        preprocessing_device="cpu",
        model_cache_directory=tmp_path / "models",
        human_parsing_enabled=False,
    )
    preprocessor = LocalImagePreprocessor(
        settings,
        human_parser=UnexpectedParser(),
    )

    assert preprocessor.warmup() == 0


def test_artifact_path_cannot_escape_job(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        preprocessing_device="cpu",
        model_cache_directory=tmp_path / "models",
    )
    preprocessor = LocalImagePreprocessor(settings)
    with pytest.raises(PreprocessingPathError):
        preprocessor._safe_artifact_path(
            tmp_path / "job",
            Path("../escape.png"),
        )
