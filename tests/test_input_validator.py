from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.core.exceptions import InputValidationError
from app.models.request_models import TryOnRequest
from app.services.input_validator import InputValidator


def request(person: Path, garment: Path, colors: list[str]) -> TryOnRequest:
    return TryOnRequest(
        person_image=person,
        garment_image=garment,
        colors=colors,
        candidates_per_color=1,
    )


def test_missing_file(settings: Settings, valid_images: tuple[Path, Path]) -> None:
    _, garment = valid_images
    with pytest.raises(InputValidationError, match="does not exist"):
        InputValidator(settings).validate(request(Path("missing.jpg"), garment, ["red"]))


def test_invalid_format(
    settings: Settings, valid_images: tuple[Path, Path], tmp_path: Path
) -> None:
    _, garment = valid_images
    invalid = tmp_path / "person.gif"
    Image.new("RGB", (128, 128)).save(invalid, "GIF")
    with pytest.raises(InputValidationError, match="extension"):
        InputValidator(settings).validate(request(invalid, garment, ["red"]))


def test_valid_hex(settings: Settings, valid_images: tuple[Path, Path]) -> None:
    person, garment = valid_images
    validated = InputValidator(settings).validate(
        request(person, garment, ["#C62828"])
    )
    assert validated.colors == ["#C62828"]


def test_original_reference_color_is_valid(
    settings: Settings,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    validated = InputValidator(settings).validate(
        TryOnRequest(
            person_image=person,
            garment_image=garment,
            product_title="تي شرت مردانه",
            candidates_per_color=1,
        )
    )
    assert validated.colors == ["original"]


def test_invalid_hex(settings: Settings, valid_images: tuple[Path, Path]) -> None:
    person, garment = valid_images
    with pytest.raises(InputValidationError, match="Invalid color"):
        InputValidator(settings).validate(request(person, garment, ["#12GG00"]))


def test_image_too_small(
    settings: Settings, valid_images: tuple[Path, Path], tmp_path: Path
) -> None:
    _, garment = valid_images
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (32, 32)).save(tiny)
    with pytest.raises(InputValidationError, match="too small"):
        InputValidator(settings).validate(request(tiny, garment, ["blue"]))


def test_corrupt_image(
    settings: Settings, valid_images: tuple[Path, Path], tmp_path: Path
) -> None:
    _, garment = valid_images
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"\xff\xd8not-an-image")
    with pytest.raises(InputValidationError, match="corrupt"):
        InputValidator(settings).validate(request(corrupt, garment, ["black"]))
