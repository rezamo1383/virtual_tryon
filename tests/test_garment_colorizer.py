from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.garment_colorizer import GarmentColorizer


@pytest.fixture
def colorizer_inputs(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    rgb = np.zeros((100, 120, 4), dtype=np.uint8)
    rgb[:, :, :3] = (90, 120, 150)
    rgb[:, :, 3] = 255
    # A luminance gradient represents texture/folds.
    rgb[20:80, 30:90, :3] += np.arange(60, dtype=np.uint8)[:, None, None] // 3
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[20:80, 30:90] = 255
    image_path = tmp_path / "garment.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path, mask_path, rgb, mask


@pytest.mark.parametrize("color", ["red", "black", "white"])
def test_color_outputs(
    colorizer_inputs: tuple[Path, Path, np.ndarray, np.ndarray],
    tmp_path: Path,
    color: str,
) -> None:
    image_path, mask_path, original, mask = colorizer_inputs
    output = tmp_path / f"{color}.png"
    GarmentColorizer().create_variant(image_path, mask_path, color, output)
    actual = np.asarray(Image.open(output).convert("RGBA"))
    assert actual.shape == original.shape
    assert output.is_file()
    assert not np.array_equal(actual[mask > 0, :3], original[mask > 0, :3])


def test_outside_mask_unchanged(
    colorizer_inputs: tuple[Path, Path, np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    image_path, mask_path, original, mask = colorizer_inputs
    output = tmp_path / "red.png"
    GarmentColorizer().create_variant(image_path, mask_path, "red", output)
    actual = np.asarray(Image.open(output).convert("RGBA"))
    np.testing.assert_array_equal(actual[mask == 0], original[mask == 0])
