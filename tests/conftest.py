"""Shared image and settings fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.core.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        min_image_width=64,
        min_image_height=64,
        output_directory=tmp_path / "outputs",
        temp_directory=tmp_path / "temp",
        log_directory=tmp_path / "logs",
        use_mock_qwen=True,
        use_mock_tryon=True,
        delete_temp_files=True,
        min_acceptance_score=0.80,
        local_preprocessing_enabled=False,
    )


@pytest.fixture
def valid_images(tmp_path: Path) -> tuple[Path, Path]:
    person_path = tmp_path / "person.jpg"
    person = Image.new("RGB", (512, 640), (190, 180, 170))
    draw = ImageDraw.Draw(person)
    draw.ellipse((190, 40, 320, 170), fill=(220, 180, 150))
    draw.rectangle((160, 170, 350, 580), fill=(80, 100, 130))
    person.save(person_path, format="JPEG", quality=92)

    garment_path = tmp_path / "hoodie.png"
    garment = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(garment)
    draw.polygon(
        [(145, 80), (367, 80), (450, 180), (390, 230), (365, 460),
         (147, 460), (122, 230), (62, 180)],
        fill=(120, 130, 150, 255),
    )
    draw.ellipse((205, 65, 307, 155), fill=(0, 0, 0, 0))
    garment.save(garment_path, format="PNG")
    return person_path, garment_path
