from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import cli
from cli import app, expand_color_arguments, labeled_garments_from_options
from app.core.config import Settings


def test_space_separated_colors_are_expanded() -> None:
    assert expand_color_arguments(
        ["run", "--colors", "red", "blue", "#000000", "--max-retries", "1"]
    ) == [
        "run",
        "--colors",
        "red",
        "--colors",
        "blue",
        "--colors",
        "#000000",
        "--max-retries",
        "1",
    ]


def test_labeled_garments_require_one_type_per_image() -> None:
    with pytest.raises(typer.BadParameter, match="exactly once"):
        labeled_garments_from_options(
            [Path("shirt.png"), Path("pants.png")],
            ["T-shirt"],
        )


def test_single_garment_keeps_product_title_compatibility() -> None:
    garments = labeled_garments_from_options(
        [Path("shirt.png")],
        None,
        product_title="Men's T-shirt",
    )
    assert garments[0].garment_type == "Men's T-shirt"


def test_clothing_command_applies_repeated_garments(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    valid_images: tuple[Path, Path],
) -> None:
    person, garment = valid_images
    test_settings = settings.model_copy(
        update={
            "analysis_provider": "mock",
            "tryon_provider": "mock",
            "use_mock_qwen": True,
            "use_mock_tryon": True,
            "tenant_config_path": (
                settings.output_directory.parent / "missing-tenants.json"
            ),
        }
    )
    monkeypatch.setattr(cli, "_settings", lambda: test_settings)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "clothing",
            "--person",
            str(person),
            "--garment",
            str(garment),
            "--garment-type",
            "T-shirt",
            "--garment",
            str(garment),
            "--garment-type",
            "Watch",
            "--candidates-per-color",
            "2",
            "--max-retries",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Applied items: T-shirt -> Watch" in result.output
    assert "Stage jobs:" in result.output
    jobs = sorted(test_settings.output_directory.glob("job_*"))
    assert len(jobs) == 1
    request_data = json.loads(
        (jobs[0] / "request.json").read_text(encoding="utf-8")
    )
    assert request_data["generation_strategy"] == "single_call_multi_reference"
    assert request_data["garment_types"] == ["T-shirt", "Watch"]
    assert request_data["candidates_per_color"] == 2
