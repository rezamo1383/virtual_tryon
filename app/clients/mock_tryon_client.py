"""Offline virtual try-on mock used by tests and local development."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.clients.tryon_api_client import TryOnAPIClient


class MockTryOnClient(TryOnAPIClient):
    """Create deterministic side-by-side composites without internet access."""

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        count = max(1, int(options.get("candidate_count", 1)))
        label = str(options.get("requested_color", "variant"))
        with Image.open(person_image) as source:
            person = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(garment_image) as source:
            garment = ImageOps.exif_transpose(source).convert("RGBA")

        garment.thumbnail((max(64, person.width // 3), max(64, person.height // 2)))
        outputs: list[bytes] = []
        for index in range(count):
            canvas = person.copy()
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            x = max(8, canvas.width - garment.width - 16)
            y = max(8, (canvas.height - garment.height) // 2)
            overlay.alpha_composite(garment, (x, y))
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            text = f"MOCK {label} #{index + 1}"
            box = draw.textbbox((0, 0), text, font=ImageFont.load_default())
            draw.rectangle(
                (8, 8, box[2] + 18, box[3] + 18), fill=(20, 20, 20)
            )
            draw.text((13, 13), text, fill=(255, 255, 255), font=ImageFont.load_default())
            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG", optimize=True)
            outputs.append(buffer.getvalue())
        return outputs
