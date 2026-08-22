"""Offline virtual try-on mock used by tests and local development."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.clients.tryon_api_client import TryOnAPIClient


class MockTryOnClient(TryOnAPIClient):
    """Create deterministic side-by-side composites without internet access."""

    supports_multi_reference = True

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        return await self.generate_multi(
            person_image,
            [garment_image],
            [str(options.get("product_title") or category)],
            category,
            options,
        )

    async def generate_multi(
        self,
        person_image: Path,
        garment_images: list[Path],
        garment_types: list[str],
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        """Render all mock references in one deterministic provider operation."""

        count = max(1, int(options.get("candidate_count", 1)))
        label = str(options.get("requested_color", "variant"))
        with Image.open(person_image) as source:
            person = ImageOps.exif_transpose(source).convert("RGB")
        garments: list[Image.Image] = []
        for garment_image in garment_images:
            with Image.open(garment_image) as source:
                garment = ImageOps.exif_transpose(source).convert("RGBA")
            garment.thumbnail(
                (
                    max(48, person.width // max(3, len(garment_images) + 1)),
                    max(64, person.height // 3),
                )
            )
            garments.append(garment)
        outputs: list[bytes] = []
        for index in range(count):
            canvas = person.copy()
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            x = 8
            for garment in garments:
                y = max(8, canvas.height - garment.height - 16)
                overlay.alpha_composite(garment, (x, y))
                x += garment.width + 8
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            text = f"MOCK {label} {len(garment_types)} items #{index + 1}"
            box = draw.textbbox((0, 0), text, font=ImageFont.load_default())
            draw.rectangle(
                (8, 8, box[2] + 18, box[3] + 18), fill=(20, 20, 20)
            )
            draw.text((13, 13), text, fill=(255, 255, 255), font=ImageFont.load_default())
            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG", optimize=True)
            outputs.append(buffer.getvalue())
        return outputs
