"""Deterministic local wallpaper generator for tests and development."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.clients.tryon_api_client import TryOnAPIClient
from app.utils.image_utils import open_image_safe


class MockWallpaperClient(TryOnAPIClient):
    """Tile the reference into the wall mask while preserving room lighting."""

    supports_mask = True
    supports_text_prompt = True

    async def generate(
        self,
        person_image: Path,
        garment_image: Path,
        category: str,
        options: dict[str, Any],
    ) -> list[bytes]:
        room = np.asarray(open_image_safe(person_image).convert("RGB"))
        reference = np.asarray(
            open_image_safe(garment_image).convert("RGB")
        )
        height, width = room.shape[:2]
        mask = self._load_mask(
            options.get("replace_mask_path"),
            (width, height),
        )
        scale = float(options.get("pattern_scale", 0.18))
        tile_width = max(32, round(width * max(0.03, min(0.75, scale))))
        tile_height = max(
            32,
            round(tile_width * reference.shape[0] / reference.shape[1]),
        )
        tile = cv2.resize(
            reference,
            (tile_width, tile_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        count = max(1, int(options.get("candidate_count", 1)))
        outputs: list[bytes] = []
        for index in range(count):
            texture = self._tile(
                tile,
                width,
                height,
                offset=(index * 13) % tile_width,
            )
            outputs.append(self._composite(room, texture, mask))
        return outputs

    @staticmethod
    def _load_mask(value: object, size: tuple[int, int]) -> np.ndarray:
        if value:
            path = Path(str(value))
            if path.is_file():
                mask = np.asarray(open_image_safe(path).convert("L"))
                if mask.shape[::-1] != size:
                    mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
                return mask
        width, height = size
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[round(height * 0.05) : round(height * 0.78),
             round(width * 0.05) : round(width * 0.95)] = 255
        return mask

    @staticmethod
    def _tile(
        tile: np.ndarray,
        width: int,
        height: int,
        *,
        offset: int,
    ) -> np.ndarray:
        repeats_y = height // tile.shape[0] + 2
        repeats_x = width // tile.shape[1] + 2
        tiled = np.tile(tile, (repeats_y, repeats_x, 1))
        return tiled[:height, offset : offset + width]

    @staticmethod
    def _composite(
        room: np.ndarray,
        texture: np.ndarray,
        mask: np.ndarray,
    ) -> bytes:
        room_float = room.astype(np.float32)
        texture_float = texture.astype(np.float32)
        alpha = (mask > 0).astype(np.float32)[:, :, None]
        output = room_float * (1.0 - alpha) + texture_float * alpha
        image = Image.fromarray(output.astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
