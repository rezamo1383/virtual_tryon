"""Garment analysis service."""

from __future__ import annotations

from pathlib import Path

from app.clients.qwen_client import QwenClient
from app.models.analysis_models import GarmentAnalysis


class GarmentAnalyzer:
    """Delegate garment classification through an injected Qwen client."""

    def __init__(self, client: QwenClient) -> None:
        self._client = client

    async def analyze(self, image_path: Path) -> GarmentAnalysis:
        return await self._client.analyze_garment(image_path)
