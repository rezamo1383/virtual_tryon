"""Person analysis service."""

from __future__ import annotations

from pathlib import Path

from app.clients.qwen_client import QwenClient
from app.models.analysis_models import PersonAnalysis


class PersonAnalyzer:
    """Delegate person analysis through an injected Qwen-compatible client."""

    def __init__(self, client: QwenClient) -> None:
        self._client = client

    async def analyze(self, image_path: Path) -> PersonAnalysis:
        return await self._client.analyze_person(image_path)
