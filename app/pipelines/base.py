"""Common pipeline interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.models.request_models import GenerationRequest
from app.tenant.models import PipelineName


class BasePipeline(ABC):
    """Domain-independent asynchronous pipeline contract."""

    pipeline_name: PipelineName
    tenant_id: str

    @abstractmethod
    async def run(self, request: GenerationRequest) -> BaseModel:
        """Execute one domain job."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release tenant-scoped providers."""
