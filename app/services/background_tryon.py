"""Simple process-local background execution for public Try-On jobs."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.runtime import PlatformRuntime
from app.models.request_models import ClothingOptions
from app.services.multi_garment_tryon import (
    LabeledGarment,
    run_multi_garment_tryon,
)
from app.tenant.models import TenantConfig
from app.utils.file_utils import remove_tree
from app.utils.json_utils import read_json, write_json

LOGGER = logging.getLogger(__name__)


class InProcessTryOnJobs:
    """Own asyncio tasks and durable lifecycle markers for one API process."""

    def __init__(self, settings: Settings, runtime: PlatformRuntime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.tasks: set[asyncio.Task[None]] = set()

    def submit(
        self,
        *,
        job_id: str,
        tenant: TenantConfig,
        person_image: Path,
        garments: list[LabeledGarment],
        options: ClothingOptions,
        input_directory: Path,
    ) -> None:
        """Persist queued state and schedule the coroutine on the current loop."""

        self._write_state(job_id, tenant, "queued")
        task = asyncio.create_task(
            self._run(
                job_id=job_id,
                tenant=tenant,
                person_image=person_image,
                garments=garments,
                options=options,
                input_directory=input_directory,
            ),
            name=f"tryon:{job_id}",
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _run(
        self,
        *,
        job_id: str,
        tenant: TenantConfig,
        person_image: Path,
        garments: list[LabeledGarment],
        options: ClothingOptions,
        input_directory: Path,
    ) -> None:
        self._write_state(job_id, tenant, "running")
        LOGGER.info(
            "background_tryon_started",
            extra={"job_id": job_id, "tenant_id": tenant.tenant_id},
        )
        try:
            result = await run_multi_garment_tryon(
                runtime=self.runtime,
                tenant=tenant,
                person_image=person_image,
                garments=garments,
                options=options,
                job_id=job_id,
            )
        except asyncio.CancelledError:
            self._mark_failed(job_id, tenant, "CancelledError")
            LOGGER.warning(
                "background_tryon_cancelled",
                extra={"job_id": job_id, "tenant_id": tenant.tenant_id},
            )
            raise
        except Exception as exc:
            self._mark_failed(job_id, tenant, type(exc).__name__)
            LOGGER.error(
                "background_tryon_failed",
                extra={
                    "job_id": job_id,
                    "tenant_id": tenant.tenant_id,
                    "error_type": type(exc).__name__,
                },
            )
        else:
            self._write_state(job_id, tenant, result.result.status)
            LOGGER.info(
                "background_tryon_finished",
                extra={
                    "job_id": job_id,
                    "tenant_id": tenant.tenant_id,
                    "status": result.result.status,
                },
            )
        finally:
            if self.settings.delete_temp_files:
                remove_tree(input_directory, self.settings.temp_directory)

    async def shutdown(self) -> None:
        """Cancel active tasks before provider clients are closed."""

        active = tuple(self.tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    async def wait_all(self) -> None:
        """Wait for the current task snapshot; intended for tests and shutdown."""

        active = tuple(self.tasks)
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    def _write_state(
        self,
        job_id: str,
        tenant: TenantConfig,
        status: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        path = self._job_directory(job_id) / "job_state.json"
        previous = read_json(path) if path.is_file() else {}
        write_json(
            path,
            {
                "job_id": job_id,
                "tenant_id": tenant.tenant_id,
                "pipeline": tenant.pipeline,
                "status": status,
                "started_at": previous.get("started_at", now),
                "completed_at": (
                    now
                    if status in {
                        "completed",
                        "completed_with_failures",
                        "failed",
                        "rejected",
                    }
                    else None
                ),
            },
        )

    def _mark_failed(
        self,
        job_id: str,
        tenant: TenantConfig,
        error_type: str,
    ) -> None:
        self._write_state(job_id, tenant, "failed")
        result_path = self._job_directory(job_id) / "results.json"
        if result_path.is_file():
            result = read_json(result_path)
            result.update(
                {
                    "job_id": job_id,
                    "tenant_id": tenant.tenant_id,
                    "pipeline": tenant.pipeline,
                    "status": "failed",
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
            result.setdefault("error", error_type)
            write_json(result_path, result)

    def _job_directory(self, job_id: str) -> Path:
        path = self.settings.output_directory / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path
