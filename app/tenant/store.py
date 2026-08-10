"""Load and validate tenant-to-pipeline mappings."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.exceptions import TenantConfigurationError
from app.tenant.models import TenantConfig


class TenantConfigDocument(BaseModel):
    """On-disk tenant configuration document."""

    model_config = ConfigDict(extra="forbid")

    tenants: list[TenantConfig] = Field(min_length=1)


class TenantConfigStore:
    """Immutable in-memory tenant registry loaded at application startup."""

    def __init__(
        self,
        *,
        config_path: Path,
        default_tenant_id: str,
        fallback_analysis_provider: str,
        fallback_generation_provider: str,
    ) -> None:
        tenants = self._load(
            config_path,
            default_tenant_id=default_tenant_id,
            fallback_analysis_provider=fallback_analysis_provider,
            fallback_generation_provider=fallback_generation_provider,
        )
        self._by_id: dict[str, TenantConfig] = {}
        hashes: set[str] = set()
        for tenant in tenants:
            if tenant.tenant_id in self._by_id:
                raise TenantConfigurationError(
                    f"Duplicate tenant id: {tenant.tenant_id}"
                )
            if tenant.api_key_sha256:
                if tenant.api_key_sha256 in hashes:
                    raise TenantConfigurationError(
                        "Tenant API key hashes must be unique."
                    )
                hashes.add(tenant.api_key_sha256)
            self._by_id[tenant.tenant_id] = tenant
        if default_tenant_id not in self._by_id:
            raise TenantConfigurationError(
                f"Default tenant '{default_tenant_id}' is not configured."
            )

    @staticmethod
    def _load(
        path: Path,
        *,
        default_tenant_id: str,
        fallback_analysis_provider: str,
        fallback_generation_provider: str,
    ) -> list[TenantConfig]:
        resolved = path.resolve(strict=False)
        if not resolved.is_file():
            tenants = [
                TenantConfig(
                    tenant_id=default_tenant_id,
                    pipeline="clothing",
                    analysis_provider=fallback_analysis_provider,
                    generation_provider=fallback_generation_provider,
                ),
                TenantConfig(
                    tenant_id="wallpaper-demo",
                    pipeline="wallpaper",
                    analysis_provider=fallback_analysis_provider,
                    generation_provider=fallback_generation_provider,
                ),
            ]
            if default_tenant_id != "wallpaper_company":
                tenants.append(
                    TenantConfig(
                        tenant_id="wallpaper_company",
                        pipeline="wallpaper",
                        analysis_provider=fallback_analysis_provider,
                        generation_provider=fallback_generation_provider,
                    )
                )
            return tenants
        try:
            data = json.loads(resolved.read_text(encoding="utf-8"))
            return TenantConfigDocument.model_validate(data).tenants
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise TenantConfigurationError(
                f"Invalid tenant configuration file: {resolved.name}"
            ) from exc

    def get(self, tenant_id: str) -> TenantConfig | None:
        """Return one configured tenant without exposing mutable registry state."""

        return self._by_id.get(tenant_id)

    def all(self) -> tuple[TenantConfig, ...]:
        """Return all tenants in stable insertion order."""

        return tuple(self._by_id.values())
