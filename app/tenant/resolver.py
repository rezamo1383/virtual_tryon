"""Resolve authenticated API keys to tenant configurations."""

from __future__ import annotations

import hashlib
import hmac

from app.core.exceptions import (
    TenantAuthenticationError,
    TenantConfigurationError,
)
from app.tenant.models import PipelineName, TenantConfig
from app.tenant.store import TenantConfigStore


class TenantResolver:
    """Authenticate tenant API keys without accepting a client task type."""

    def __init__(
        self,
        store: TenantConfigStore,
        *,
        default_tenant_id: str,
        authentication_required: bool,
    ) -> None:
        self._store = store
        self._default_tenant_id = default_tenant_id
        self._authentication_required = authentication_required

    def resolve(self, api_key: str | None) -> TenantConfig:
        """Resolve an API key or use the configured compatibility tenant."""

        supplied = (api_key or "").strip()
        if supplied:
            digest = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
            for tenant in self._store.all():
                expected = tenant.api_key_sha256
                if expected and hmac.compare_digest(digest, expected):
                    return self._ensure_enabled(tenant)
            raise TenantAuthenticationError("Invalid tenant API key.")
        if self._authentication_required:
            raise TenantAuthenticationError("A tenant API key is required.")
        tenant = self._store.get(self._default_tenant_id)
        if tenant is None:
            raise TenantConfigurationError("The default tenant is unavailable.")
        return self._ensure_enabled(tenant)

    def resolve_for_cli(
        self,
        *,
        tenant_id: str | None,
        pipeline: PipelineName,
    ) -> TenantConfig:
        """Resolve an explicit or first matching tenant for trusted local CLI."""

        if tenant_id:
            tenant = self._store.get(tenant_id)
            if tenant is None:
                raise TenantConfigurationError(
                    f"Unknown tenant id: {tenant_id}"
                )
            tenant = self._ensure_enabled(tenant)
            if tenant.pipeline != pipeline:
                raise TenantConfigurationError(
                    f"Tenant '{tenant.tenant_id}' owns pipeline "
                    f"'{tenant.pipeline}', not '{pipeline}'."
                )
            return tenant
        default = self._store.get(self._default_tenant_id)
        if default and default.enabled and default.pipeline == pipeline:
            return default
        for candidate in self._store.all():
            if candidate.enabled and candidate.pipeline == pipeline:
                return candidate
        raise TenantConfigurationError(
            f"No enabled tenant is configured for pipeline '{pipeline}'."
        )

    @staticmethod
    def _ensure_enabled(tenant: TenantConfig) -> TenantConfig:
        if not tenant.enabled:
            raise TenantAuthenticationError("Tenant is disabled.")
        return tenant
