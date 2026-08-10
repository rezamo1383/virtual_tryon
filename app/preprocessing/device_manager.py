"""Safe compute-device selection with a real CUDA probe."""

from __future__ import annotations

import logging
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)


def _cuda_works(torch_module: Any) -> bool:
    """Run a tiny CUDA operation instead of trusting availability metadata."""

    if not torch_module.cuda.is_available():
        return False
    tensor = torch_module.tensor([1.0], device="cuda")
    result = tensor + 1.0
    torch_module.cuda.synchronize()
    return float(result.cpu().item()) == 2.0


def get_compute_device(
    preference: Literal["auto", "cpu", "cuda"] = "auto",
) -> Literal["cpu", "cuda"]:
    """Return CUDA only after successful allocation and computation."""

    if preference == "cpu":
        LOGGER.info("preprocessing_device_selected", extra={"device": "cpu"})
        return "cpu"
    try:
        import torch

        if _cuda_works(torch):
            LOGGER.info(
                "preprocessing_device_selected",
                extra={"device": "cuda"},
            )
            return "cuda"
        warning = "CUDA is unavailable; using CPU."
    except Exception as exc:
        warning = f"CUDA probe failed ({type(exc).__name__}); using CPU."
    LOGGER.warning(
        "preprocessing_cuda_fallback",
        extra={"device": "cpu", "warning": warning, "fallback_used": True},
    )
    return "cpu"
