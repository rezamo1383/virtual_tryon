"""Platform and domain exceptions with safe, user-facing messages."""


class AIPlatformError(Exception):
    """Base exception for expected multi-domain platform failures."""


class TenantAuthenticationError(AIPlatformError):
    """Raised when an API key cannot be mapped to an enabled tenant."""


class TenantConfigurationError(AIPlatformError):
    """Raised when tenant configuration is missing, unsafe, or inconsistent."""


class PipelineRoutingError(AIPlatformError):
    """Raised when no compatible pipeline can serve a tenant request."""


class VirtualTryOnError(AIPlatformError):
    """Base exception for all expected application failures."""


class InputValidationError(VirtualTryOnError):
    """Raised when an input file or request is unsafe or invalid."""


class QwenAPIError(VirtualTryOnError):
    """Raised for Qwen transport, protocol, or validation failures."""


class OpenRouterAPIError(VirtualTryOnError):
    """Raised for OpenRouter transport, protocol, or validation failures."""


class GapGPTAPIError(VirtualTryOnError):
    """Raised for GapGPT transport, protocol, or validation failures."""


class TryOnAPIError(VirtualTryOnError):
    """Raised for try-on provider transport or response failures."""


class ImageProcessingError(VirtualTryOnError):
    """Raised when segmentation or colorization cannot be completed."""


class OutputEvaluationError(VirtualTryOnError):
    """Raised when a generated candidate cannot be evaluated."""
