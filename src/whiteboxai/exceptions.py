"""
SDK Exceptions
"""


class WhiteBoxAIError(Exception):
    """Base exception for WhiteBoxAI SDK."""

    pass


class APIError(WhiteBoxAIError):
    """API request error."""

    pass


class AuthenticationError(WhiteBoxAIError):
    """Authentication error."""

    pass


class RateLimitError(WhiteBoxAIError):
    """Rate limit exceeded error."""

    pass


class ValidationError(WhiteBoxAIError):
    """Validation error."""

    pass


class NotFoundError(WhiteBoxAIError):
    """Resource not found error."""

    pass


class ConfigurationError(WhiteBoxAIError):
    """Configuration error."""

    pass


class IntegrationError(WhiteBoxAIError):
    """Framework integration error."""

    pass


class CacheError(WhiteBoxAIError):
    """Cache operation error."""

    pass
