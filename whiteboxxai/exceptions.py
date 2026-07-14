"""
SDK Exceptions
"""

from typing import Any, Dict, Optional


class WhiteBoxXAIError(Exception):
    """Base exception for WhiteBoxXAI SDK."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class APIError(WhiteBoxXAIError):
    """API request error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response = response
        self.request_id = request_id


class AuthenticationError(WhiteBoxXAIError):
    """Authentication error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response = response


class RateLimitError(WhiteBoxXAIError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


class ValidationError(WhiteBoxXAIError):
    """Validation error."""

    def __init__(
        self,
        message: str,
        fields: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.fields = fields or {}


class NotFoundError(WhiteBoxXAIError):
    """Resource not found error."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConfigurationError(WhiteBoxXAIError):
    """Configuration error."""

    pass


class IntegrationError(WhiteBoxXAIError):
    """Framework integration error."""

    pass


class CacheError(WhiteBoxXAIError):
    """Cache operation error."""

    pass
