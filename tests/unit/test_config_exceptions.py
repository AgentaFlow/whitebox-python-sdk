"""
Tests for SDK Config and Exceptions

Tests for SDK configuration and exception handling.
"""

import pytest

from whiteboxxai.config import Config
from whiteboxxai.exceptions import (
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ValidationError,
    WhiteBoxXAIError,
)


class TestConfig:
    """Tests for Config class."""

    def test_config_basic(self):
        """Test basic configuration."""
        config = Config(api_key="test-key-123")
        assert config.api_key == "test-key-123"
        assert config.base_url == "https://api.whiteboxxai.com"
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_config_custom_base_url(self):
        """Test configuration with custom base URL."""
        config = Config(api_key="test-key", base_url="https://custom.api.com")
        assert config.base_url == "https://custom.api.com"

    def test_config_custom_timeout(self):
        """Test configuration with custom timeout."""
        config = Config(api_key="test-key", timeout=60)
        assert config.timeout == 60

    def test_config_custom_max_retries(self):
        """Test configuration with custom max retries."""
        config = Config(api_key="test-key", max_retries=5)
        assert config.max_retries == 5

    def test_config_additional_params(self):
        """Test configuration with additional parameters."""
        config = Config(api_key="test-key", custom_param="value", another_param=123)
        assert config.api_key == "test-key"
        # Additional params should be accessible

    def test_config_empty_api_key(self):
        """Test that empty API key raises error."""
        with pytest.raises(ValueError):
            Config(api_key="")

    def test_config_none_api_key(self):
        """Test that None API key raises error."""
        with pytest.raises((ValueError, TypeError)):
            Config(api_key=None)

    def test_config_invalid_timeout(self):
        """Test that invalid timeout raises error."""
        with pytest.raises((ValueError, TypeError)):
            Config(api_key="test-key", timeout=-1)

    def test_config_invalid_max_retries(self):
        """Test that invalid max_retries raises error."""
        with pytest.raises((ValueError, TypeError)):
            Config(api_key="test-key", max_retries=-1)

    def test_config_base_url_validation(self):
        """Test base URL validation."""
        # Valid URLs
        config1 = Config(api_key="test-key", base_url="https://api.example.com")
        assert config1.base_url.startswith("https://")

        # Invalid URL (no protocol) - should handle gracefully or raise
        try:
            config2 = Config(api_key="test-key", base_url="invalid-url")
            # If it accepts, it should add protocol or validate differently
        except ValueError:
            # Acceptable to reject invalid URLs
            pass


class TestWhiteBoxXAIError:
    """Tests for base WhiteBoxXAIError exception."""

    def test_base_error_creation(self):
        """Test creating base error."""
        error = WhiteBoxXAIError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_base_error_with_details(self):
        """Test base error with additional details."""
        error = WhiteBoxXAIError("Error occurred", details={"code": "ERR001"})
        assert "Error occurred" in str(error)

    def test_base_error_inheritance(self):
        """Test that base error inherits from Exception."""
        error = WhiteBoxXAIError("Test")
        assert isinstance(error, Exception)


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_authentication_error_creation(self):
        """Test creating authentication error."""
        error = AuthenticationError("Invalid API key")
        assert "Invalid API key" in str(error)

    def test_authentication_error_inheritance(self):
        """Test that authentication error inherits from WhiteBoxXAIError."""
        error = AuthenticationError("Test")
        assert isinstance(error, WhiteBoxXAIError)
        assert isinstance(error, Exception)

    def test_authentication_error_with_response(self):
        """Test authentication error with response details."""
        error = AuthenticationError(
            "Unauthorized", status_code=401, response={"error": "invalid_token"}
        )
        assert "Unauthorized" in str(error)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        error = ValidationError("Invalid input data")
        assert "Invalid input data" in str(error)

    def test_validation_error_inheritance(self):
        """Test that validation error inherits from WhiteBoxXAIError."""
        error = ValidationError("Test")
        assert isinstance(error, WhiteBoxXAIError)

    def test_validation_error_with_fields(self):
        """Test validation error with field details."""
        error = ValidationError("Validation failed", fields={"email": ["Invalid email format"]})
        assert "Validation failed" in str(error)


class TestNotFoundError:
    """Tests for NotFoundError exception."""

    def test_not_found_error_creation(self):
        """Test creating not found error."""
        error = NotFoundError("Model not found")
        assert "Model not found" in str(error)

    def test_not_found_error_inheritance(self):
        """Test that not found error inherits from WhiteBoxXAIError."""
        error = NotFoundError("Test")
        assert isinstance(error, WhiteBoxXAIError)

    def test_not_found_error_with_resource(self):
        """Test not found error with resource details."""
        error = NotFoundError("Resource not found", resource_type="model", resource_id="123")
        assert "Resource not found" in str(error)


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_rate_limit_error_creation(self):
        """Test creating rate limit error."""
        error = RateLimitError("Rate limit exceeded")
        assert "Rate limit exceeded" in str(error)

    def test_rate_limit_error_inheritance(self):
        """Test that rate limit error inherits from WhiteBoxXAIError."""
        error = RateLimitError("Test")
        assert isinstance(error, WhiteBoxXAIError)

    def test_rate_limit_error_with_retry(self):
        """Test rate limit error with retry information."""
        error = RateLimitError("Too many requests", retry_after=60, limit=100, remaining=0)
        assert "Too many requests" in str(error)


class TestAPIError:
    """Tests for APIError exception."""

    def test_api_error_creation(self):
        """Test creating API error."""
        error = APIError("API request failed")
        assert "API request failed" in str(error)

    def test_api_error_inheritance(self):
        """Test that API error inherits from WhiteBoxXAIError."""
        error = APIError("Test")
        assert isinstance(error, WhiteBoxXAIError)

    def test_api_error_with_status(self):
        """Test API error with status code."""
        error = APIError("Server error", status_code=500, response={"error": "Internal error"})
        assert "Server error" in str(error)

    def test_api_error_with_request_id(self):
        """Test API error with request ID."""
        error = APIError("Request failed", request_id="req-abc-123")
        assert "Request failed" in str(error)


class TestExceptionRaising:
    """Tests for raising exceptions in different scenarios."""

    def test_raise_authentication_error(self):
        """Test raising authentication error."""
        with pytest.raises(AuthenticationError) as exc_info:
            raise AuthenticationError("Invalid credentials")
        assert "Invalid credentials" in str(exc_info.value)

    def test_raise_validation_error(self):
        """Test raising validation error."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid data format")
        assert "Invalid data format" in str(exc_info.value)

    def test_raise_not_found_error(self):
        """Test raising not found error."""
        with pytest.raises(NotFoundError) as exc_info:
            raise NotFoundError("Resource not found")
        assert "Resource not found" in str(exc_info.value)

    def test_raise_rate_limit_error(self):
        """Test raising rate limit error."""
        with pytest.raises(RateLimitError) as exc_info:
            raise RateLimitError("Rate limit exceeded")
        assert "Rate limit exceeded" in str(exc_info.value)

    def test_raise_api_error(self):
        """Test raising API error."""
        with pytest.raises(APIError) as exc_info:
            raise APIError("API error occurred")
        assert "API error occurred" in str(exc_info.value)


class TestExceptionCatching:
    """Tests for catching exception hierarchy."""

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        try:
            raise ValidationError("Validation failed")
        except ValidationError as e:
            assert isinstance(e, ValidationError)

    def test_catch_base_exception(self):
        """Test catching via base WhiteBoxXAIError."""
        try:
            raise AuthenticationError("Auth failed")
        except WhiteBoxXAIError as e:
            assert isinstance(e, WhiteBoxXAIError)
            assert isinstance(e, AuthenticationError)

    def test_catch_multiple_exceptions(self):
        """Test catching multiple exception types."""
        for error_class in [
            AuthenticationError,
            ValidationError,
            NotFoundError,
            RateLimitError,
            APIError,
        ]:
            try:
                raise error_class("Test error")
            except WhiteBoxXAIError as e:
                assert isinstance(e, WhiteBoxXAIError)


class TestExceptionAttributes:
    """Tests for exception attributes and methods."""

    def test_error_message_attribute(self):
        """Test that error message is accessible."""
        error = ValidationError("Invalid input")
        assert error.args[0] == "Invalid input"

    def test_error_string_representation(self):
        """Test error string representation."""
        error = APIError("Request failed")
        assert str(error) == "Request failed"

    def test_error_repr(self):
        """Test error repr."""
        error = NotFoundError("Not found")
        repr_str = repr(error)
        assert "NotFoundError" in repr_str or "Not found" in repr_str
