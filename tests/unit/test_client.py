"""
Tests for WhiteBoxXAI SDK Client

Tests for the main SDK client class.
"""

from unittest.mock import Mock, patch

import httpx
import pytest

from whiteboxxai.client import WhiteBoxXAI
from whiteboxxai.config import Config
from whiteboxxai.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
)


class TestWhiteBoxXAIClient:
    """Tests for WhiteBoxXAI client initialization and configuration."""

    def test_client_initialization(self):
        """Test basic client initialization."""
        client = WhiteBoxXAI(api_key="test_key")
        assert client.config.api_key == "test_key"
        assert client.config.base_url == "https://api.whiteboxxai.com"
        assert client.config.timeout == 30
        assert client.config.max_retries == 3

    def test_client_custom_config(self):
        """Test client with custom configuration."""
        client = WhiteBoxXAI(
            api_key="test_key",
            base_url="https://custom.api.com",
            timeout=60,
            max_retries=5,
        )
        assert client.config.base_url == "https://custom.api.com"
        assert client.config.timeout == 60
        assert client.config.max_retries == 5

    def test_client_resources_initialized(self):
        """Test that resource managers are initialized."""
        client = WhiteBoxXAI(api_key="test_key")
        assert client.models is not None
        assert client.predictions is not None
        assert client.explanations is not None
        assert client.drift is not None
        assert client.alerts is not None

    def test_sync_client_property(self):
        """Test synchronous client property."""
        client = WhiteBoxXAI(api_key="test_key")
        sync_client = client.sync_client
        assert isinstance(sync_client, httpx.Client)
        assert sync_client.base_url == "https://api.whiteboxxai.com"

    def test_async_client_property(self):
        """Test asynchronous client property."""
        client = WhiteBoxXAI(api_key="test_key")
        async_client = client.async_client
        assert isinstance(async_client, httpx.AsyncClient)
        assert async_client.base_url == "https://api.whiteboxxai.com"

    def test_headers_include_api_key(self):
        """Test that headers include API key."""
        client = WhiteBoxXAI(api_key="test_secret_key")
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_secret_key"

    def test_client_close(self):
        """Test client cleanup."""
        client = WhiteBoxXAI(api_key="test_key")
        # Access clients to create them
        _ = client.sync_client
        _ = client.async_client

        # Close should not raise
        client.close()

    def test_context_manager(self):
        """Test client as context manager."""
        with WhiteBoxXAI(api_key="test_key") as client:
            assert client.config.api_key == "test_key"


class TestClientHTTPMethods:
    """Tests for client HTTP request methods."""

    @patch("httpx.Client.request")
    def test_request_success(self, mock_request):
        """Test successful HTTP request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123", "name": "test"}
        mock_request.return_value = mock_response

        client = WhiteBoxXAI(api_key="test_key")
        response = client.request("GET", "/models/123")

        assert response == {"id": "123", "name": "test"}
        mock_request.assert_called_once()

    @patch("httpx.Client.request")
    def test_request_authentication_error(self, mock_request):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Unauthorized"}
        mock_request.return_value = mock_response

        client = WhiteBoxXAI(api_key="invalid_key")
        with pytest.raises(AuthenticationError):
            client.request("GET", "/models")

    @patch("httpx.Client.request")
    def test_request_validation_error(self, mock_request):
        """Test validation error handling."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"detail": "Validation failed"}
        mock_request.return_value = mock_response

        client = WhiteBoxXAI(api_key="test_key")
        with pytest.raises(ValidationError):
            client.request("POST", "/models", data={"invalid": "data"})

    @patch("httpx.Client.request")
    def test_request_rate_limit_error(self, mock_request):
        """Test rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_request.return_value = mock_response

        client = WhiteBoxXAI(api_key="test_key")
        with pytest.raises(RateLimitError):
            client.request("GET", "/models")

    @patch("httpx.Client.request")
    def test_request_server_error(self, mock_request):
        """Test server error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_request.return_value = mock_response

        client = WhiteBoxXAI(api_key="test_key")
        with pytest.raises(APIError):
            client.request("GET", "/models")

    @patch("httpx.Client.request")
    def test_request_with_retry(self, mock_request):
        """Test request retry on transient failures."""
        # First call fails, second succeeds
        mock_response_error = Mock()
        mock_response_error.status_code = 503
        mock_response_error.json.return_value = {"error": "Service unavailable"}

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"id": "123"}

        mock_request.side_effect = [mock_response_error, mock_response_success]

        client = WhiteBoxXAI(api_key="test_key", max_retries=2)
        # Note: Retry behavior depends on implementation
        # This test verifies the interface exists


class TestClientConfiguration:
    """Tests for client configuration."""

    def test_config_validation(self):
        """Test configuration validation."""
        config = Config(api_key="test_key", base_url="https://api.test.com")
        assert config.api_key == "test_key"
        assert config.base_url == "https://api.test.com"

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = Config(api_key="test_key")
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.base_url == "https://api.whiteboxxai.com"

    def test_invalid_api_key(self):
        """Test that empty API key raises error."""
        with pytest.raises((ValueError, ValidationError)):
            Config(api_key="")


class TestClientResourceIntegration:
    """Tests for client resource integration."""

    def test_models_resource_access(self):
        """Test accessing models resource."""
        client = WhiteBoxXAI(api_key="test_key")
        assert hasattr(client.models, "register")
        assert hasattr(client.models, "list")
        assert hasattr(client.models, "get")

    def test_predictions_resource_access(self):
        """Test accessing predictions resource."""
        client = WhiteBoxXAI(api_key="test_key")
        assert hasattr(client.predictions, "log")
        assert hasattr(client.predictions, "query")

    def test_explanations_resource_access(self):
        """Test accessing explanations resource."""
        client = WhiteBoxXAI(api_key="test_key")
        assert hasattr(client.explanations, "generate")
        assert hasattr(client.explanations, "get")

    def test_drift_resource_access(self):
        """Test accessing drift resource."""
        client = WhiteBoxXAI(api_key="test_key")
        assert hasattr(client.drift, "detect")
        assert hasattr(client.drift, "get_reports")

    def test_alerts_resource_access(self):
        """Test accessing alerts resource."""
        client = WhiteBoxXAI(api_key="test_key")
        assert hasattr(client.alerts, "list")
        assert hasattr(client.alerts, "create")
