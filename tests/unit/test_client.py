"""Unit tests for WhiteBoxAI client."""

import pytest
from whiteboxai import WhiteBoxAI
from whiteboxai.exceptions import AuthenticationError


class TestWhiteBoxAIClient:
    """Test cases for WhiteBoxAI client."""

    def test_client_initialization(self, mock_api_key):
        """Test client can be initialized with API key."""
        client = WhiteBoxAI(api_key=mock_api_key)
        assert client is not None

    def test_client_requires_api_key(self):
        """Test client raises error without API key."""
        with pytest.raises((ValueError, AuthenticationError)):
            WhiteBoxAI(api_key=None)

    def test_client_with_custom_base_url(self, mock_api_key):
        """Test client accepts custom base URL."""
        client = WhiteBoxAI(
            api_key=mock_api_key,
            base_url="https://custom.api.example.com"
        )
        assert client is not None

    def test_offline_mode_configuration(self, mock_api_key):
        """Test offline mode can be enabled."""
        client = WhiteBoxAI(
            api_key=mock_api_key,
            enable_offline=True,
            offline_dir="./test_offline"
        )
        assert client is not None
