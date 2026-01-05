"""Test configuration and fixtures."""

import pytest
from whiteboxai import WhiteBoxAI


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def client(mock_api_key):
    """Create a WhiteBoxAI client for testing."""
    return WhiteBoxAI(
        api_key=mock_api_key,
        base_url="http://localhost:8000",
        timeout=10
    )


@pytest.fixture
def sample_prediction_data():
    """Provide sample prediction data for tests."""
    return {
        "inputs": {"feature1": 1.0, "feature2": 2.0},
        "output": {"prediction": 1, "probability": 0.85}
    }
