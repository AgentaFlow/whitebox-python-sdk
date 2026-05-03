"""Unit tests for ModelMonitor."""

import pytest

from whiteboxai import ModelMonitor


class TestModelMonitor:
    """Test cases for ModelMonitor."""

    def test_monitor_initialization(self, client):
        """Test monitor can be initialized with client."""
        monitor = ModelMonitor(client)
        assert monitor is not None

    def test_monitor_with_sampling(self, client):
        """Test monitor with sampling rate."""
        monitor = ModelMonitor(client, sampling_rate=0.5)
        assert monitor is not None

    @pytest.mark.asyncio
    async def test_async_operations(self, client):
        """Test async monitor operations."""
        monitor = ModelMonitor(client)
        # Add async tests when implementing
        assert monitor is not None
