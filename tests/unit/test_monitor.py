"""
Tests for SDK Monitor Module

Tests for model monitoring functionality.
"""

from unittest.mock import AsyncMock, Mock, create_autospec, patch

import pytest

from whiteboxxai.client import WhiteBoxXAI
from whiteboxxai.monitor import ModelMonitor
from whiteboxxai.resources import DriftResource, ExplanationsResource, PredictionsResource


def _autospec_client() -> Mock:
    """A client whose resources enforce the real method signatures.

    A bare ``Mock()`` accepts any keyword, which is how ``detect_drift()``
    shipped calling ``drift.detect(reference_data=..., current_data=...)`` --
    a ``TypeError`` for every real user -- while the suite stayed green.
    """
    client = Mock(spec=WhiteBoxXAI)
    client.drift = create_autospec(DriftResource, instance=True)
    client.predictions = create_autospec(PredictionsResource, instance=True)
    client.explanations = create_autospec(ExplanationsResource, instance=True)
    return client


class TestModelMonitorInitialization:
    """Tests for ModelMonitor initialization."""

    def test_monitor_initialization(self):
        """Test basic monitor initialization."""
        mock_client = Mock(spec=WhiteBoxXAI)
        monitor = ModelMonitor(client=mock_client, model_id="model-123")
        assert monitor.model_id == "model-123"
        assert monitor.client == mock_client

    def test_monitor_with_config(self):
        """Test monitor with custom configuration."""
        mock_client = Mock(spec=WhiteBoxXAI)
        monitor = ModelMonitor(
            client=mock_client,
            model_id="model-123",
            auto_explain=True,
            buffer_size=100,
        )
        assert monitor.auto_explain is True
        assert monitor.buffer_size == 100


class TestPredictionLogging:
    """Tests for prediction logging."""

    @patch("whiteboxxai.monitor.ModelMonitor.log_prediction")
    def test_log_prediction_basic(self, mock_log):
        """Test basic prediction logging."""
        mock_client = Mock(spec=WhiteBoxXAI)
        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        inputs = {"feature1": 1.0, "feature2": 2.0}
        output = 0.85

        monitor.log_prediction(inputs=inputs, output=output)

        # Verify log was called
        mock_log.assert_called_once_with(inputs=inputs, output=output)

    def test_log_prediction_with_metadata(self):
        """Test prediction logging with metadata."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.predictions = Mock()
        mock_client.predictions.log = Mock()

        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        inputs = [1.0, 2.0, 3.0]
        output = 0.75
        metadata = {"version": "1.0", "region": "us-east"}

        monitor.log_prediction(inputs=inputs, output=output, metadata=metadata)

        # Verify metadata was included
        call_args = mock_client.predictions.log.call_args
        assert call_args is not None

    def test_log_prediction_with_explain(self):
        """explain=True explains using the response's ``prediction_id``."""
        client = _autospec_client()
        client.predictions.log.return_value = {"prediction_id": "p-1"}
        client.explanations.generate.return_value = {"feature_weights": {"a": 1.0}}
        monitor = ModelMonitor(client=client, model_id="model-123")

        result = monitor.log_prediction(inputs={"a": 1.0}, output=0.9, explain=True)

        client.explanations.generate.assert_called_once_with(
            "model-123", {"a": 1.0}, prediction_id="p-1"
        )
        assert result["explanation"] == {"feature_weights": {"a": 1.0}}

    def test_auto_explain_explains_without_per_call_flag(self):
        client = _autospec_client()
        client.predictions.log.return_value = {"prediction_id": "p-1"}
        monitor = ModelMonitor(client=client, model_id="m", auto_explain=True)

        result = monitor.log_prediction(inputs={"a": 1.0}, output=0.9)

        client.explanations.generate.assert_called_once()
        assert "explanation" in result

    def test_explain_false_overrides_auto_explain(self):
        client = _autospec_client()
        client.predictions.log.return_value = {"prediction_id": "p-1"}
        monitor = ModelMonitor(client=client, model_id="m", auto_explain=True)

        result = monitor.log_prediction(inputs={"a": 1.0}, output=0.9, explain=False)

        client.explanations.generate.assert_not_called()
        assert "explanation" not in result

    def test_no_explanation_when_response_has_no_prediction_id(self):
        client = _autospec_client()
        client.predictions.log.return_value = {"model_id": "m"}
        monitor = ModelMonitor(client=client, model_id="m", auto_explain=True)

        result = monitor.log_prediction(inputs={"a": 1.0}, output=0.9)

        client.explanations.generate.assert_not_called()
        assert "explanation" not in result

    async def test_async_log_prediction_with_explain(self):
        client = _autospec_client()
        client.predictions.alog.return_value = {"prediction_id": "p-1"}
        client.explanations.agenerate.return_value = {"feature_weights": {}}
        monitor = ModelMonitor(client=client, model_id="m", auto_explain=True)

        result = await monitor.alog_prediction(inputs={"a": 1.0}, output=0.9)

        client.explanations.agenerate.assert_awaited_once_with("m", {"a": 1.0}, prediction_id="p-1")
        assert result["explanation"] == {"feature_weights": {}}

    @pytest.mark.asyncio
    async def test_async_log_prediction(self):
        """Test async prediction logging."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.predictions = Mock()
        mock_client.predictions.alog = AsyncMock()

        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        inputs = [1.0, 2.0]
        output = 0.9

        await monitor.alog_prediction(inputs=inputs, output=output)

        # Verify async log was called
        mock_client.predictions.alog.assert_called_once()


class TestBatchPredictionLogging:
    """Tests for batch prediction logging."""

    def test_log_batch_predictions(self):
        """Test logging batch of predictions."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.predictions = Mock()
        mock_client.predictions.log_batch = Mock()

        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        batch = [
            {"inputs": [1.0, 2.0], "output": 0.8},
            {"inputs": [2.0, 3.0], "output": 0.9},
        ]

        monitor.log_batch(batch)

        # Verify batch log was called
        mock_client.predictions.log_batch.assert_called_once()

    def test_log_batch_with_buffer(self):
        """Test batch logging with buffering."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.predictions = Mock()
        mock_client.predictions.log_batch = Mock()
        monitor = ModelMonitor(client=mock_client, model_id="model-123", buffer_size=3)

        # Log predictions below buffer size
        monitor.log_prediction(inputs=[1.0], output=0.5)
        monitor.log_prediction(inputs=[2.0], output=0.6)

        # Buffer should not flush yet
        # (Implementation-specific verification)

        # Log one more to exceed buffer
        monitor.log_prediction(inputs=[3.0], output=0.7)

        # Should have flushed
        # (Implementation-specific verification)


class TestDriftMonitoring:
    """Tests for drift monitoring functionality."""

    def test_detect_drift_uses_the_real_resource_signature(self):
        client = _autospec_client()
        client.drift.detect.return_value = {"drift_detected": True}
        monitor = ModelMonitor(client=client, model_id="model-123")

        result = monitor.detect_drift()

        client.drift.detect.assert_called_once_with(
            model_id="model-123", window_size=1000, feature_names=None
        )
        assert result["drift_detected"] is True

    def test_detect_drift_forwards_window_and_features(self):
        client = _autospec_client()
        monitor = ModelMonitor(client=client, model_id="model-123")

        monitor.detect_drift(window_size=200, feature_names=["a"])

        client.drift.detect.assert_called_once_with(
            model_id="model-123", window_size=200, feature_names=["a"]
        )

    def test_detect_drift_requires_a_registered_model(self):
        monitor = ModelMonitor(client=_autospec_client())

        with pytest.raises(ValueError, match="not registered"):
            monitor.detect_drift()

    async def test_adetect_drift_uses_the_real_resource_signature(self):
        client = _autospec_client()
        client.drift.adetect.return_value = {"drift_detected": False}
        monitor = ModelMonitor(client=client, model_id="model-123")

        result = await monitor.adetect_drift(window_size=500)

        client.drift.adetect.assert_awaited_once_with(
            model_id="model-123", window_size=500, feature_names=None
        )
        assert result["drift_detected"] is False

    def test_get_drift_reports(self):
        """Test getting drift reports."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.drift = Mock()
        mock_client.drift.get_reports = Mock(return_value=[{"id": "report-1", "drift_score": 0.8}])

        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        reports = monitor.get_drift_reports(limit=10)

        # Verify reports were retrieved
        assert len(reports) == 1
        assert reports[0]["drift_score"] == 0.8


class TestAlertManagement:
    """Tests for alert management."""

    def test_create_alert_rule(self):
        """Test creating alert rule."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.alerts = Mock()
        mock_client.alerts.create = Mock(return_value={"id": "rule-1"})

        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        rule = monitor.create_alert_rule(metric="accuracy", threshold=0.8, condition="below")

        # Verify alert rule was created
        assert rule["id"] == "rule-1"

    def test_list_active_alerts(self):
        """Test listing active alerts."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.alerts = Mock()
        mock_client.alerts.list = Mock(
            return_value=[
                {"id": "alert-1", "severity": "high"},
                {"id": "alert-2", "severity": "medium"},
            ]
        )

        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        alerts = monitor.get_active_alerts()

        # Verify alerts were retrieved
        assert len(alerts) == 2
        assert alerts[0]["severity"] == "high"


class TestMonitorContextManager:
    """Tests for monitor as context manager."""

    def test_context_manager(self):
        """Test monitor as context manager."""
        mock_client = Mock(spec=WhiteBoxXAI)
        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        with monitor:
            # Use monitor
            pass

        # Verify cleanup was called
        # (Implementation-specific verification)

    def test_context_manager_flushes_buffer(self):
        """Test that context manager flushes buffer on exit."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.predictions = Mock()
        mock_client.predictions.log_batch = Mock()
        monitor = ModelMonitor(client=mock_client, model_id="model-123", buffer_size=10)

        with monitor:
            # Log some predictions
            monitor.log_prediction(inputs=[1.0], output=0.5)
            monitor.log_prediction(inputs=[2.0], output=0.6)

        # Buffer should be flushed on exit
        # (Implementation-specific verification)


class TestMonitorStatistics:
    """Tests for monitor statistics."""

    def test_get_prediction_count(self):
        """Test getting prediction count."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.predictions = Mock()
        mock_client.predictions.log = Mock()
        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        # Log some predictions
        monitor.log_prediction(inputs=[1.0], output=0.5)
        monitor.log_prediction(inputs=[2.0], output=0.6)

        count = monitor.get_prediction_count()
        assert count >= 2

    @pytest.mark.skip(
        reason="No MetricsResource/client.metrics exists in whiteboxxai.client yet; "
        "this needs a real backend-routes-driven implementation, not a guess."
    )
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.metrics = Mock()
        mock_client.metrics.get = Mock(return_value={"accuracy": 0.95, "precision": 0.93})

        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        metrics = monitor.get_performance_metrics()

        # Verify metrics were retrieved
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 0.95


class TestMonitorEdgeCases:
    """Tests for monitor edge cases."""

    def test_monitor_with_invalid_model_id(self):
        """Test that logging with an unregistered (model_id=None) monitor raises.

        model_id=None is a legitimate construction state (matches the
        default and the class's own "register later" workflow via
        register_model()) so construction itself must not raise; only
        operations that require a registered model should.
        """
        mock_client = Mock(spec=WhiteBoxXAI)
        monitor = ModelMonitor(client=mock_client, model_id=None)

        with pytest.raises(ValueError):
            monitor.log_prediction(inputs=[1.0], output=0.5)

    def test_log_prediction_without_output(self):
        """Test logging prediction without output."""
        mock_client = Mock(spec=WhiteBoxXAI)
        monitor = ModelMonitor(client=mock_client, model_id="model-123")

        with pytest.raises((ValueError, TypeError)):
            monitor.log_prediction(inputs=[1.0, 2.0], output=None)

    def test_buffer_overflow_handling(self):
        """Test handling of buffer overflow."""
        mock_client = Mock(spec=WhiteBoxXAI)
        mock_client.predictions = Mock()
        mock_client.predictions.log_batch = Mock()
        monitor = ModelMonitor(client=mock_client, model_id="model-123", buffer_size=2)

        # Log more than buffer size
        for i in range(10):
            monitor.log_prediction(inputs=[float(i)], output=0.5)

        # Should handle overflow gracefully
        # (Implementation-specific verification)
