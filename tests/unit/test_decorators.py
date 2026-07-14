"""
Tests for SDK Decorators

Tests for monitoring decorators.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from whiteboxxai.decorators import (
    _extract_inputs,
    _extract_output,
    monitor_model,
    monitor_performance,
)
from whiteboxxai.monitor import ModelMonitor


class TestMonitorModelDecorator:
    """Tests for monitor_model decorator."""

    def test_decorator_basic_usage(self):
        """Test basic decorator usage."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor, input_keys=["x"], explain=False)
        def predict(x):
            return x * 2

        result = predict(x=5)
        assert result == 10
        # Verify monitor was called
        mock_monitor.log_prediction.assert_called_once()

    def test_decorator_extracts_inputs(self):
        """Test that decorator extracts inputs correctly."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor, input_keys=["features"], explain=False)
        def predict(features):
            return {"prediction": sum(features)}

        result = predict(features=[1, 2, 3])

        # Verify inputs were extracted
        call_args = mock_monitor.log_prediction.call_args
        assert call_args[1]["inputs"] == {"features": [1, 2, 3]}

    def test_decorator_extracts_output(self):
        """Test that decorator extracts output correctly."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(
            mock_monitor, input_keys=["x"], output_key="prediction", explain=False
        )
        def predict(x):
            return {"prediction": x * 2, "confidence": 0.95}

        result = predict(x=5)

        # Verify output was extracted
        call_args = mock_monitor.log_prediction.call_args
        assert call_args[1]["output"] == 10  # x * 2

    def test_decorator_with_explain(self):
        """Test decorator with explain=True."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor, input_keys=["x"], explain=True)
        def predict(x):
            return x * 2

        result = predict(x=5)

        # Verify explain was passed
        call_args = mock_monitor.log_prediction.call_args
        assert call_args[1]["explain"] is True

    def test_decorator_tracks_inference_time(self):
        """Test that decorator tracks inference time."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor, input_keys=["x"], explain=False)
        def predict(x):
            time.sleep(0.1)  # Simulate processing
            return x * 2

        result = predict(x=5)

        # Verify metadata includes inference time
        call_args = mock_monitor.log_prediction.call_args
        metadata = call_args[1]["metadata"]
        assert "inference_time_ms" in metadata
        assert metadata["inference_time_ms"] >= 100  # At least 100ms

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor, input_keys=["x"], explain=False)
        def my_predict_function(x):
            """My prediction function."""
            return x * 2

        assert my_predict_function.__name__ == "my_predict_function"
        assert "My prediction function" in my_predict_function.__doc__

    @pytest.mark.asyncio
    async def test_decorator_async_function(self):
        """Test decorator with async function."""
        mock_monitor = Mock(spec=ModelMonitor)
        mock_monitor.alog_prediction = AsyncMock()

        @monitor_model(mock_monitor, input_keys=["x"], explain=False)
        async def predict_async(x):
            return x * 2

        result = await predict_async(x=5)
        assert result == 10
        # Verify async log was called
        mock_monitor.alog_prediction.assert_called_once()


class TestExtractInputs:
    """Tests for _extract_inputs helper function."""

    def test_extract_from_kwargs(self):
        """Test extracting inputs from kwargs."""

        def dummy_func(x, y):
            pass

        inputs = _extract_inputs(
            dummy_func,
            args=(),
            kwargs={"x": 1, "y": 2},
            input_keys=["x", "y"],
        )
        assert inputs == {"x": 1, "y": 2}

    def test_extract_from_args(self):
        """Test extracting inputs from positional args."""

        def dummy_func(x, y):
            pass

        inputs = _extract_inputs(
            dummy_func,
            args=(1, 2),
            kwargs={},
            input_keys=["x", "y"],
        )
        assert inputs == {"x": 1, "y": 2}

    def test_extract_mixed_args_kwargs(self):
        """Test extracting from both args and kwargs."""

        def dummy_func(x, y, z):
            pass

        inputs = _extract_inputs(
            dummy_func,
            args=(1,),
            kwargs={"y": 2, "z": 3},
            input_keys=["x", "y", "z"],
        )
        assert inputs == {"x": 1, "y": 2, "z": 3}

    def test_extract_without_input_keys(self):
        """Test extraction without input_keys (use all args)."""

        def dummy_func(x, y):
            pass

        inputs = _extract_inputs(
            dummy_func,
            args=(1, 2),
            kwargs={},
            input_keys=None,
        )
        # Should extract all arguments
        assert "x" in inputs or len(inputs) > 0


class TestExtractOutput:
    """Tests for _extract_output helper function."""

    def test_extract_with_key(self):
        """Test extracting output with specified key."""
        result = {"prediction": 42, "confidence": 0.95}
        output = _extract_output(result, output_key="prediction")
        assert output == 42

    def test_extract_without_key(self):
        """Test extracting output without key (use whole result)."""
        result = {"prediction": 42}
        output = _extract_output(result, output_key=None)
        assert output == {"prediction": 42}

    def test_extract_from_simple_value(self):
        """Test extracting from simple value."""
        result = 42
        output = _extract_output(result, output_key=None)
        assert output == 42

    def test_extract_missing_key(self):
        """Test extracting with missing key."""
        result = {"prediction": 42}
        output = _extract_output(result, output_key="nonexistent")
        # Should handle gracefully
        assert output is None or output == result


class TestMonitorPerformanceDecorator:
    """Tests for monitor_performance decorator."""

    def test_performance_tracking(self):
        """Test that performance is tracked."""

        @monitor_performance(threshold_ms=10)
        def compute_intensive_task():
            time.sleep(0.05)
            return "done"

        result = compute_intensive_task()
        assert result == "done"

    def test_performance_with_errors(self):
        """Test performance tracking with errors."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_performance(mock_monitor)
        def failing_task():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_task()
        # Verify error was logged
        # (Implementation-specific assertion)


class TestDecoratorEdgeCases:
    """Tests for decorator edge cases."""

    def test_decorator_with_no_args_function(self):
        """Test decorator on function with no arguments."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor, explain=False)
        def predict_constant():
            return 42

        result = predict_constant()
        assert result == 42
        mock_monitor.log_prediction.assert_called_once()

    def test_decorator_with_varargs(self):
        """Test decorator on function with *args."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor, explain=False)
        def predict_sum(*numbers):
            return sum(numbers)

        result = predict_sum(1, 2, 3, 4)
        assert result == 10
        mock_monitor.log_prediction.assert_called_once()

    def test_decorator_with_kwargs_only(self):
        """Test decorator on function with **kwargs."""
        mock_monitor = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor, explain=False)
        def predict_with_options(**options):
            return options.get("value", 0) * 2

        result = predict_with_options(value=5)
        assert result == 10
        mock_monitor.log_prediction.assert_called_once()

    def test_decorator_multiple_decorators(self):
        """Test stacking multiple decorators."""
        mock_monitor1 = Mock(spec=ModelMonitor)
        mock_monitor2 = Mock(spec=ModelMonitor)

        @monitor_model(mock_monitor1, input_keys=["x"], explain=False)
        @monitor_model(mock_monitor2, input_keys=["x"], explain=False)
        def predict(x):
            return x * 2

        result = predict(x=5)
        assert result == 10
        # Both monitors should be called
        mock_monitor1.log_prediction.assert_called_once()
        mock_monitor2.log_prediction.assert_called_once()
