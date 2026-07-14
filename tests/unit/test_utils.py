"""
Tests for SDK Utils Module

Tests for SDK utility functions.
"""

import numpy as np
import pytest

from whiteboxxai.utils import (
    batch_iterator,
    compute_metrics,
    deserialize_numpy,
    format_features,
    serialize_numpy,
    validate_features,
    validate_model_type,
    validate_prediction,
)


class TestFeatureValidation:
    """Tests for feature validation functions."""

    def test_validate_features_list(self):
        """Test validating list of features."""
        features = [1.0, 2.0, 3.0]
        result = validate_features(features)
        assert result is True

    def test_validate_features_numpy(self):
        """Test validating numpy array features."""
        features = np.array([1.0, 2.0, 3.0])
        result = validate_features(features)
        assert result is True

    def test_validate_features_dict(self):
        """Test validating dictionary features."""
        features = {"feature1": 1.0, "feature2": 2.0}
        result = validate_features(features)
        assert result is True

    def test_validate_features_invalid(self):
        """Test validating invalid features."""
        with pytest.raises(ValueError):
            validate_features("invalid")

    def test_validate_features_empty(self):
        """Test validating empty features."""
        with pytest.raises(ValueError):
            validate_features([])


class TestFeatureFormatting:
    """Tests for feature formatting functions."""

    def test_format_features_list(self):
        """Test formatting list features."""
        features = [1, 2, 3]
        formatted = format_features(features)
        assert isinstance(formatted, list)
        assert formatted == [1, 2, 3]

    def test_format_features_numpy(self):
        """Test formatting numpy array features."""
        features = np.array([1.0, 2.0, 3.0])
        formatted = format_features(features)
        assert isinstance(formatted, list)
        assert formatted == [1.0, 2.0, 3.0]

    def test_format_features_dict(self):
        """Test formatting dictionary features."""
        features = {"a": 1, "b": 2}
        formatted = format_features(features)
        assert isinstance(formatted, dict)
        assert formatted == {"a": 1, "b": 2}

    def test_format_features_2d_array(self):
        """Test formatting 2D numpy array."""
        features = np.array([[1, 2], [3, 4]])
        formatted = format_features(features)
        assert isinstance(formatted, list)
        assert len(formatted) == 2


class TestModelTypeValidation:
    """Tests for model type validation."""

    def test_validate_model_type_classification(self):
        """Test validating classification model type."""
        result = validate_model_type("classification")
        assert result is True

    def test_validate_model_type_regression(self):
        """Test validating regression model type."""
        result = validate_model_type("regression")
        assert result is True

    def test_validate_model_type_invalid(self):
        """Test validating invalid model type."""
        with pytest.raises(ValueError):
            validate_model_type("invalid_type")

    def test_validate_model_type_case_insensitive(self):
        """Test that validation is case-insensitive."""
        result = validate_model_type("CLASSIFICATION")
        assert result is True


class TestPredictionValidation:
    """Tests for prediction validation."""

    def test_validate_prediction_number(self):
        """Test validating numeric prediction."""
        result = validate_prediction(42.5)
        assert result is True

    def test_validate_prediction_string(self):
        """Test validating string prediction (class label)."""
        result = validate_prediction("fraud")
        assert result is True

    def test_validate_prediction_list(self):
        """Test validating list prediction (probabilities)."""
        result = validate_prediction([0.2, 0.8])
        assert result is True

    def test_validate_prediction_numpy(self):
        """Test validating numpy array prediction."""
        result = validate_prediction(np.array([0.2, 0.8]))
        assert result is True

    def test_validate_prediction_none(self):
        """Test validating None prediction."""
        with pytest.raises(ValueError):
            validate_prediction(None)


class TestNumpySerialization:
    """Tests for numpy serialization functions."""

    def test_serialize_numpy_array(self):
        """Test serializing numpy array."""
        arr = np.array([1, 2, 3])
        serialized = serialize_numpy(arr)
        assert isinstance(serialized, list)
        assert serialized == [1, 2, 3]

    def test_serialize_2d_array(self):
        """Test serializing 2D numpy array."""
        arr = np.array([[1, 2], [3, 4]])
        serialized = serialize_numpy(arr)
        assert isinstance(serialized, list)
        assert len(serialized) == 2
        assert serialized[0] == [1, 2]

    def test_serialize_non_numpy(self):
        """Test serializing non-numpy object."""
        data = [1, 2, 3]
        serialized = serialize_numpy(data)
        assert serialized == [1, 2, 3]

    def test_deserialize_to_numpy(self):
        """Test deserializing to numpy array."""
        data = [1, 2, 3]
        arr = deserialize_numpy(data)
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([1, 2, 3]))

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization/deserialization."""
        original = np.array([1.0, 2.0, 3.0])
        serialized = serialize_numpy(original)
        deserialized = deserialize_numpy(serialized)
        assert np.array_equal(original, deserialized)


class TestMetricsComputation:
    """Tests for metrics computation."""

    def test_compute_metrics_classification(self):
        """Test computing classification metrics."""
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        metrics = compute_metrics(y_true, y_pred, task="classification")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_compute_metrics_regression(self):
        """Test computing regression metrics."""
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 2.1, 2.9, 4.2]
        metrics = compute_metrics(y_true, y_pred, task="regression")
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0

    def test_compute_metrics_invalid_task(self):
        """Test computing metrics with invalid task."""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        with pytest.raises(ValueError):
            compute_metrics(y_true, y_pred, task="invalid")

    def test_compute_metrics_mismatched_lengths(self):
        """Test computing metrics with mismatched array lengths."""
        y_true = [1, 2, 3]
        y_pred = [1, 2]
        with pytest.raises(ValueError):
            compute_metrics(y_true, y_pred, task="classification")


class TestBatchIterator:
    """Tests for batch iterator utility."""

    def test_batch_iterator_basic(self):
        """Test basic batch iteration."""
        data = list(range(10))
        batches = list(batch_iterator(data, batch_size=3))
        assert len(batches) == 4  # 3, 3, 3, 1
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]

    def test_batch_iterator_exact_size(self):
        """Test batch iteration with exact batch size."""
        data = list(range(9))
        batches = list(batch_iterator(data, batch_size=3))
        assert len(batches) == 3
        assert all(len(batch) == 3 for batch in batches)

    def test_batch_iterator_single_batch(self):
        """Test batch iteration with size larger than data."""
        data = [1, 2, 3]
        batches = list(batch_iterator(data, batch_size=10))
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_batch_iterator_numpy(self):
        """Test batch iteration with numpy array."""
        data = np.array(range(10))
        batches = list(batch_iterator(data, batch_size=4))
        assert len(batches) == 3
        assert len(batches[0]) == 4


class TestUtilsEdgeCases:
    """Tests for utils edge cases."""

    def test_format_features_with_nans(self):
        """Test formatting features with NaN values."""
        features = [1.0, np.nan, 3.0]
        formatted = format_features(features)
        assert len(formatted) == 3
        assert np.isnan(formatted[1])

    def test_serialize_numpy_with_inf(self):
        """Test serializing numpy with infinity."""
        arr = np.array([1.0, np.inf, -np.inf])
        serialized = serialize_numpy(arr)
        assert len(serialized) == 3

    def test_validate_features_with_none_values(self):
        """Test validating features with None values."""
        features = [1.0, None, 3.0]
        # Should handle gracefully or raise ValueError
        try:
            result = validate_features(features)
            # If it passes, it should still be True
            assert result is True
        except ValueError:
            # Acceptable to reject None values
            pass
