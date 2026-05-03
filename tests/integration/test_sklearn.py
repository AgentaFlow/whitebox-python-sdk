"""Integration tests for sklearn integration."""

import pytest

# Skip entire module if sklearn not available
pytest.importorskip("sklearn")

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from whiteboxai.integrations.sklearn import SklearnMonitor  # noqa: E402


class TestSklearnIntegration:
    """Test cases for sklearn integration."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y

    @pytest.fixture
    def trained_model(self, sample_data):
        """Train a sample model."""
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        return model

    def test_sklearn_monitor_creation(self, client, trained_model):
        """Test SklearnMonitor can be created."""
        monitor = SklearnMonitor(client=client, model=trained_model, model_name="test_model")
        assert monitor is not None

    def test_model_wrapping(self, client, trained_model, sample_data):
        """Test model wrapping for automatic monitoring."""
        X, _ = sample_data
        monitor = SklearnMonitor(client=client, model=trained_model)
        wrapped_model = monitor.wrap_model(trained_model)

        # Should be able to make predictions
        predictions = wrapped_model.predict(X[:10])
        assert predictions is not None
        assert len(predictions) == 10
