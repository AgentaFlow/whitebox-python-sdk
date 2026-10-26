"""
Unit tests for TensorFlow/Keras integration.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Mock TensorFlow if not available
try:
    import tensorflow as tf
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestKerasMonitor:
    """Test KerasMonitor class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock WhiteBoxXAI client."""
        client = Mock()
        client.models.register = Mock(return_value={"id": 123})
        return client

    @pytest.fixture
    def simple_model(self):
        """Create a simple Keras model for testing."""
        model = keras.Sequential(
            [
                keras.layers.Dense(64, activation="relu", input_shape=(10,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    @pytest.fixture
    def classification_model(self):
        """Create a simple classification model."""
        model = keras.Sequential(
            [
                keras.layers.Dense(64, activation="relu", input_shape=(10,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        return model

    def test_init(self, mock_client, simple_model):
        """Test KerasMonitor initialization."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(
            client=mock_client,
            model=simple_model,
            model_name="test_model",
            model_type="regression",
        )

        assert monitor.model == simple_model
        assert monitor._model_name == "test_model"
        assert monitor._model_type == "regression"
        assert not monitor._registered

    def test_init_without_tensorflow(self, mock_client):
        """Test initialization fails without TensorFlow."""
        with patch("whiteboxxai.integrations.tensorflow.TENSORFLOW_AVAILABLE", False):
            from whiteboxxai.integrations.tensorflow import KerasMonitor

            with pytest.raises(ImportError, match="TensorFlow is not installed"):
                KerasMonitor(client=mock_client)

    def test_register_from_model(self, mock_client, simple_model):
        """Test registering a Keras model."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")

        model_id = monitor.register_from_model(
            name="custom_name", model_type="regression", version="1.0.0"
        )

        assert model_id == 123
        assert monitor.model_id == 123
        assert monitor._registered
        mock_client.models.register.assert_called_once()

        # Check the payload: model_type is the API's enum, the task is metadata,
        # and the fields ModelCreate requires are present.
        call_args = mock_client.models.register.call_args
        assert call_args[1]["name"] == "custom_name"
        assert call_args[1]["model_type"] == "neural_network"
        assert call_args[1]["framework"] == "tensorflow"
        assert call_args[1]["version"] == "1.0.0"
        assert call_args[1]["features"] == [f"feature_{i}" for i in range(10)]
        assert call_args[1]["baseline_metrics"] == {}
        assert "metadata" in call_args[1]
        assert call_args[1]["metadata"]["framework"] == "tensorflow"
        assert call_args[1]["metadata"]["task"] == "regression"

    def test_register_from_model_auto_name(self, mock_client, simple_model):
        """Test auto-generating model name."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="auto_model")

        monitor.register_from_model(model_type="regression")

        call_args = mock_client.models.register.call_args
        assert call_args[1]["name"] == "auto_model"

    def test_predict_single(self, mock_client, simple_model):
        """Test making a single prediction."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")
        monitor.log_prediction = Mock()

        # Create input data
        X = np.random.randn(1, 10).astype(np.float32)

        # Make prediction
        predictions = monitor.predict(X, log=True)

        assert predictions.shape == (1, 1)
        assert monitor._registered  # Auto-registered
        monitor.log_prediction.assert_called_once()

    def test_predict_batch(self, mock_client, simple_model):
        """Test making batch predictions."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")
        monitor.log_batch = Mock()

        # Create batch input data
        X = np.random.randn(10, 10).astype(np.float32)

        # Make predictions
        predictions = monitor.predict(X, log=True)

        assert predictions.shape == (10, 1)
        monitor.log_batch.assert_called_once()

    def test_predict_with_actuals(self, mock_client, simple_model):
        """Test predictions with actual values."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")
        monitor.log_batch = Mock()

        X = np.random.randn(5, 10).astype(np.float32)
        y = np.random.randn(5, 1).astype(np.float32)

        predictions = monitor.predict(X, log=True, actuals=y)

        assert predictions.shape == (5, 1)
        batch = monitor.log_batch.call_args[0][0]
        assert all("actual" in item for item in batch)

    def test_predict_without_logging(self, mock_client, simple_model):
        """Test predictions without logging."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")
        monitor._registered = True  # Skip auto-registration
        monitor.log_batch = Mock()
        monitor.log_prediction = Mock()

        X = np.random.randn(5, 10).astype(np.float32)
        predictions = monitor.predict(X, log=False)

        assert predictions.shape == (5, 1)
        monitor.log_batch.assert_not_called()
        monitor.log_prediction.assert_not_called()

    def test_predict_with_tensor_input(self, mock_client, simple_model):
        """Test predictions with TensorFlow tensor input."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")

        # Create tensor input
        X = tf.random.normal((5, 10))

        predictions = monitor.predict(X, log=True)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (5, 1)

    def test_set_baseline(self, mock_client, simple_model):
        """Test setting baseline data."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")
        monitor._registered = True

        X_baseline = np.random.randn(100, 10).astype(np.float32)
        y_baseline = np.random.randn(100, 1).astype(np.float32)

        monitor.set_baseline(X_baseline, y_baseline)

        assert monitor._baseline_data is not None
        assert monitor._baseline_labels is not None

    def test_set_baseline_with_tensor(self, mock_client, simple_model):
        """Test setting baseline with tensor input."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")
        monitor._registered = True

        X_baseline = tf.random.normal((100, 10))
        y_baseline = tf.random.normal((100, 1))

        monitor.set_baseline(X_baseline, y_baseline)

        assert monitor._baseline_data is not None
        assert monitor._baseline_labels is not None

    def test_log_epoch(self, mock_client, simple_model):
        """Test logging epoch metrics."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")
        monitor._registered = True
        monitor.log_custom_metric = Mock()

        monitor.log_epoch(epoch=5, train_loss=0.5, val_loss=0.6, accuracy=0.85, val_accuracy=0.83)

        monitor.log_custom_metric.assert_called_once()
        call_args = monitor.log_custom_metric.call_args
        assert call_args[0][0] == "training_epoch"
        assert call_args[0][1]["epoch"] == 5
        assert call_args[0][1]["train_loss"] == 0.5

    def test_log_checkpoint(self, mock_client, simple_model):
        """Test logging model checkpoint."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=simple_model, model_name="test_model")
        monitor._registered = True
        monitor.log_custom_metric = Mock()

        monitor.log_checkpoint(
            epoch=10,
            checkpoint_path="checkpoints/model_10.h5",
            metrics={"accuracy": 0.9},
        )

        monitor.log_custom_metric.assert_called_once()

    def test_register_saved_model(self, mock_client):
        """Test registering a SavedModel."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=None, model_name="saved_model")

        monitor.register_saved_model(
            model_path="saved_models/my_model",
            metadata={"saved_by": "ci"},
            features=["age", "income"],
        )

        assert monitor._registered
        mock_client.models.register.assert_called_once()
        call_kwargs = mock_client.models.register.call_args[1]
        assert call_kwargs["name"] == "saved_model"
        assert call_kwargs["model_type"] == "neural_network"
        assert call_kwargs["features"] == ["age", "income"]
        assert call_kwargs["metadata"]["format"] == "SavedModel"

    def test_register_saved_model_requires_features_without_a_model(self, mock_client):
        """A SavedModel path alone has no feature names to send."""
        from whiteboxxai.integrations.tensorflow import KerasMonitor

        monitor = KerasMonitor(client=mock_client, model=None, model_name="saved_model")

        with pytest.raises(ValueError, match="features"):
            monitor.register_saved_model(model_path="saved_models/my_model")

        mock_client.models.register.assert_not_called()


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestWhiteBoxXAICallback:
    """Test WhiteBoxXAICallback class."""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock monitor."""
        monitor = Mock()
        monitor._registered = True
        monitor.log_epoch = Mock()
        monitor.log_custom_metric = Mock()
        return monitor

    def test_init(self, mock_monitor):
        """Test callback initialization."""
        from whiteboxxai.integrations.tensorflow import WhiteBoxXAICallback

        callback = WhiteBoxXAICallback(
            monitor=mock_monitor,
            log_frequency=5,
            log_weights=True,
            log_gradients=False,
            log_validation=True,
        )

        assert callback.monitor == mock_monitor
        assert callback.log_frequency == 5
        assert callback.log_weights is True
        assert callback.log_gradients is False

    def test_on_epoch_end(self, mock_monitor):
        """Test epoch end callback."""
        from whiteboxxai.integrations.tensorflow import WhiteBoxXAICallback

        callback = WhiteBoxXAICallback(monitor=mock_monitor, log_frequency=1)

        logs = {"loss": 0.5, "accuracy": 0.85, "val_loss": 0.6, "val_accuracy": 0.83}

        callback.on_epoch_end(epoch=5, logs=logs)

        mock_monitor.log_epoch.assert_called_once()
        call_args = mock_monitor.log_epoch.call_args
        assert call_args[1]["epoch"] == 5
        assert call_args[1]["train_loss"] == 0.5
        assert call_args[1]["val_loss"] == 0.6

    def test_on_epoch_end_frequency(self, mock_monitor):
        """Test epoch logging frequency."""
        from whiteboxxai.integrations.tensorflow import WhiteBoxXAICallback

        callback = WhiteBoxXAICallback(monitor=mock_monitor, log_frequency=5)

        # Should not log on epoch 3
        callback.on_epoch_end(epoch=3, logs={"loss": 0.5})
        mock_monitor.log_epoch.assert_not_called()

        # Should log on epoch 4 (every 5th: 4, 9, 14...)
        callback.on_epoch_end(epoch=4, logs={"loss": 0.5})
        mock_monitor.log_epoch.assert_called_once()

    def test_on_train_end(self, mock_monitor):
        """Test training completion callback."""
        from whiteboxxai.integrations.tensorflow import WhiteBoxXAICallback

        callback = WhiteBoxXAICallback(monitor=mock_monitor)

        logs = {"loss": 0.3, "accuracy": 0.92}
        callback.on_train_end(logs=logs)

        mock_monitor.log_custom_metric.assert_called_once()


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestWrapKerasModel:
    """Test wrap_keras_model function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        model = keras.Sequential([keras.layers.Dense(10, input_shape=(5,)), keras.layers.Dense(1)])
        return model

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock monitor."""
        monitor = Mock()
        monitor.log_batch = Mock()
        return monitor

    def test_wrap_model(self, simple_model, mock_monitor):
        """Test wrapping a Keras model."""
        from whiteboxxai.integrations.tensorflow import wrap_keras_model

        wrapped_model = wrap_keras_model(simple_model, mock_monitor)

        assert wrapped_model == simple_model
        assert wrapped_model.predict != keras.Model.predict

    def test_wrapped_model_logs_predictions(self, simple_model, mock_monitor):
        """Test wrapped model logs predictions."""
        from whiteboxxai.integrations.tensorflow import wrap_keras_model

        wrapped_model = wrap_keras_model(simple_model, mock_monitor)

        # Make predictions
        X = np.random.randn(10, 5).astype(np.float32)
        wrapped_model.predict(X, verbose=0)

        # Check logging was called
        mock_monitor.log_batch.assert_called_once()
        batch = mock_monitor.log_batch.call_args[0][0]
        assert len(batch) == 10
        assert "inputs" in batch[0]
        assert "output" in batch[0]


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
def test_integration_end_to_end():
    """Test complete integration workflow."""
    from whiteboxxai.integrations.tensorflow import KerasMonitor, WhiteBoxXAICallback

    # Create mock client
    mock_client = Mock()
    mock_client.models.register = Mock(return_value={"id": 123})

    # Create model
    model = keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu", input_shape=(10,)),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Create monitor
    monitor = KerasMonitor(
        client=mock_client,
        model=model,
        model_name="integration_test_model",
        model_type="classification",
    )

    # Register model
    model_id = monitor.register_from_model()
    assert model_id == 123

    # Generate data
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)
    X_test = np.random.randn(20, 10).astype(np.float32)

    # Set baseline
    monitor.set_baseline(X_train, y_train)
    assert monitor._baseline_data is not None

    # Train with callback
    callback = WhiteBoxXAICallback(monitor, log_frequency=1)
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0, callbacks=[callback])

    # Make predictions
    predictions = monitor.predict(X_test, log=True)
    assert predictions.shape == (20, 1)
    mock_client.predictions.log_batch.assert_called()
