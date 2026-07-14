"""
TensorFlow/Keras Integration

Integration for monitoring TensorFlow and Keras models.
"""

import warnings
from typing import Any, Dict, Optional, Union

try:
    import tensorflow as tf
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None

import numpy as np

from whiteboxai.monitor import ModelMonitor


class KerasMonitor(ModelMonitor):
    """
    Monitor for TensorFlow/Keras models.

    Example:
        ```python
        from tensorflow import keras
        from whiteboxai import WhiteBoxAI
        from whiteboxai.integrations.tensorflow import KerasMonitor

        # Build model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Setup monitoring
        client = WhiteBoxAI(api_key="your-api-key")
        monitor = KerasMonitor(client, model=model, model_name="keras_model")

        # Train with monitoring callback
        from whiteboxai.integrations.tensorflow import WhiteBoxAICallback

        model.fit(X_train, y_train,
                  callbacks=[WhiteBoxAICallback(monitor)])

        # Make predictions with automatic logging
        predictions = monitor.predict(X_test)
        ```
    """

    def __init__(
        self,
        client,
        model: Optional["keras.Model"] = None,
        model_name: Optional[str] = None,
        model_type: str = "regression",
        **kwargs,
    ):
        """
        Initialize Keras monitor.

        Args:
            client: WhiteBoxAI client instance
            model: Keras model to monitor
            model_name: Name for the model
            model_type: Type of model (classification, regression)
            **kwargs: Additional arguments for ModelMonitor
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")

        super().__init__(client, **kwargs)
        self.model = model
        self._model_name = model_name
        self._model_type = model_type
        self._registered = False

    def register_from_model(
        self,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        version: Optional[str] = None,
        framework_version: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """
        Register Keras model with WhiteBoxAI.

        Args:
            name: Model name (default: model class name)
            model_type: Model type (classification, regression)
            version: Model version
            framework_version: TensorFlow version
            **kwargs: Additional metadata

        Returns:
            Model ID
        """
        if self.model is None:
            raise ValueError("No model provided")

        # Auto-detect model name
        if name is None:
            name = self._model_name or self.model.__class__.__name__

        # Use provided or instance model type
        model_type = model_type or self._model_type

        # Get TensorFlow version
        if framework_version is None:
            framework_version = tf.__version__

        # Extract model metadata
        metadata = {
            "framework": "tensorflow",
            "framework_version": framework_version,
            "model_class": self.model.__class__.__name__,
            **kwargs,
        }

        # Get model configuration
        try:
            config = self.model.get_config()
            metadata["model_config"] = str(config)
        except Exception:
            pass

        # Get input/output shapes
        try:
            if hasattr(self.model, "input_shape"):
                metadata["input_shape"] = str(self.model.input_shape)
            if hasattr(self.model, "output_shape"):
                metadata["output_shape"] = str(self.model.output_shape)
        except Exception:
            pass

        # Register model
        self.model_id = self.client.register_model(
            name=name,
            model_type=model_type,
            framework="tensorflow",
            version=version,
            metadata=metadata,
        )

        self._registered = True
        return self.model_id

    def predict(
        self,
        inputs: Union[np.ndarray, tf.Tensor],
        log: bool = True,
        actuals: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Make predictions and optionally log them.

        Args:
            inputs: Input data (numpy array or TensorFlow tensor)
            log: Whether to log the prediction
            actuals: Actual values (if available)
            **kwargs: Additional metadata to log

        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("No model provided")

        # Ensure model is registered
        if not self._registered:
            self.register_from_model()

        # Convert to numpy if needed
        if isinstance(inputs, tf.Tensor):
            inputs_np = inputs.numpy()
        else:
            inputs_np = inputs

        # Make prediction
        predictions = self.model.predict(inputs, verbose=0)

        # Convert predictions to numpy
        if isinstance(predictions, tf.Tensor):
            predictions_np = predictions.numpy()
        else:
            predictions_np = predictions

        # Log if requested
        if log:
            # Determine if batch or single
            if len(inputs_np.shape) == 1 or inputs_np.shape[0] == 1:
                # Single prediction
                self.log_prediction(
                    inputs=inputs_np[0] if len(inputs_np.shape) > 1 else inputs_np,
                    prediction=(
                        predictions_np[0] if len(predictions_np.shape) > 1 else predictions_np
                    ),
                    actual=actuals[0] if actuals is not None else None,
                    **kwargs,
                )
            else:
                # Batch prediction
                self.log_batch(
                    inputs=inputs_np,
                    predictions=predictions_np,
                    actuals=actuals,
                    **kwargs,
                )

        return predictions_np

    def set_baseline(
        self,
        baseline_data: Union[np.ndarray, tf.Tensor],
        baseline_labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
    ) -> None:
        """
        Set baseline data for drift detection.

        Args:
            baseline_data: Baseline input data
            baseline_labels: Baseline labels (optional)
        """
        # Convert to numpy
        if isinstance(baseline_data, tf.Tensor):
            baseline_data = baseline_data.numpy()
        if baseline_labels is not None and isinstance(baseline_labels, tf.Tensor):
            baseline_labels = baseline_labels.numpy()

        # Make baseline predictions if labels not provided
        if baseline_labels is None and self.model is not None:
            baseline_predictions = self.model.predict(baseline_data, verbose=0)
            if isinstance(baseline_predictions, tf.Tensor):
                baseline_predictions = baseline_predictions.numpy()
        else:
            baseline_predictions = baseline_labels

        # Set baseline through parent class
        super().set_baseline(baseline_data, baseline_predictions)

    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        **metrics: Any,
    ) -> None:
        """
        Log metrics from a training epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            **metrics: Additional metrics (accuracy, etc.)
        """
        metric_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }

        self.log_custom_metric("training_epoch", metric_data)

    def log_checkpoint(
        self,
        epoch: int,
        checkpoint_path: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a model checkpoint.

        Args:
            epoch: Epoch number
            checkpoint_path: Path to saved checkpoint
            metrics: Metrics at checkpoint
        """
        checkpoint_data = {
            "epoch": epoch,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics or {},
        }

        self.log_custom_metric("checkpoint", checkpoint_data)

    def register_saved_model(
        self,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a SavedModel with WhiteBoxAI.

        Args:
            model_path: Path to SavedModel directory
            metadata: Additional metadata
        """
        model_metadata = {
            "model_path": model_path,
            "format": "SavedModel",
            **(metadata or {}),
        }

        if not self._registered:
            self.register_from_model(**model_metadata)


class WhiteBoxAICallback(keras.callbacks.Callback):
    """
    Keras callback for automatic monitoring during training.

    Example:
        ```python
        monitor = KerasMonitor(client, model=model, model_name="my_model")
        callback = WhiteBoxAICallback(monitor, log_frequency=1)

        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  callbacks=[callback],
                  epochs=50)
        ```
    """

    def __init__(
        self,
        monitor: KerasMonitor,
        log_frequency: int = 1,
        log_weights: bool = False,
        log_gradients: bool = False,
        log_validation: bool = True,
    ):
        """
        Initialize callback.

        Args:
            monitor: KerasMonitor instance
            log_frequency: Log metrics every N epochs
            log_weights: Whether to log model weights
            log_gradients: Whether to log gradients
            log_validation: Whether to log validation predictions
        """
        super().__init__()
        self.monitor = monitor
        self.log_frequency = log_frequency
        self.log_weights = log_weights
        self.log_gradients = log_gradients
        self.log_validation = log_validation

        # Ensure model is registered
        if not self.monitor._registered:
            self.monitor.register_from_model()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        if logs is None:
            logs = {}

        # Log metrics at specified frequency
        if (epoch + 1) % self.log_frequency == 0:
            # Extract train and val metrics
            train_loss = logs.get("loss")
            val_loss = logs.get("val_loss")

            # Extract additional metrics
            metrics = {}
            for key, value in logs.items():
                if key not in ["loss", "val_loss"]:
                    metrics[key] = float(value) if value is not None else None

            # Log epoch metrics
            self.monitor.log_epoch(
                epoch=epoch,
                train_loss=float(train_loss) if train_loss is not None else None,
                val_loss=float(val_loss) if val_loss is not None else None,
                **metrics,
            )

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        if logs is None:
            logs = {}

        # Log final metrics
        self.monitor.log_custom_metric(
            "training_complete",
            {
                "final_metrics": {k: float(v) if v is not None else None for k, v in logs.items()},
            },
        )


class TorchMonitor(ModelMonitor):
    """
    Monitor for PyTorch models (kept for backwards compatibility).

    Note: This is a placeholder. Use the PyTorch integration module instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TorchMonitor in tensorflow module is deprecated. "
            "Use whiteboxai.integrations.pytorch.TorchMonitor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def wrap_keras_model(
    model: "keras.Model",
    monitor: KerasMonitor,
) -> "keras.Model":
    """
    Wrap a Keras model to automatically log predictions.

    Args:
        model: Keras model to wrap
        monitor: KerasMonitor instance

    Returns:
        Wrapped model that logs predictions

    Example:
        ```python
        wrapped_model = wrap_keras_model(model, monitor)
        predictions = wrapped_model.predict(X_test)  # Automatically logged
        ```
    """
    original_predict = model.predict

    def logged_predict(x, *args, **kwargs):
        # Make prediction
        predictions = original_predict(x, *args, **kwargs)

        # Log to WhiteBoxAI
        try:
            if isinstance(x, tf.Tensor):
                x_np = x.numpy()
            else:
                x_np = x

            if isinstance(predictions, tf.Tensor):
                pred_np = predictions.numpy()
            else:
                pred_np = predictions

            monitor.log_batch(
                inputs=x_np,
                predictions=pred_np,
            )
        except Exception as e:
            warnings.warn(f"Failed to log predictions: {e}")

        return predictions

    # Replace predict method
    model.predict = logged_predict
    return model


__all__ = [
    "KerasMonitor",
    "WhiteBoxAICallback",
    "TorchMonitor",  # Deprecated
    "wrap_keras_model",
]
