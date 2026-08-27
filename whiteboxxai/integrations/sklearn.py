"""
Scikit-learn Integration

Integration for monitoring scikit-learn models.
"""

from typing import Any, Dict, Optional

try:
    import sklearn
    from sklearn.base import BaseEstimator

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = object

import numpy as np

from whiteboxxai.monitor import ModelMonitor


class SklearnMonitor(ModelMonitor):
    """
    Monitor for scikit-learn models.

    Example:
        ```python
        from sklearn.ensemble import RandomForestClassifier
        from whiteboxxai import WhiteBoxXAI
        from whiteboxxai.integrations.sklearn import SklearnMonitor

        # Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Setup monitoring
        client = WhiteBoxXAI(api_key="your-api-key")
        monitor = SklearnMonitor(client, model=model)
        monitor.register_from_model(model_type="classification")

        # Wrap model for automatic monitoring
        monitored_model = monitor.wrap_model(model)

        # Predictions are automatically logged
        predictions = monitored_model.predict(X_test)
        ```
    """

    def __init__(self, client, model: Optional[BaseEstimator] = None, **kwargs):
        """Initialize sklearn monitor."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install with: pip install scikit-learn"
            )

        super().__init__(client, **kwargs)
        self.model = model

    def register_from_model(
        self,
        name: Optional[str] = None,
        model_type: str = "classification",
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """
        Register model from sklearn estimator.

        Args:
            name: Model name (default: class name)
            model_type: Model type (classification, regression, clustering)
            version: Model version
            **kwargs: Additional metadata

        Returns:
            Model ID
        """
        if self.model is None:
            raise ValueError("No model provided")

        # Generate name from model class
        if name is None:
            name = self.model.__class__.__name__

        # Extract model metadata
        metadata = self._extract_model_metadata()
        metadata.update(kwargs)

        return self.register_model(
            name=name,
            model_type=model_type,
            framework="sklearn",
            version=version,
            **metadata,
        )

    async def aregister_from_model(
        self,
        name: Optional[str] = None,
        model_type: str = "classification",
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """Async version of register_from_model()."""
        if self.model is None:
            raise ValueError("No model provided")

        if name is None:
            name = self.model.__class__.__name__

        metadata = self._extract_model_metadata()
        metadata.update(kwargs)

        return await self.aregister_model(
            name=name,
            model_type=model_type,
            framework="sklearn",
            version=version,
            **metadata,
        )

    def wrap_model(self, model: BaseEstimator) -> "SklearnWrapper":
        """
        Wrap sklearn model for automatic monitoring.

        Args:
            model: Sklearn estimator

        Returns:
            Wrapped model
        """
        return SklearnWrapper(model, self)

    def _extract_model_metadata(self) -> Dict[str, Any]:
        """Extract metadata from sklearn model."""
        metadata = {}

        if self.model is None:
            return metadata

        # Get model parameters
        if hasattr(self.model, "get_params"):
            metadata["parameters"] = self.model.get_params()

        # Get feature names
        if hasattr(self.model, "feature_names_in_"):
            metadata["feature_names"] = self.model.feature_names_in_.tolist()

        # Get feature importances (for tree-based models)
        if hasattr(self.model, "feature_importances_"):
            metadata["feature_importances"] = self.model.feature_importances_.tolist()

        # Get classes (for classifiers)
        if hasattr(self.model, "classes_"):
            metadata["classes"] = self.model.classes_.tolist()

        # Get number of features
        if hasattr(self.model, "n_features_in_"):
            metadata["n_features"] = self.model.n_features_in_

        # Get sklearn version
        metadata["sklearn_version"] = sklearn.__version__

        return metadata


class SklearnWrapper:
    """
    Wrapper for sklearn models with automatic monitoring.

    This wrapper intercepts predict/predict_proba calls and logs predictions
    to WhiteBoxXAI automatically.
    """

    def __init__(self, model: BaseEstimator, monitor: SklearnMonitor):
        """Initialize wrapper."""
        self.model = model
        self.monitor = monitor

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict with automatic logging."""
        predictions = self.model.predict(X, **kwargs)

        # Log predictions
        self._log_batch_predictions(X, predictions)

        return predictions

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict probabilities with automatic logging."""
        probas = self.model.predict_proba(X, **kwargs)

        # Log predictions
        self._log_batch_predictions(X, probas)

        return probas

    def _log_batch_predictions(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        """Log batch of predictions."""
        # Convert to list format
        predictions = []
        for i in range(len(inputs)):
            pred = {
                "inputs": (inputs[i].tolist() if isinstance(inputs[i], np.ndarray) else inputs[i]),
                "output": (
                    outputs[i].tolist() if isinstance(outputs[i], np.ndarray) else outputs[i]
                ),
            }
            predictions.append(pred)

        # Log batch
        try:
            self.monitor.log_batch(predictions)
        except Exception as e:
            # Don't fail prediction if logging fails
            print(f"Warning: Failed to log predictions: {e}")

    def __getattr__(self, name: str):
        """Delegate other methods to wrapped model."""
        return getattr(self.model, name)
