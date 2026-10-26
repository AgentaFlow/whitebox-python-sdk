"""
Scikit-learn Integration

Integration for monitoring scikit-learn models.
"""

from __future__ import annotations

from typing import Any

try:
    import sklearn
    from sklearn.base import BaseEstimator

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = object

import numpy as np

from whiteboxxai.integrations._registration import positional_feature_names, registration_fields
from whiteboxxai.monitor import ModelMonitor


def _default_model_type(estimator: Any) -> str:
    """Map an estimator to the nearest API model type (``custom`` if none fits)."""
    module = type(estimator).__module__
    class_name = type(estimator).__name__
    if module.startswith(("sklearn.ensemble", "sklearn.tree")) and (
        "Forest" in class_name or "Tree" in class_name
    ):
        return "random_forest"
    if module.startswith("sklearn.linear_model"):
        return "linear_model"
    if module.startswith("sklearn.neural_network"):
        return "neural_network"
    return "custom"


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
        monitor.register_from_model(
            version="1.0.0", baseline_metrics={"accuracy": 0.94}
        )

        # Wrap model for automatic monitoring
        monitored_model = monitor.wrap_model(model)

        # Predictions are automatically logged
        predictions = monitored_model.predict(X_test)
        ```
    """

    def __init__(self, client, model: BaseEstimator | None = None, **kwargs):
        """Initialize sklearn monitor."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install with: pip install scikit-learn"
            )

        super().__init__(client, **kwargs)
        self.model = model

    def register_from_model(
        self,
        name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        features: list[str] | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> int:
        """
        Register model from sklearn estimator.

        Args:
            name: Model name (default: class name)
            model_type: One of the API's model types (``xgboost``,
                ``random_forest``, ``neural_network``, ``linear_model``, ``llm``,
                ``custom``). Default: inferred from the estimator's class. A task
                name such as ``"classification"`` is kept as ``task`` metadata.
            version: Model version (default ``1.0.0``)
            features: Feature names (default: the estimator's
                ``feature_names_in_``, else ``feature_0..feature_{n-1}`` from
                ``n_features_in_``)
            baseline_metrics: Baseline metrics, e.g. ``{"accuracy": 0.94}``
            **kwargs: Additional metadata

        Returns:
            Model ID

        Raises:
            ValueError: No model, an invalid ``model_type``, or no feature names
                could be determined.
        """
        name, fields, metadata = self._prepare_registration(
            name, model_type, version, features, baseline_metrics, kwargs
        )
        return self.register_model(name=name, framework="sklearn", **fields, **metadata)

    async def aregister_from_model(
        self,
        name: str | None = None,
        model_type: str | None = None,
        version: str | None = None,
        features: list[str] | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> int:
        """Async version of register_from_model()."""
        name, fields, metadata = self._prepare_registration(
            name, model_type, version, features, baseline_metrics, kwargs
        )
        return await self.aregister_model(name=name, framework="sklearn", **fields, **metadata)

    def _prepare_registration(
        self,
        name: str | None,
        model_type: str | None,
        version: str | None,
        features: list[str] | None,
        baseline_metrics: dict[str, Any] | None,
        extra_metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Resolve (name, API-required fields, metadata) for registration."""
        if self.model is None:
            raise ValueError("No model provided")

        if name is None:
            name = self.model.__class__.__name__

        metadata = self._extract_model_metadata()
        metadata.update(extra_metadata)

        inferred = metadata.get("feature_names")
        if not inferred and metadata.get("n_features"):
            inferred = positional_feature_names(metadata["n_features"])

        fields = registration_fields(
            model_type=model_type,
            default_model_type=_default_model_type(self.model),
            version=version,
            features=features,
            inferred_features=inferred,
            baseline_metrics=baseline_metrics,
            metadata=metadata,
        )
        return name, fields, metadata

    def wrap_model(self, model: BaseEstimator) -> "SklearnWrapper":
        """
        Wrap sklearn model for automatic monitoring.

        Args:
            model: Sklearn estimator

        Returns:
            Wrapped model
        """
        return SklearnWrapper(model, self)

    def _extract_model_metadata(self) -> dict[str, Any]:
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
