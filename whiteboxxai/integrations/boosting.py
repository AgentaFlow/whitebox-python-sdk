"""
XGBoost and LightGBM integration for WhiteBoxXAI.

This module provides monitoring capabilities for gradient boosting models
from XGBoost and LightGBM frameworks. It supports both frameworks through
a unified interface with automatic feature importance tracking, prediction
logging, and performance monitoring.

Example:
    Basic XGBoost monitoring:

    >>> import xgboost as xgb
    >>> from whiteboxxai import WhiteBoxXAI
    >>> from whiteboxxai.integrations.boosting import XGBoostMonitor
    >>>
    >>> client = WhiteBoxXAI(api_key="your-api-key")
    >>> monitor = XGBoostMonitor(client=client, model_name="fraud_detector")
    >>>
    >>> model = xgb.XGBClassifier()
    >>> model.fit(X_train, y_train)
    >>>
    >>> monitor.register_from_model(model, X_train, y_train)
    >>> predictions = monitor.predict(model, X_test, y_test)

    LightGBM monitoring:

    >>> import lightgbm as lgb
    >>> from whiteboxxai.integrations.boosting import LightGBMMonitor
    >>>
    >>> monitor = LightGBMMonitor(client=client, model_name="churn_predictor")
    >>>
    >>> model = lgb.LGBMClassifier()
    >>> model.fit(X_train, y_train)
    >>>
    >>> monitor.register_from_model(model, X_train, y_train)
    >>> predictions = monitor.predict(model, X_test, y_test)
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from whiteboxxai.monitor import ModelMonitor

logger = logging.getLogger(__name__)

# Optional imports - graceful degradation
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None


def _build_prediction_batch(
    inputs: np.ndarray,
    predictions: np.ndarray,
    actuals: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Build per-sample dicts for ModelMonitor.log_batch() from parallel arrays."""

    def _item(value: Any) -> Any:
        return value.tolist() if isinstance(value, np.ndarray) else value

    batch = []
    for i in range(len(inputs)):
        item: Dict[str, Any] = {
            "inputs": _item(inputs[i]),
            "output": _item(predictions[i]),
        }
        if actuals is not None:
            item["actual"] = _item(actuals[i])
        if probabilities is not None:
            item["probability"] = _item(probabilities[i])
        batch.append(item)
    return batch


class XGBoostMonitor(ModelMonitor):
    """
    Monitor for XGBoost models.

    Provides comprehensive monitoring for XGBoost models including prediction
    logging, feature importance tracking, and automatic model registration.

    Attributes:
        client: WhiteBoxXAI client instance
        model_name: Name of the model being monitored
        track_feature_importance: Whether to track feature importance with predictions
        importance_type: Type of importance to track ('weight', 'gain', 'cover', 'total_gain', 'total_cover')

    Example:
        >>> monitor = XGBoostMonitor(
        ...     client=client,
        ...     model_name="xgb_classifier",
        ...     track_feature_importance=True,
        ...     importance_type="gain"
        ... )
        >>> monitor.register_from_model(model, X_train, y_train)
        >>> predictions = monitor.predict(model, X_test, y_test)
    """

    def __init__(
        self,
        client: Any,
        model_name: str = "xgboost_model",
        track_feature_importance: bool = True,
        importance_type: str = "gain",
        **kwargs,
    ):
        """
        Initialize XGBoost monitor.

        Args:
            client: WhiteBoxXAI client instance
            model_name: Name of the model
            track_feature_importance: Whether to track feature importance
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
            **kwargs: Additional arguments passed to ModelMonitor
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        super().__init__(client=client, model_name=model_name, **kwargs)
        self.track_feature_importance = track_feature_importance
        self.importance_type = importance_type
        self._feature_names: Optional[List[str]] = None

    def register_from_model(
        self,
        model: Any,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        model_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register XGBoost model with WhiteBoxXAI.

        Automatically extracts model metadata including feature names,
        number of features, number of trees, and feature importance.

        Args:
            model: XGBoost model (Booster, XGBClassifier, XGBRegressor, etc.)
            X_train: Training features for baseline
            y_train: Training labels for baseline
            model_type: Model type override (auto-detected if None)
            metadata: Additional metadata

        Returns:
            Model ID from registration
        """
        # Extract model information
        model_metadata = metadata or {}

        # Get feature names
        if hasattr(model, "feature_names_in_"):
            self._feature_names = list(model.feature_names_in_)
            model_metadata["feature_names"] = self._feature_names
        elif hasattr(model, "feature_names"):
            self._feature_names = model.feature_names
            model_metadata["feature_names"] = self._feature_names
        elif X_train is not None and hasattr(X_train, "columns"):
            self._feature_names = list(X_train.columns)
            model_metadata["feature_names"] = self._feature_names

        # Get number of features
        if hasattr(model, "n_features_in_"):
            model_metadata["num_features"] = model.n_features_in_
        elif self._feature_names:
            model_metadata["num_features"] = len(self._feature_names)

        # Get number of trees/boosting rounds
        if hasattr(model, "n_estimators"):
            model_metadata["num_trees"] = model.n_estimators
        elif hasattr(model, "best_iteration"):
            model_metadata["num_trees"] = model.best_iteration

        # Get XGBoost parameters
        if hasattr(model, "get_params"):
            params = model.get_params()
            model_metadata["xgboost_params"] = {
                k: str(v) for k, v in params.items() if v is not None
            }

        # Get feature importance
        if self.track_feature_importance:
            try:
                importance = self._get_feature_importance(model)
                if importance:
                    model_metadata["feature_importance"] = importance
            except Exception as e:
                warnings.warn(f"Failed to extract feature importance: {e}", stacklevel=2)

        # Detect model type
        if model_type is None:
            if hasattr(model, "_estimator_type"):
                model_type = model._estimator_type
            elif isinstance(model, xgb.XGBClassifier):
                model_type = "classification"
            elif isinstance(model, xgb.XGBRegressor):
                model_type = "regression"
            elif isinstance(model, xgb.XGBRanker):
                model_type = "ranking"
            else:
                model_type = "classification"  # Default

        model_metadata["framework"] = "xgboost"
        model_metadata["xgboost_version"] = xgb.__version__

        # Register model
        model_id = self.register_model(
            name=self.model_name, model_type=model_type, metadata=model_metadata
        )

        # Set baseline if training data provided
        if X_train is not None and y_train is not None:
            self.set_baseline(X_train, y_train)

        return model_id

    def predict(
        self,
        model: Any,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        log_predictions: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Make predictions and log to WhiteBoxXAI.

        Args:
            model: XGBoost model
            X: Input features
            y_true: True labels (optional)
            log_predictions: Whether to log predictions
            metadata: Additional metadata

        Returns:
            Model predictions
        """
        # Make predictions
        predictions = model.predict(X)

        # Get prediction probabilities if classification
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X)
            except Exception as e:
                logger.debug(f"predict_proba failed, continuing without probabilities: {e}")

        # Log predictions
        if log_predictions:
            self.log_batch(
                _build_prediction_batch(
                    inputs=X,
                    predictions=predictions,
                    actuals=y_true,
                    probabilities=probabilities,
                )
            )

        return predictions

    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from XGBoost model.

        Args:
            model: XGBoost model

        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            # Try sklearn-style feature_importances_
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                if self._feature_names and len(importances) == len(self._feature_names):
                    return dict(zip(self._feature_names, importances.tolist()))
                else:
                    return {f"f{i}": float(v) for i, v in enumerate(importances)}

            # Try get_score method (native XGBoost)
            if hasattr(model, "get_score"):
                importance_dict = model.get_score(importance_type=self.importance_type)
                return {k: float(v) for k, v in importance_dict.items()}

            # Try get_booster for sklearn API
            if hasattr(model, "get_booster"):
                booster = model.get_booster()
                importance_dict = booster.get_score(importance_type=self.importance_type)
                return {k: float(v) for k, v in importance_dict.items()}

            return None
        except Exception:
            return None


class LightGBMMonitor(ModelMonitor):
    """
    Monitor for LightGBM models.

    Provides comprehensive monitoring for LightGBM models including prediction
    logging, feature importance tracking, and automatic model registration.

    Attributes:
        client: WhiteBoxXAI client instance
        model_name: Name of the model being monitored
        track_feature_importance: Whether to track feature importance with predictions
        importance_type: Type of importance to track ('split' or 'gain')

    Example:
        >>> monitor = LightGBMMonitor(
        ...     client=client,
        ...     model_name="lgbm_classifier",
        ...     track_feature_importance=True,
        ...     importance_type="gain"
        ... )
        >>> monitor.register_from_model(model, X_train, y_train)
        >>> predictions = monitor.predict(model, X_test, y_test)
    """

    def __init__(
        self,
        client: Any,
        model_name: str = "lightgbm_model",
        track_feature_importance: bool = True,
        importance_type: str = "gain",
        **kwargs,
    ):
        """
        Initialize LightGBM monitor.

        Args:
            client: WhiteBoxXAI client instance
            model_name: Name of the model
            track_feature_importance: Whether to track feature importance
            importance_type: Type of importance ('split' or 'gain')
            **kwargs: Additional arguments passed to ModelMonitor
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        super().__init__(client=client, model_name=model_name, **kwargs)
        self.track_feature_importance = track_feature_importance
        self.importance_type = importance_type
        self._feature_names: Optional[List[str]] = None

    def register_from_model(
        self,
        model: Any,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        model_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register LightGBM model with WhiteBoxXAI.

        Automatically extracts model metadata including feature names,
        number of features, number of trees, and feature importance.

        Args:
            model: LightGBM model (Booster, LGBMClassifier, LGBMRegressor, etc.)
            X_train: Training features for baseline
            y_train: Training labels for baseline
            model_type: Model type override (auto-detected if None)
            metadata: Additional metadata

        Returns:
            Model ID from registration
        """
        # Extract model information
        model_metadata = metadata or {}

        # Get feature names
        if hasattr(model, "feature_name_"):
            self._feature_names = model.feature_name_
            model_metadata["feature_names"] = self._feature_names
        elif hasattr(model, "feature_names_in_"):
            self._feature_names = list(model.feature_names_in_)
            model_metadata["feature_names"] = self._feature_names
        elif X_train is not None and hasattr(X_train, "columns"):
            self._feature_names = list(X_train.columns)
            model_metadata["feature_names"] = self._feature_names

        # Get number of features
        if hasattr(model, "n_features_in_"):
            model_metadata["num_features"] = model.n_features_in_
        elif self._feature_names:
            model_metadata["num_features"] = len(self._feature_names)

        # Get number of trees
        if hasattr(model, "n_estimators"):
            model_metadata["num_trees"] = model.n_estimators
        elif hasattr(model, "best_iteration_"):
            model_metadata["num_trees"] = model.best_iteration_
        elif hasattr(model, "num_trees"):
            model_metadata["num_trees"] = model.num_trees()

        # Get LightGBM parameters
        if hasattr(model, "get_params"):
            params = model.get_params()
            model_metadata["lightgbm_params"] = {
                k: str(v) for k, v in params.items() if v is not None
            }

        # Get feature importance
        if self.track_feature_importance:
            try:
                importance = self._get_feature_importance(model)
                if importance:
                    model_metadata["feature_importance"] = importance
            except Exception as e:
                warnings.warn(f"Failed to extract feature importance: {e}", stacklevel=2)

        # Detect model type
        if model_type is None:
            if hasattr(model, "_estimator_type"):
                model_type = model._estimator_type
            elif isinstance(model, lgb.LGBMClassifier):
                model_type = "classification"
            elif isinstance(model, lgb.LGBMRegressor):
                model_type = "regression"
            elif isinstance(model, lgb.LGBMRanker):
                model_type = "ranking"
            else:
                model_type = "classification"  # Default

        model_metadata["framework"] = "lightgbm"
        model_metadata["lightgbm_version"] = lgb.__version__

        # Register model
        model_id = self.register_model(
            name=self.model_name, model_type=model_type, metadata=model_metadata
        )

        # Set baseline if training data provided
        if X_train is not None and y_train is not None:
            self.set_baseline(X_train, y_train)

        return model_id

    def predict(
        self,
        model: Any,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        log_predictions: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Make predictions and log to WhiteBoxXAI.

        Args:
            model: LightGBM model
            X: Input features
            y_true: True labels (optional)
            log_predictions: Whether to log predictions
            metadata: Additional metadata

        Returns:
            Model predictions
        """
        # Make predictions
        predictions = model.predict(X)

        # Get prediction probabilities if classification
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X)
            except Exception as e:
                logger.debug(f"predict_proba failed, continuing without probabilities: {e}")

        # Log predictions
        if log_predictions:
            self.log_batch(
                _build_prediction_batch(
                    inputs=X,
                    predictions=predictions,
                    actuals=y_true,
                    probabilities=probabilities,
                )
            )

        return predictions

    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from LightGBM model.

        Args:
            model: LightGBM model

        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            # Try sklearn-style feature_importances_
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                if self._feature_names and len(importances) == len(self._feature_names):
                    return dict(zip(self._feature_names, importances.tolist()))
                else:
                    return {f"f{i}": float(v) for i, v in enumerate(importances)}

            # Try feature_importance method (native LightGBM)
            if hasattr(model, "feature_importance"):
                importances = model.feature_importance(importance_type=self.importance_type)
                if self._feature_names and len(importances) == len(self._feature_names):
                    return dict(zip(self._feature_names, importances.tolist()))
                else:
                    return {f"f{i}": float(v) for i, v in enumerate(importances)}

            # Try booster
            if hasattr(model, "booster_"):
                booster = model.booster_
                importances = booster.feature_importance(importance_type=self.importance_type)
                feature_names = booster.feature_name()
                if len(importances) == len(feature_names):
                    return dict(zip(feature_names, importances.tolist()))

            return None
        except Exception:
            return None


# Unified wrapper functions
def wrap_xgboost_model(model: Any, monitor: XGBoostMonitor, auto_register: bool = True) -> Any:
    """
    Wrap an XGBoost model for automatic monitoring.

    This creates a wrapper that automatically logs predictions when
    the model's predict method is called.

    Args:
        model: XGBoost model to wrap
        monitor: XGBoostMonitor instance
        auto_register: Whether to auto-register the model

    Returns:
        Wrapped model with monitoring

    Example:
        >>> monitor = XGBoostMonitor(client=client, model_name="my_model")
        >>> wrapped_model = wrap_xgboost_model(model, monitor)
        >>> predictions = wrapped_model.predict(X_test)  # Auto-logged
    """
    if auto_register and not monitor.model_id:
        monitor.register_from_model(model)

    # Store original methods
    original_predict = model.predict
    if hasattr(model, "predict_proba"):
        original_predict_proba = model.predict_proba
    else:
        original_predict_proba = None

    # Wrap predict method
    def wrapped_predict(X, *args, **kwargs):
        predictions = original_predict(X, *args, **kwargs)

        # Log predictions
        try:
            monitor.log_batch(_build_prediction_batch(inputs=X, predictions=predictions))
        except Exception as e:
            warnings.warn(f"Failed to log predictions: {e}", stacklevel=2)

        return predictions

    # Wrap predict_proba if available
    if original_predict_proba:

        def wrapped_predict_proba(X, *args, **kwargs):
            probabilities = original_predict_proba(X, *args, **kwargs)

            # Log with probabilities
            try:
                predictions = np.argmax(probabilities, axis=1)
                monitor.log_batch(
                    _build_prediction_batch(
                        inputs=X, predictions=predictions, probabilities=probabilities
                    )
                )
            except Exception as e:
                warnings.warn(f"Failed to log predictions: {e}", stacklevel=2)

            return probabilities

        model.predict_proba = wrapped_predict_proba

    model.predict = wrapped_predict

    return model


def wrap_lightgbm_model(model: Any, monitor: LightGBMMonitor, auto_register: bool = True) -> Any:
    """
    Wrap a LightGBM model for automatic monitoring.

    This creates a wrapper that automatically logs predictions when
    the model's predict method is called.

    Args:
        model: LightGBM model to wrap
        monitor: LightGBMMonitor instance
        auto_register: Whether to auto-register the model

    Returns:
        Wrapped model with monitoring

    Example:
        >>> monitor = LightGBMMonitor(client=client, model_name="my_model")
        >>> wrapped_model = wrap_lightgbm_model(model, monitor)
        >>> predictions = wrapped_model.predict(X_test)  # Auto-logged
    """
    if auto_register and not monitor.model_id:
        monitor.register_from_model(model)

    # Store original methods
    original_predict = model.predict
    if hasattr(model, "predict_proba"):
        original_predict_proba = model.predict_proba
    else:
        original_predict_proba = None

    # Wrap predict method
    def wrapped_predict(X, *args, **kwargs):
        predictions = original_predict(X, *args, **kwargs)

        # Log predictions
        try:
            monitor.log_batch(_build_prediction_batch(inputs=X, predictions=predictions))
        except Exception as e:
            warnings.warn(f"Failed to log predictions: {e}", stacklevel=2)

        return predictions

    # Wrap predict_proba if available
    if original_predict_proba:

        def wrapped_predict_proba(X, *args, **kwargs):
            probabilities = original_predict_proba(X, *args, **kwargs)

            # Log with probabilities
            try:
                predictions = np.argmax(probabilities, axis=1)
                monitor.log_batch(
                    _build_prediction_batch(
                        inputs=X, predictions=predictions, probabilities=probabilities
                    )
                )
            except Exception as e:
                warnings.warn(f"Failed to log predictions: {e}", stacklevel=2)

            return probabilities

        model.predict_proba = wrapped_predict_proba

    model.predict = wrapped_predict

    return model
