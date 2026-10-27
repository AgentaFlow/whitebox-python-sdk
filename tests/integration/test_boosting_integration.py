"""
Unit tests for XGBoost and LightGBM integration.

Tests cover:
- XGBoostMonitor functionality
- LightGBMMonitor functionality
- Model registration and metadata extraction
- Prediction logging
- Feature importance tracking
- Model wrapping
- Error handling
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Test with mock frameworks if not available
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# The monitors raise ImportError when instantiated without xgboost/lightgbm
# installed, so skip the whole module when neither is available rather than
# failing outright.
if not (XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE):
    pytest.skip("xgboost/lightgbm are optional dependencies", allow_module_level=True)


class TestXGBoostMonitor(unittest.TestCase):
    """Test suite for XGBoostMonitor."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_client.predict = Mock()
        self.mock_client.models.register = Mock(return_value={"id": "model-123"})

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_init_with_xgboost(self):
        """Test XGBoostMonitor initialization when XGBoost is available."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        monitor = XGBoostMonitor(
            client=self.mock_client,
            model_name="test_model",
            track_feature_importance=True,
            importance_type="gain",
        )

        self.assertEqual(monitor.model_name, "test_model")
        self.assertTrue(monitor.track_feature_importance)
        self.assertEqual(monitor.importance_type, "gain")

    def test_init_without_xgboost(self):
        """Test XGBoostMonitor initialization fails without XGBoost."""
        from whiteboxxai.integrations import boosting

        # patch.object restores the real flag on exit, unlike mutating it
        # directly or reloading the module under a poisoned sys.modules
        # (which previously left XGBOOST_AVAILABLE=False for every other
        # test in the session).
        with patch.object(boosting, "XGBOOST_AVAILABLE", False):
            with self.assertRaises(ImportError):
                boosting.XGBoostMonitor(client=self.mock_client)

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_register_from_model_classifier(self):
        """Test registering XGBoost classifier."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        monitor = XGBoostMonitor(client=self.mock_client, model_name="xgb_clf")

        # Create mock model
        mock_model = Mock(spec=xgb.XGBClassifier)
        mock_model.n_estimators = 100
        mock_model.n_features_in_ = 10
        mock_model.feature_names_in_ = [f"f{i}" for i in range(10)]
        mock_model._estimator_type = "classifier"
        mock_model.get_params = Mock(return_value={"max_depth": 5})
        mock_model.feature_importances_ = np.random.random(10)

        # Mock training data
        X_train = np.random.random((100, 10))
        y_train = np.random.randint(0, 2, 100)

        # Register model
        model_id = monitor.register_from_model(model=mock_model, X_train=X_train, y_train=y_train)

        self.assertEqual(model_id, "model-123")
        self.mock_client.models.register.assert_called_once()

        # Check metadata
        call_args = self.mock_client.models.register.call_args
        metadata = call_args[1]["metadata"]
        self.assertEqual(metadata["framework"], "xgboost")
        self.assertEqual(metadata["num_features"], 10)
        self.assertEqual(metadata["num_trees"], 100)
        self.assertIn("feature_importance", metadata)

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_register_from_model_regressor(self):
        """Test registering XGBoost regressor."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        monitor = XGBoostMonitor(client=self.mock_client, model_name="xgb_reg")

        # Create mock model
        mock_model = Mock(spec=xgb.XGBRegressor)
        mock_model.n_estimators = 50
        mock_model.n_features_in_ = 5
        mock_model.feature_names_in_ = [f"f{i}" for i in range(5)]
        mock_model._estimator_type = "regressor"
        mock_model.get_params = Mock(return_value={"learning_rate": 0.1})
        mock_model.feature_importances_ = np.random.random(5)

        # Register without training data
        model_id = monitor.register_from_model(model=mock_model)

        self.assertEqual(model_id, "model-123")

        # Check metadata
        call_args = self.mock_client.models.register.call_args
        metadata = call_args[1]["metadata"]
        self.assertEqual(metadata["framework"], "xgboost")
        self.assertEqual(metadata["task"], "regression")

        # model_type is the API's enum, not the task
        self.assertEqual(call_args[1]["model_type"], "xgboost")
        self.assertEqual(call_args[1]["version"], "1.0.0")
        self.assertEqual(call_args[1]["features"], [f"f{i}" for i in range(5)])
        self.assertEqual(call_args[1]["baseline_metrics"], {})

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_predict_classifier(self):
        """Test prediction logging for classifier."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        monitor = XGBoostMonitor(client=self.mock_client, model_name="xgb_clf")
        monitor.model_id = "model-123"

        # Create mock model
        mock_model = Mock(spec=xgb.XGBClassifier)
        X_test = np.random.random((10, 5))
        y_test = np.random.randint(0, 2, 10)
        y_pred = np.random.randint(0, 2, 10)
        y_proba = np.random.random((10, 2))

        mock_model.predict = Mock(return_value=y_pred)
        mock_model.predict_proba = Mock(return_value=y_proba)
        mock_model.feature_importances_ = np.random.random(5)

        # Make predictions
        predictions = monitor.predict(
            model=mock_model, X=X_test, y_true=y_test, log_predictions=True
        )

        self.assertEqual(len(predictions), 10)
        self.mock_client.predictions.log_batch.assert_called_once()

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_predict_regressor(self):
        """Test prediction logging for regressor."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        monitor = XGBoostMonitor(client=self.mock_client, model_name="xgb_reg")
        monitor.model_id = "model-123"

        # Create mock model
        mock_model = Mock(spec=xgb.XGBRegressor)
        X_test = np.random.random((10, 5))
        y_pred = np.random.random(10)

        mock_model.predict = Mock(return_value=y_pred)
        mock_model.feature_importances_ = np.random.random(5)

        # Make predictions
        predictions = monitor.predict(model=mock_model, X=X_test, log_predictions=True)

        self.assertEqual(len(predictions), 10)
        self.mock_client.predictions.log_batch.assert_called_once()

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_feature_importance_extraction(self):
        """Test feature importance extraction."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        monitor = XGBoostMonitor(client=self.mock_client)

        # Mock model with feature_importances_
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        monitor._feature_names = ["f0", "f1", "f2"]

        importance = monitor._get_feature_importance(mock_model)

        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 3)
        self.assertIn("f0", importance)
        self.assertAlmostEqual(importance["f0"], 0.5)

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_feature_importance_different_types(self):
        """Test different feature importance types."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        for imp_type in ["weight", "gain", "cover", "total_gain", "total_cover"]:
            monitor = XGBoostMonitor(client=self.mock_client, importance_type=imp_type)

            self.assertEqual(monitor.importance_type, imp_type)


class TestLightGBMMonitor(unittest.TestCase):
    """Test suite for LightGBMMonitor."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_client.predict = Mock()
        self.mock_client.models.register = Mock(return_value={"id": "model-456"})

    @unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM not installed")
    def test_init_with_lightgbm(self):
        """Test LightGBMMonitor initialization when LightGBM is available."""
        from whiteboxxai.integrations.boosting import LightGBMMonitor

        monitor = LightGBMMonitor(
            client=self.mock_client,
            model_name="test_model",
            track_feature_importance=True,
            importance_type="gain",
        )

        self.assertEqual(monitor.model_name, "test_model")
        self.assertTrue(monitor.track_feature_importance)
        self.assertEqual(monitor.importance_type, "gain")

    def test_init_without_lightgbm(self):
        """Test LightGBMMonitor initialization fails without LightGBM."""
        from whiteboxxai.integrations import boosting

        # patch.object restores the real flag on exit, unlike mutating it
        # directly or reloading the module under a poisoned sys.modules
        # (which previously left LIGHTGBM_AVAILABLE=False for every other
        # test in the session).
        with patch.object(boosting, "LIGHTGBM_AVAILABLE", False):
            with self.assertRaises(ImportError):
                boosting.LightGBMMonitor(client=self.mock_client)

    @unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM not installed")
    def test_register_from_model_classifier(self):
        """Test registering LightGBM classifier."""
        from whiteboxxai.integrations.boosting import LightGBMMonitor

        monitor = LightGBMMonitor(client=self.mock_client, model_name="lgb_clf")

        # Create mock model
        mock_model = Mock(spec=lgb.LGBMClassifier)
        mock_model.n_estimators = 100
        mock_model.n_features_in_ = 10
        mock_model.feature_name_ = [f"f{i}" for i in range(10)]
        mock_model._estimator_type = "classifier"
        mock_model.get_params = Mock(return_value={"max_depth": 5})
        mock_model.feature_importances_ = np.random.random(10)

        # Mock training data
        X_train = np.random.random((100, 10))
        y_train = np.random.randint(0, 2, 100)

        # Register model
        model_id = monitor.register_from_model(model=mock_model, X_train=X_train, y_train=y_train)

        self.assertEqual(model_id, "model-456")
        self.mock_client.models.register.assert_called_once()

        # Check metadata
        call_args = self.mock_client.models.register.call_args
        metadata = call_args[1]["metadata"]
        self.assertEqual(metadata["framework"], "lightgbm")
        self.assertEqual(metadata["num_features"], 10)

    @unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM not installed")
    def test_register_from_model_regressor(self):
        """Test registering LightGBM regressor."""
        from whiteboxxai.integrations.boosting import LightGBMMonitor

        monitor = LightGBMMonitor(client=self.mock_client, model_name="lgb_reg")

        # Create mock model
        mock_model = Mock(spec=lgb.LGBMRegressor)
        mock_model.n_estimators = 50
        mock_model.n_features_in_ = 5
        mock_model.feature_name_ = [f"f{i}" for i in range(5)]
        mock_model._estimator_type = "regressor"
        mock_model.get_params = Mock(return_value={"learning_rate": 0.1})
        mock_model.feature_importances_ = np.random.random(5)

        # Register without training data
        model_id = monitor.register_from_model(model=mock_model)

        self.assertEqual(model_id, "model-456")

        # model_type is the API's enum (LightGBM has no member of its own, so
        # "custom"); the regression task is recorded as metadata.
        call_kwargs = self.mock_client.models.register.call_args[1]
        self.assertEqual(call_kwargs["model_type"], "custom")
        self.assertEqual(call_kwargs["metadata"]["task"], "regression")
        self.assertEqual(call_kwargs["version"], "1.0.0")
        self.assertEqual(call_kwargs["features"], [f"f{i}" for i in range(5)])

    @unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM not installed")
    def test_predict_classifier(self):
        """Test prediction logging for classifier."""
        from whiteboxxai.integrations.boosting import LightGBMMonitor

        monitor = LightGBMMonitor(client=self.mock_client, model_name="lgb_clf")
        monitor.model_id = "model-456"

        # Create mock model
        mock_model = Mock(spec=lgb.LGBMClassifier)
        X_test = np.random.random((10, 5))
        y_test = np.random.randint(0, 2, 10)
        y_pred = np.random.randint(0, 2, 10)
        y_proba = np.random.random((10, 2))

        mock_model.predict = Mock(return_value=y_pred)
        mock_model.predict_proba = Mock(return_value=y_proba)
        mock_model.feature_importances_ = np.random.random(5)

        # Make predictions
        predictions = monitor.predict(
            model=mock_model, X=X_test, y_true=y_test, log_predictions=True
        )

        self.assertEqual(len(predictions), 10)
        self.mock_client.predictions.log_batch.assert_called_once()

    @unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM not installed")
    def test_feature_importance_extraction(self):
        """Test feature importance extraction."""
        from whiteboxxai.integrations.boosting import LightGBMMonitor

        monitor = LightGBMMonitor(client=self.mock_client)

        # Mock model with feature_importances_
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        monitor._feature_names = ["f0", "f1", "f2"]

        importance = monitor._get_feature_importance(mock_model)

        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 3)
        self.assertIn("f0", importance)
        self.assertAlmostEqual(importance["f0"], 0.5)


class TestWrapXGBoostModel(unittest.TestCase):
    """Test suite for wrap_xgboost_model function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_client.predict = Mock()
        self.mock_client.models.register = Mock(return_value={"id": "model-123"})

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_wrap_model_basic(self):
        """Test basic model wrapping."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor, wrap_xgboost_model

        # Create monitor
        monitor = XGBoostMonitor(client=self.mock_client)
        monitor.model_id = "model-123"

        # Create mock model
        mock_model = Mock()
        y_pred = np.array([0, 1, 0])
        mock_model.predict = Mock(return_value=y_pred)

        # Wrap model
        wrapped_model = wrap_xgboost_model(model=mock_model, monitor=monitor, auto_register=False)

        # Test predictions
        X_test = np.random.random((3, 5))
        predictions = wrapped_model.predict(X_test)

        self.assertTrue(np.array_equal(predictions, y_pred))
        self.mock_client.predictions.log_batch.assert_called_once()

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_wrap_model_with_auto_register(self):
        """Test model wrapping with auto-registration."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor, wrap_xgboost_model

        # Create monitor without model_id
        monitor = XGBoostMonitor(client=self.mock_client)
        monitor.register_from_model = Mock(return_value="model-789")

        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0, 1]))

        # Wrap with auto-register
        wrap_xgboost_model(model=mock_model, monitor=monitor, auto_register=True)

        # Should have called register_from_model
        monitor.register_from_model.assert_called_once()


class TestWrapLightGBMModel(unittest.TestCase):
    """Test suite for wrap_lightgbm_model function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_client.predict = Mock()
        self.mock_client.models.register = Mock(return_value={"id": "model-456"})

    @unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM not installed")
    def test_wrap_model_basic(self):
        """Test basic model wrapping."""
        from whiteboxxai.integrations.boosting import LightGBMMonitor, wrap_lightgbm_model

        # Create monitor
        monitor = LightGBMMonitor(client=self.mock_client)
        monitor.model_id = "model-456"

        # Create mock model
        mock_model = Mock()
        y_pred = np.array([0, 1, 0])
        mock_model.predict = Mock(return_value=y_pred)

        # Wrap model
        wrapped_model = wrap_lightgbm_model(model=mock_model, monitor=monitor, auto_register=False)

        # Test predictions
        X_test = np.random.random((3, 5))
        predictions = wrapped_model.predict(X_test)

        self.assertTrue(np.array_equal(predictions, y_pred))
        self.mock_client.predictions.log_batch.assert_called_once()


class TestEndToEndBoosting(unittest.TestCase):
    """Integration tests for complete boosting workflow."""

    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_xgboost_end_to_end(self):
        """Test complete XGBoost workflow."""
        from whiteboxxai.integrations.boosting import XGBoostMonitor

        # Create mock client
        mock_client = Mock()
        mock_client.models.register = Mock(return_value={"id": "model-e2e-xgb"})
        mock_client.predict = Mock()

        # Create monitor
        monitor = XGBoostMonitor(client=mock_client, model_name="e2e_xgb_test")

        # Create mock model. spec'd to a fixed attribute list (rather than a
        # plain Mock()) so hasattr() checks for optional attributes the mock
        # doesn't set (feature_names_in_, predict_proba) correctly return
        # False instead of Mock always answering True.
        mock_model = Mock(
            spec=[
                "n_estimators",
                "n_features_in_",
                "feature_importances_",
                "get_params",
                "predict",
            ]
        )
        mock_model.n_estimators = 10
        mock_model.n_features_in_ = 5
        mock_model.feature_importances_ = np.random.random(5)
        mock_model.get_params = Mock(return_value={})
        mock_model.predict = Mock(return_value=np.array([0, 1, 0, 1]))

        # Register
        X_train = np.random.random((20, 5))
        y_train = np.random.randint(0, 2, 20)
        model_id = monitor.register_from_model(mock_model, X_train, y_train)

        self.assertEqual(model_id, "model-e2e-xgb")

        # Predict
        X_test = np.random.random((4, 5))
        y_test = np.random.randint(0, 2, 4)
        predictions = monitor.predict(mock_model, X_test, y_test)

        self.assertEqual(len(predictions), 4)
        mock_client.predictions.log_batch.assert_called_once()

    @unittest.skipUnless(LIGHTGBM_AVAILABLE, "LightGBM not installed")
    def test_lightgbm_end_to_end(self):
        """Test complete LightGBM workflow."""
        from whiteboxxai.integrations.boosting import LightGBMMonitor

        # Create mock client
        mock_client = Mock()
        mock_client.models.register = Mock(return_value={"id": "model-e2e-lgb"})
        mock_client.predict = Mock()

        # Create monitor
        monitor = LightGBMMonitor(client=mock_client, model_name="e2e_lgb_test")

        # Create mock model. spec'd to a fixed attribute list (rather than a
        # plain Mock()) so hasattr() checks for optional attributes the mock
        # doesn't set (feature_names_in_, predict_proba) correctly return
        # False instead of Mock always answering True.
        mock_model = Mock(
            spec=[
                "n_estimators",
                "n_features_in_",
                "feature_importances_",
                "get_params",
                "predict",
            ]
        )
        mock_model.n_estimators = 10
        mock_model.n_features_in_ = 5
        mock_model.feature_importances_ = np.random.random(5)
        mock_model.get_params = Mock(return_value={})
        mock_model.predict = Mock(return_value=np.array([0, 1, 0, 1]))

        # Register
        X_train = np.random.random((20, 5))
        y_train = np.random.randint(0, 2, 20)
        model_id = monitor.register_from_model(mock_model, X_train, y_train)

        self.assertEqual(model_id, "model-e2e-lgb")

        # Predict
        X_test = np.random.random((4, 5))
        y_test = np.random.randint(0, 2, 4)
        predictions = monitor.predict(mock_model, X_test, y_test)

        self.assertEqual(len(predictions), 4)
        mock_client.predictions.log_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
