"""
XGBoost and LightGBM Integration Examples

This script demonstrates how to monitor gradient boosting models
(XGBoost and LightGBM) using WhiteBoxXAI.

Examples include:
1. XGBoost binary classification
2. XGBoost regression
3. LightGBM binary classification
4. LightGBM regression
5. Feature importance tracking
6. Model comparison
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# WhiteBoxXAI imports
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.boosting import (
    LightGBMMonitor,
    XGBoostMonitor,
    wrap_lightgbm_model,
    wrap_xgboost_model,
)


def example_xgboost_classification():
    """
    Example: XGBoost binary classification with monitoring.
    """
    print("\n" + "=" * 60)
    print("Example 1: XGBoost Binary Classification")
    print("=" * 60 + "\n")

    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
        return

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key="demo-api-key")

    # Create monitor
    monitor = XGBoostMonitor(
        client=client,
        model_name="xgboost_fraud_detector",
        track_feature_importance=True,
        importance_type="gain",
    )

    # Train XGBoost model
    print("Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # Register model with WhiteBoxXAI
    print("Registering model with WhiteBoxXAI...")
    model_id = monitor.register_from_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        metadata={
            "description": "Fraud detection model using XGBoost",
            "dataset": "synthetic_fraud_data",
            "features": 20,
        },
    )
    print(f"Model registered with ID: {model_id}")

    # Make predictions with monitoring
    print("\nMaking predictions...")
    predictions = monitor.predict(model, X_test, y_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Predictions logged to WhiteBoxXAI")

    # Get feature importance
    importance = model.feature_importances_
    top_features = np.argsort(importance)[-5:][::-1]
    print(f"\nTop 5 features: {top_features.tolist()}")
    print(f"Feature importance tracked in metadata")


def example_xgboost_regression():
    """
    Example: XGBoost regression with automatic wrapper.
    """
    print("\n" + "=" * 60)
    print("Example 2: XGBoost Regression with Wrapper")
    print("=" * 60 + "\n")

    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Install with: pip install xgboost")
        return

    # Generate synthetic data
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=8, noise=10, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key="demo-api-key")

    # Create monitor
    monitor = XGBoostMonitor(
        client=client,
        model_name="xgboost_price_predictor",
        track_feature_importance=True,
    )

    # Train XGBoost regressor
    print("Training XGBoost regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # Wrap model for automatic monitoring
    print("Wrapping model for automatic monitoring...")
    wrapped_model = wrap_xgboost_model(model=model, monitor=monitor, auto_register=True)

    # Predictions automatically logged
    print("\nMaking predictions (auto-logged)...")
    predictions = wrapped_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Predictions automatically logged via wrapper")


def example_lightgbm_classification():
    """
    Example: LightGBM binary classification with monitoring.
    """
    print("\n" + "=" * 60)
    print("Example 3: LightGBM Binary Classification")
    print("=" * 60 + "\n")

    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Install with: pip install lightgbm")
        return

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key="demo-api-key")

    # Create monitor
    monitor = LightGBMMonitor(
        client=client,
        model_name="lightgbm_churn_predictor",
        track_feature_importance=True,
        importance_type="gain",
    )

    # Train LightGBM model
    print("Training LightGBM classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # Register model with WhiteBoxXAI
    print("Registering model with WhiteBoxXAI...")
    model_id = monitor.register_from_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        metadata={
            "description": "Customer churn prediction using LightGBM",
            "dataset": "customer_data",
            "features": 20,
        },
    )
    print(f"Model registered with ID: {model_id}")

    # Make predictions with monitoring
    print("\nMaking predictions...")
    predictions = monitor.predict(model, X_test, y_test)

    # Get probabilities
    probabilities = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Predictions logged with probabilities")

    # Feature importance
    importance = model.feature_importances_
    top_features = np.argsort(importance)[-5:][::-1]
    print(f"\nTop 5 features: {top_features.tolist()}")


def example_lightgbm_regression():
    """
    Example: LightGBM regression with automatic wrapper.
    """
    print("\n" + "=" * 60)
    print("Example 4: LightGBM Regression with Wrapper")
    print("=" * 60 + "\n")

    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Install with: pip install lightgbm")
        return

    # Generate synthetic data
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=8, noise=10, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key="demo-api-key")

    # Create monitor
    monitor = LightGBMMonitor(
        client=client,
        model_name="lightgbm_sales_predictor",
        track_feature_importance=True,
    )

    # Train LightGBM regressor
    print("Training LightGBM regressor...")
    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    # Wrap model for automatic monitoring
    print("Wrapping model for automatic monitoring...")
    wrapped_model = wrap_lightgbm_model(
        model=model, monitor=monitor, auto_register=True
    )

    # Predictions automatically logged
    print("\nMaking predictions (auto-logged)...")
    predictions = wrapped_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Predictions automatically logged via wrapper")


def example_feature_importance_tracking():
    """
    Example: Detailed feature importance tracking.
    """
    print("\n" + "=" * 60)
    print("Example 5: Feature Importance Tracking")
    print("=" * 60 + "\n")

    try:
        import lightgbm as lgb
        import xgboost as xgb
    except ImportError:
        print("XGBoost or LightGBM not installed")
        return

    # Generate synthetic data with feature names
    import pandas as pd

    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=7, random_state=42
    )

    # Create DataFrame with feature names
    feature_names = [f"feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key="demo-api-key")

    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Monitor with different importance types
    for importance_type in ["weight", "gain", "cover"]:
        print(f"\nXGBoost Feature Importance (type={importance_type}):")
        monitor = XGBoostMonitor(
            client=client,
            model_name=f"xgb_importance_{importance_type}",
            importance_type=importance_type,
        )
        monitor.register_from_model(xgb_model, X_train, y_train)

        # Get importance
        importance_dict = monitor._get_feature_importance(xgb_model)
        if importance_dict:
            sorted_features = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for feat, score in sorted_features:
                print(f"  {feat}: {score:.4f}")

    # Train LightGBM model
    print("\n\nTraining LightGBM model...")
    lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=42)
    lgb_model.fit(X_train, y_train)

    # Monitor with different importance types
    for importance_type in ["split", "gain"]:
        print(f"\nLightGBM Feature Importance (type={importance_type}):")
        monitor = LightGBMMonitor(
            client=client,
            model_name=f"lgb_importance_{importance_type}",
            importance_type=importance_type,
        )
        monitor.register_from_model(lgb_model, X_train, y_train)

        # Get importance
        importance_dict = monitor._get_feature_importance(lgb_model)
        if importance_dict:
            sorted_features = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for feat, score in sorted_features:
                print(f"  {feat}: {score:.4f}")


def example_model_comparison():
    """
    Example: Compare XGBoost and LightGBM models.
    """
    print("\n" + "=" * 60)
    print("Example 6: Model Comparison (XGBoost vs LightGBM)")
    print("=" * 60 + "\n")

    try:
        import lightgbm as lgb
        import xgboost as xgb
    except ImportError:
        print("XGBoost or LightGBM not installed")
        return

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize WhiteBoxXAI client
    client = WhiteBoxXAI(api_key="demo-api-key")

    # Train and monitor XGBoost
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    xgb_model.fit(X_train, y_train)

    xgb_monitor = XGBoostMonitor(client=client, model_name="xgb_comparison")
    xgb_monitor.register_from_model(xgb_model, X_train, y_train)
    xgb_preds = xgb_monitor.predict(xgb_model, X_test, y_test)
    xgb_accuracy = accuracy_score(y_test, xgb_preds)

    # Train and monitor LightGBM
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    lgb_model.fit(X_train, y_train)

    lgb_monitor = LightGBMMonitor(client=client, model_name="lgb_comparison")
    lgb_monitor.register_from_model(lgb_model, X_train, y_train)
    lgb_preds = lgb_monitor.predict(lgb_model, X_test, y_test)
    lgb_accuracy = accuracy_score(y_test, lgb_preds)

    # Compare results
    print("\n" + "-" * 60)
    print("Model Comparison Results")
    print("-" * 60)
    print(f"XGBoost Accuracy:  {xgb_accuracy:.4f}")
    print(f"LightGBM Accuracy: {lgb_accuracy:.4f}")
    print(f"\nBetter model: {'XGBoost' if xgb_accuracy > lgb_accuracy else 'LightGBM'}")
    print("Both models logged to WhiteBoxXAI for detailed analysis")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WhiteBoxXAI - Gradient Boosting Integration Examples")
    print("=" * 60)

    # Run examples
    example_xgboost_classification()
    example_xgboost_regression()
    example_lightgbm_classification()
    example_lightgbm_regression()
    example_feature_importance_tracking()
    example_model_comparison()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")
