"""
Scikit-learn Integration Example

This example demonstrates monitoring scikit-learn models.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.sklearn import SklearnMonitor


def main():
    # Generate sample dataset
    print("Generating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.3f}")

    # Setup monitoring
    print("\nSetting up WhiteBoxXAI monitoring...")
    client = WhiteBoxXAI(api_key="your-api-key")
    monitor = SklearnMonitor(client, model=model)

    # Register model with automatic metadata extraction
    model_id = monitor.register_from_model(
        name="fraud_classifier_rf",
        model_type="classification",
        version="1.0.0",
    )
    print(f"Model registered with ID: {model_id}")

    # Wrap model for automatic monitoring
    print("\nWrapping model for automatic prediction logging...")
    monitored_model = monitor.wrap_model(model)

    # Make predictions (automatically logged)
    print("Making predictions...")
    predictions = monitored_model.predict(X_test[:10])
    print(f"Predictions: {predictions}")

    # Get prediction probabilities (also automatically logged)
    print("\nGetting prediction probabilities...")
    probas = monitored_model.predict_proba(X_test[:5])
    print(f"Probabilities shape: {probas.shape}")

    # Set baseline for drift detection
    print("\nSetting baseline data...")
    monitor.set_baseline(X_train)

    # Detect drift on test data
    print("Detecting drift...")
    drift_result = monitor.detect_drift(X_test)
    print(f"Drift detected: {drift_result}")

    # Close client
    client.close()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
