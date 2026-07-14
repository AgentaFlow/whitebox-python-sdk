"""
TensorFlow/Keras Integration Example

This example demonstrates how to use WhiteBoxXAI with TensorFlow/Keras models.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    exit(1)

from whiteboxxai import WhiteBoxXAI
from whiteboxxai.integrations.tensorflow import KerasMonitor, WhiteBoxXAICallback


def main():
    print("=" * 60)
    print("TensorFlow/Keras Integration Example")
    print("=" * 60)

    # Generate synthetic classification data
    print("\n1. Generating synthetic data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")

    # Build Keras model
    print("\n2. Building Keras model...")
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(20,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC()],
    )

    print(f"   Model built with {model.count_params():,} parameters")

    # Initialize WhiteBoxXAI
    print("\n3. Initializing WhiteBoxXAI monitoring...")
    client = WhiteBoxXAI(api_key="demo-api-key", base_url="http://localhost:8000")

    # Create Keras monitor
    monitor = KerasMonitor(
        client=client,
        model=model,
        model_name="keras_binary_classifier",
        model_type="classification",
    )

    # Register model
    model_id = monitor.register_from_model(
        version="1.0.0",
        description="Binary classifier using Keras Sequential API",
    )
    print(f"   Model registered with ID: {model_id}")

    # Set baseline for drift detection
    print("\n4. Setting baseline data...")
    monitor.set_baseline(X_train, y_train)
    print("   ✓ Baseline set")

    # Create WhiteBoxXAI callback
    print("\n5. Training model with WhiteBoxXAI monitoring...")
    callback = WhiteBoxXAICallback(
        monitor=monitor, log_frequency=5, log_validation=True  # Log every 5 epochs
    )

    # Additional callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[callback, early_stopping, reduce_lr],
        verbose=0,
    )

    final_epoch = len(history.history["loss"])
    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    print(f"   Training completed in {final_epoch} epochs")
    print(f"   Final training accuracy: {final_acc:.4f}")
    print(f"   Final validation accuracy: {final_val_acc:.4f}")

    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test AUC: {test_auc:.4f}")

    # Make predictions with automatic logging
    print("\n7. Making predictions with automatic logging...")
    predictions = monitor.predict(
        X_test, log=True, actuals=y_test, metadata={"phase": "test_evaluation"}
    )
    print(f"   ✓ Logged {len(predictions)} predictions")

    # Log individual predictions
    print("\n8. Logging individual predictions...")
    for i in range(5):
        prob = predictions[i][0]
        pred_class = 1 if prob > 0.5 else 0
        actual = y_test[i]
        print(f"   Sample {i+1}: Predicted={pred_class} (prob={prob:.3f}), Actual={actual}")

    # Save model
    print("\n9. Saving model...")
    model.save("models/keras_binary_classifier")
    print("   ✓ Model saved to 'models/keras_binary_classifier'")

    # Register saved model
    monitor.register_saved_model(
        model_path="models/keras_binary_classifier", metadata={"format": "SavedModel"}
    )
    print("   ✓ SavedModel registered with WhiteBoxXAI")

    # Check drift
    print("\n10. Checking for data drift...")
    try:
        drift_report = monitor.check_drift()
        if drift_report:
            print(f"   Drift detected: {drift_report.get('drift_detected', False)}")
            if drift_report.get("drift_score"):
                print(f"   Drift score: {drift_report['drift_score']:.4f}")
        else:
            print("   No drift detected")
    except Exception as e:
        print(f"   Drift check not available: {e}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
