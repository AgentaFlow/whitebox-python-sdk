"""
Basic Monitoring Example

This example demonstrates basic model registration and prediction logging.
"""

from whiteboxai import WhiteBoxAI, ModelMonitor


def main():
    # Initialize client with API key
    client = WhiteBoxAI(api_key="your-api-key")

    # Create monitor instance
    monitor = ModelMonitor(client)

    # Register a new model
    print("Registering model...")
    model_id = monitor.register_model(
        name="fraud_detection_v1",
        model_type="classification",
        framework="sklearn",
        version="1.0.0",
        description="Fraud detection model using Random Forest",
    )
    print(f"Model registered with ID: {model_id}")

    # Log a single prediction
    print("\nLogging single prediction...")
    monitor.log_prediction(
        inputs={
            "transaction_amount": 150.50,
            "merchant_id": "merchant_123",
            "customer_age": 35,
            "transaction_hour": 14,
        },
        output={"fraud_probability": 0.15, "prediction": "legitimate"},
        metadata={"model_version": "1.0.0", "timestamp": "2024-01-15T14:30:00Z"},
    )
    print("Prediction logged successfully")

    # Log multiple predictions in batch
    print("\nLogging batch predictions...")
    predictions = [
        {
            "inputs": {"amount": 50.0, "merchant": "store_1"},
            "output": {"fraud_prob": 0.05, "prediction": "legitimate"},
        },
        {
            "inputs": {"amount": 500.0, "merchant": "store_2"},
            "output": {"fraud_prob": 0.85, "prediction": "fraud"},
        },
        {
            "inputs": {"amount": 75.0, "merchant": "store_3"},
            "output": {"fraud_prob": 0.12, "prediction": "legitimate"},
        },
    ]

    result = monitor.log_batch(predictions)
    print(f"Batch logged: {result}")

    # Get model details
    print("\nRetrieving model details...")
    model = client.models.get(model_id)
    print(f"Model: {model['name']}")
    print(f"Type: {model['model_type']}")
    print(f"Framework: {model['framework']}")

    # Close client
    client.close()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
