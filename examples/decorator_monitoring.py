"""
Decorator-based Monitoring Example

This example demonstrates zero-code-change monitoring using decorators.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from whiteboxxai import ModelMonitor, WhiteBoxXAI, monitor_model, monitor_prediction

# Global monitor instance
client = WhiteBoxXAI(api_key="your-api-key")
monitor = ModelMonitor(client, model_id=123)


@monitor_model(monitor, input_keys=["features"], explain=True)
def predict_fraud(features):
    """Predict fraud probability."""
    # Simulate model prediction
    model = RandomForestClassifier()
    # ... (assume model is trained)
    prediction = np.random.choice([0, 1])
    probability = np.random.random()

    return {"prediction": prediction, "probability": probability}


@monitor_prediction(
    monitor,
    input_extractor=lambda args, kwargs: {"data": kwargs.get("data")},
    output_extractor=lambda result: result["score"],
)
def score_transaction(data):
    """Score transaction for fraud."""
    # Complex scoring logic
    score = sum(data.values()) / len(data)
    return {"score": score, "factors": data}


class ModelPredictor:
    """Class-based predictor with monitoring."""

    def __init__(self, model):
        self.model = model

    @monitor_model(monitor, explain=False)
    def predict(self, features):
        """Make prediction."""
        return self.model.predict([features])[0]

    @monitor_prediction(monitor)
    async def async_predict(self, features):
        """Make async prediction."""
        # Simulate async prediction
        import asyncio

        await asyncio.sleep(0.1)
        return self.model.predict([features])[0]


def main():
    """Run decorator examples."""
    print("=== Function Decorator ===")

    # Decorated function - predictions are automatically logged
    result = predict_fraud(features=[1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Fraud prediction: {result}")

    print("\n=== Custom Extractors ===")

    # Custom input/output extraction
    result = score_transaction(data={"amount": 100.0, "velocity": 5.0, "location_risk": 0.3})
    print(f"Transaction score: {result}")

    print("\n=== Class Method Decorator ===")

    # Class-based predictor
    model = RandomForestClassifier()
    # ... (assume model is trained)

    predictor = ModelPredictor(model)

    # Monitored method
    prediction = predictor.predict([1.0, 2.0, 3.0])
    print(f"Class prediction: {prediction}")

    print("\n=== Async Decorator ===")

    import asyncio

    async def async_example():
        prediction = await predictor.async_predict([1.0, 2.0, 3.0])
        print(f"Async prediction: {prediction}")

    asyncio.run(async_example())

    # Close client
    client.close()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
