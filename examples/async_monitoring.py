"""
Async Monitoring Example

This example demonstrates asynchronous API usage for better performance.
"""

import asyncio

import numpy as np

from whiteboxai import ModelMonitor, WhiteBoxAI


async def register_and_log():
    """Register model and log predictions asynchronously."""
    # Create client with async context manager
    async with WhiteBoxAI(api_key="your-api-key") as client:
        monitor = ModelMonitor(client)

        # Register model asynchronously
        print("Registering model...")
        model_id = await monitor.aregister_model(
            name="async_fraud_detector",
            model_type="classification",
            framework="sklearn",
            version="1.0.0",
        )
        print(f"Model registered with ID: {model_id}")

        # Log single prediction
        print("\nLogging single prediction...")
        await monitor.alog_prediction(
            inputs={"amount": 100.0, "merchant": "store_1"},
            output={"fraud_prob": 0.15, "prediction": "legitimate"},
        )
        print("Prediction logged")

        # Log batch predictions
        print("\nLogging batch predictions...")
        predictions = [
            {"inputs": {"amount": 50.0}, "output": {"fraud_prob": 0.05}} for _ in range(100)
        ]

        await monitor.alog_batch(predictions)
        print("Batch logged")

        # Get model details
        print("\nRetrieving model details...")
        model = await client.models.aget(model_id)
        print(f"Model: {model['name']}")


async def parallel_logging():
    """Log predictions in parallel for better throughput."""
    async with WhiteBoxAI(api_key="your-api-key") as client:
        monitor = ModelMonitor(client, model_id=123)

        # Create multiple prediction logging tasks
        tasks = []
        for i in range(10):
            task = monitor.alog_prediction(
                inputs={"feature": float(i)}, output={"prediction": i % 2}
            )
            tasks.append(task)

        # Execute all tasks in parallel
        print("Logging 10 predictions in parallel...")
        results = await asyncio.gather(*tasks)
        print(f"Logged {len(results)} predictions")


async def drift_detection():
    """Detect drift asynchronously."""
    async with WhiteBoxAI(api_key="your-api-key") as client:
        monitor = ModelMonitor(client, model_id=123)

        # Set baseline
        baseline = np.random.randn(1000, 10)
        monitor.set_baseline(baseline)

        # Detect drift
        print("Detecting drift...")
        current_data = np.random.randn(100, 10)
        drift_result = await monitor.adetect_drift(current_data)
        print(f"Drift result: {drift_result}")


async def main():
    """Run all async examples."""
    print("=== Register and Log ===")
    await register_and_log()

    print("\n=== Parallel Logging ===")
    await parallel_logging()

    print("\n=== Drift Detection ===")
    await drift_detection()

    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
