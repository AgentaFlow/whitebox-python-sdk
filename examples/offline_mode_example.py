"""
Offline Mode Examples for WhiteBoxXAI SDK

This example demonstrates how to use offline mode for queueing operations
when the API is unavailable and syncing when connection is restored.
"""

import os
import time
from typing import Dict, List

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from whiteboxxai import WhiteBoxXAI
from whiteboxxai.offline import OperationPriority, OperationType


def example_1_basic_offline_mode():
    """Example 1: Basic offline mode with auto-sync."""
    print("=" * 80)
    print("Example 1: Basic Offline Mode")
    print("=" * 80)

    # Initialize client with offline mode enabled
    client = WhiteBoxXAI(
        api_key=os.getenv("WHITEBOXXAI_API_KEY", "test_key"),
        enable_offline=True,
        offline_dir="./offline_queue",
        offline_auto_sync=True,  # Auto-sync every 60 seconds
        offline_sync_interval=60,
        offline_max_queue_size=10000,
    )

    print(f"Offline mode enabled: {client.is_offline_enabled()}")

    # Check offline status
    status = client.get_offline_status()
    print(f"\nOffline Status:")
    print(f"  Queue size: {status['queue_size']}")
    print(f"  Statistics: {status['statistics']}")

    # When API is unavailable, operations are queued automatically
    # The offline manager will retry syncing in the background

    client.close()
    print("\n✓ Example 1 complete\n")


def example_2_manual_sync():
    """Example 2: Manual sync control."""
    print("=" * 80)
    print("Example 2: Manual Sync Control")
    print("=" * 80)

    # Initialize with auto-sync disabled for manual control
    client = WhiteBoxXAI(
        api_key=os.getenv("WHITEBOXXAI_API_KEY", "test_key"),
        enable_offline=True,
        offline_dir="./offline_queue",
        offline_auto_sync=False,  # Disable auto-sync
    )

    # Simulate queueing operations
    # (In real usage, these would be queued when API is unavailable)
    print("\nQueueing operations...")
    client._offline_manager._queue.enqueue(
        OperationType.PREDICT,
        {
            "model_id": "model_123",
            "inputs": {"feature1": 1.0, "feature2": 2.0},
            "outputs": [0.8, 0.2],
        },
        OperationPriority.HIGH,
    )

    client._offline_manager._queue.enqueue(
        OperationType.LOG_BATCH,
        {
            "model_id": "model_123",
            "predictions": [
                {"inputs": {"f1": 1}, "outputs": [0.7, 0.3]},
                {"inputs": {"f1": 2}, "outputs": [0.6, 0.4]},
            ],
        },
        OperationPriority.NORMAL,
    )

    # Check queue status
    status = client.get_offline_status()
    print(f"Queue size before sync: {status['queue_size']}")

    # Manually trigger sync when connection is available
    print("\nTriggering manual sync...")
    result = client.sync_offline_queue(batch_size=50)
    print(f"Sync result:")
    print(f"  Synced: {result['synced']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Pending: {result['pending']}")

    client.close()
    print("\n✓ Example 2 complete\n")


def example_3_priority_based_syncing():
    """Example 3: Priority-based operation syncing."""
    print("=" * 80)
    print("Example 3: Priority-Based Syncing")
    print("=" * 80)

    client = WhiteBoxXAI(
        api_key=os.getenv("WHITEBOXXAI_API_KEY", "test_key"),
        enable_offline=True,
        offline_dir="./offline_queue",
        offline_auto_sync=False,
    )

    # Queue operations with different priorities
    print("\nQueueing operations with different priorities...")

    # Low priority - batch logging
    client._offline_manager._queue.enqueue(
        OperationType.LOG_BATCH,
        {"model_id": "model_123", "batch": []},
        OperationPriority.LOW,
    )
    print("  ✓ Queued LOW priority: batch logging")

    # Normal priority - prediction logging
    client._offline_manager._queue.enqueue(
        OperationType.PREDICT,
        {"model_id": "model_123", "prediction": [0.5, 0.5]},
        OperationPriority.NORMAL,
    )
    print("  ✓ Queued NORMAL priority: prediction")

    # High priority - model registration
    client._offline_manager._queue.enqueue(
        OperationType.REGISTER_MODEL,
        {"name": "critical_model", "model_type": "classification"},
        OperationPriority.HIGH,
    )
    print("  ✓ Queued HIGH priority: model registration")

    # Critical priority - urgent prediction
    client._offline_manager._queue.enqueue(
        OperationType.PREDICT,
        {"model_id": "model_123", "urgent": True, "prediction": [0.9, 0.1]},
        OperationPriority.CRITICAL,
    )
    print("  ✓ Queued CRITICAL priority: urgent prediction")

    # Operations will be synced in priority order: CRITICAL > HIGH > NORMAL > LOW
    print("\nOperations will be synced in order: CRITICAL → HIGH → NORMAL → LOW")

    status = client.get_offline_status()
    print(f"\nTotal queued: {status['queue_size']}")

    client.close()
    print("\n✓ Example 3 complete\n")


def example_4_queue_management():
    """Example 4: Queue management and cleanup."""
    print("=" * 80)
    print("Example 4: Queue Management")
    print("=" * 80)

    client = WhiteBoxXAI(
        api_key=os.getenv("WHITEBOXXAI_API_KEY", "test_key"),
        enable_offline=True,
        offline_dir="./offline_queue",
        offline_auto_sync=False,
        offline_max_queue_size=100,  # Limit queue size
    )

    # Get queue statistics
    status = client.get_offline_status()
    print(f"\nInitial Queue Status:")
    print(f"  Total: {status['statistics']['total']}")
    print(f"  Pending: {status['statistics']['pending']}")
    print(f"  Completed: {status['statistics']['completed']}")
    print(f"  Failed: {status['statistics']['failed']}")

    # Clean up old completed operations (older than 7 days)
    print("\nCleaning up old operations...")
    client.cleanup_offline_queue(older_than_days=7)
    print("  ✓ Cleanup complete")

    # Check queue size limit
    print(f"\nQueue size limit: {client._offline_manager._max_queue_size}")
    print("When queue is full, new operations will fail with ValueError")

    client.close()
    print("\n✓ Example 4 complete\n")


def example_5_ml_model_with_offline():
    """Example 5: Using offline mode with ML model monitoring."""
    print("=" * 80)
    print("Example 5: ML Model with Offline Mode")
    print("=" * 80)

    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8, n_redundant=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("  ✓ Model trained")

    # Initialize client with offline mode
    client = WhiteBoxXAI(
        api_key=os.getenv("WHITEBOXXAI_API_KEY", "test_key"),
        enable_offline=True,
        offline_dir="./ml_offline_queue",
        offline_auto_sync=True,
        offline_sync_interval=30,  # Sync every 30 seconds
    )

    print(f"\nOffline mode: {client.is_offline_enabled()}")

    # In production, if API is unavailable:
    # 1. Model registration would be queued
    # 2. Predictions would be queued
    # 3. Operations sync automatically when connection restored

    # Simulate predictions
    print("\nMaking predictions (will queue if offline)...")
    predictions = model.predict_proba(X_test[:5])

    for i, pred in enumerate(predictions):
        # In real usage, this would automatically queue if API unavailable
        print(f"  Prediction {i+1}: {pred}")

    # Check queue status
    status = client.get_offline_status()
    print(f"\nQueue Status:")
    print(f"  Pending operations: {status['statistics']['pending']}")
    print(f"  Completed: {status['statistics']['completed']}")

    client.close()
    print("\n✓ Example 5 complete\n")


def example_6_error_handling():
    """Example 6: Error handling and retry logic."""
    print("=" * 80)
    print("Example 6: Error Handling and Retry")
    print("=" * 80)

    client = WhiteBoxXAI(
        api_key=os.getenv("WHITEBOXXAI_API_KEY", "test_key"),
        enable_offline=True,
        offline_dir="./offline_queue",
        offline_auto_sync=False,
    )

    print("\nOffline mode retry behavior:")
    print("  - Failed operations are automatically retried")
    print("  - Default max retries: 3")
    print("  - After max retries, marked as permanently failed")
    print("  - Failed operations can be retrieved for investigation")

    # Simulate a failed operation
    op_id = client._offline_manager._queue.enqueue(
        OperationType.PREDICT,
        {"model_id": "test", "prediction": [0.5, 0.5]},
        OperationPriority.NORMAL,
    )

    # Mark as failed multiple times (simulating retries)
    print(f"\nSimulating retry attempts for operation {op_id}...")
    for attempt in range(3):
        client._offline_manager._queue.mark_failure(
            op_id, f"Simulated error (attempt {attempt + 1})", max_retries=3
        )
        print(f"  Attempt {attempt + 1}: Failed")

    # Check failed operations
    failed_ops = client._offline_manager._queue.get_failed_operations()
    print(f"\nPermanently failed operations: {len(failed_ops)}")

    if failed_ops:
        print("\nFailed operation details:")
        for op in failed_ops:
            print(f"  ID: {op['id']}")
            print(f"  Type: {op['operation_type']}")
            print(f"  Retry count: {op['retry_count']}")
            print(f"  Last error: {op['last_error']}")

    client.close()
    print("\n✓ Example 6 complete\n")


def example_7_context_manager():
    """Example 7: Using client as context manager with offline mode."""
    print("=" * 80)
    print("Example 7: Context Manager Usage")
    print("=" * 80)

    print("\nUsing WhiteBoxXAI client as context manager:")
    print("  - Automatically starts auto-sync")
    print("  - Properly stops sync on exit")
    print("  - Ensures resource cleanup")

    with WhiteBoxXAI(
        api_key=os.getenv("WHITEBOXXAI_API_KEY", "test_key"),
        enable_offline=True,
        offline_dir="./offline_queue",
        offline_auto_sync=True,
        offline_sync_interval=60,
    ) as client:
        print(f"\n✓ Client initialized with offline mode")
        print(f"  Auto-sync running: {client._offline_manager._sync_running}")

        status = client.get_offline_status()
        print(f"  Queue size: {status['queue_size']}")

        # Operations would be queued here if API unavailable
        # Auto-sync runs in background

    print("\n✓ Context exited - auto-sync stopped automatically")
    print("✓ Example 7 complete\n")


def run_all_examples():
    """Run all offline mode examples."""
    print("\n" + "=" * 80)
    print("WhiteBoxXAI SDK - Offline Mode Examples")
    print("=" * 80 + "\n")

    examples = [
        example_1_basic_offline_mode,
        example_2_manual_sync,
        example_3_priority_based_syncing,
        example_4_queue_management,
        example_5_ml_model_with_offline,
        example_6_error_handling,
        example_7_context_manager,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ Example failed: {e}\n")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Set environment variable if not set
    if "WHITEBOXXAI_API_KEY" not in os.environ:
        print("Note: WHITEBOXXAI_API_KEY not set, using 'test_key' for examples")
        print("Set your API key: export WHITEBOXXAI_API_KEY=your_key_here\n")

    run_all_examples()
