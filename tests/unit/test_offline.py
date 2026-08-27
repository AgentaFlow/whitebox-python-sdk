"""
Tests for WhiteBoxXAI SDK offline mode.

Covers the offline queue's core correctness guarantees: a connection
failure gets queued instead of raised, queued operations replay
correctly, and concurrent dequeue() calls can never claim the same row
twice (PR6).
"""

import threading
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from whiteboxxai.client import WhiteBoxXAI
from whiteboxxai.offline import OfflineManager, OfflineQueue, OperationType


@pytest.fixture
def offline_dir(tmp_path):
    return str(tmp_path / "whiteboxxai_offline")


@pytest.fixture
def queue(offline_dir):
    return OfflineQueue(db_path=str(Path(offline_dir) / "queue.db"))


class TestOfflineQueueBasics:
    """Enqueue/dequeue/mark_* round trip."""

    def test_enqueue_then_dequeue_returns_the_operation(self, queue):
        op_id = queue.enqueue(OperationType.PREDICT, {"data": {"model_id": "m1"}})

        operations = queue.dequeue(limit=10)

        assert len(operations) == 1
        got_id, op_type, data = operations[0]
        assert got_id == op_id
        assert op_type == OperationType.PREDICT
        assert data == {"data": {"model_id": "m1"}}

    def test_dequeue_claims_rows_as_processing(self, queue):
        queue.enqueue(OperationType.PREDICT, {"data": {}})

        queue.dequeue(limit=10)

        stats = queue.get_statistics()
        assert stats["processing"] == 1
        assert stats["pending"] == 0

    def test_dequeue_does_not_return_already_claimed_rows(self, queue):
        queue.enqueue(OperationType.PREDICT, {"data": {}})

        first = queue.dequeue(limit=10)
        second = queue.dequeue(limit=10)

        assert len(first) == 1
        assert second == []

    def test_mark_success_completes_the_operation(self, queue):
        op_id = queue.enqueue(OperationType.PREDICT, {"data": {}})
        queue.dequeue(limit=10)

        queue.mark_success(op_id)

        stats = queue.get_statistics()
        assert stats["completed"] == 1
        assert stats["processing"] == 0

    def test_mark_failure_with_retries_remaining_releases_to_pending(self, queue):
        op_id = queue.enqueue(OperationType.PREDICT, {"data": {}})
        queue.dequeue(limit=10)

        queue.mark_failure(op_id, "connection reset", max_retries=3)

        stats = queue.get_statistics()
        assert stats["pending"] == 1
        assert stats["processing"] == 0
        # Released back to pending, so a later dequeue() can retry it.
        assert len(queue.dequeue(limit=10)) == 1

    def test_mark_failure_exhausting_retries_marks_failed(self, queue):
        op_id = queue.enqueue(OperationType.PREDICT, {"data": {}})

        for _ in range(3):
            queue.dequeue(limit=10)
            queue.mark_failure(op_id, "still broken", max_retries=3)

        stats = queue.get_statistics()
        assert stats["failed"] == 1
        assert stats["pending"] == 0
        assert stats["processing"] == 0

    def test_get_statistics_includes_processing_key(self, queue):
        stats = queue.get_statistics()
        assert "processing" in stats
        assert stats["processing"] == 0


class TestOrphanedClaimRecovery:
    """A crash between dequeue() and mark_success()/mark_failure() must not
    strand an operation in 'processing' forever."""

    def test_reopening_the_queue_resets_orphaned_processing_rows(self, offline_dir):
        import sqlite3

        db_path = str(Path(offline_dir) / "queue.db")
        queue = OfflineQueue(db_path=db_path)
        op_id = queue.enqueue(OperationType.PREDICT, {"data": {}})
        queue.dequeue(limit=10)  # Claims it, then "the process crashes" -- no
        # mark_success()/mark_failure() ever runs.
        assert queue.get_statistics()["processing"] == 1

        # Backdate the claim well past _STALE_CLAIM_MINUTES so it reads as
        # genuinely orphaned rather than a live sibling's in-progress claim
        # (a claim only a few milliseconds old, as a real crash-then-
        # immediate-restart would produce, is deliberately NOT reclaimed --
        # see TestConcurrentDequeue -- so this test has to simulate real
        # staleness rather than just an instant restart).
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE queue SET claimed_at = datetime('now', '-1 hour') " "WHERE id = ?",
                (op_id,),
            )
            conn.commit()

        # Simulates the next process startup, pointed at the same db file.
        reopened = OfflineQueue(db_path=db_path)

        stats = reopened.get_statistics()
        assert stats["processing"] == 0
        assert stats["pending"] == 1
        operations = reopened.dequeue(limit=10)
        assert len(operations) == 1
        assert operations[0][0] == op_id


class TestConcurrentDequeue:
    """The actual race the audit flagged: two callers dequeuing at once
    must never both claim the same row."""

    def test_concurrent_dequeue_never_returns_overlapping_rows(self, offline_dir):
        db_path = str(Path(offline_dir) / "queue.db")
        setup_queue = OfflineQueue(db_path=db_path)
        for i in range(20):
            setup_queue.enqueue(OperationType.PREDICT, {"data": {"i": i}})

        # Each thread gets its own OfflineQueue instance (its own sqlite3
        # connection, its own threading.Lock) pointed at the same db file --
        # this is what actually exercises the BEGIN IMMEDIATE cross-connection
        # locking rather than just this process's in-memory lock.
        results = []
        results_lock = threading.Lock()

        def worker():
            worker_queue = OfflineQueue(db_path=db_path)
            claimed = worker_queue.dequeue(limit=5)
            with results_lock:
                results.append(claimed)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_claimed_ids = [op_id for batch in results for op_id, _, _ in batch]
        assert len(all_claimed_ids) == len(set(all_claimed_ids)), (
            "the same operation id was claimed by more than one concurrent " "dequeue() call"
        )


class TestOfflineManagerSync:
    """OfflineManager.sync() must replay each operation type against the
    matching WhiteBoxXAI._api_* method with the exact enqueued kwargs."""

    def test_sync_replays_predict_via_api_predict(self, offline_dir):
        manager = OfflineManager(offline_dir=offline_dir, auto_sync=False)
        mock_client = Mock()
        manager.set_client(mock_client)
        manager.queue.enqueue(OperationType.PREDICT, {"data": {"model_id": "m1"}})

        stats = manager.sync()

        mock_client._api_predict.assert_called_once_with(data={"model_id": "m1"})
        assert stats["synced"] == 1
        assert stats["failed"] == 0

    def test_sync_replays_register_model_via_api_register_model(self, offline_dir):
        manager = OfflineManager(offline_dir=offline_dir, auto_sync=False)
        mock_client = Mock()
        manager.set_client(mock_client)
        manager.queue.enqueue(OperationType.REGISTER_MODEL, {"data": {"name": "fraud-model"}})

        manager.sync()

        mock_client._api_register_model.assert_called_once_with(data={"name": "fraud-model"})

    def test_sync_replays_update_baseline_via_api_update_baseline(self, offline_dir):
        manager = OfflineManager(offline_dir=offline_dir, auto_sync=False)
        mock_client = Mock()
        manager.set_client(mock_client)
        manager.queue.enqueue(
            OperationType.UPDATE_BASELINE,
            {"model_id": "m1", "data": {"accuracy": 0.9}, "params": {}},
        )

        manager.sync()

        mock_client._api_update_baseline.assert_called_once_with(
            model_id="m1", data={"accuracy": 0.9}, params={}
        )

    def test_sync_replays_log_batch_via_api_log_batch(self, offline_dir):
        manager = OfflineManager(offline_dir=offline_dir, auto_sync=False)
        mock_client = Mock()
        manager.set_client(mock_client)
        manager.queue.enqueue(
            OperationType.LOG_BATCH, {"data": {"model_id": "m1", "predictions": []}}
        )

        manager.sync()

        mock_client._api_log_batch.assert_called_once_with(
            data={"model_id": "m1", "predictions": []}
        )

    def test_sync_marks_success_and_clears_the_queue(self, offline_dir):
        manager = OfflineManager(offline_dir=offline_dir, auto_sync=False)
        mock_client = Mock()
        manager.set_client(mock_client)
        manager.queue.enqueue(OperationType.PREDICT, {"data": {}})

        manager.sync()

        assert manager.queue.get_statistics()["completed"] == 1
        assert manager.queue.get_statistics()["pending"] == 0


class TestClientConnectionFailureQueuesInsteadOfRaising:
    """End-to-end: a real WhiteBoxXAI client with offline mode enabled must
    queue on connection failure rather than raising, and the resulting
    queued operation must be replayable."""

    @patch("httpx.Client.request")
    def test_predictions_log_queues_on_connection_error(self, mock_request, offline_dir):
        mock_request.side_effect = httpx.ConnectError("Connection refused")

        client = WhiteBoxXAI(
            api_key="test_key",
            enable_offline=True,
            offline_dir=offline_dir,
            offline_auto_sync=False,
        )

        result = client.predictions.log(
            model_id="m1",
            input_data={"x": 1},
            output_data={"y": 2},
        )

        assert result["status"] == "queued"
        assert client.get_offline_status()["queue_stats"]["pending"] == 1

    @patch("httpx.Client.request")
    def test_connection_error_reraised_when_offline_mode_disabled(self, mock_request):
        mock_request.side_effect = httpx.ConnectError("Connection refused")

        client = WhiteBoxXAI(api_key="test_key")  # enable_offline defaults False

        from whiteboxxai.exceptions import APIConnectionError

        with pytest.raises(APIConnectionError):
            client.predictions.log(model_id="m1", input_data={"x": 1}, output_data={"y": 2})
