"""
Tests for SDK Offline Queue Module

Tests for offline mode queueing functionality.
"""

import sqlite3

from whiteboxxai.offline import OfflineQueue, OperationPriority, OperationType


class TestOfflineQueueClearCompleted:
    """Tests for OfflineQueue.clear_completed()."""

    def test_clears_records_exactly_older_than_days_old(self, tmp_path):
        """A completed record whose age is exactly `older_than_days` must be
        cleared (boundary is inclusive: created_at <= cutoff, not <)."""
        db_path = str(tmp_path / "queue.db")
        queue = OfflineQueue(db_path=db_path)

        op_id = queue.enqueue(OperationType.PREDICT, {"id": 1}, OperationPriority.NORMAL)
        queue.mark_success(op_id)

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE queue SET created_at = datetime('now', '-7 days') WHERE id = ?",
                (op_id,),
            )
            conn.commit()

        queue.clear_completed(older_than_days=7)

        assert queue.get_statistics()["total"] == 0

    def test_keeps_records_younger_than_older_than_days(self, tmp_path):
        """A completed record younger than `older_than_days` must be kept."""
        db_path = str(tmp_path / "queue.db")
        queue = OfflineQueue(db_path=db_path)

        op_id = queue.enqueue(OperationType.PREDICT, {"id": 1}, OperationPriority.NORMAL)
        queue.mark_success(op_id)

        queue.clear_completed(older_than_days=7)

        assert queue.get_statistics()["total"] == 1
