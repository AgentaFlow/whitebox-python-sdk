"""
Offline mode support for WhiteBoxXAI SDK.

This module provides offline queueing capabilities for the SDK, allowing
operations to be queued locally when the API is unavailable and synced
when connectivity is restored.

Features:
- Persistent queue storage (SQLite-based)
- Automatic retry on connection restoration
- Configurable queue limits
- Operation prioritization
- Failed operation tracking

Example:
    Enable offline mode:

    >>> from whiteboxxai import WhiteBoxXAI
    >>> client = WhiteBoxXAI(
    ...     api_key="your-api-key",
    ...     enable_offline=True,
    ...     offline_dir="./whiteboxxai_offline"
    ... )
    >>>
    >>> # Operations are queued when offline
    >>> client.predict(...)  # Queued if API unavailable
    >>>
    >>> # Manually sync
    >>> client.sync_offline_queue()
"""

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be queued."""

    PREDICT = "predict"
    REGISTER_MODEL = "register_model"
    UPDATE_BASELINE = "update_baseline"
    LOG_BATCH = "log_batch"


class OperationPriority(Enum):
    """Priority levels for queued operations."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class OfflineQueue:
    """
    Persistent queue for offline operations.

    Uses SQLite for storage to ensure operations are not lost even if
    the application crashes or restarts.

    Attributes:
        db_path: Path to the SQLite database file
        max_queue_size: Maximum number of operations to queue
        auto_sync: Whether to automatically sync when connection available
    """

    def __init__(self, db_path: str, max_queue_size: int = 10000, auto_sync: bool = True):
        """
        Initialize offline queue.

        Args:
            db_path: Path to SQLite database file
            max_queue_size: Maximum operations to queue (0 = unlimited)
            auto_sync: Enable automatic syncing
        """
        self.db_path = db_path
        self.max_queue_size = max_queue_size
        self.auto_sync = auto_sync
        self._lock = threading.Lock()
        self._ensure_db()

    def _ensure_db(self):
        """Create database and tables if they don't exist."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    last_error TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """
            )

            # Create index for efficient querying
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status_priority
                ON queue(status, priority DESC, created_at ASC)
            """
            )

            conn.commit()

    def enqueue(
        self,
        operation_type: OperationType,
        data: Dict[str, Any],
        priority: OperationPriority = OperationPriority.NORMAL,
    ) -> int:
        """
        Add an operation to the queue.

        Args:
            operation_type: Type of operation
            data: Operation data (will be JSON serialized)
            priority: Operation priority

        Returns:
            Operation ID

        Raises:
            ValueError: If queue is full
        """
        with self._lock:
            # Check queue size
            if self.max_queue_size > 0:
                current_size = self.get_queue_size()
                if current_size >= self.max_queue_size:
                    raise ValueError(f"Queue is full ({current_size}/{self.max_queue_size})")

            # Insert operation
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO queue (operation_type, priority, data)
                    VALUES (?, ?, ?)
                    """,
                    (operation_type.value, priority.value, json.dumps(data)),
                )
                conn.commit()
                op_id = cursor.lastrowid

            logger.info(f"Queued operation {op_id}: {operation_type.value}")
            return op_id

    def dequeue(self, limit: int = 100) -> List[Tuple[int, OperationType, Dict[str, Any]]]:
        """
        Get pending operations from queue.

        Returns operations ordered by priority (highest first) and
        creation time (oldest first).

        Args:
            limit: Maximum number of operations to return

        Returns:
            List of (id, operation_type, data) tuples
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, operation_type, data
                    FROM queue
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )

                operations = []
                for row in cursor.fetchall():
                    op_id, op_type, data_json = row
                    operations.append((op_id, OperationType(op_type), json.loads(data_json)))

                return operations

    def mark_success(self, operation_id: int):
        """
        Mark operation as successfully completed.

        Args:
            operation_id: Operation ID
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE queue
                    SET status = 'completed'
                    WHERE id = ?
                    """,
                    (operation_id,),
                )
                conn.commit()

        logger.debug(f"Marked operation {operation_id} as completed")

    def mark_failure(self, operation_id: int, error: str, max_retries: int = 3):
        """
        Mark operation as failed and increment retry count.

        Args:
            operation_id: Operation ID
            error: Error message
            max_retries: Maximum retry attempts
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Increment retry count
                cursor = conn.execute(
                    """
                    UPDATE queue
                    SET retry_count = retry_count + 1,
                        last_error = ?
                    WHERE id = ?
                    """,
                    (error, operation_id),
                )

                # Check if max retries exceeded
                cursor = conn.execute(
                    """
                    SELECT retry_count FROM queue WHERE id = ?
                    """,
                    (operation_id,),
                )
                retry_count = cursor.fetchone()[0]

                if retry_count >= max_retries:
                    conn.execute(
                        """
                        UPDATE queue
                        SET status = 'failed'
                        WHERE id = ?
                        """,
                        (operation_id,),
                    )
                    logger.error(
                        f"Operation {operation_id} permanently failed after "
                        f"{retry_count} retries: {error}"
                    )
                else:
                    logger.warning(
                        f"Operation {operation_id} failed (retry {retry_count}/"
                        f"{max_retries}): {error}"
                    )

                conn.commit()

    def get_queue_size(self, status: str = "pending") -> int:
        """
        Get number of operations in queue.

        Args:
            status: Filter by status ('pending', 'completed', 'failed', or None for all)

        Returns:
            Number of operations
        """
        with sqlite3.connect(self.db_path) as conn:
            if status:
                cursor = conn.execute("SELECT COUNT(*) FROM queue WHERE status = ?", (status,))
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM queue")

            return cursor.fetchone()[0]

    def get_statistics(self) -> Dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dictionary with counts by status
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM queue
                GROUP BY status
                """
            )

            stats = {"total": 0, "pending": 0, "completed": 0, "failed": 0}

            for row in cursor.fetchall():
                status, count = row
                stats[status] = count
                stats["total"] += count

            return stats

    def clear_completed(self, older_than_days: int = 7):
        """
        Remove completed operations older than specified days.

        Args:
            older_than_days: Remove operations older than this many days
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM queue
                    WHERE status = 'completed'
                    AND created_at < datetime('now', '-' || ? || ' days')
                    """,
                    (older_than_days,),
                )
                deleted = cursor.rowcount
                conn.commit()

        if deleted > 0:
            logger.info(f"Cleared {deleted} completed operations older than {older_than_days} days")

    def clear_all(self):
        """Clear all operations from queue (use with caution)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM queue")
                conn.commit()

        logger.warning("Cleared all operations from queue")

    def get_failed_operations(self) -> List[Dict[str, Any]]:
        """
        Get all permanently failed operations.

        Returns:
            List of failed operation details
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, operation_type, data, created_at, retry_count, last_error
                FROM queue
                WHERE status = 'failed'
                ORDER BY created_at DESC
                """
            )

            failed = []
            for row in cursor.fetchall():
                op_id, op_type, data_json, created_at, retry_count, last_error = row
                failed.append(
                    {
                        "id": op_id,
                        "operation_type": op_type,
                        "data": json.loads(data_json),
                        "created_at": created_at,
                        "retry_count": retry_count,
                        "last_error": last_error,
                    }
                )

            return failed


class OfflineManager:
    """
    Manages offline mode for WhiteBoxXAI SDK.

    Handles automatic queueing when offline and syncing when online.

    Attributes:
        queue: OfflineQueue instance
        sync_interval: Seconds between automatic sync attempts
        max_retries: Maximum retry attempts for failed operations
    """

    def __init__(
        self,
        offline_dir: str = "./whiteboxxai_offline",
        max_queue_size: int = 10000,
        auto_sync: bool = True,
        sync_interval: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize offline manager.

        Args:
            offline_dir: Directory for offline storage
            max_queue_size: Maximum operations to queue
            auto_sync: Enable automatic syncing
            sync_interval: Seconds between sync attempts
            max_retries: Maximum retry attempts
        """
        self.offline_dir = Path(offline_dir)
        self.offline_dir.mkdir(parents=True, exist_ok=True)

        db_path = str(self.offline_dir / "queue.db")
        self.queue = OfflineQueue(
            db_path=db_path, max_queue_size=max_queue_size, auto_sync=auto_sync
        )

        self.sync_interval = sync_interval
        self.max_retries = max_retries
        self._sync_thread = None
        self._stop_sync = threading.Event()
        self._client = None  # Will be set by WhiteBoxXAI client

        if auto_sync:
            self.start_auto_sync()

    def set_client(self, client):
        """
        Set the WhiteBoxXAI client for syncing.

        Args:
            client: WhiteBoxXAI client instance
        """
        self._client = client

    def start_auto_sync(self):
        """Start automatic sync thread."""
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._stop_sync.clear()
            self._sync_thread = threading.Thread(target=self._auto_sync_loop, daemon=True)
            self._sync_thread.start()
            logger.info("Started automatic sync thread")

    def stop_auto_sync(self):
        """Stop automatic sync thread."""
        if self._sync_thread and self._sync_thread.is_alive():
            self._stop_sync.set()
            self._sync_thread.join(timeout=5)
            logger.info("Stopped automatic sync thread")

    def _auto_sync_loop(self):
        """Background thread for automatic syncing."""
        while not self._stop_sync.is_set():
            try:
                if self._client:
                    self.sync(batch_size=100)
            except Exception as e:
                logger.error(f"Error in auto-sync: {e}")

            # Wait for next sync interval
            self._stop_sync.wait(self.sync_interval)

    def sync(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Sync queued operations with API.

        Args:
            batch_size: Number of operations to sync per batch

        Returns:
            Dictionary with sync statistics
        """
        if not self._client:
            logger.warning("No client set, cannot sync")
            return {"synced": 0, "failed": 0, "pending": self.queue.get_queue_size()}

        stats = {"synced": 0, "failed": 0}

        # Get pending operations
        operations = self.queue.dequeue(limit=batch_size)

        if not operations:
            return {**stats, "pending": 0}

        logger.info(f"Syncing {len(operations)} operations...")

        for op_id, op_type, data in operations:
            try:
                # Execute operation
                if op_type == OperationType.PREDICT:
                    self._client._api_predict(**data)
                elif op_type == OperationType.REGISTER_MODEL:
                    self._client._api_register_model(**data)
                elif op_type == OperationType.UPDATE_BASELINE:
                    self._client._api_update_baseline(**data)
                elif op_type == OperationType.LOG_BATCH:
                    self._client._api_log_batch(**data)

                # Mark success
                self.queue.mark_success(op_id)
                stats["synced"] += 1

            except Exception as e:
                # Mark failure
                error_msg = str(e)
                self.queue.mark_failure(op_id, error_msg, self.max_retries)
                stats["failed"] += 1

        stats["pending"] = self.queue.get_queue_size()

        if stats["synced"] > 0:
            logger.info(
                f"Synced {stats['synced']} operations, {stats['failed']} failed, {stats['pending']} pending"
            )

        return stats

    def get_status(self) -> Dict[str, Any]:
        """
        Get offline mode status.

        Returns:
            Status dictionary with queue stats and sync info
        """
        stats = self.queue.get_statistics()

        return {
            "enabled": True,
            "auto_sync": self.queue.auto_sync,
            "queue_stats": stats,
            "sync_interval": self.sync_interval,
            "offline_dir": str(self.offline_dir),
            "auto_sync_running": self._sync_thread and self._sync_thread.is_alive(),
        }

    def cleanup(self, older_than_days: int = 7):
        """
        Clean up old completed operations.

        Args:
            older_than_days: Remove operations older than this many days
        """
        self.queue.clear_completed(older_than_days)
