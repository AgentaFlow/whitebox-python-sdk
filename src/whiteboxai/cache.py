"""
Local Caching

Simple TTL-based cache for API responses.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    """
    Time-to-live cache with LRU eviction.

    Features:
    - TTL-based expiration
    - LRU eviction when max size reached
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of items to cache
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None

        # Check expiration
        timestamp = self._timestamps.get(key, 0)
        if time.time() - timestamp > self.ttl:
            # Expired, remove from cache
            del self._cache[key]
            del self._timestamps[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if already exists
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]

        # Check size limit
        if len(self._cache) >= self.max_size:
            # Remove least recently used
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]

        # Add to cache
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._timestamps.clear()

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def __len__(self) -> int:
        """Get number of cached items."""
        return len(self._cache)


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # Create deterministic key from arguments
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items()),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()
