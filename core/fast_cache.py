"""
Fast Cache Implementation with XXHash
=====================================

High-performance content-addressed caching using xxhash3 for 10-50x speedup
over SHA256. Safe for mutable arrays through content hashing with metadata.

Key features:
- XXHash3 for fast non-cryptographic hashing
- Zero-copy operations using memoryview
- LRU eviction when cache exceeds capacity
- Optional thread safety
- Feature flag support for gradual rollout
"""

import os
import threading
from collections import OrderedDict
from typing import Optional, Any
import numpy as np
import logging

# Feature flag for using XXHash
USE_XXHASH = os.getenv("TEJAS_FAST_CACHE", "1") == "1"

# Try to import xxhash, fallback to SHA256 if not available
try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    import hashlib

logger = logging.getLogger(__name__)


def _as_contiguous_memoryview(arr: np.ndarray) -> memoryview:
    """
    Get a zero-copy memoryview of array data.
    Makes array contiguous if needed to ensure correct hashing.

    Args:
        arr: NumPy array to view

    Returns:
        Memoryview of contiguous array data
    """
    # Ensure C-contiguous for consistent hashing
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    # Create memoryview without copying
    return memoryview(arr)


def fast_content_hash(arr: np.ndarray) -> str:
    """
    Fast content hash of numpy array using XXHash3 or SHA256 fallback.

    Args:
        arr: NumPy array to hash

    Returns:
        Hex string hash of array content
    """
    # Get zero-copy view of array
    mv = _as_contiguous_memoryview(arr)

    if USE_XXHASH and XXHASH_AVAILABLE:
        # Use XXHash3 64-bit for speed (10-50x faster than SHA256)
        return xxhash.xxh3_64(mv).hexdigest()
    else:
        # Fallback to SHA256 (slower but always available)
        # Only use first 16 bytes for compatibility with 64-bit XXHash
        return hashlib.sha256(mv).hexdigest()[:16]


def make_array_key(arr: np.ndarray, prefix: str = "") -> str:
    """
    Create comprehensive cache key for numpy array including metadata.

    Includes shape, dtype, strides, and content hash to ensure:
    - Different views of same data have different keys
    - Modified arrays get new keys
    - No false cache hits from metadata changes

    Args:
        arr: NumPy array to create key for
        prefix: Optional prefix for key namespacing

    Returns:
        String key uniquely identifying array content and metadata
    """
    # Include all metadata that affects array interpretation
    metadata_parts = [
        prefix,
        str(arr.shape),
        str(arr.dtype),
        str(arr.strides),
        str(arr.flags["C_CONTIGUOUS"]),
        str(arr.nbytes),
    ]

    # Get content hash
    content_hash = fast_content_hash(arr)

    # Combine metadata and content hash
    metadata = "_".join(filter(None, metadata_parts))
    return f"{metadata}_{content_hash}"


class FastCache:
    """
    High-performance LRU cache with XXHash content addressing.

    Features:
    - Content-based caching safe for mutable arrays
    - LRU eviction when capacity exceeded
    - Optional thread safety
    - Zero-copy operations where possible

    Example:
        cache = FastCache(capacity=100, thread_safe=True)
        key = cache.get_key(my_array)
        result = cache.get(my_array)
        if result is None:
            result = expensive_computation(my_array)
            cache.set(my_array, result)
    """

    def __init__(
        self, capacity: int = None, max_size: int = None, thread_safe: bool = False
    ):
        """
        Initialize FastCache.

        Args:
            capacity: Maximum number of items to cache (preferred)
            max_size: Alias for capacity (for compatibility)
            thread_safe: Enable thread-safe operations
        """
        # Handle both capacity and max_size for compatibility
        if capacity is None and max_size is None:
            capacity = 100
        elif capacity is None:
            capacity = max_size

        self.capacity = max(1, capacity)
        self._cache = OrderedDict()
        self._lock = threading.RLock() if thread_safe else None

        # Metrics for monitoring
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        if USE_XXHASH and XXHASH_AVAILABLE:
            logger.info(f"FastCache initialized with XXHash3 (capacity={capacity})")
        else:
            logger.warning("FastCache using SHA256 fallback (XXHash not available)")

    def get_key(self, array: np.ndarray, prefix: str = "") -> str:
        """
        Get cache key for array.

        Args:
            array: NumPy array to get key for
            prefix: Optional prefix for namespacing

        Returns:
            Cache key string
        """
        return make_array_key(array, prefix)

    def get(self, array_or_key, prefix: str = "", default=None) -> Optional[Any]:
        """
        Get cached value for array or key.

        Args:
            array_or_key: NumPy array or string key to look up
            prefix: Optional prefix used when storing (ignored if key is string)
            default: Default value if not found

        Returns:
            Cached value or default if not found
        """
        # Handle both array and string keys
        if isinstance(array_or_key, str):
            key = array_or_key
        elif isinstance(array_or_key, np.ndarray):
            key = self.get_key(array_or_key, prefix)
        else:
            # Try to convert to string for simple types
            key = str(array_or_key)

        if self._lock:
            with self._lock:
                result = self._get_unsafe(key)
        else:
            result = self._get_unsafe(key)

        return result if result is not None else default

    def _get_unsafe(self, key: str) -> Optional[Any]:
        """Internal get without locking."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        else:
            self.misses += 1
            return None

    def set(self, array_or_key, value: Any, prefix: str = "") -> None:
        """
        Store value in cache for array or key.

        Args:
            array_or_key: NumPy array or string key
            value: Value to cache
            prefix: Optional prefix for namespacing (ignored if key is string)
        """
        # Handle both array and string keys
        if isinstance(array_or_key, str):
            key = array_or_key
        elif isinstance(array_or_key, np.ndarray):
            key = self.get_key(array_or_key, prefix)
        else:
            # Try to convert to string for simple types
            key = str(array_or_key)

        if self._lock:
            with self._lock:
                self._set_unsafe(key, value)
        else:
            self._set_unsafe(key, value)

    def _set_unsafe(self, key: str, value: Any) -> None:
        """Internal set without locking."""
        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            # Add new entry
            self._cache[key] = value

            # Evict oldest if over capacity
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        """Clear all cached items."""
        if self._lock:
            with self._lock:
                self._cache.clear()
        else:
            self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached items."""
        if self._lock:
            with self._lock:
                return len(self._cache)
        else:
            return len(self._cache)

    def get_stats(self) -> dict:
        """
        Get cache performance statistics.

        Returns:
            Dict with hits, misses, hit rate, size, evictions
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "capacity": self.capacity,
            "evictions": self.evictions,
            "using_xxhash": USE_XXHASH and XXHASH_AVAILABLE,
        }

    def __contains__(self, array: np.ndarray) -> bool:
        """Check if array is in cache."""
        key = self.get_key(array)
        return key in self._cache


# Convenience functions for drop-in replacement
def create_cache(capacity: int = 100, thread_safe: bool = False) -> FastCache:
    """
    Create a new FastCache instance.

    Args:
        capacity: Maximum cache size
        thread_safe: Enable thread safety

    Returns:
        FastCache instance
    """
    return FastCache(capacity=capacity, thread_safe=thread_safe)


# Global default cache instance for simple usage
_default_cache = None


def get_default_cache() -> FastCache:
    """Get or create default global cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FastCache(capacity=1000, thread_safe=True)
    return _default_cache
