#!/usr/bin/env python3
"""
Fixed Parallel Search without Numba JIT compilation issues.
Uses threading and numpy for parallelization instead of numba.
"""

import numpy as np
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for parallel search."""

    n_threads: int = 4
    chunk_size: int = 1000
    use_simd: bool = True
    prefetch: bool = False
    cache_friendly: bool = True
    batch_size: int = 32


def hamming_distance_batch_numpy(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Calculate Hamming distances using numpy (no numba).

    Args:
        query: Query fingerprint (n_bits,)
        database: Database fingerprints (n_samples, n_bits)

    Returns:
        Array of Hamming distances
    """
    # Use XOR and count differing bits
    xor_result = np.logical_xor(query, database)
    distances = np.sum(xor_result, axis=1)
    return distances.astype(np.int32)


def search_chunk(
    query: np.ndarray, chunk: np.ndarray, chunk_start_idx: int, k: int
) -> List[Tuple[int, int]]:
    """
    Search within a chunk of the database.

    Args:
        query: Query fingerprint
        chunk: Chunk of database
        chunk_start_idx: Starting index in full database
        k: Number of nearest neighbors

    Returns:
        List of (index, distance) tuples
    """
    distances = hamming_distance_batch_numpy(query, chunk)

    # Get top k within this chunk
    chunk_k = min(k, len(distances))
    top_k_indices = np.argpartition(distances, chunk_k - 1)[:chunk_k]

    # Convert to global indices
    results = [(chunk_start_idx + idx, distances[idx]) for idx in top_k_indices]

    # Sort by distance
    results.sort(key=lambda x: x[1])

    return results


class ParallelSearchOptimized:
    """
    Optimized parallel search without numba dependencies.
    """

    def __init__(
        self,
        n_threads: int = 4,
        chunk_size: int = 1000,
        use_simd: bool = True,
        prefetch: bool = False,
        cache_friendly: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Initialize parallel search.

        Args:
            n_threads: Number of threads
            chunk_size: Size of chunks for parallel processing
            use_simd: Use SIMD optimizations (numpy)
            prefetch: Prefetch data (not implemented)
            cache_friendly: Use cache-friendly access patterns
            batch_size: Batch size for batch operations
        """
        self.n_threads = max(1, n_threads)
        self.chunk_size = max(100, chunk_size)
        self.use_simd = use_simd
        self.prefetch = prefetch
        self.cache_friendly = cache_friendly
        self.batch_size = batch_size

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        logger.info(f"Initialized ParallelSearch with {self.n_threads} threads")

    def search(
        self, query: np.ndarray, database: np.ndarray, k: int = 10
    ) -> List[Tuple[int, int]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query fingerprint (n_bits,)
            database: Database fingerprints (n_samples, n_bits)
            k: Number of nearest neighbors

        Returns:
            List of (index, distance) tuples
        """
        n_samples = len(database)

        # Single-threaded for small databases
        if n_samples <= self.chunk_size or self.n_threads == 1:
            return self._search_single_thread(query, database, k)

        # Multi-threaded search
        return self._search_multi_thread(query, database, k)

    def _search_single_thread(
        self, query: np.ndarray, database: np.ndarray, k: int
    ) -> List[Tuple[int, int]]:
        """Single-threaded search."""
        distances = hamming_distance_batch_numpy(query, database)

        # Get top k
        k = min(k, len(distances))
        top_k_indices = np.argpartition(distances, k - 1)[:k]

        results = [(int(idx), int(distances[idx])) for idx in top_k_indices]
        results.sort(key=lambda x: x[1])

        return results

    def _search_multi_thread(
        self, query: np.ndarray, database: np.ndarray, k: int
    ) -> List[Tuple[int, int]]:
        """Multi-threaded search."""
        n_samples = len(database)

        # Calculate chunks
        chunks = []
        for i in range(0, n_samples, self.chunk_size):
            chunk_end = min(i + self.chunk_size, n_samples)
            chunks.append((i, database[i:chunk_end]))

        # Search in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            for chunk_start, chunk_data in chunks:
                future = executor.submit(
                    search_chunk, query, chunk_data, chunk_start, k
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk search failed: {e}")

        # Merge and get final top k
        all_results.sort(key=lambda x: x[1])
        return all_results[:k]

    def batch_search(
        self, queries: np.ndarray, database: np.ndarray, k: int = 10
    ) -> List[List[Tuple[int, int]]]:
        """
        Batch search for multiple queries.

        Args:
            queries: Query fingerprints (n_queries, n_bits)
            database: Database fingerprints (n_samples, n_bits)
            k: Number of nearest neighbors

        Returns:
            List of result lists
        """
        results = []
        for query in queries:
            results.append(self.search(query, database, k))
        return results

    def _calculate_chunks(self, n_samples: int) -> List[Tuple[int, int]]:
        """Calculate chunk boundaries."""
        chunks = []
        for i in range(0, n_samples, self.chunk_size):
            chunk_end = min(i + self.chunk_size, n_samples)
            chunks.append((i, chunk_end))
        return chunks

    def _search_chunk(self, args):
        """Search within a chunk (for compatibility)."""
        query, chunk, chunk_start_idx, k = args
        return search_chunk(query, chunk, chunk_start_idx, k)


# Compatibility function
def create_parallel_search(n_threads: int = 4, **kwargs) -> ParallelSearchOptimized:
    """Create a parallel search instance."""
    return ParallelSearchOptimized(n_threads=n_threads, **kwargs)
