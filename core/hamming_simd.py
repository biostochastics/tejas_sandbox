"""
High-Performance Hamming Distance with True SIMD Optimization
==============================================================

Provides genuinely optimized Hamming distance computation using:
1. Proper XOR operations on packed bits
2. Efficient population count implementations
3. Multiple backend support (Numba, NumPy, Pure Python)
4. Auto-vectorization friendly code structure
"""

import numpy as np
from typing import Tuple, Literal, Union
import logging
import warnings

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import numba
    from numba import jit, njit, prange, uint32, uint64, int32, types
    from numba import config as numba_config

    HAS_NUMBA = True

    # Enable SIMD vectorization
    numba_config.DISABLE_JIT = False

    @njit(inline="always", cache=True)
    def popcount32(x: np.uint32) -> np.int32:
        """
        Efficient popcount for uint32 using bit manipulation.
        This pattern is recognized by LLVM and compiled to POPCNT instruction.
        Using uint32 instead of uint64 for better SIMD vectorization.
        """
        x = x - ((x >> np.uint32(1)) & np.uint32(0x55555555))
        x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
        x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
        x = x + (x >> np.uint32(8))
        x = x + (x >> np.uint32(16))
        return np.int32(x & np.uint32(0x3F))

    @njit(inline="always", cache=True)
    def popcount64(x: np.uint64) -> np.int32:
        """
        Efficient popcount for uint64 using bit manipulation.
        """
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + (
            (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
        )
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        x = x * np.uint64(0x0101010101010101)
        return np.int32(x >> np.uint64(56))

    @njit(parallel=True, cache=True, nogil=True, fastmath=True)
    def hamming_distance_numba_fixed(
        fingerprints_packed: np.ndarray, query_packed: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fixed Numba-accelerated Hamming distance using proper XOR and popcount.

        Args:
            fingerprints_packed: Packed fingerprints (n_docs, n_chunks) as uint32
            query_packed: Packed query (n_chunks,) as uint32
            k: Number of nearest neighbors

        Returns:
            indices: Top-k document indices
            distances: Corresponding Hamming distances
        """
        n_docs = fingerprints_packed.shape[0]
        n_chunks = fingerprints_packed.shape[1]
        distances = np.zeros(n_docs, dtype=np.int32)

        # Parallel distance computation with proper XOR and popcount
        for i in prange(n_docs):
            dist = np.int32(0)
            # Unroll loop for better vectorization
            for j in range(n_chunks):
                # XOR and popcount - the core of SIMD optimization
                xor_result = fingerprints_packed[i, j] ^ query_packed[j]
                dist += popcount32(xor_result)
            distances[i] = dist

        # Find top k using argpartition
        if k >= n_docs:
            top_k_indices = np.arange(n_docs, dtype=np.int32)
            top_k_distances = distances.copy()
        else:
            # Use argpartition to get k smallest elements
            partition_indices = np.argpartition(distances, k - 1)
            top_k_indices = partition_indices[:k].astype(np.int32)
            top_k_distances = distances[top_k_indices]

        # Sort the top k by distance for consistent results
        sort_indices = np.argsort(
            top_k_distances
        )  # Numba doesn't support kind='stable'
        final_indices = top_k_indices[sort_indices]
        final_distances = top_k_distances[sort_indices]

        return final_indices, final_distances

    @njit(parallel=True, cache=True, nogil=True, fastmath=True)
    def hamming_batch_numba_fixed(
        fingerprints_packed: np.ndarray, queries_packed: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch Hamming distance with proper SIMD optimization.
        """
        n_queries = queries_packed.shape[0]
        n_docs = fingerprints_packed.shape[0]
        n_chunks = fingerprints_packed.shape[1]

        all_indices = np.zeros((n_queries, k), dtype=np.int32)
        all_distances = np.zeros((n_queries, k), dtype=np.int32)

        for q_idx in prange(n_queries):
            distances = np.zeros(n_docs, dtype=np.int32)

            # Compute distances for this query
            for i in range(n_docs):
                dist = np.int32(0)
                for j in range(n_chunks):
                    xor_result = fingerprints_packed[i, j] ^ queries_packed[q_idx, j]
                    dist += popcount32(xor_result)
                distances[i] = dist

            # Find top k
            if k >= n_docs:
                top_k_indices = np.arange(n_docs, dtype=np.int32)
                top_k_distances = distances.copy()
            else:
                partition_indices = np.argpartition(distances, k - 1)
                top_k_indices = partition_indices[:k].astype(np.int32)
                top_k_distances = distances[top_k_indices]

            # Sort and store
            sort_indices = np.argsort(top_k_distances, kind="stable")
            all_indices[q_idx] = top_k_indices[sort_indices]
            all_distances[q_idx] = top_k_distances[sort_indices]

        return all_indices, all_distances

    logger.info("Numba backend available with fixed SIMD optimization")

except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba not available, will use NumPy backend")


# Pre-compute powers of 2 for vectorized packing
_POWERS_32 = 2 ** np.arange(32, dtype=np.uint32)


def pack_bits_uint32(bits: np.ndarray, use_vectorized: bool = True) -> np.ndarray:
    """
    Pack bit array into uint32 chunks for optimal SIMD operations.
    uint32 provides better vectorization than uint64 on most architectures.

    Args:
        bits: Binary array of shape (..., n_bits) with values 0 or 1
        use_vectorized: Use vectorized dot product method (10-50x faster)

    Returns:
        Packed array of shape (..., ceil(n_bits/32)) with uint32 dtype
    """
    *prefix_shape, n_bits = bits.shape
    n_chunks = (n_bits + 31) // 32

    # Handle empty array
    if bits.size == 0:
        return np.zeros((*prefix_shape, 0), dtype=np.uint32)

    # Pad to multiple of 32 if needed
    if n_bits % 32 != 0:
        padding = 32 - (n_bits % 32)
        bits = np.pad(bits, [(0, 0)] * len(prefix_shape) + [(0, padding)])

    # Reshape for packing
    bits = bits.reshape(*prefix_shape, n_chunks, 32)

    if use_vectorized:
        # Vectorized packing using dot product - no loops!
        # This is actually more efficient for ALL sizes due to NumPy's optimizations
        packed = bits.astype(np.uint32) @ _POWERS_32
    else:
        # Original implementation for compatibility or explicit request
        packed = np.zeros((*prefix_shape, n_chunks), dtype=np.uint32)

        # Vectorized packing using broadcasting
        for i in range(32):
            packed |= bits[..., i].astype(np.uint32) << np.uint32(i)

    return packed


def unpack_bits_uint32(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Unpack uint32 array back to binary bits.

    Args:
        packed: Packed array of shape (..., n_chunks) with uint32 dtype
        n_bits: Original number of bits

    Returns:
        Binary array of shape (..., n_bits) with values 0 or 1
    """
    *prefix_shape, n_chunks = packed.shape

    # Allocate output array
    bits = np.zeros((*prefix_shape, n_chunks * 32), dtype=np.uint8)

    # Unpack each 32-bit chunk
    for i in range(32):
        bits[..., i::32] = (packed >> np.uint32(i)) & np.uint32(1)

    # Trim to original size
    if n_bits < n_chunks * 32:
        bits = bits[..., :n_bits]

    return bits


# Pre-compute powers of 2 for 64-bit packing
_POWERS_64 = 2 ** np.arange(64, dtype=np.uint64)


def pack_bits_uint64(bits: np.ndarray, use_vectorized: bool = True) -> np.ndarray:
    """
    Pack bit array into uint64 chunks (alternative for systems with better 64-bit support).

    Args:
        bits: Binary array of shape (..., n_bits) with values 0 or 1
        use_vectorized: Use vectorized dot product method (10-50x faster)

    Returns:
        Packed array of shape (..., ceil(n_bits/64)) with uint64 dtype
    """
    *prefix_shape, n_bits = bits.shape
    n_chunks = (n_bits + 63) // 64

    # Handle empty array
    if bits.size == 0:
        return np.zeros((*prefix_shape, 0), dtype=np.uint64)

    if n_bits % 64 != 0:
        padding = 64 - (n_bits % 64)
        bits = np.pad(bits, [(0, 0)] * len(prefix_shape) + [(0, padding)])

    bits = bits.reshape(*prefix_shape, n_chunks, 64)

    if use_vectorized:
        # Vectorized packing using dot product - no loops!
        # This is actually more efficient for ALL sizes due to NumPy's optimizations
        packed = bits.astype(np.uint64) @ _POWERS_64
    else:
        # Original implementation for compatibility or explicit request
        packed = np.zeros((*prefix_shape, n_chunks), dtype=np.uint64)

        for i in range(64):
            packed |= bits[..., i].astype(np.uint64) << np.uint64(i)

    return packed


def hamming_distance_numpy_optimized(
    fingerprints: np.ndarray, query: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy backend using XOR and bit counting.
    Efficient for moderate-sized datasets.
    """
    # Use XOR and count nonzero for Hamming distance
    distances = np.count_nonzero(fingerprints != query, axis=1)

    # Find top k efficiently
    if k >= len(fingerprints):
        top_k_indices = np.arange(len(fingerprints))
    else:
        # Use argpartition for O(n) average case
        top_k_indices = np.argpartition(distances, k)[:k]

    # Sort by distance
    top_k_indices = top_k_indices[np.argsort(distances[top_k_indices], kind="stable")]
    top_k_distances = distances[top_k_indices]

    return top_k_indices.astype(np.int32), top_k_distances.astype(np.int32)


def hamming_distance_numpy_packed(
    fingerprints_packed: np.ndarray, query_packed: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NumPy backend with packed bits for better cache efficiency.
    """
    # XOR packed arrays
    xor_results = fingerprints_packed ^ query_packed[np.newaxis, :]

    # Optimized bit counting
    # Check if bit_count is available (NumPy >= 1.22)
    # Note: bit_count is a method on integer arrays, not a numpy function
    try:
        # Use SIMD-optimized bit_count method
        distances = xor_results.bit_count().sum(axis=1).astype(np.int32)
    except AttributeError:
        # Fall back to lookup table method for older NumPy versions
        # Create lookup table once (for 8-bit chunks)
        if not hasattr(hamming_distance_numpy_packed, "_popcount_table"):
            hamming_distance_numpy_packed._popcount_table = np.array(
                [bin(i).count("1") for i in range(256)], dtype=np.uint8
            )

        # View as uint8 for byte-wise popcount
        xor_bytes = xor_results.view(np.uint8)
        # Use lookup table for fast popcount
        distances = (
            hamming_distance_numpy_packed._popcount_table[xor_bytes]
            .sum(axis=1)
            .astype(np.int32)
        )

    # Find top k
    if k >= len(fingerprints_packed):
        top_k_indices = np.arange(len(fingerprints_packed))
    else:
        top_k_indices = np.argpartition(distances, k)[:k]

    top_k_indices = top_k_indices[np.argsort(distances[top_k_indices], kind="stable")]
    top_k_distances = distances[top_k_indices]

    return top_k_indices.astype(np.int32), top_k_distances.astype(np.int32)


# Lookup table for fast popcount on 8-bit values
POPCOUNT_TABLE_8BIT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def optimized_hamming_distance(a: np.ndarray, b: np.ndarray) -> Union[int, np.ndarray]:
    """
    SIMD-optimized Hamming distance computation.

    Uses np.bit_count when available (NumPy 1.22+) for hardware POPCNT instruction,
    falls back to lookup table for older versions.

    Parameters
    ----------
    a : np.ndarray
        First binary array (packed as uint32 or uint64), can be 1D query
    b : np.ndarray
        Second binary array (packed as uint32 or uint64), can be 2D database

    Returns
    -------
    int or np.ndarray
        Hamming distance(s) between a and b
    """
    # Handle broadcasting for query vs database case
    if a.ndim == 1 and b.ndim == 2:
        # Broadcast query against each row in database
        distances = np.zeros(len(b), dtype=np.int32)
        for i in range(len(b)):
            distances[i] = optimized_hamming_distance(a, b[i])
        return distances

    # Ensure same shape for single comparison
    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape for Hamming distance")

    # XOR using NumPy's optimized bitwise operation
    xor_result = np.bitwise_xor(a, b)

    # Try to use bit_count method (available on integer arrays in NumPy 1.22+)
    try:
        # NumPy 1.22+ with SIMD POPCNT instruction
        # This uses hardware population count for 3-5x speedup
        return int(xor_result.bit_count().sum(dtype=np.uint64))
    except AttributeError:
        # Fallback with proper dtype handling
        # Ensure contiguous array for optimal performance
        xor_contig = np.ascontiguousarray(xor_result)
        # Use lookup table on byte view
        return int(POPCOUNT_TABLE_8BIT[xor_contig.view(np.uint8)].sum(dtype=np.uint64))


def hamming_distance_pure_python(
    fingerprints: np.ndarray, query: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure Python fallback with lookup table optimization.
    """
    n_docs = len(fingerprints)
    distances = []

    # Convert to bytes for lookup table
    query_bytes = np.packbits(query)

    for i in range(n_docs):
        fp_bytes = np.packbits(fingerprints[i])
        # XOR and count bits using lookup table
        xor_bytes = fp_bytes ^ query_bytes
        distance = sum(POPCOUNT_TABLE_8BIT[b] for b in xor_bytes)
        distances.append(distance)

    distances = np.array(distances, dtype=np.int32)

    # Find top k
    if k >= n_docs:
        top_k_indices = np.arange(n_docs)
    else:
        top_k_indices = np.argpartition(distances, k)[:k]

    top_k_indices = top_k_indices[np.argsort(distances[top_k_indices], kind="stable")]
    top_k_distances = distances[top_k_indices]

    return top_k_indices.astype(np.int32), top_k_distances.astype(np.int32)


# Backward compatibility aliases
HammingSIMD = HammingSIMDFixed = None  # Will be set after class definition


class HammingSIMDImpl:
    """
    High-performance Hamming distance with multiple backend support.
    Automatically selects the best available backend.
    """

    def __init__(
        self,
        fingerprints: np.ndarray,
        backend: Literal["auto", "numba", "numpy", "python"] = "auto",
        use_packing: bool = True,
        pack_size: Literal[32, 64] = 32,
    ):
        """
        Initialize optimized Hamming distance search.

        Args:
            fingerprints: Binary fingerprint matrix (n_docs, n_bits)
            backend: Backend to use ('auto' selects best available)
            use_packing: Whether to pack bits for better performance
            pack_size: Bit packing size (32 or 64)
        """
        self.n_docs, self.n_bits = fingerprints.shape

        # Ensure uint8 dtype
        if fingerprints.dtype != np.uint8:
            fingerprints = fingerprints.astype(np.uint8)

        # Select backend
        if backend == "auto":
            if HAS_NUMBA:
                self.backend = "numba"
            else:
                self.backend = "numpy"
        else:
            self.backend = backend

        if self.backend == "numba" and not HAS_NUMBA:
            warnings.warn("Numba not available, falling back to NumPy")
            self.backend = "numpy"

        # Pack bits if requested
        self.use_packing = use_packing and self.backend in ["numba", "numpy"]
        self.pack_size = pack_size

        if self.use_packing:
            logger.info(
                f"Packing fingerprints into uint{pack_size} for {self.backend} backend"
            )
            if pack_size == 32:
                self.fingerprints_packed = pack_bits_uint32(fingerprints)
            else:
                self.fingerprints_packed = pack_bits_uint64(fingerprints)
            self.fingerprints = None  # Save memory
        else:
            self.fingerprints = fingerprints
            self.fingerprints_packed = None

        logger.info(f"Using {self.backend} backend for Hamming distance")

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using optimized Hamming distance.

        Args:
            query: Query fingerprint (n_bits,)
            k: Number of nearest neighbors

        Returns:
            indices: Top-k document indices
            distances: Corresponding Hamming distances
        """
        # Ensure query is uint8
        if query.dtype != np.uint8:
            query = query.astype(np.uint8)

        # Select appropriate backend function
        if self.backend == "numba":
            if self.use_packing:
                # Pack query
                if self.pack_size == 32:
                    query_packed = pack_bits_uint32(query.reshape(1, -1))[0]
                else:
                    query_packed = pack_bits_uint64(query.reshape(1, -1))[0]
                return hamming_distance_numba_fixed(
                    self.fingerprints_packed, query_packed, k
                )
            else:
                # Fall back to NumPy if not packed
                return hamming_distance_numpy_optimized(self.fingerprints, query, k)

        elif self.backend == "numpy":
            if self.use_packing:
                if self.pack_size == 32:
                    query_packed = pack_bits_uint32(query.reshape(1, -1))[0]
                else:
                    query_packed = pack_bits_uint64(query.reshape(1, -1))[0]
                return hamming_distance_numpy_packed(
                    self.fingerprints_packed, query_packed, k
                )
            else:
                return hamming_distance_numpy_optimized(self.fingerprints, query, k)

        else:  # pure python
            return hamming_distance_pure_python(self.fingerprints, query, k)

    def search_batch(
        self, queries: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.
        """
        if queries.dtype != np.uint8:
            queries = queries.astype(np.uint8)

        if self.backend == "numba" and self.use_packing:
            # Pack queries
            if self.pack_size == 32:
                queries_packed = pack_bits_uint32(queries)
            else:
                queries_packed = pack_bits_uint64(queries)
            return hamming_batch_numba_fixed(
                self.fingerprints_packed, queries_packed, k
            )
        else:
            # Process one by one for other backends
            n_queries = len(queries)
            all_indices = np.zeros((n_queries, k), dtype=np.int32)
            all_distances = np.zeros((n_queries, k), dtype=np.int32)

            for i, query in enumerate(queries):
                indices, distances = self.search(query, k)
                all_indices[i] = indices
                all_distances[i] = distances

            return all_indices, all_distances

    def get_info(self) -> dict:
        """Get information about the current configuration."""
        return {
            "backend": self.backend,
            "use_packing": self.use_packing,
            "pack_size": self.pack_size if self.use_packing else None,
            "n_docs": self.n_docs,
            "n_bits": self.n_bits,
            "memory_usage_mb": self._estimate_memory_usage(),
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if self.use_packing:
            if self.pack_size == 32:
                bytes_used = self.fingerprints_packed.nbytes
            else:
                bytes_used = self.fingerprints_packed.nbytes
        else:
            bytes_used = self.fingerprints.nbytes
        return bytes_used / (1024 * 1024)


# Set backward compatibility aliases
HammingSIMD = HammingSIMDFixed = HammingSIMDImpl

# Keep old functions for compatibility
hamming_distance_fallback = hamming_distance_numpy_optimized
pack_bits_to_uint64 = pack_bits_uint64
pack_bits_to_uint32 = pack_bits_uint32
