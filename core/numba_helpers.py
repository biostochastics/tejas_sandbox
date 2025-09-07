"""
Numba JIT-compiled helper functions for performance optimization.
Falls back to pure NumPy if Numba is not available.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import numba
try:
    import numba
    from numba import jit, prange

    NUMBA_AVAILABLE = True
    logger.info("Numba JIT compilation available - performance optimizations enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba not available - using pure NumPy implementations")

    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range


# ============================================================================
# Binarization Functions
# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def binarize_cyclic_numba(X_normalized: np.ndarray, target_bits: int) -> np.ndarray:
    """
    Numba-optimized cyclic tiling binarization.

    Args:
        X_normalized: Normalized input data (n_samples, n_features)
        target_bits: Target number of bits

    Returns:
        Binary codes (n_samples, target_bits)
    """
    n_samples, n_features = X_normalized.shape
    binary = np.zeros((n_samples, target_bits), dtype=np.uint8)

    for i in prange(n_samples):
        for j in range(target_bits):
            source_idx = j % n_features
            binary[i, j] = 1 if X_normalized[i, source_idx] > 0 else 0

    return binary


@jit(nopython=True, parallel=True, cache=True)
def pack_bits_uint32_numba(binary: np.ndarray) -> np.ndarray:
    """
    Numba-optimized bit packing into uint32 arrays.

    Args:
        binary: Binary codes (n_samples, n_bits) as uint8

    Returns:
        Packed codes (n_samples, ceil(n_bits/32)) as uint32
    """
    n_samples, n_bits = binary.shape
    n_uint32 = (n_bits + 31) // 32
    packed = np.zeros((n_samples, n_uint32), dtype=np.uint32)

    for i in prange(n_samples):
        for j in range(n_uint32):
            value = np.uint32(0)
            start_bit = j * 32
            end_bit = min(start_bit + 32, n_bits)

            for k in range(start_bit, end_bit):
                if binary[i, k]:
                    bit_position = k - start_bit
                    value |= np.uint32(1) << np.uint32(bit_position)

            packed[i, j] = value

    return packed


@jit(nopython=True, parallel=True, cache=True)
def unpack_bits_uint32_numba(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Numba-optimized bit unpacking from uint32 arrays.

    Args:
        packed: Packed codes (n_samples, n_uint32) as uint32
        n_bits: Number of bits to unpack

    Returns:
        Binary codes (n_samples, n_bits) as uint8
    """
    n_samples = packed.shape[0]
    binary = np.zeros((n_samples, n_bits), dtype=np.uint8)

    for i in prange(n_samples):
        for j in range(packed.shape[1]):
            start_bit = j * 32
            end_bit = min(start_bit + 32, n_bits)
            value = packed[i, j]

            for k in range(start_bit, end_bit):
                bit_position = k - start_bit
                binary[i, k] = 1 if (value >> bit_position) & 1 else 0

    return binary


# ============================================================================
# Distance Functions
# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def hamming_distance_packed_numba(
    query: np.ndarray, database: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized Hamming distance for packed uint32 codes.

    Args:
        query: Single query (n_uint32,) as uint32
        database: Database codes (n_samples, n_uint32) as uint32

    Returns:
        Hamming distances (n_samples,)
    """
    n_samples = database.shape[0]
    n_uint32 = database.shape[1]
    distances = np.zeros(n_samples, dtype=np.int32)

    for i in prange(n_samples):
        dist = 0
        for j in range(n_uint32):
            xor = query[j] ^ database[i, j]
            # Count bits (popcount)
            while xor:
                dist += 1
                xor &= xor - 1
        distances[i] = dist

    return distances


@jit(nopython=True, cache=True)
def hamming_search_packed_numba(
    query: np.ndarray, database: np.ndarray, k: int
) -> tuple:
    """
    Numba-optimized k-NN search using Hamming distance on packed codes.

    Args:
        query: Single query (n_uint32,) as uint32
        database: Database codes (n_samples, n_uint32) as uint32
        k: Number of nearest neighbors

    Returns:
        Tuple of (indices, distances) for k nearest neighbors
    """
    distances = hamming_distance_packed_numba(query, database)

    # Get k smallest distances (partial sort)
    k_actual = min(k, len(distances))
    indices = np.argpartition(distances, k_actual - 1)[:k_actual]

    # Sort the k results
    sorted_idx = np.argsort(distances[indices])
    indices = indices[sorted_idx]

    return indices, distances[indices]


# ============================================================================
# Matrix Operations
# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def normalize_rows_numba(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Numba-optimized L2 row normalization.

    Args:
        X: Input matrix (n_samples, n_features)
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized matrix
    """
    n_samples = X.shape[0]
    X_norm = np.empty_like(X)

    for i in prange(n_samples):
        norm = 0.0
        for j in range(X.shape[1]):
            norm += X[i, j] * X[i, j]
        norm = np.sqrt(norm) + epsilon

        for j in range(X.shape[1]):
            X_norm[i, j] = X[i, j] / norm

    return X_norm


# ============================================================================
# Fallback Functions (when Numba not available)
# ============================================================================


def binarize_cyclic_numpy(X_normalized: np.ndarray, target_bits: int) -> np.ndarray:
    """Pure NumPy cyclic tiling binarization."""
    n_samples, n_features = X_normalized.shape
    binary = np.zeros((n_samples, target_bits), dtype=np.uint8)

    for i in range(target_bits):
        source_idx = i % n_features
        binary[:, i] = (X_normalized[:, source_idx] > 0).astype(np.uint8)

    return binary


def pack_bits_uint32_numpy(binary: np.ndarray) -> np.ndarray:
    """Pure NumPy bit packing into uint32."""
    n_samples, n_bits = binary.shape
    n_uint32 = (n_bits + 31) // 32
    packed = np.zeros((n_samples, n_uint32), dtype=np.uint32)

    for i in range(0, n_bits, 32):
        byte_idx = i // 32
        bits_to_pack = min(32, n_bits - i)
        for j in range(bits_to_pack):
            if i + j < n_bits:
                packed[:, byte_idx] |= binary[:, i + j].astype(np.uint32) << j

    return packed


def hamming_distance_packed_numpy(
    query: np.ndarray, database: np.ndarray
) -> np.ndarray:
    """Pure NumPy Hamming distance for packed codes."""
    xor = np.bitwise_xor(database, query)
    # Count bits using Brian Kernighan's algorithm
    distances = np.zeros(database.shape[0], dtype=np.int32)
    for i in range(database.shape[1]):
        for j in range(32):
            distances += (xor[:, i] >> j) & 1
    return distances


# ============================================================================
# Exported Functions (automatically use Numba when available)
# ============================================================================


def binarize_cyclic(X_normalized: np.ndarray, target_bits: int) -> np.ndarray:
    """Cyclic tiling binarization with automatic Numba acceleration."""
    if NUMBA_AVAILABLE:
        return binarize_cyclic_numba(X_normalized, target_bits)
    return binarize_cyclic_numpy(X_normalized, target_bits)


def pack_bits_uint32(binary: np.ndarray) -> np.ndarray:
    """Bit packing with automatic Numba acceleration."""
    if NUMBA_AVAILABLE:
        return pack_bits_uint32_numba(binary)
    return pack_bits_uint32_numpy(binary)


def hamming_distance_packed(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Hamming distance with automatic Numba acceleration."""
    if NUMBA_AVAILABLE:
        return hamming_distance_packed_numba(query, database)
    return hamming_distance_packed_numpy(query, database)


def hamming_search_packed(query: np.ndarray, database: np.ndarray, k: int) -> tuple:
    """k-NN search with automatic Numba acceleration."""
    if NUMBA_AVAILABLE:
        return hamming_search_packed_numba(query, database, k)
    else:
        distances = hamming_distance_packed_numpy(query, database)
        k_actual = min(k, len(distances))
        indices = np.argpartition(distances, k_actual - 1)[:k_actual]
        sorted_idx = np.argsort(distances[indices])
        indices = indices[sorted_idx]
        return indices, distances[indices]


def normalize_rows(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """L2 row normalization with automatic Numba acceleration."""
    if NUMBA_AVAILABLE:
        return normalize_rows_numba(X, epsilon)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + epsilon)
