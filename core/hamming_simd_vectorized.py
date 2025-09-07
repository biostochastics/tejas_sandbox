"""
Vectorized SIMD-optimized Hamming distance operations.
Provides highly optimized bit packing and Hamming distance computation.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# Pre-compute powers of 2 for different bit widths
POWERS_32 = 2 ** np.arange(32, dtype=np.uint32)
POWERS_64 = 2 ** np.arange(64, dtype=np.uint64)


def pack_bits_uint32_vectorized(bits: np.ndarray) -> np.ndarray:
    """
    Vectorized bit packing into uint32 using dot product.
    10-50x faster than loop-based implementation.

    Args:
        bits: Binary array of shape (..., n_bits) with values 0 or 1

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

    # Reshape for vectorized packing
    bits = bits.reshape(*prefix_shape, n_chunks, 32)

    # Vectorized packing using dot product - no loops!
    packed = bits.astype(np.uint32) @ POWERS_32

    return packed


def pack_bits_uint32_packbits(bits: np.ndarray) -> np.ndarray:
    """
    Alternative packing using numpy.packbits + view casting.
    More memory efficient for very large arrays.

    Args:
        bits: Binary array of shape (..., n_bits) with values 0 or 1

    Returns:
        Packed array of shape (..., ceil(n_bits/32)) with uint32 dtype
    """
    *prefix_shape, n_bits = bits.shape
    n_chunks = (n_bits + 31) // 32

    # Handle empty array
    if bits.size == 0:
        return np.zeros((*prefix_shape, 0), dtype=np.uint32)

    # Pad to multiple of 32
    target_bits = n_chunks * 32
    if n_bits < target_bits:
        padding = target_bits - n_bits
        bits = np.pad(bits, [(0, 0)] * len(prefix_shape) + [(0, padding)])

    # Ensure binary values
    bits = (bits & 1).astype(np.uint8)

    # Reshape for packbits (32 bits = 4 bytes of 8 bits each)
    original_shape = bits.shape[:-1]
    bits = bits.reshape(*original_shape, n_chunks, 4, 8)

    # Pack to bytes
    packed_bytes = np.packbits(bits, axis=-1, bitorder="little")

    # Ensure contiguous and view as uint32
    packed_bytes = np.ascontiguousarray(packed_bytes)
    packed_bytes = packed_bytes.reshape(*original_shape, n_chunks, 4)

    # View as uint32 (little endian)
    packed = packed_bytes.view("<u4")
    packed = packed.reshape(*original_shape, n_chunks)

    return packed


def unpack_bits_uint32_vectorized(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Vectorized unpacking from uint32 to bits.

    Args:
        packed: Packed array of shape (..., n_chunks) with uint32 dtype
        n_bits: Original number of bits

    Returns:
        Binary array of shape (..., n_bits) with values 0 or 1
    """
    *prefix_shape, n_chunks = packed.shape

    # Allocate output
    total_bits = n_chunks * 32
    bits = np.zeros((*prefix_shape, total_bits), dtype=np.uint8)

    # Vectorized unpacking using broadcasting
    for i in range(32):
        bits[..., i::32] = (packed >> i) & 1

    # Trim to original size
    if n_bits < total_bits:
        bits = bits[..., :n_bits]

    return bits


def pack_bits_uint64_vectorized(bits: np.ndarray) -> np.ndarray:
    """
    Vectorized bit packing into uint64 using dot product.

    Args:
        bits: Binary array of shape (..., n_bits) with values 0 or 1

    Returns:
        Packed array of shape (..., ceil(n_bits/64)) with uint64 dtype
    """
    *prefix_shape, n_bits = bits.shape
    n_chunks = (n_bits + 63) // 64

    # Handle empty array
    if bits.size == 0:
        return np.zeros((*prefix_shape, 0), dtype=np.uint64)

    # Pad to multiple of 64 if needed
    if n_bits % 64 != 0:
        padding = 64 - (n_bits % 64)
        bits = np.pad(bits, [(0, 0)] * len(prefix_shape) + [(0, padding)])

    # Reshape for vectorized packing
    bits = bits.reshape(*prefix_shape, n_chunks, 64)

    # Vectorized packing using dot product
    packed = bits.astype(np.uint64) @ POWERS_64

    return packed


def hamming_distance_vectorized(
    packed_a: np.ndarray, packed_b: np.ndarray
) -> np.ndarray:
    """
    Compute Hamming distances using np.bitwise_count (NumPy >= 2.0).
    2-5x faster than lookup table methods.

    Args:
        packed_a: Packed array shape (n_chunks,) or (1, n_chunks)
        packed_b: Packed array shape (m, n_chunks)

    Returns:
        Hamming distances of shape (m,)
    """
    # Ensure 2D for broadcasting
    if packed_a.ndim == 1:
        packed_a = packed_a[None, :]

    # XOR to find differences
    xor_result = packed_a ^ packed_b  # Shape: (m, n_chunks)

    # Use bitwise_count if available (NumPy >= 2.0)
    if hasattr(np, "bitwise_count"):
        distances = np.sum(np.bitwise_count(xor_result), axis=1)
    else:
        # Fallback to byte-wise LUT
        logger.warning("np.bitwise_count not available, using LUT fallback")
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        xor_bytes = xor_result.view(np.uint8)
        distances = np.sum(lut[xor_bytes], axis=1)

    return distances.astype(np.int32)


def search_hamming_vectorized(
    query_packed: np.ndarray, database_packed: np.ndarray, k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k-nearest neighbors using vectorized Hamming distance.

    Args:
        query_packed: Query fingerprint (n_chunks,)
        database_packed: Database fingerprints (n_docs, n_chunks)
        k: Number of nearest neighbors

    Returns:
        indices: Top-k document indices
        distances: Corresponding Hamming distances
    """
    # Compute all distances vectorized
    distances = hamming_distance_vectorized(query_packed, database_packed)

    # Find top-k efficiently
    if k >= len(distances):
        indices = np.arange(len(distances))
        sorted_idx = np.argsort(distances, kind="stable")
        return sorted_idx, distances[sorted_idx]

    # Use argpartition for efficiency with large arrays
    partition_idx = np.argpartition(distances, k)[:k]
    top_k_distances = distances[partition_idx]

    # Sort the top-k
    sorted_within_k = np.argsort(top_k_distances)
    final_indices = partition_idx[sorted_within_k]
    final_distances = top_k_distances[sorted_within_k]

    return final_indices, final_distances


def benchmark_vectorization():
    """Benchmark vectorized vs original implementations."""
    import time
    from .hamming_simd import pack_bits_uint32

    print("=" * 60)
    print("VECTORIZATION BENCHMARK")
    print("=" * 60)

    # Test different array sizes
    sizes = [(100, 128), (1000, 256), (10000, 512)]

    for n_docs, n_bits in sizes:
        print(f"\nArray size: {n_docs} x {n_bits}")
        print("-" * 40)

        # Generate test data
        np.random.seed(42)
        bits = np.random.randint(0, 2, (n_docs, n_bits), dtype=np.uint8)

        # Benchmark original implementation
        start = time.time()
        packed_orig = pack_bits_uint32(bits)
        time_orig = time.time() - start

        # Benchmark vectorized dot product
        start = time.time()
        packed_vec = pack_bits_uint32_vectorized(bits)
        time_vec = time.time() - start

        # Benchmark packbits method
        start = time.time()
        packed_pb = pack_bits_uint32_packbits(bits)
        time_pb = time.time() - start

        # Verify correctness
        np.testing.assert_array_equal(packed_orig, packed_vec)
        np.testing.assert_array_equal(packed_orig, packed_pb)

        # Print results
        print(f"Original (loop):     {time_orig * 1000:8.2f} ms")
        print(
            f"Vectorized (dot):    {time_vec * 1000:8.2f} ms ({time_orig / time_vec:.1f}x faster)"
        )
        print(
            f"Packbits+view:       {time_pb * 1000:8.2f} ms ({time_orig / time_pb:.1f}x faster)"
        )

    # Test Hamming distance
    print("\n" + "=" * 60)
    print("HAMMING DISTANCE BENCHMARK")
    print("=" * 60)

    n_docs = 10000
    n_bits = 256
    bits_db = np.random.randint(0, 2, (n_docs, n_bits), dtype=np.uint8)
    bits_query = np.random.randint(0, 2, n_bits, dtype=np.uint8)

    # Pack using vectorized method
    db_packed = pack_bits_uint32_vectorized(bits_db)
    query_packed = pack_bits_uint32_vectorized(bits_query.reshape(1, -1))[0]

    # Benchmark Hamming distance
    start = time.time()
    indices, distances = search_hamming_vectorized(query_packed, db_packed, k=100)
    time_search = time.time() - start

    throughput = n_docs / time_search
    print(f"\nSearched {n_docs} documents in {time_search * 1000:.2f} ms")
    print(f"Throughput: {throughput:,.0f} comparisons/second")
    print(f"Target (>1M): {'✓ PASS' if throughput > 1_000_000 else '✗ FAIL'}")


if __name__ == "__main__":
    benchmark_vectorization()
