"""
High-Performance Bit Operations for Tejas
=========================================

Optimized bit packing and Hamming distance computation with multiple backends.
Achieves >1M comparisons/second through vectorized XOR+popcount operations.

Key Features:
- 8x memory reduction through bit packing
- LUT-based popcount for maximum throughput
- Multiple backends: numpy, numba, native, auto
- SIMD-friendly memory layouts
- Comprehensive edge case handling
"""

import numpy as np
import logging
from typing import Union, Literal, Optional, Tuple
import time
import sys

logger = logging.getLogger(__name__)

# Pre-compute popcount lookup table for all 256 uint8 values
# Note: Popcount is bit-order invariant (bit reversal preserves number of 1s)
POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
    
    @jit(nopython=True, parallel=True)
    def _hamming_distance_numba(query_packed, database_packed, lut):
        """Numba-accelerated Hamming distance computation."""
        n_items, n_bytes = database_packed.shape
        distances = np.zeros(n_items, dtype=np.int32)
        
        for i in prange(n_items):
            dist = 0
            for j in range(n_bytes):
                xor_val = query_packed[j] ^ database_packed[i, j]
                dist += lut[xor_val]
            distances[i] = dist
        return distances
    
    logger.info("Numba backend available for accelerated bit operations")
    
except ImportError:
    HAS_NUMBA = False
    _hamming_distance_numba = None
    logger.info("Numba not available, falling back to numpy backend")


def pack_bits_rows(binary_matrix: np.ndarray, 
                   bitorder: Literal['little', 'big'] = 'little',
                   n_bits: Optional[int] = None) -> np.ndarray:
    """
    Pack binary matrix to uint8 bytes with specified bit ordering.
    
    Args:
        binary_matrix: (N, M) bool/int array with values {0, 1}
        bitorder: 'little' or 'big' endian bit packing
        n_bits: Target bits (pad/truncate if needed). If None, use matrix width.
    
    Returns:
        packed: (N, ceil(n_bits/8)) uint8 array
        
    Examples:
        >>> binary = np.array([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
        >>> packed = pack_bits_rows(binary, bitorder='little')
        >>> packed.shape
        (1, 1)
        >>> bin(packed[0, 0])  # With 'little' bitorder, leftmost bit becomes LSB
        '0b10101010'  # 170 in decimal
        
        Note: Bitorder must match between pack/unpack for roundtrip integrity.
    """
    if binary_matrix.size == 0:
        return np.array([], dtype=np.uint8).reshape(0, 0)
    
    if binary_matrix.ndim != 2:
        raise ValueError(f"binary_matrix must be 2D, got {binary_matrix.ndim}D")
    
    n_rows, n_cols = binary_matrix.shape
    
    # Handle target bits
    if n_bits is None:
        n_bits = n_cols
    
    # Prepare matrix for packing
    if n_bits != n_cols:
        if n_bits > n_cols:
            # Pad with zeros
            padding = np.zeros((n_rows, n_bits - n_cols), dtype=binary_matrix.dtype)
            binary_matrix = np.concatenate([binary_matrix, padding], axis=1)
        else:
            # Truncate
            binary_matrix = binary_matrix[:, :n_bits]
    
    # Ensure values are strictly 0 or 1
    binary_matrix = (binary_matrix & 1).astype(np.uint8)
    
    # Pack bits using numpy.packbits
    packed = np.packbits(binary_matrix, axis=1, bitorder=bitorder)
    
    logger.debug(f"Packed {n_rows} rows from {n_bits} bits to {packed.shape[1]} bytes")
    return packed


def unpack_bits_rows(packed_matrix: np.ndarray,
                     bitorder: Literal['little', 'big'] = 'little',
                     n_bits: Optional[int] = None) -> np.ndarray:
    """
    Unpack uint8 bytes back to binary matrix.
    
    Args:
        packed_matrix: (N, n_bytes) uint8 array
        bitorder: 'little' or 'big' (must match packing)
        n_bits: Number of valid bits to extract. If None, extract all.
    
    Returns:
        binary: (N, n_bits) uint8 array with values {0, 1}
        
    Examples:
        >>> packed = np.array([[170]], dtype=np.uint8)  # 0b10101010 in binary
        >>> binary = unpack_bits_rows(packed, bitorder='little', n_bits=8)
        >>> binary[0].tolist()  # With 'little' bitorder, LSB becomes leftmost element
        [0, 1, 0, 1, 0, 1, 0, 1]  # Matches the input from pack_bits_rows example
    """
    if packed_matrix.size == 0:
        return np.array([], dtype=np.uint8).reshape(0, 0)
    
    if packed_matrix.ndim != 2:
        raise ValueError(f"packed_matrix must be 2D, got {packed_matrix.ndim}D")
    
    n_rows, n_bytes = packed_matrix.shape
    
    # Calculate default n_bits if not specified
    if n_bits is None:
        n_bits = n_bytes * 8
    
    # Unpack bits
    unpacked = np.unpackbits(packed_matrix, axis=1, count=n_bits, bitorder=bitorder)
    
    logger.debug(f"Unpacked {n_rows} rows from {n_bytes} bytes to {n_bits} bits")
    return unpacked.astype(np.uint8)


def hamming_distance_packed(query_packed: np.ndarray,
                           database_packed: np.ndarray,
                           backend: Literal['numpy', 'numba', 'native', 'auto'] = 'auto',
                           bitorder: str = 'little') -> np.ndarray:
    """
    Compute Hamming distances between packed fingerprints using optimized backends.
    
    Args:
        query_packed: (n_bits//8,) uint8 query fingerprint
        database_packed: (N, n_bits//8) uint8 database fingerprints
        backend: Backend to use for computation
    
    Returns:
        distances: (N,) int32 Hamming distances
        
    Performance Targets:
        - >1M comparisons/second on consumer hardware
        - Memory-efficient for large databases
        - Vectorized operations where possible
    """
    # Input validation
    if query_packed.ndim != 1:
        raise ValueError(f"query_packed must be 1D, got {query_packed.ndim}D")
    
    if database_packed.ndim != 2:
        raise ValueError(f"database_packed must be 2D, got {database_packed.ndim}D")
    
    if query_packed.shape[0] != database_packed.shape[1]:
        raise ValueError(f"Dimension mismatch: query has {query_packed.shape[0]} bytes, "
                        f"database has {database_packed.shape[1]} bytes per item")
    
    if database_packed.size == 0:
        return np.array([], dtype=np.int32)
    
    # Backend selection
    if backend == 'auto':
        # NumPy is often faster than Numba for this workload due to:
        # 1. Highly optimized BLAS operations
        # 2. Small inner loop (just XOR + LUT lookup)
        # 3. Numba JIT overhead
        # Use Numba only when explicitly requested
        backend = 'numpy'
    
    # Route to appropriate implementation
    start_time = time.time()
    
    if backend == 'numpy':
        distances = _hamming_distance_numpy(query_packed, database_packed, bitorder=bitorder)
    elif backend == 'numba':
        if not HAS_NUMBA:
            raise RuntimeError("Numba backend requested but numba is not installed. "
                             "Please install numba or use 'numpy' or 'native' backend.")
        distances = _hamming_distance_numba(query_packed, database_packed, POPCOUNT_LUT)
    elif backend == 'native':
        distances = _hamming_distance_native(query_packed, database_packed, bitorder=bitorder)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    compute_time = time.time() - start_time
    n_comparisons = len(database_packed)
    throughput = n_comparisons / compute_time if compute_time > 0 else float('inf')
    
    logger.debug(f"Computed {n_comparisons} Hamming distances in {compute_time*1000:.2f}ms "
                f"({throughput:,.0f} comparisons/sec) using {backend} backend")
    
    return distances


def _hamming_distance_numpy(query_packed: np.ndarray, database_packed: np.ndarray, bitorder: str = 'little') -> np.ndarray:
    """Numpy-based Hamming distance calculation for packed arrays."""
    # XOR and popcount (popcount is bitorder-invariant)
    xor_result = query_packed[None, :] ^ database_packed
    distances = np.sum(POPCOUNT_LUT[xor_result], axis=1)
    return distances.astype(np.int32)


def _hamming_distance_native(query_packed: np.ndarray, database_packed: np.ndarray, bitorder: str = 'little') -> np.ndarray:
    """Optimized native Python implementation using int.bit_count() for better performance.
    
    This version groups bytes into 64-bit integers for faster bit counting.
    Python 3.10+ required for int.bit_count() method.
    """
    n_items, n_bytes = database_packed.shape
    distances = np.zeros(n_items, dtype=np.int32)
    
    # Check if we can use int.bit_count() (Python 3.10+)
    try:
        # Group bytes into 64-bit chunks for efficiency
        bytes_per_chunk = 8
        n_chunks = (n_bytes + bytes_per_chunk - 1) // bytes_per_chunk
        
        # Pad arrays to multiples of 8 bytes if needed
        if n_bytes % bytes_per_chunk != 0:
            pad_size = bytes_per_chunk - (n_bytes % bytes_per_chunk)
            query_padded = np.pad(query_packed, (0, pad_size), constant_values=0)
            db_padded = np.pad(database_packed, ((0, 0), (0, pad_size)), constant_values=0)
        else:
            query_padded = query_packed
            db_padded = database_packed
        
        # Ensure contiguous memory layout before view
        query_padded = np.ascontiguousarray(query_padded)
        db_padded = np.ascontiguousarray(db_padded)
        
        # View as uint64 for efficient processing
        query_64 = query_padded.view(np.uint64)
        db_64 = db_padded.view(np.uint64).reshape(n_items, -1)
        
        for i in range(n_items):
            dist = 0
            for j in range(len(query_64)):
                xor_val = int(query_64[j] ^ db_64[i, j])
                dist += xor_val.bit_count()
            distances[i] = dist
            
    except AttributeError:
        # Fallback to LUT-based approach for older Python versions
        for i in range(n_items):
            dist = 0
            for j in range(n_bytes):
                xor_val = query_packed[j] ^ database_packed[i, j]
                dist += POPCOUNT_LUT[xor_val]
            distances[i] = dist
    
    return distances


def search_packed_vectorized(query_packed: np.ndarray, database_packed: np.ndarray, k: int = 10, backend: str = 'auto', bitorder: str = 'little') -> Tuple[np.ndarray, np.ndarray]:
    """Find the k-nearest neighbors in a packed binary database."""
    start_time = time.time()
    
    # Compute all Hamming distances
    distances = hamming_distance_packed(query_packed, database_packed, backend=backend, bitorder=bitorder)
    
    # Top-k selection using stable sort for consistent ordering
    if k >= len(distances):
        # Return all results sorted with stable sorting for consistent ordering
        sort_indices = np.argsort(distances, kind='stable')
        return sort_indices, distances[sort_indices]
    
    # For k < len, use full stable sort then slice
    # This ensures consistent ordering when there are ties
    sort_indices = np.argsort(distances, kind='stable')[:k]
    final_distances = distances[sort_indices]
    
    return sort_indices, final_distances


def validate_packing_roundtrip(binary_matrix: np.ndarray,
                             bitorder: str = 'little',
                             n_bits: Optional[int] = None) -> bool:
    """
    Validate that pack → unpack preserves data integrity.
    
    Args:
        binary_matrix: Test matrix to validate
        bitorder: Bit order for packing
        n_bits: Target bits (None = use matrix width)
    
    Returns:
        True if roundtrip preserves data, False otherwise
    """
    if binary_matrix.size == 0:
        return True
    
    # Ensure binary values
    original = (binary_matrix & 1).astype(np.uint8)
    
    try:
        # Pack and unpack
        packed = pack_bits_rows(original, bitorder=bitorder, n_bits=n_bits)
        unpacked = unpack_bits_rows(packed, bitorder=bitorder, 
                                  n_bits=n_bits or original.shape[1])
        
        # Compare (handle potential shape differences due to n_bits)
        min_cols = min(original.shape[1], unpacked.shape[1])
        return np.array_equal(original[:, :min_cols], unpacked[:, :min_cols])
        
    except Exception as e:
        logger.error(f"Roundtrip validation failed: {e}")
        return False


def benchmark_backends(n_items: int = 50000, n_bits: Optional[Union[int, list]] = None) -> dict:
    """
    Benchmark different backends for performance comparison.
    
    Args:
        n_items: Number of database items
        n_bits: Bits per fingerprint (single value or list for multiple tests)
    
    Returns:
        Performance results for each backend and bit configuration
    """
    if n_bits is None:
        n_bits = [64, 128, 256, 512, 1024]
    elif isinstance(n_bits, int):
        n_bits = [n_bits]
    
    results = {}
    backends = ['numpy', 'native']
    if HAS_NUMBA:
        backends.append('numba')
    
    for bits in n_bits:
        logger.info(f"Benchmarking backends with {n_items} items, {bits} bits each")
        n_bytes = (bits + 7) // 8
        
        # Generate test data
        database_packed = np.random.randint(0, 256, (n_items, n_bytes), dtype=np.uint8)
        query_packed = np.random.randint(0, 256, n_bytes, dtype=np.uint8)
        
        results[f"{bits}_bits"] = {}
        
        for backend in backends:
            try:
                # Warmup
                _ = hamming_distance_packed(query_packed, database_packed[:100], backend=backend)
                
                # Multiple runs for stability
                times = []
                for _ in range(3):
                    start_time = time.time()
                    distances = hamming_distance_packed(query_packed, database_packed, backend=backend)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                
                # Use median time for stability
                median_time = np.median(times)
                throughput = n_items / median_time
                
                results[f"{bits}_bits"][backend] = {
                    'elapsed_ms': median_time * 1000,
                    'throughput_per_sec': throughput,
                    'memory_mb': database_packed.nbytes / (1024 * 1024),
                    'p25_ms': np.percentile(times, 25) * 1000,
                    'p75_ms': np.percentile(times, 75) * 1000
                }
                
                logger.info(f"  {backend:>8}: {median_time*1000:6.2f}ms, {throughput:>10,.0f} comp/sec")
                
            except Exception as e:
                logger.error(f"Backend {backend} failed: {e}")
                results[f"{bits}_bits"][backend] = {'error': str(e)}
    
    return results


def print_benchmark_table(results: dict):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Extract all backends and bit configurations
    bit_configs = sorted(results.keys(), key=lambda x: int(x.split('_')[0]))
    if not bit_configs:
        print("No results to display")
        return
    
    backends = list(results[bit_configs[0]].keys())
    
    # Print header
    header = f"{'Bits':<10} | {'Backend':<10} | {'Time (ms)':<12} | {'Throughput/sec':<15} | {'Memory (MB)':<12}"
    print(header)
    print("-" * len(header))
    
    # Print results
    for bit_config in bit_configs:
        bits = bit_config.split('_')[0]
        for backend in backends:
            if backend in results[bit_config]:
                res = results[bit_config][backend]
                if 'error' in res:
                    print(f"{bits:<10} | {backend:<10} | {'ERROR':<12} | {res['error'][:40]}")
                else:
                    print(f"{bits:<10} | {backend:<10} | {res['elapsed_ms']:<12.2f} | "
                          f"{res['throughput_per_sec']:<15,.0f} | {res['memory_mb']:<12.2f}")
        print("-" * len(header))
    
    print("\nSummary:")
    # Find best throughput
    best_throughput = 0
    best_config = ""
    for bit_config in bit_configs:
        for backend, res in results[bit_config].items():
            if 'throughput_per_sec' in res and res['throughput_per_sec'] > best_throughput:
                best_throughput = res['throughput_per_sec']
                best_config = f"{bit_config.split('_')[0]} bits, {backend}"
    
    print(f"  Best throughput: {best_throughput:,.0f} comp/sec ({best_config})")
    print(f"  Target (>1M): {'✓ PASS' if best_throughput > 1_000_000 else '✗ FAIL'}")
    print("="*80)


if __name__ == "__main__":
    # Quick demonstration
    print("Tejas Bit Operations Demo")
    print("=" * 40)
    
    # Test packing/unpacking
    print("\n1. Testing pack/unpack roundtrip:")
    test_matrix = np.random.randint(0, 2, (1000, 128), dtype=np.uint8)
    is_valid = validate_packing_roundtrip(test_matrix)
    print(f"   Roundtrip validation: {'PASS' if is_valid else 'FAIL'}")
    
    # Memory reduction demo
    original_size = test_matrix.nbytes
    packed = pack_bits_rows(test_matrix)
    packed_size = packed.nbytes
    reduction = original_size / packed_size
    print(f"   Memory reduction: {original_size} → {packed_size} bytes ({reduction:.1f}x)")
    
    # Performance benchmark with multiple bit sizes
    print("\n2. Performance benchmark (with varying n_bits):")
    results = benchmark_backends(n_items=10000, n_bits=[64, 128, 256])
    
    # Print formatted table
    print_benchmark_table(results)
