"""
Binary Fingerprint Operations and Search
=======================================

High-performance binary operations for semantic fingerprints.
Implements XOR-based Hamming distance for hardware speed search.
Also provides utilities to pack/unpack fingerprints into bytes for
compact storage without changing consumer APIs.
"""

import torch
import numpy as np
import time
import logging
from typing import List, Tuple, Union, Optional, Literal
from pathlib import Path
from core.bitops import (
    pack_bits_rows, unpack_bits_rows, hamming_distance_packed,
    search_packed_vectorized, POPCOUNT_LUT
)
logger = logging.getLogger(__name__)


class BinaryFingerprintSearch:
    """
    Ultra-fast search using binary fingerprints and XOR operations.
    Supports both packed and unpacked formats with automatic detection.
    Achieves >1M comparisons/second through optimized bit operations.
    """
    
    def __init__(self, fingerprints: Union[torch.Tensor, np.ndarray], 
                 titles: Optional[List[str]] = None, 
                 device: str = 'auto',
                 format_hint: Literal['unpacked', 'packed', 'auto'] = 'auto',
                 bitorder: Literal['little', 'big'] = 'little',
                 backend: str = 'auto'):
        """
        Initialize search engine with automatic format detection.
        
        Args:
            fingerprints: Either (N, n_bits) bool/int or (N, n_bytes) uint8
            titles: List of titles corresponding to fingerprints
            device: Device for computation ('cpu', 'cuda', or 'auto')
            format_hint: 'packed', 'unpacked', 'auto' (auto-detect)
            bitorder: 'little' or 'big' (endianness for packed format)
            backend: Backend for packed operations ('numpy', 'numba', 'auto')
        """
        # Convert to numpy if needed for format detection
        if isinstance(fingerprints, torch.Tensor):
            fp_np = fingerprints.detach().cpu().numpy()
        else:
            fp_np = fingerprints
        
        # Handle titles=None gracefully
        if titles is None:
            self.titles = [f"Item {i}" for i in range(fp_np.shape[0])]
            logger.info("Generated default titles since none provided")
        else:
            self.titles = titles
        
        self.backend = backend
        
        self.bitorder = bitorder
        self.format_hint = format_hint

        # Tentatively assume unpacked to get n_bits, then detect format
        self.n_bits = fp_np.shape[1]
        self.format = self._detect_format(fp_np, format_hint)

        if self.format == 'packed':
            self.fingerprints_packed = fp_np.astype(np.uint8)
            self.n_bits = self.fingerprints_packed.shape[1] * 8
            self.fingerprints = None
            logger.info(f"Detected packed format: {self.fingerprints_packed.shape} → {self.n_bits} bits")
        else:
            self.fingerprints = torch.from_numpy(fp_np.astype(np.uint8))
            # n_bits is already correctly set from shape[1]
            self.fingerprints_packed = None
            logger.info(f"Detected unpacked format: {self.fingerprints.shape}")
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move unpacked fingerprints to device if available
        if self.fingerprints is not None:
            self.fingerprints = self.fingerprints.to(self.device)
        
        logger.info(f"Loaded {len(self.titles):,} fingerprints")
        logger.info(f"Format: {self.format}, Backend: {self.backend}, Device: {self.device}")
        logger.info("Ready for search!")
    
    def _detect_format(self, fp: np.ndarray, hint: Optional[str] = None) -> str:
        """
        Auto-detect packed vs unpacked format based on dtype and dimensions.
        
        Args:
            fp: Fingerprint array to analyze
            hint: User-provided hint ('packed', 'unpacked', None)
        
        Returns:
            'packed' or 'unpacked'
        """
        if hint in ['packed', 'unpacked']:
            logger.info(f"Using format hint: {hint}")
            return hint
        
        # Detection logic:
        # - dtype uint8 AND small width (< 32) → likely packed
        # - dtype bool/int AND large width (> 32) → likely unpacked
        # - Edge cases: validate against reasonable bounds
        
        dtype = fp.dtype
        n_cols = fp.shape[1] if fp.ndim > 1 else 1
        
        if dtype == np.uint8 and n_cols < 32:
            detected = 'packed'
            logger.info(f"Auto-detected packed format: dtype={dtype}, width={n_cols}")
        elif dtype in [bool, np.bool_, np.uint8, np.int8, np.int32] and n_cols >= 32:
            detected = 'unpacked'
            logger.info(f"Auto-detected unpacked format: dtype={dtype}, width={n_cols}")
        else:
            # Ambiguous case: use heuristics
            if dtype == np.uint8:
                # Could be either; use width as tie-breaker
                detected = 'packed' if n_cols <= 16 else 'unpacked'
                logger.warning(f"Ambiguous format, guessing {detected} based on width={n_cols}")
            else:
                # Non-uint8 likely means unpacked
                detected = 'unpacked'
                logger.info(f"Non-uint8 dtype, assuming unpacked: dtype={dtype}")
        
        # Critical fix: Check actual data values to prevent misclassification
        # Unpacked format should only have values 0 or 1
        if detected == 'unpacked' and len(fp) > 0:
            # Sample a subset for efficiency on large arrays
            sample_size = min(1000, len(fp))
            sample = fp[:sample_size] if fp.ndim == 1 else fp[:sample_size].flat[:1000]
            max_val = np.max(sample)
            min_val = np.min(sample)
            unique_vals = np.unique(sample)
            
            # More robust detection: unpacked should be binary (0,1)
            if max_val > 1 or min_val < 0 or len(unique_vals) > 2:
                detected = 'packed'
                logger.info(f"Overriding to 'packed' due to non-binary values (range=[{min_val},{max_val}], unique={len(unique_vals)})")
        
        return detected
    
    def search(self, query_fingerprint: Union[torch.Tensor, np.ndarray], 
               k: int = 10, 
               show_pattern_analysis: bool = True,
               return_distances: bool = True,
               backend: Optional[str] = None) -> List[Tuple[str, float, int]]:
        """
        Unified search supporting both packed and unpacked formats.
        
        Args:
            query_fingerprint: Query fingerprint (torch.Tensor or np.ndarray)
            k: Number of results to return
            show_pattern_analysis: Show pattern family analysis
            return_distances: Include distances in results
            backend: Override backend for search (None uses instance default)
            
        Returns:
            List of (title, similarity, distance) tuples
        """
        start_time = time.time()
        
        # Use provided backend or instance default
        search_backend = backend if backend is not None else self.backend
        
        # Convert query to numpy for format detection
        if isinstance(query_fingerprint, torch.Tensor):
            query_np = query_fingerprint.detach().cpu().numpy()
        else:
            query_np = query_fingerprint
        
        # Standardize search path: convert query to match database format
        query_format = self._detect_format(query_np.reshape(1, -1))

        if self.format == 'unpacked':
            # Database is unpacked. If query is packed, unpack it.
            if query_format == 'packed':
                query_tensor = unpack_fingerprints(query_np.reshape(1, -1), n_bits=self.n_bits, bitorder=self.bitorder)[0]
            else:
                query_tensor = torch.from_numpy(query_np) if isinstance(query_np, np.ndarray) else query_fingerprint
            distances, indices = self._search_unpacked(query_tensor, k)
        else: # self.format == 'packed'
            # Database is packed. If query is unpacked, pack it.
            if query_format == 'unpacked':
                query_packed = pack_fingerprints(query_np.reshape(1, -1), bitorder=self.bitorder)[0]
            else:
                query_packed = query_np
            distances, indices = self._search_packed(query_packed, k, backend=search_backend)
        
        search_time = time.time() - start_time
        
        # Convert to similarities
        similarities = 1.0 - (distances.astype(np.float32) / self.n_bits)
        
        # Prepare results
        results = []
        for idx, sim, dist in zip(indices, similarities, distances):
            results.append((
                self.titles[int(idx)],
                float(sim),
                int(dist)
            ))
        
        # Log performance
        comparisons_per_sec = len(self.titles) / search_time if search_time > 0 else float('inf')
        logger.info(f"Search time: {search_time*1000:.2f} ms ({self.format} format)")
        logger.info(f"Comparisons/sec: {comparisons_per_sec:,.0f}")
        
        # Pattern analysis
        if show_pattern_analysis:
            self._analyze_patterns(results)
        
        return results
    
    def _search_packed(self, query_np: np.ndarray, k: int, backend: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using packed format with optimized XOR+popcount.
        
        Args:
            query_np: Query fingerprint as numpy array
            k: Number of results
            backend: Backend for search operations
        
        Returns:
            (distances, indices) as numpy arrays
        """
        # Ensure we have packed database
        if self.fingerprints_packed is None:
            logger.info("Converting unpacked fingerprints to packed format for search")
            fp_np = self.fingerprints.detach().cpu().numpy()
            self.fingerprints_packed = pack_fingerprints(fp_np, bitorder=self.bitorder)
        
        # Ensure query is in packed format
        if query_np.size == self.n_bits:
            # Query is unpacked, need to pack it
            query_packed = pack_fingerprints(query_np.reshape(1, -1), bitorder=self.bitorder)[0]
        else:
            # Query is already packed
            query_packed = query_np.astype(np.uint8)
        
        # Ensure database is numpy for search functions
        db_packed_np = self.fingerprints_packed
        if isinstance(db_packed_np, torch.Tensor):
            db_packed_np = db_packed_np.detach().cpu().numpy()

        # Use optimized packed search with specified backend
        search_backend = backend if backend is not None else self.backend
        indices, distances = search_packed_vectorized(
            query_packed, db_packed_np, k=k, backend=search_backend, bitorder=self.bitorder
        )
        
        return distances, indices
    
    def _search_unpacked(self, query_fingerprint: Union[torch.Tensor, np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using unpacked format (legacy path).
        
        Args:
            query_fingerprint: Query fingerprint
            k: Number of results
        
        Returns:
            (distances, indices) as numpy arrays
        """
        # Ensure we have unpacked fingerprints
        if self.fingerprints is None:
            logger.info("Converting packed fingerprints to unpacked format for search")
            self.fingerprints = unpack_fingerprints(self.fingerprints_packed, 
                                                  bitorder=self.bitorder, n_bits=self.n_bits).to(self.device)
        
        # Convert query to tensor and move to device
        if isinstance(query_fingerprint, np.ndarray):
            query_tensor = torch.from_numpy(query_fingerprint.astype(np.uint8))
        else:
            query_tensor = query_fingerprint
        
        query_tensor = query_tensor.to(self.device)
        
        # Original XOR-based search
        xor_result = self.fingerprints ^ query_tensor.unsqueeze(0)
        hamming_distances = xor_result.sum(dim=1)
        
        # Get top-k nearest using a stable sort to handle ties consistently
        if k >= len(hamming_distances):
            sorted_indices = torch.argsort(hamming_distances, stable=True)
        else:
            # For k < N, full sort is efficient enough and guarantees stability.
            sorted_indices = torch.argsort(hamming_distances, stable=True)[:k]

        indices_torch = sorted_indices
        distances_torch = hamming_distances[indices_torch]

        return distances_torch.cpu().numpy(), indices_torch.cpu().numpy()
    
    def search_pattern(self, pattern: str, encoder, max_results: int = 100) -> List[Tuple[str, float, int]]:
        """
        Search for titles containing a specific pattern.
        Demonstrates zero false positives for pattern matching.
        
        Args:
            pattern: Pattern to search for (e.g., "List of", "University of")
            encoder: Encoder object with .encode_single(str) -> binary array method
            max_results: Maximum results to return
            
        Returns:
            Matching titles with similarities
        """
        # Validate encoder has required method
        if not hasattr(encoder, 'encode_single'):
            raise ValueError("Encoder must have an 'encode_single' method")
        
        logger.info(f"Pattern search for: '{pattern}'")
        
        # Encode the pattern
        pattern_fingerprint = encoder.encode_single(pattern)
        
        # Search with larger k to find true matches
        results = self.search(pattern_fingerprint, k=min(1000, len(self.titles)), show_pattern_analysis=False)
        
        # Filter to only those that ACTUALLY contain the pattern
        pattern_matches = []
        false_positives = []
        
        for title, sim, dist in results:
            if pattern.lower() in title.lower():
                pattern_matches.append((title, sim, dist))
            else:
                false_positives.append((title, sim, dist))
            
            if len(pattern_matches) >= max_results:
                break
        
        # Report findings
        logger.info(f"Pattern Match Analysis:")
        logger.info(f"  Checked: {len(results)} similar fingerprints")
        logger.info(f"  True matches: {len(pattern_matches)}")
        logger.info(f"  False positives: {len(false_positives)}")
        if len(pattern_matches) + len(false_positives) > 0:
            logger.info(f"  Precision: {len(pattern_matches)/(len(pattern_matches)+len(false_positives))*100:.1f}%")
        
        return pattern_matches[:max_results]
    
    def _analyze_patterns(self, results: List[Tuple[str, float, int]]):
        """Analyze pattern families in search results."""
        # Common patterns to check
        patterns = {
            'List of': 0,
            'University': 0,
            'County': 0,
            'Battle of': 0,
            '(disambiguation)': 0,
            '(film)': 0,
            '(album)': 0,
            'History of': 0
        }
        
        # Count patterns in results
        for title, _, _ in results:
            for pattern in patterns:
                if pattern in title:
                    patterns[pattern] += 1
        
        # Show if any patterns dominate
        if any(count > len(results) * 0.3 for count in patterns.values()):
            logger.info("Pattern Family Analysis:")
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    logger.info(f"  {pattern}: {count}/{len(results)} ({count/len(results)*100:.0f}%)")
    
    def benchmark(self, n_queries: int = 100):
        """
        Benchmark search performance.
        
        Args:
            n_queries: Number of random queries to test
        """
        logger.info(f"Benchmarking with {n_queries} random queries...")
        
        # Select random fingerprints as queries
        query_indices = torch.randperm(len(self.titles))[:n_queries]
        
        # Time searches
        search_times = []
        
        for idx in query_indices:
            # Handle both packed and unpacked formats
            if self.format == 'unpacked':
                query = self.fingerprints[int(idx)]
            else:
                # For packed format, unpack one row to use as query
                packed_row = self.fingerprints_packed[int(idx):int(idx)+1]
                query = unpack_bits_rows(packed_row, bitorder=self.bitorder, n_bits=self.n_bits)[0]
            
            start = time.time()
            _ = self.search(query, k=10, show_pattern_analysis=False)
            search_times.append(time.time() - start)
        
        # Calculate statistics
        search_times = torch.tensor(search_times) * 1000  # Convert to ms
        
        logger.info(f"Benchmark Results:")
        logger.info(f"  Average search time: {search_times.mean():.2f} ms")
        logger.info(f"  Median search time: {search_times.median():.2f} ms")
        logger.info(f"  Min search time: {search_times.min():.2f} ms")
        logger.info(f"  Max search time: {search_times.max():.2f} ms")
        logger.info(f"  Comparisons/sec: {len(self.titles)/search_times.mean()*1000:,.0f}")


def demonstrate_fingerprint_search():
    """
    Demonstrate fingerprint search capabilities.
    """
    # Create sample data
    n_items = 10000
    n_bits = 128
    
    # Generate random fingerprints and titles
    fingerprints = torch.randint(0, 2, (n_items, n_bits), dtype=torch.uint8)
    titles = [f"Sample Title {i}" for i in range(n_items)]
    
    # Create search engine
    search_engine = BinaryFingerprintSearch(fingerprints, titles)
    
    print("\nBinary Fingerprint Search Demo:")
    print("=" * 50)
    print(f"Database: {n_items:,} items, {n_bits} bits each")
    
    # Perform search
    query = fingerprints[0]
    results = search_engine.search(query, k=5)
    
    print(f"\nSearch results:")
    for i, (title, sim, dist) in enumerate(results):
        print(f"  {i+1}. {title}: similarity={sim:.3f}, distance={dist}")
    
    # Benchmark
    search_engine.benchmark(n_queries=10)


if __name__ == "__main__":
    pass


# -----------------------------
# Packing/Unpacking Utilities
# -----------------------------

def pack_fingerprints(fingerprints, bitorder: str = 'little'):
    """
    Pack binary fingerprints (0/1) into bytes along the bit dimension.

    Args:
        fingerprints: Array/Tensor of shape (n_items, n_bits), dtype uint8 with values {0,1}
        bitorder: 'little' or 'big' bit order for packing (numpy semantics)

    Returns:
        Packed array of shape (n_items, ceil(n_bits/8)), dtype uint8 (numpy array)
    """
    # Convert to numpy if torch tensor
    if hasattr(fingerprints, 'detach'):
        fp_np = fingerprints.detach().cpu().numpy()
    else:
        fp_np = np.asarray(fingerprints, dtype=np.uint8)
    
    if fp_np.ndim != 2:
        raise ValueError("fingerprints must be 2D (n_items, n_bits)")
    # Ensure strictly 0/1
    fp_np = (fp_np & 1).astype(np.uint8)
    packed = np.packbits(fp_np, axis=1, bitorder=bitorder)
    return packed


def unpack_fingerprints(packed, n_bits: int = None, bitorder: str = 'little'):
    """
    Unpack byte-packed fingerprints back to binary (0/1) matrix.

    Args:
        packed: Array/Tensor of shape (n_items, n_bytes), dtype uint8
        n_bits: Number of valid bits per item to return (if None, returns all bits)
        bitorder: 'little' or 'big' (must match packing)

    Returns:
        Array of shape (n_items, n_bits), dtype uint8 with values {0,1}
    """
    # Convert to numpy if torch tensor
    if hasattr(packed, 'detach'):
        pk_np = packed.detach().cpu().numpy()
    else:
        pk_np = np.asarray(packed, dtype=np.uint8)
    
    if pk_np.ndim != 2:
        raise ValueError("packed must be 2D (n_items, n_bytes)")
    
    if n_bits is None:
        unpacked_full = np.unpackbits(pk_np, axis=1, bitorder=bitorder).astype(np.uint8)
    else:
        unpacked_full = np.unpackbits(pk_np, axis=1, count=n_bits, bitorder=bitorder).astype(np.uint8)
    return torch.from_numpy(unpacked_full)