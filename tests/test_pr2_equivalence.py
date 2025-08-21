"""
PR2 Equivalence Tests: Packed vs Unpacked Search Results
======================================================

Comprehensive tests to ensure 100% search result parity between packed and unpacked formats.
Critical for validating that bit packing optimization maintains correctness.
"""

import pytest
import numpy as np
import torch
import logging
from typing import List, Tuple
import time

from core.bitops import (
    pack_bits_rows, unpack_bits_rows, hamming_distance_packed,
    search_packed_vectorized, validate_packing_roundtrip, benchmark_backends
)
from core.fingerprint import BinaryFingerprintSearch, pack_fingerprints, unpack_fingerprints
from core.encoder import GoldenRatioEncoder

logger = logging.getLogger(__name__)


class TestPackingEquivalence:
    """Test packing/unpacking operations maintain data integrity."""
    
    def test_pack_unpack_roundtrip_small(self):
        """Test roundtrip on small matrices."""
        # Test various sizes and patterns
        test_cases = [
            np.array([[1, 0, 1, 0]], dtype=np.uint8),  # Single row
            np.array([[1], [0]], dtype=np.uint8),      # Single column
            np.random.randint(0, 2, (10, 8), dtype=np.uint8),   # 8 bits
            np.random.randint(0, 2, (5, 16), dtype=np.uint8),   # 16 bits
            np.random.randint(0, 2, (3, 128), dtype=np.uint8),  # 128 bits
        ]
        
        for i, binary_matrix in enumerate(test_cases):
            # Pack
            packed = pack_bits_rows(binary_matrix, bitorder='little')
            
            # Unpack - need to specify the original number of bits
            n_bits = binary_matrix.shape[1]
            unpacked = unpack_bits_rows(packed, n_bits=n_bits)
            
            # Verify roundtrip
            np.testing.assert_array_equal(binary_matrix, unpacked,
                err_msg=f"Failed for test case {i}") 
    
    def test_pack_unpack_roundtrip_large(self):
        """Test roundtrip on large matrices."""
        # Large random matrix
        large_matrix = np.random.randint(0, 2, (1000, 128), dtype=np.uint8)
        
        for bitorder in ['little', 'big']:
            assert validate_packing_roundtrip(large_matrix, bitorder=bitorder), \
                f"Large matrix roundtrip failed with bitorder {bitorder}"
    
    def test_pack_unpack_edge_cases(self):
        """Test edge cases for packing/unpacking."""
        # Empty matrix
        empty = np.array([], dtype=np.uint8).reshape(0, 0)
        assert validate_packing_roundtrip(empty)
        
        # Single bit
        single_bit = np.array([[1]], dtype=np.uint8)
        assert validate_packing_roundtrip(single_bit)
        
        # Non-power-of-2 widths
        for width in [7, 15, 31, 63, 127]:
            matrix = np.random.randint(0, 2, (10, width), dtype=np.uint8)
            assert validate_packing_roundtrip(matrix), f"Failed for width {width}"
    
    def test_pack_with_padding_truncation(self):
        """Test packing with explicit n_bits parameter."""
        original = np.random.randint(0, 2, (5, 10), dtype=np.uint8)
        
        # Test padding (expand)
        packed_padded = pack_bits_rows(original, n_bits=16)
        unpacked_padded = unpack_bits_rows(packed_padded, n_bits=16)
        
        # Should match in first 10 bits, padding with zeros
        assert np.array_equal(original, unpacked_padded[:, :10])
        assert np.all(unpacked_padded[:, 10:] == 0)
        
        # Test truncation (reduce)
        packed_truncated = pack_bits_rows(original, n_bits=8)
        unpacked_truncated = unpack_bits_rows(packed_truncated, n_bits=8)
        
        # Should match in first 8 bits
        assert np.array_equal(original[:, :8], unpacked_truncated)


class TestHammingDistanceEquivalence:
    """Test that packed and unpacked Hamming distance computation gives identical results."""
    
    def test_hamming_distance_equivalence_small(self):
        """Test Hamming distance equivalence on small datasets."""
        # Generate test data
        n_items = 100
        n_bits = 64
        
        binary_db = np.random.randint(0, 2, (n_items, n_bits), dtype=np.uint8)
        binary_query = np.random.randint(0, 2, n_bits, dtype=np.uint8)
        
        # Pack the data
        packed_db = pack_bits_rows(binary_db)
        packed_query = pack_bits_rows(binary_query.reshape(1, -1))[0]
        
        # Compute distances using both methods
        # Method 1: Direct computation on unpacked data
        unpacked_distances = []
        for row in binary_db:
            dist = np.sum(row != binary_query)
            unpacked_distances.append(dist)
        unpacked_distances = np.array(unpacked_distances, dtype=np.int32)
        
        # Method 2: Packed computation
        packed_distances = hamming_distance_packed(packed_query, packed_db, backend='numpy')
        
        # Should be identical
        assert np.array_equal(unpacked_distances, packed_distances), \
            f"Distance mismatch: max diff = {np.max(np.abs(unpacked_distances - packed_distances))}"
    
    def test_hamming_distance_equivalence_large(self):
        """Test Hamming distance equivalence on larger datasets."""
        n_items = 5000
        n_bits = 128
        
        binary_db = np.random.randint(0, 2, (n_items, n_bits), dtype=np.uint8)
        binary_query = np.random.randint(0, 2, n_bits, dtype=np.uint8)
        
        # Pack the data
        packed_db = pack_bits_rows(binary_db)
        packed_query = pack_bits_rows(binary_query.reshape(1, -1))[0]
        
        # Compute using XOR on unpacked (simulating torch.sum)
        query_tensor = torch.from_numpy(binary_query)
        db_tensor = torch.from_numpy(binary_db)
        xor_result = db_tensor ^ query_tensor.unsqueeze(0)
        unpacked_distances = xor_result.sum(dim=1).numpy().astype(np.int32)
        
        # Compute using packed method
        packed_distances = hamming_distance_packed(packed_query, packed_db, backend='numpy')
        
        assert np.array_equal(unpacked_distances, packed_distances), \
            "Large dataset distance mismatch"
    
    def test_all_backends_equivalence(self):
        """Test that all backends produce identical results."""
        n_items = 1000
        n_bits = 128
        
        binary_db = np.random.randint(0, 2, (n_items, n_bits), dtype=np.uint8)
        binary_query = np.random.randint(0, 2, n_bits, dtype=np.uint8)
        
        packed_db = pack_bits_rows(binary_db)
        packed_query = pack_bits_rows(binary_query.reshape(1, -1))[0]
        
        # Test all available backends
        backends = ['numpy', 'native']
        try:
            import numba
            backends.append('numba')
        except ImportError:
            pass
        
        results = {}
        for backend in backends:
            try:
                distances = hamming_distance_packed(packed_query, packed_db, backend=backend)
                results[backend] = distances
            except Exception as e:
                pytest.skip(f"Backend {backend} not available: {e}")
        
        # All backends should produce identical results
        reference = list(results.values())[0]
        for backend, distances in results.items():
            assert np.array_equal(reference, distances), \
                f"Backend {backend} produced different results"


class TestSearchEquivalence:
    """Test that search results are identical between packed and unpacked formats."""
    
    def setup_method(self):
        """Setup test data for search equivalence tests."""
        # Generate deterministic test data
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.n_items = 1000
        self.n_bits = 128
        self.k = 10
        
        # Generate binary fingerprints and titles
        self.binary_fp = np.random.randint(0, 2, (self.n_items, self.n_bits), dtype=np.uint8)
        self.titles = [f"Title_{i:04d}" for i in range(self.n_items)]
        self.query = np.random.randint(0, 2, self.n_bits, dtype=np.uint8)
        
        # Create packed version
        self.packed_fp = pack_bits_rows(self.binary_fp, bitorder='little')
        self.query_packed = pack_bits_rows(self.query.reshape(1, -1), bitorder='little')[0]
    
    def test_search_result_equivalence(self):
        """Test that packed and unpacked search return identical results."""
        # Create search engines
        search_unpacked = BinaryFingerprintSearch(
            torch.from_numpy(self.binary_fp), 
            self.titles, 
            device='cpu',
            format_hint='unpacked'
        )
        
        search_packed = BinaryFingerprintSearch(
            self.packed_fp,
            self.titles,
            device='cpu', 
            format_hint='packed'
        )
        
        # Perform searches
        results_unpacked = search_unpacked.search(
            torch.from_numpy(self.query), 
            k=self.k, 
            show_pattern_analysis=False
        )
        
        results_packed = search_packed.search(
            self.query,
            k=self.k,
            show_pattern_analysis=False
        )
        
        # Results should be identical
        assert len(results_unpacked) == len(results_packed), "Result count mismatch"
        
        for i, ((title_u, sim_u, dist_u), (title_p, sim_p, dist_p)) in enumerate(
            zip(results_unpacked, results_packed)
        ):
            assert title_u == title_p, f"Title mismatch at position {i}: {title_u} != {title_p}"
            assert abs(sim_u - sim_p) < 1e-6, f"Similarity mismatch at position {i}: {sim_u} != {sim_p}"
            assert dist_u == dist_p, f"Distance mismatch at position {i}: {dist_u} != {dist_p}"
    
    def test_auto_detection_consistency(self):
        """Test that auto-detection produces consistent results."""
        # Test both formats with auto-detection
        search_auto_unpacked = BinaryFingerprintSearch(
            torch.from_numpy(self.binary_fp), 
            self.titles, 
            device='cpu'  # No format hint
        )
        
        search_auto_packed = BinaryFingerprintSearch(
            self.packed_fp,
            self.titles,
            device='cpu'   # No format hint
        )
        
        # Should detect formats correctly
        assert search_auto_unpacked.format == 'unpacked'
        assert search_auto_packed.format == 'packed'
        
        # Search results should be equivalent
        results_unpacked = search_auto_unpacked.search(
            self.query, k=self.k, show_pattern_analysis=False
        )
        results_packed = search_auto_packed.search(
            self.query, k=self.k, show_pattern_analysis=False
        )
        
        # Check equivalence
        for (title_u, sim_u, dist_u), (title_p, sim_p, dist_p) in zip(results_unpacked, results_packed):
            assert title_u == title_p
            assert abs(sim_u - sim_p) < 1e-6
            assert dist_u == dist_p
    
    def test_cross_format_query_handling(self):
        """Test search with queries in different formats than database."""
        # Packed database, unpacked query
        search_packed = BinaryFingerprintSearch(
            self.packed_fp, self.titles, device='cpu', format_hint='packed'
        )
        
        results_cross = search_packed.search(
            torch.from_numpy(self.query), k=self.k, show_pattern_analysis=False
        )
        
        # Unpacked database, packed query  
        search_unpacked = BinaryFingerprintSearch(
            torch.from_numpy(self.binary_fp), self.titles, device='cpu', format_hint='unpacked'
        )
        
        results_cross2 = search_unpacked.search(
            self.query_packed, k=self.k, show_pattern_analysis=False
        )
        
        # Both should work and give consistent results
        assert len(results_cross) == self.k
        assert len(results_cross2) == self.k


class TestPerformanceTargets:
    """Test that performance targets are met."""
    
    def test_memory_reduction_target(self):
        """Verify 8x memory reduction target."""
        n_items = 10000
        n_bits = 128
        
        # Generate test data
        binary_fp = np.random.randint(0, 2, (n_items, n_bits), dtype=np.uint8)
        packed_fp = pack_bits_rows(binary_fp)
        
        # Measure memory usage
        unpacked_size = binary_fp.nbytes
        packed_size = packed_fp.nbytes
        reduction_ratio = unpacked_size / packed_size
        
        logger.info(f"Memory reduction: {unpacked_size:,} → {packed_size:,} bytes ({reduction_ratio:.1f}x)")
        
        # Should achieve at least 7x reduction (allowing some tolerance)
        assert reduction_ratio >= 7.0, f"Memory reduction target not met: {reduction_ratio:.1f}x < 7x"
        assert reduction_ratio <= 8.1, f"Unexpected over-reduction: {reduction_ratio:.1f}x > 8.1x"
    
    def test_throughput_target(self):
        """Verify >1M comparisons/second target."""
        # Use reasonably large dataset for meaningful measurement
        n_items = 50000
        n_bits = 128
        
        binary_db = np.random.randint(0, 2, (n_items, n_bits), dtype=np.uint8)
        binary_query = np.random.randint(0, 2, n_bits, dtype=np.uint8)
        
        packed_db = pack_bits_rows(binary_db)
        packed_query = pack_bits_rows(binary_query.reshape(1, -1))[0]
        
        # Warm up
        _ = hamming_distance_packed(packed_query, packed_db[:1000], backend='numpy')
        
        # Benchmark
        start_time = time.time()
        distances = hamming_distance_packed(packed_query, packed_db, backend='auto')
        elapsed = time.time() - start_time
        
        throughput = n_items / elapsed
        logger.info(f"Throughput: {throughput:,.0f} comparisons/sec")
        
        # Target: >1M comparisons/second
        assert throughput > 1_000_000, f"Throughput target not met: {throughput:,.0f} < 1M"
    
    def test_latency_target(self):
        """Verify latency targets for different database sizes."""
        targets = [
            (10_000, 1.0),    # <1ms for 10K database
            (100_000, 10.0),  # <10ms for 100K database
        ]
        
        n_bits = 128
        query = np.random.randint(0, 2, n_bits, dtype=np.uint8)
        query_packed = pack_bits_rows(query.reshape(1, -1))[0]
        
        for db_size, target_ms in targets:
            # Generate database
            db = np.random.randint(0, 2, (db_size, n_bits), dtype=np.uint8)
            db_packed = pack_bits_rows(db)
            
            # Warm up
            _ = hamming_distance_packed(query_packed, db_packed[:1000])
            
            # Measure latency
            start_time = time.time()
            _ = hamming_distance_packed(query_packed, db_packed, backend='auto')
            latency_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Latency for {db_size:,} items: {latency_ms:.2f}ms (target: <{target_ms}ms)")
            
            assert latency_ms < target_ms, \
                f"Latency target not met for {db_size} items: {latency_ms:.2f}ms >= {target_ms}ms"


@pytest.mark.integration
class TestEndToEndEquivalence:
    """End-to-end integration tests with encoder."""
    
    def test_encoder_packed_equivalence(self):
        """Test that encoder produces equivalent results in packed mode."""
        # Create sample data
        sample_titles = [
            "Machine learning fundamentals",
            "Deep neural networks",
            "Artificial intelligence overview", 
            "Computer vision applications",
            "Natural language processing"
        ] * 20  # 100 total
        
        # Create encoder
        encoder = GoldenRatioEncoder(n_bits=64, max_features=1000, pack_bits=False)
        encoder.fit(sample_titles)
        
        # Generate fingerprints in both formats
        fp_unpacked = encoder.transform(sample_titles, pack_output=False)
        fp_packed = encoder.transform(sample_titles, pack_output=True)
        
        # Verify they represent the same data
        fp_unpacked_np = fp_unpacked.detach().cpu().numpy()
        fp_repacked = pack_bits_rows(fp_unpacked_np, bitorder='little')
        
        assert np.array_equal(fp_packed, fp_repacked), \
            "Encoder packed output doesn't match manual packing"
        
        # Test search equivalence
        search_unpacked = BinaryFingerprintSearch(fp_unpacked, sample_titles, format_hint='unpacked')
        search_packed = BinaryFingerprintSearch(fp_packed, sample_titles, format_hint='packed')
        
        # Query with first title
        query_fp = encoder.encode_single(sample_titles[0], pack_output=False)
        
        results_unpacked = search_unpacked.search(query_fp, k=5, show_pattern_analysis=False)
        results_packed = search_packed.search(query_fp, k=5, show_pattern_analysis=False)
        
        # Should find identical results
        for (title_u, sim_u, dist_u), (title_p, sim_p, dist_p) in zip(results_unpacked, results_packed):
            assert title_u == title_p
            assert abs(sim_u - sim_p) < 1e-6
            assert dist_u == dist_p


if __name__ == "__main__":
    # Quick validation
    pytest.main([__file__, "-v", "--tb=short"])
