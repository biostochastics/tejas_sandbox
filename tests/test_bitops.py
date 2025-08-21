"""
Unit tests for bitops module.
Tests packing/unpacking, Hamming distance computation, and all backends.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bitops import (
    pack_bits_rows,
    unpack_bits_rows,
    hamming_distance_packed,
    validate_packing_roundtrip,
    search_packed_vectorized,
    POPCOUNT_LUT
)


class TestBitPacking:
    """Test bit packing and unpacking operations."""
    
    def test_pack_unpack_roundtrip_little(self):
        """Test roundtrip integrity with little endian."""
        binary = np.array([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
        packed = pack_bits_rows(binary, bitorder='little')
        unpacked = unpack_bits_rows(packed, bitorder='little', n_bits=8)
        np.testing.assert_array_equal(binary, unpacked)
    
    def test_pack_unpack_roundtrip_big(self):
        """Test roundtrip integrity with big endian."""
        binary = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        packed = pack_bits_rows(binary, bitorder='big')
        unpacked = unpack_bits_rows(packed, bitorder='big', n_bits=8)
        np.testing.assert_array_equal(binary, unpacked)
    
    def test_pack_with_padding(self):
        """Test packing with padding to target bits."""
        binary = np.array([[1, 0, 1]], dtype=np.uint8)
        packed = pack_bits_rows(binary, n_bits=16)
        assert packed.shape == (1, 2)  # 16 bits = 2 bytes
        
        # Verify padded bits are zeros
        unpacked = unpack_bits_rows(packed, n_bits=16)
        expected = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(unpacked, expected)
    
    def test_pack_with_truncation(self):
        """Test packing with truncation to target bits."""
        binary = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 1]], dtype=np.uint8)
        packed = pack_bits_rows(binary, n_bits=8)
        unpacked = unpack_bits_rows(packed, n_bits=8)
        np.testing.assert_array_equal(binary[:, :8], unpacked)
    
    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        empty = np.array([], dtype=np.uint8).reshape(0, 0)
        packed = pack_bits_rows(empty)
        assert packed.shape == (0, 0)
        unpacked = unpack_bits_rows(packed)
        assert unpacked.shape == (0, 0)
    
    def test_dimension_validation(self):
        """Test dimension validation."""
        with pytest.raises(ValueError, match="must be 2D"):
            pack_bits_rows(np.array([1, 0, 1]))
        
        with pytest.raises(ValueError, match="must be 2D"):
            unpack_bits_rows(np.array([170]))
    
    def test_multiple_rows(self):
        """Test packing/unpacking multiple rows."""
        binary = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
        packed = pack_bits_rows(binary)
        assert packed.shape == (100, 16)  # 128 bits = 16 bytes
        unpacked = unpack_bits_rows(packed, n_bits=128)
        np.testing.assert_array_equal(binary, unpacked)
    
    def test_non_byte_aligned_bits(self):
        """Test with non-byte-aligned bit counts."""
        for n_bits in [7, 15, 23, 31, 63, 127]:
            binary = np.random.randint(0, 2, (10, n_bits), dtype=np.uint8)
            packed = pack_bits_rows(binary)
            expected_bytes = (n_bits + 7) // 8
            assert packed.shape == (10, expected_bytes)
            unpacked = unpack_bits_rows(packed, n_bits=n_bits)
            np.testing.assert_array_equal(binary, unpacked)


class TestHammingDistance:
    """Test Hamming distance computation."""
    
    def test_identical_vectors(self):
        """Test distance between identical vectors is 0."""
        query = np.array([170, 85], dtype=np.uint8)
        database = np.array([[170, 85]], dtype=np.uint8)
        distances = hamming_distance_packed(query, database)
        assert distances[0] == 0
    
    def test_opposite_vectors(self):
        """Test distance between opposite vectors."""
        query = np.array([0xFF], dtype=np.uint8)  # All 1s
        database = np.array([[0x00]], dtype=np.uint8)  # All 0s
        distances = hamming_distance_packed(query, database)
        assert distances[0] == 8  # All 8 bits different
    
    def test_known_distance(self):
        """Test with known Hamming distances."""
        query = np.array([0b10101010], dtype=np.uint8)
        database = np.array([
            [0b10101010],  # Same: distance 0
            [0b10101011],  # 1 bit diff: distance 1
            [0b10101000],  # 1 bit diff: distance 1
            [0b11111111],  # 4 bits diff: distance 4
            [0b00000000],  # 4 bits diff: distance 4
            [0b01010101],  # All bits diff: distance 8
        ], dtype=np.uint8)
        
        distances = hamming_distance_packed(query, database)
        expected = np.array([0, 1, 1, 4, 4, 8], dtype=np.int32)
        np.testing.assert_array_equal(distances, expected)
    
    def test_backend_consistency(self):
        """Test that all backends produce the same results."""
        np.random.seed(42)
        query = np.random.randint(0, 256, 16, dtype=np.uint8)
        database = np.random.randint(0, 256, (100, 16), dtype=np.uint8)
        
        # Test available backends
        backends = ['numpy', 'native']
        try:
            import numba
            backends.append('numba')
        except ImportError:
            pass
        
        results = {}
        for backend in backends:
            results[backend] = hamming_distance_packed(query, database, backend=backend)
        
        # All backends should produce identical results
        reference = results['numpy']
        for backend, distances in results.items():
            np.testing.assert_array_equal(
                distances, reference,
                err_msg=f"Backend {backend} produces different results"
            )
    
    def test_dimension_mismatch(self):
        """Test error handling for dimension mismatches."""
        query = np.array([1, 2, 3], dtype=np.uint8)
        database = np.array([[1, 2]], dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            hamming_distance_packed(query, database)
    
    def test_empty_database(self):
        """Test handling of empty database."""
        query = np.array([170], dtype=np.uint8)
        database = np.array([], dtype=np.uint8).reshape(0, 1)
        distances = hamming_distance_packed(query, database)
        assert len(distances) == 0
    
    def test_large_scale(self):
        """Test with large-scale data."""
        np.random.seed(42)
        n_items = 10000
        n_bytes = 128 // 8
        
        query = np.random.randint(0, 256, n_bytes, dtype=np.uint8)
        database = np.random.randint(0, 256, (n_items, n_bytes), dtype=np.uint8)
        
        distances = hamming_distance_packed(query, database)
        assert len(distances) == n_items
        assert distances.dtype == np.int32
        
        # Check distance range (0 to n_bytes * 8)
        assert np.all(distances >= 0)
        assert np.all(distances <= n_bytes * 8)


class TestSearch:
    """Test k-nearest neighbor search."""
    
    def test_top_k_search(self):
        """Test finding top-k nearest neighbors."""
        query = np.array([0b10101010], dtype=np.uint8)
        database = np.array([
            [0b10101010],  # Distance 0
            [0b10101011],  # Distance 1
            [0b10101000],  # Distance 1
            [0b11101010],  # Distance 1
            [0b11111111],  # Distance 4
        ], dtype=np.uint8)
        
        indices, distances = search_packed_vectorized(query, database, k=3)
        
        # Should return indices [0, 1, 2] or [0, 1, 3] or [0, 2, 3] (all have same distances)
        assert len(indices) == 3
        assert indices[0] == 0  # Closest match
        assert distances[0] == 0
        assert np.all(distances[1:3] == 1)
    
    def test_k_larger_than_database(self):
        """Test when k is larger than database size."""
        query = np.array([170], dtype=np.uint8)
        database = np.array([[170], [171], [168]], dtype=np.uint8)
        
        indices, distances = search_packed_vectorized(query, database, k=10)
        assert len(indices) == 3  # Returns all available
        assert indices[0] == 0  # Exact match first


class TestPopcountLUT:
    """Test popcount lookup table."""
    
    def test_popcount_values(self):
        """Test that popcount LUT has correct values."""
        for i in range(256):
            expected = bin(i).count('1')
            assert POPCOUNT_LUT[i] == expected


class TestRoundtripValidation:
    """Test roundtrip validation function."""
    
    def test_valid_roundtrip(self):
        """Test validation with valid roundtrip."""
        binary = np.random.randint(0, 2, (50, 64), dtype=np.uint8)
        assert validate_packing_roundtrip(binary, bitorder='little')
        assert validate_packing_roundtrip(binary, bitorder='big')
    
    def test_roundtrip_with_padding(self):
        """Test validation with padding."""
        binary = np.random.randint(0, 2, (10, 50), dtype=np.uint8)
        assert validate_packing_roundtrip(binary, n_bits=64)
    
    def test_roundtrip_with_truncation(self):
        """Test validation with truncation."""
        binary = np.random.randint(0, 2, (10, 100), dtype=np.uint8)
        assert validate_packing_roundtrip(binary, n_bits=64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
