"""
Test suite for fingerprint module.
Tests format detection, search functionality, and edge cases.
"""

import pytest
import numpy as np
import torch
from core.fingerprint import (
    BinaryFingerprintSearch,
    pack_fingerprints,
    unpack_fingerprints
)


class TestFormatDetection:
    """Test automatic format detection logic."""
    
    def test_small_packed_detection(self):
        """Test detection of small packed arrays."""
        packed = np.random.randint(0, 256, (10, 16), dtype=np.uint8)
        search = BinaryFingerprintSearch(packed, format_hint='auto')
        assert search.format == 'packed'
        assert search.n_bits == 16 * 8
    
    def test_large_packed_detection(self):
        """Test detection of large packed arrays (n_bits >= 256)."""
        # This was the bug case - large packed misclassified as unpacked
        packed = np.random.randint(0, 256, (10, 128), dtype=np.uint8)  # 1024 bits
        search = BinaryFingerprintSearch(packed, format_hint='auto')
        assert search.format == 'packed'
        assert search.n_bits == 128 * 8
    
    def test_unpacked_detection(self):
        """Test detection of unpacked binary arrays."""
        unpacked = np.random.randint(0, 2, (10, 256), dtype=np.uint8)
        search = BinaryFingerprintSearch(unpacked, format_hint='auto')
        assert search.format == 'unpacked'
        assert search.n_bits == 256
    
    def test_format_hint_override(self):
        """Test that explicit format hints are respected."""
        data = np.random.randint(0, 2, (10, 64), dtype=np.uint8)
        
        # Force packed interpretation
        search_packed = BinaryFingerprintSearch(data, format_hint='packed')
        assert search_packed.format == 'packed'
        
        # Force unpacked interpretation
        search_unpacked = BinaryFingerprintSearch(data, format_hint='unpacked')
        assert search_unpacked.format == 'unpacked'
    
    def test_value_based_detection_override(self):
        """Test that values > 1 force packed format."""
        # Create data that looks unpacked by shape but has values > 1
        data = np.random.randint(0, 256, (10, 256), dtype=np.uint8)
        search = BinaryFingerprintSearch(data, format_hint='auto')
        # Should detect as packed due to values > 1
        assert search.format == 'packed'


class TestInitialization:
    """Test initialization and edge cases."""
    
    def test_titles_none_handling(self):
        """Test that titles=None generates default titles."""
        data = np.random.randint(0, 2, (5, 128), dtype=np.uint8)
        search = BinaryFingerprintSearch(data, titles=None)
        assert len(search.titles) == 5
        assert search.titles[0] == "Item 0"
        assert search.titles[4] == "Item 4"
    
    def test_torch_tensor_input(self):
        """Test initialization with PyTorch tensors."""
        tensor_data = torch.randint(0, 2, (10, 128), dtype=torch.uint8)
        search = BinaryFingerprintSearch(tensor_data)
        assert len(search.titles) == 10
    
    def test_device_selection(self):
        """Test device selection logic."""
        data = np.random.randint(0, 2, (10, 128), dtype=np.uint8)
        
        # Test CPU explicit
        search_cpu = BinaryFingerprintSearch(data, device='cpu')
        assert search_cpu.device.type == 'cpu'
        
        # Test auto (will be CPU if no CUDA)
        search_auto = BinaryFingerprintSearch(data, device='auto')
        assert search_auto.device.type in ['cpu', 'cuda']


class TestSearch:
    """Test search functionality."""
    
    def test_search_identical_unpacked(self):
        """Test search finds identical fingerprint with similarity=1.0."""
        data = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
        titles = [f"Item {i}" for i in range(100)]
        search = BinaryFingerprintSearch(data, titles=titles, format_hint='unpacked')
        
        # Search for first item
        results = search.search(data[0], k=5, show_pattern_analysis=False)
        
        # First result should be exact match
        assert results[0][0] == "Item 0"
        assert results[0][1] == 1.0  # similarity
        assert results[0][2] == 0     # distance
    
    def test_search_identical_packed(self):
        """Test search on packed format."""
        data = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
        packed = pack_fingerprints(data)
        titles = [f"Item {i}" for i in range(100)]
        search = BinaryFingerprintSearch(packed, titles=titles, format_hint='packed')
        
        # Search for first item (unpacked query)
        results = search.search(data[0], k=5, show_pattern_analysis=False)
        
        # First result should be exact match
        assert results[0][0] == "Item 0"
        assert results[0][1] == 1.0  # similarity
        assert results[0][2] == 0     # distance
    
    def test_mixed_format_search(self):
        """Test searching packed database with unpacked query."""
        data = np.random.randint(0, 2, (50, 128), dtype=np.uint8)
        packed = pack_fingerprints(data)
        
        search = BinaryFingerprintSearch(packed, format_hint='packed')
        
        # Search with unpacked query
        results = search.search(data[0], k=3, show_pattern_analysis=False)
        assert len(results) == 3
        assert results[0][2] == 0  # First match should be exact
    
    def test_k_parameter(self):
        """Test that k parameter limits results correctly."""
        data = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
        search = BinaryFingerprintSearch(data)
        
        for k in [1, 5, 10, 50]:
            results = search.search(data[0], k=k, show_pattern_analysis=False)
            assert len(results) == k


class TestBenchmark:
    """Test benchmark functionality."""
    
    def test_benchmark_unpacked(self):
        """Test benchmark works for unpacked format."""
        data = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
        search = BinaryFingerprintSearch(data, format_hint='unpacked')
        
        # Should not raise error
        search.benchmark(n_queries=5)
    
    def test_benchmark_packed(self):
        """Test benchmark works for packed format (previously failed)."""
        data = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
        packed = pack_fingerprints(data)
        search = BinaryFingerprintSearch(packed, format_hint='packed')
        
        # Should not raise error (this was the bug)
        search.benchmark(n_queries=5)


class TestPackingUtilities:
    """Test packing/unpacking utilities."""
    
    def test_pack_returns_numpy(self):
        """Test that pack_fingerprints returns numpy array, not torch tensor."""
        data = np.random.randint(0, 2, (10, 128), dtype=np.uint8)
        packed = pack_fingerprints(data)
        assert isinstance(packed, np.ndarray)
        assert not isinstance(packed, torch.Tensor)
        assert packed.dtype == np.uint8
    
    def test_pack_unpack_roundtrip(self):
        """Test packing and unpacking preserves data."""
        original = np.random.randint(0, 2, (10, 128), dtype=np.uint8)
        
        for bitorder in ['little', 'big']:
            packed = pack_fingerprints(original, bitorder=bitorder)
            unpacked = unpack_fingerprints(packed, n_bits=128, bitorder=bitorder)
            
            # Convert torch tensor result to numpy for comparison
            if isinstance(unpacked, torch.Tensor):
                unpacked = unpacked.numpy()
            
            np.testing.assert_array_equal(original, unpacked)
    
    def test_pack_dimension_validation(self):
        """Test that packing validates input dimensions."""
        # 1D array should raise error
        data_1d = np.array([0, 1, 0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 2D"):
            pack_fingerprints(data_1d)


class TestPatternSearch:
    """Test pattern search with encoder validation."""
    
    def test_encoder_validation(self):
        """Test that invalid encoder raises error."""
        data = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
        search = BinaryFingerprintSearch(data)
        
        # Object without encode_single method
        class BadEncoder:
            pass
        
        with pytest.raises(ValueError, match="must have an 'encode_single' method"):
            search.search_pattern("test", BadEncoder())
    
    def test_encoder_with_valid_method(self):
        """Test pattern search with valid encoder."""
        data = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
        titles = [f"List of Item {i}" if i < 10 else f"Item {i}" for i in range(100)]
        search = BinaryFingerprintSearch(data, titles=titles)
        
        # Mock encoder that returns first fingerprint
        class MockEncoder:
            def encode_single(self, text):
                # Return a fingerprint similar to first few items
                return data[0]
        
        # Should not raise error
        results = search.search_pattern("List of", MockEncoder(), max_results=5)
        assert isinstance(results, list)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_database(self):
        """Test handling of empty database."""
        # Empty array should work but have no results
        data = np.zeros((0, 128), dtype=np.uint8)
        search = BinaryFingerprintSearch(data)
        assert len(search.titles) == 0
    
    def test_single_item_database(self):
        """Test database with single item."""
        data = np.random.randint(0, 2, (1, 128), dtype=np.uint8)
        search = BinaryFingerprintSearch(data)
        
        results = search.search(data[0], k=1, show_pattern_analysis=False)
        assert len(results) == 1
        assert results[0][1] == 1.0  # Perfect match
    
    def test_non_byte_aligned_bits(self):
        """Test handling of non-byte-aligned bit counts."""
        # 130 bits is not byte-aligned
        data = np.random.randint(0, 2, (10, 130), dtype=np.uint8)
        search = BinaryFingerprintSearch(data, format_hint='unpacked')
        
        # Should handle correctly
        results = search.search(data[0], k=3, show_pattern_analysis=False)
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
