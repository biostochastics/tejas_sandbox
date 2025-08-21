"""
Performance Benchmarking and Regression Testing
===============================================

Comprehensive performance testing to ensure Tejas v2 meets performance
requirements and detects regressions across different configurations.
"""

import pytest
import time
import psutil
import threading
import statistics
import json
from pathlib import Path
import tempfile
import torch
import numpy as np
from typing import Dict, Any, List, Tuple

# Import system components
from core.encoder import GoldenRatioEncoder
from core.fingerprint import BinaryFingerprintSearch, pack_fingerprints, unpack_fingerprints
from utils.benchmark import BenchmarkSuite


class PerformanceTestSuite:
    """Comprehensive performance testing suite."""
    
    def __init__(self):
        self.baseline_metrics = {
            'encoding_throughput_min': 100,
            'search_throughput_min': 10000,
            'memory_usage_max': 500,
            'search_latency_max': 100,
            'pattern_precision_min': 0.95,
            'linear_scaling_factor': 1.2
        }
        self.test_results = {}
        self.temp_dir = None
    
    def setup_test_environment(self) -> Path:
        """Setup performance test environment."""
        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)
        
        # Create test datasets
        data_path = temp_path / 'data'
        data_path.mkdir()
        
        # Medium dataset (10K titles)
        medium_titles = []
        patterns = ['University of', 'List of', 'History of']
        for i in range(10000):
            if i % 1000 < len(patterns):
                pattern = patterns[i % 1000]
                medium_titles.append(f"{pattern} Topic {i}")
            else:
                medium_titles.append(f"Article {i}")
        
        with open(data_path / 'medium_10k.txt', 'w') as f:
            for title in medium_titles:
                f.write(f"{title}\n")
        
        return temp_path


class TestPerformanceRegression:
    """Performance and regression test cases."""
    
    @pytest.fixture
    def perf_suite(self):
        """Create performance test suite."""
        suite = PerformanceTestSuite()
        temp_path = suite.setup_test_environment()
        yield suite
        
        # Cleanup
        import shutil
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    def test_encoding_performance(self, perf_suite):
        """Test encoding performance."""
        print("\n⚡ Testing encoding performance...")
        
        temp_path = Path(perf_suite.temp_dir)
        with open(temp_path / 'data' / 'medium_10k.txt') as f:
            titles = [line.strip() for line in f if line.strip()]
        
        # Test standard configuration
        encoder = GoldenRatioEncoder(n_bits=256, max_features=5000)
        encoder.fit(titles)
        
        # Test encoding throughput
        encode_start = time.time()
        fingerprints = encoder.encode(titles[:1000])
        encode_time = time.time() - encode_start
        
        throughput = 1000 / encode_time
        
        assert throughput >= perf_suite.baseline_metrics['encoding_throughput_min'], \
            f"Encoding throughput {throughput:.0f} below minimum"
        
        perf_suite.test_results['encoding'] = {'throughput': throughput}
        print(f"   Encoding throughput: {throughput:.0f} titles/sec")
        print("✅ Encoding performance validation passed")
    
    def test_search_performance(self, perf_suite):
        """Test search performance."""
        print("\n🔍 Testing search performance...")
        
        temp_path = Path(perf_suite.temp_dir)
        with open(temp_path / 'data' / 'medium_10k.txt') as f:
            titles = [line.strip() for line in f if line.strip()]
        
        # Setup search engine
        encoder = GoldenRatioEncoder(n_bits=256, max_features=5000)
        encoder.fit(titles)
        fingerprints = encoder.encode(titles)
        search_engine = BinaryFingerprintSearch(fingerprints, titles)
        
        # Test search performance
        query_fp = encoder.encode_single(titles[0])
        
        # Warm up
        for _ in range(10):
            search_engine.search(query_fp, k=10)
        
        # Measure latency
        search_times = []
        for i in range(100):
            test_query = encoder.encode_single(titles[i])
            
            start_time = time.time()
            search_engine.search(test_query, k=10)
            search_time = (time.time() - start_time) * 1000
            search_times.append(search_time)
        
        avg_latency = statistics.mean(search_times)
        
        # Calculate throughput
        comparisons_per_sec = len(titles) / (avg_latency / 1000)
        
        assert avg_latency <= perf_suite.baseline_metrics['search_latency_max'], \
            f"Search latency {avg_latency:.2f}ms above maximum"
        
        assert comparisons_per_sec >= perf_suite.baseline_metrics['search_throughput_min'], \
            f"Search throughput {comparisons_per_sec:.0f} below minimum"
        
        perf_suite.test_results['search'] = {
            'avg_latency_ms': avg_latency,
            'comparisons_per_sec': comparisons_per_sec
        }
        
        print(f"   Search latency: {avg_latency:.2f}ms")
        print(f"   Comparisons/sec: {comparisons_per_sec:,.0f}")
        print("✅ Search performance validation passed")
    
    def test_memory_efficiency(self, perf_suite):
        """Test memory usage."""
        print("\n💾 Testing memory efficiency...")
        
        temp_path = Path(perf_suite.temp_dir)
        with open(temp_path / 'data' / 'medium_10k.txt') as f:
            titles = [line.strip() for line in f if line.strip()]
        
        # Measure memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        encoder = GoldenRatioEncoder(n_bits=256, max_features=5000)
        encoder.fit(titles)
        fingerprints = encoder.encode(titles)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_delta = memory_after - memory_before
        
        # Test compression
        packed_fp = pack_fingerprints(fingerprints, bitorder='little')
        unpacked_memory = fingerprints.numel() * fingerprints.element_size() / 1024 / 1024
        packed_memory = packed_fp.size * packed_fp.itemsize / 1024 / 1024
        compression_ratio = unpacked_memory / packed_memory
        
        assert memory_delta <= perf_suite.baseline_metrics['memory_usage_max'], \
            f"Memory usage {memory_delta:.1f}MB above maximum"
        
        perf_suite.test_results['memory'] = {
            'memory_delta_mb': memory_delta,
            'compression_ratio': compression_ratio
        }
        
        print(f"   Memory usage: {memory_delta:.1f}MB")
        print(f"   Compression: {compression_ratio:.1f}x")
        print("✅ Memory efficiency validation passed")


def test_performance_regression_comprehensive():
    """Run performance regression testing."""
    print("\n" + "="*60)
    print("⚡ STARTING PERFORMANCE REGRESSION TESTING")
    print("="*60)
    
    suite = PerformanceTestSuite()
    temp_path = suite.setup_test_environment()
    
    try:
        test_instance = TestPerformanceRegression()
        
        test_methods = [
            test_instance.test_encoding_performance,
            test_instance.test_search_performance, 
            test_instance.test_memory_efficiency
        ]
        
        passed_tests = 0
        for test_method in test_methods:
            try:
                test_method(suite)
                passed_tests += 1
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed: {e}")
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE REGRESSION RESULTS")
        print("="*60)
        print(f"✅ Passed: {passed_tests}/{len(test_methods)} tests")
        
        return passed_tests == len(test_methods)
        
    finally:
        import shutil
        if temp_path.exists():
            shutil.rmtree(temp_path)


if __name__ == "__main__":
    success = test_performance_regression_comprehensive()
    exit(0 if success else 1)
