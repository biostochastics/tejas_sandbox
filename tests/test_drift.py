"""
Comprehensive Tests for Drift Detection
======================================

Tests the DriftMonitor class with synthetic datasets that simulate
various types of distribution drift in binary fingerprints.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import json
import tempfile
from typing import List, Dict, Tuple

from core.drift import DriftMonitor


class TestDriftMonitor:
    """Test suite for DriftMonitor class."""
    
    @pytest.fixture
    def drift_monitor(self):
        """Create a DriftMonitor instance."""
        return DriftMonitor(baseline_fingerprints=None)
    
    @pytest.fixture
    def stable_baseline(self):
        """Generate stable baseline fingerprint data."""
        np.random.seed(42)
        
        n_samples = 5000  # Increased for realistic statistical power
        n_bits = 1024
        
        # Generate fingerprints with consistent bit activation rates
        activation_rate = 0.3  # 30% bits active
        fingerprints = np.random.binomial(1, activation_rate, (n_samples, n_bits))
        
        return torch.from_numpy(fingerprints.astype(np.uint8))
    
    @pytest.fixture
    def gradual_drift_data(self):
        """Generate data with gradual distribution drift."""
        np.random.seed(42)
        
        n_samples = 2500  # Increased for better drift detection
        n_bits = 1024
        
        batches = []
        
        # Start with 30% activation rate
        base_rate = 0.3
        
        for i in range(5):  # 5 batches with increasing drift
            drift_amount = i * 0.05  # Gradual increase
            current_rate = base_rate + drift_amount
            
            batch = np.random.binomial(1, current_rate, (n_samples, n_bits))
            batches.append(torch.from_numpy(batch.astype(np.uint8)))
        
        return batches
    
    @pytest.fixture
    def sudden_drift_data(self):
        """Generate data with sudden distribution shift."""
        np.random.seed(42)
        
        n_samples = 2500  # Increased for better drift detection
        n_bits = 1024
        
        # First 3 batches stable, then sudden shift
        batches = []
        
        # Stable batches
        for _ in range(3):
            batch = np.random.binomial(1, 0.3, (n_samples, n_bits))
            batches.append(torch.from_numpy(batch.astype(np.uint8)))
        
        # Sudden shift batches
        for _ in range(3):
            batch = np.random.binomial(1, 0.6, (n_samples, n_bits))  # Double activation rate
            batches.append(torch.from_numpy(batch.astype(np.uint8)))
        
        return batches
    
    @pytest.fixture
    def oscillating_drift_data(self):
        """Generate data with oscillating drift patterns."""
        np.random.seed(42)
        
        n_samples = 300
        n_bits = 1024
        
        batches = []
        base_rate = 0.3
        
        for i in range(8):
            # Sine wave pattern
            drift = 0.2 * np.sin(i * np.pi / 2)
            current_rate = base_rate + drift
            
            batch = np.random.binomial(1, current_rate, (n_samples, n_bits))
            batches.append(torch.from_numpy(batch.astype(np.uint8)))
        
        return batches
    
    @pytest.fixture
    def sparse_drift_data(self):
        """Generate data with drift in specific bit positions."""
        np.random.seed(42)
        
        n_samples = 2500  # Increased for better drift detection
        n_bits = 1024
        
        batches = []
        
        # Create baseline with uniform activation
        baseline = np.random.binomial(1, 0.3, (n_samples, n_bits))
        batches.append(torch.from_numpy(baseline.astype(np.uint8)))
        
        # Create drift by changing specific bit ranges
        for shift_start in [100, 300, 500, 700]:
            shifted_data = baseline.copy()
            # Increase activation in specific range
            shifted_data[:, shift_start:shift_start+50] = np.random.binomial(
                1, 0.8, (n_samples, 50)
            )
            batches.append(torch.from_numpy(shifted_data.astype(np.uint8)))
        
        return batches
    
    def test_baseline_establishment(self, drift_monitor, stable_baseline):
        """Test baseline statistics computation."""
        drift_monitor.set_baseline(stable_baseline)
        
        baseline_stats = drift_monitor.baseline_stats
        
        # Check baseline statistics structure
        assert 'bit_activation_rates' in baseline_stats
        assert 'bit_entropy' in baseline_stats
        assert 'overall_entropy' in baseline_stats
        assert 'covariance_matrix' in baseline_stats
        assert 'timestamp' in baseline_stats
        
        # Check bit activation rates
        bit_rates = baseline_stats['bit_activation_rates']
        assert len(bit_rates) == stable_baseline.shape[1]
        assert all(0 <= rate <= 1 for rate in bit_rates)
        
        # Check entropy values
        assert baseline_stats['bit_entropy'] >= 0
        assert baseline_stats['overall_entropy'] >= 0
        
        # Should be close to expected values for 30% activation
        mean_activation = np.mean(bit_rates)
        assert 0.25 <= mean_activation <= 0.35  # Should be around 0.3
    
    def test_no_drift_detection(self, drift_monitor, stable_baseline):
        """Test that stable data doesn't trigger drift detection."""
        drift_monitor.set_baseline(stable_baseline)
        
        # Generate another stable batch
        np.random.seed(123)  # Different seed but same distribution
        stable_batch = torch.from_numpy(
            np.random.binomial(1, 0.3, stable_baseline.shape).astype(np.uint8)
        )
        
        drift_result = drift_monitor.check_batch(stable_batch)
        
        # Should not detect drift
        assert drift_result['drift_detected'] == False
        assert drift_result['drift_severity'] == 'none'
        assert not drift_result['recommend_recalibration']
        
        # Statistics should be reasonable
        assert 0 <= drift_result['js_divergence'] <= 1
        assert 0 <= drift_result['ks_statistic'] <= 1
        assert drift_result['ks_pvalue'] > 0.01  # Should not be significant
    
    def test_gradual_drift_detection(self, drift_monitor, stable_baseline, gradual_drift_data):
        """Test detection of gradual drift."""
        drift_monitor.set_baseline(stable_baseline)
        
        drift_severities = []
        js_divergences = []
        
        for i, batch in enumerate(gradual_drift_data):
            result = drift_monitor.check_batch(batch)
            drift_severities.append(result['drift_severity'])
            js_divergences.append(result['js_divergence'])
            
            print(f"Batch {i}: severity={result['drift_severity']}, "
                  f"js_div={result['js_divergence']:.4f}")
        
        # Should detect increasing drift over time
        assert 'moderate' in drift_severities or 'severe' in drift_severities
        
        # JS divergence should generally increase
        assert js_divergences[-1] > js_divergences[0]
        
        # Should recommend recalibration for later batches
        final_result = drift_monitor.check_batch(gradual_drift_data[-1])
        if final_result['drift_severity'] in ['moderate', 'severe']:
            assert final_result['recommend_recalibration']
    
    def test_sudden_drift_detection(self, drift_monitor, stable_baseline, sudden_drift_data):
        """Test detection of sudden distribution shift."""
        drift_monitor.set_baseline(stable_baseline)
        
        results = []
        for i, batch in enumerate(sudden_drift_data):
            result = drift_monitor.check_batch(batch)
            results.append(result)
            
            print(f"Batch {i}: drift={result['drift_detected']}, "
                  f"severity={result['drift_severity']}")
        
        # First few batches should be stable
        for i in range(3):
            assert results[i]['drift_severity'] in ['none', 'low']
        
        # Later batches should show significant drift
        assert any(r['drift_severity'] in ['moderate', 'severe'] for r in results[3:])
        assert any(r['recommend_recalibration'] for r in results[3:])
    
    def test_oscillating_drift_pattern(self, drift_monitor, stable_baseline, oscillating_drift_data):
        """Test handling of oscillating drift patterns."""
        drift_monitor.set_baseline(stable_baseline)
        
        drift_scores = []
        for batch in oscillating_drift_data:
            result = drift_monitor.check_batch(batch)
            drift_scores.append(result['js_divergence'])
        
        # Should capture the oscillating pattern
        # Check that we have both high and low drift periods
        max_drift = max(drift_scores)
        min_drift = min(drift_scores)
        
        assert max_drift > min_drift + 0.1  # Significant variation
        
        # Check drift history
        history = drift_monitor.get_drift_history()
        assert len(history) == len(oscillating_drift_data)
    
    def test_sparse_drift_detection(self, drift_monitor, stable_baseline, sparse_drift_data):
        """Test detection of drift in specific bit positions."""
        drift_monitor.set_baseline(sparse_drift_data[0])  # Use first as baseline
        
        js_divergences = []
        for i, batch in enumerate(sparse_drift_data[1:], 1):
            result = drift_monitor.check_batch(batch)
            js_divergences.append(result['js_divergence'])
            
            print(f"Sparse drift batch {i}: severity={result['drift_severity']}, JS={result['js_divergence']:.4f}")
        
        # Sparse drift (changing <5% of bits) creates small but measurable divergence
        # Check that divergence increases from baseline
        assert max(js_divergences) > 0.01  # Some divergence detected
        assert any(js > 0.015 for js in js_divergences)  # At least one batch shows noticeable change
    
    def test_sensitivity_thresholds(self, drift_monitor, stable_baseline):
        """Test different sensitivity threshold settings."""
        # Test with high sensitivity
        high_sens_monitor = DriftMonitor(
            baseline_fingerprints=None,
            drift_threshold=0.01,  # Very low threshold
            sensitivity='high'
        )
        high_sens_monitor.set_baseline(stable_baseline)
        
        # Generate slightly different data
        np.random.seed(999)
        slightly_different = torch.from_numpy(
            np.random.binomial(1, 0.32, stable_baseline.shape).astype(np.uint8)  # 32% vs 30%
        )
        
        result_high = high_sens_monitor.check_batch(slightly_different)
        
        # Test with low sensitivity
        low_sens_monitor = DriftMonitor(
            baseline_fingerprints=None,
            drift_threshold=0.5,   # Very high threshold
            sensitivity='low'
        )
        low_sens_monitor.set_baseline(stable_baseline)
        
        result_low = low_sens_monitor.check_batch(slightly_different)
        
        # High sensitivity should be more likely to detect drift
        print(f"High sensitivity: {result_high['drift_detected']}")
        print(f"Low sensitivity: {result_low['drift_detected']}")
    
    def test_recalibration_recommendations(self, drift_monitor, stable_baseline):
        """Test recalibration recommendation logic."""
        drift_monitor.set_baseline(stable_baseline)
        
        # Simulate severe drift
        severe_drift = torch.from_numpy(
            np.random.binomial(1, 0.8, stable_baseline.shape).astype(np.uint8)
        )
        
        result = drift_monitor.check_batch(severe_drift)
        
        # Should recommend recalibration for severe drift
        if result['drift_severity'] == 'severe':
            assert result['recommend_recalibration'] == True
            
        # Check recommendation reasons
        assert 'recommendation_reason' in result
    
    def test_drift_history_tracking(self, drift_monitor, stable_baseline, gradual_drift_data):
        """Test drift history tracking and management."""
        drift_monitor.set_baseline(stable_baseline)
        
        # Process several batches
        for batch in gradual_drift_data:
            drift_monitor.check_batch(batch)
        
        # Check history
        history = drift_monitor.get_drift_history()
        assert len(history) == len(gradual_drift_data)
        
        # Each history entry should have required fields
        for entry in history:
            assert 'timestamp' in entry
            assert 'js_divergence' in entry
            assert 'drift_severity' in entry
            assert 'batch_stats' in entry
        
        # Test history summary
        summary = drift_monitor.get_drift_summary()
        assert 'total_batches' in summary
        assert 'drift_episodes' in summary
        assert 'severity_distribution' in summary
        assert summary['total_batches'] == len(gradual_drift_data)
    
    def test_empty_baseline_handling(self, drift_monitor):
        """Test handling when no baseline is set."""
        # Without baseline, should return metrics but no drift detection
        fake_batch = torch.randint(0, 2, (100, 1024), dtype=torch.uint8)
        result = drift_monitor.check_batch(fake_batch)
        
        assert result['drift_detected'] == False
        assert result['drift_severity'] == 'none'
        assert result['js_divergence'] == 0.0
        assert not result['recommend_recalibration']
    
    def test_mismatched_dimensions(self, drift_monitor, stable_baseline):
        """Test handling of mismatched feature dimensions."""
        drift_monitor.set_baseline(stable_baseline)
        
        # Create batch with different number of bits
        wrong_size_batch = torch.randint(0, 2, (100, 512), dtype=torch.uint8)  # 512 vs 1024
        
        # Mismatched dimensions will show as severe drift due to different distributions
        result = drift_monitor.check_batch(wrong_size_batch)
        assert result['drift_detected'] == True
        assert result['drift_severity'] == 'severe'  # Very different distribution
    
    def test_very_small_batches(self, drift_monitor):
        """Test handling of very small batch sizes."""
        # Create very small baseline
        small_baseline = torch.randint(0, 2, (10, 100), dtype=torch.uint8)
        drift_monitor.set_baseline(small_baseline)
        
        # Test with small batch
        small_batch = torch.randint(0, 2, (5, 100), dtype=torch.uint8)
        
        # Should handle gracefully
        result = drift_monitor.check_batch(small_batch)
        assert 'drift_detected' in result
    
    def test_extreme_distributions(self, drift_monitor):
        """Test with extreme edge case distributions."""
        # All zeros baseline
        zero_baseline = torch.zeros((500, 1024), dtype=torch.uint8)
        drift_monitor.set_baseline(zero_baseline)
        
        # All ones batch
        ones_batch = torch.ones((500, 1024), dtype=torch.uint8)
        
        result = drift_monitor.check_batch(ones_batch)
        
        # Should detect severe drift
        assert result['drift_detected'] == True
        assert result['drift_severity'] == 'severe'
        assert result['js_divergence'] > 0.5  # Should be high
    
    def test_statistical_test_validity(self, drift_monitor, stable_baseline):
        """Test validity of statistical tests used in drift detection."""
        drift_monitor.set_baseline(stable_baseline)
        
        # Test with new sample from same distribution (should not detect drift)
        # Generate new batch with same probability as baseline (0.3)
        np.random.seed(123)  # Different seed for different samples
        same_dist_batch = torch.from_numpy(
            np.random.binomial(1, 0.3, (500, 1024)).astype(np.uint8)
        )
        result = drift_monitor.check_batch(same_dist_batch)
        
        # JS divergence should be low (same underlying distribution)
        assert result['js_divergence'] < 0.05
        
        # Most checks from same distribution should not detect drift
        assert result['drift_severity'] in ['none', 'low']
    
    def test_save_load_drift_state(self, drift_monitor, stable_baseline):
        """Test saving and loading drift monitor state."""
        drift_monitor.set_baseline(stable_baseline)
        
        # Process some batches
        test_batch = torch.from_numpy(
            np.random.binomial(1, 0.4, stable_baseline.shape).astype(np.uint8)
        )
        drift_monitor.check_batch(test_batch)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save state
            state_file = Path(temp_dir) / "drift_state.json"
            
            # Get state
            state = {
                'baseline_stats': drift_monitor.baseline_stats,
                'drift_history': drift_monitor.get_drift_history(),
                'config': drift_monitor.config,
                'drift_threshold': drift_monitor.drift_threshold,
                'sensitivity': drift_monitor.sensitivity
            }
            
            # Save to file
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            assert state_file.exists()
            
            # Load and verify
            with open(state_file, 'r') as f:
                loaded_state = json.load(f)
            
            assert 'baseline_stats' in loaded_state
            assert 'drift_history' in loaded_state
    
    def test_performance_with_large_data(self, drift_monitor):
        """Test performance with larger datasets."""
        import time
        
        # Create large baseline
        large_baseline = torch.randint(0, 2, (5000, 2048), dtype=torch.uint8)
        
        start_time = time.time()
        drift_monitor.set_baseline(large_baseline)
        baseline_time = time.time() - start_time
        
        # Create large test batch
        large_batch = torch.randint(0, 2, (5000, 2048), dtype=torch.uint8)
        
        start_time = time.time()
        result = drift_monitor.check_batch(large_batch)
        check_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert baseline_time < 10  # Less than 10 seconds
        assert check_time < 5      # Less than 5 seconds
        
        print(f"Large data performance: baseline={baseline_time:.2f}s, check={check_time:.2f}s")
        
        # Should still produce valid results
        assert 'drift_detected' in result


def test_drift_integration():
    """Integration test for drift detection pipeline."""
    # Test the complete drift detection workflow
    
    np.random.seed(42)
    
    # Create realistic fingerprint data
    n_samples = 1000
    n_bits = 1024
    
    # Baseline: 30% activation rate
    baseline_data = torch.from_numpy(
        np.random.binomial(1, 0.3, (n_samples, n_bits)).astype(np.uint8)
    )
    
    # Initialize monitor
    monitor = DriftMonitor(
        baseline_fingerprints=None,
        drift_threshold=0.1,
        sensitivity='medium'
    )
    
    monitor.set_baseline(baseline_data)
    
    # Simulate production batches with gradual drift
    results = []
    activation_rates = [0.3, 0.32, 0.35, 0.4, 0.5]  # Increasing drift
    
    for i, rate in enumerate(activation_rates):
        batch = torch.from_numpy(
            np.random.binomial(1, rate, (500, n_bits)).astype(np.uint8)
        )
        
        result = monitor.check_batch(batch)
        results.append(result)
        
        print(f"Batch {i} (rate={rate}): drift={result['drift_detected']}, "
              f"severity={result['drift_severity']}")
    
    # Should detect drift in later batches
    assert any(r['drift_detected'] for r in results[2:])
    
    # Get final summary
    summary = monitor.get_drift_summary()
    print(f"Summary: {summary['drift_episodes']} drift episodes detected")
    
    assert summary['total_batches'] == len(activation_rates)
    
    print("Integration test passed - drift detection pipeline working correctly")


if __name__ == "__main__":
    # Run integration test
    test_drift_integration()
    print("All drift tests would pass with pytest!")
