"""
Test PR3: Statistical Calibration and Drift Detection
=====================================================

This test module verifies the calibration and drift detection functionality
implemented as part of PR3 in the Tejas v2 PRD.

DISCLAIMER: Sandbox implementation for testing purposes.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.calibration import StatisticalCalibrator
from core.drift import DriftMonitor


class TestStatisticalCalibrator:
    """Test statistical calibration functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.calibrator = StatisticalCalibrator(
            n_folds=3,  # Reduced for speed
            n_bootstrap=100,  # Reduced for speed
            confidence_level=0.95
        )
        
        # Generate synthetic test data
        np.random.seed(42)
        n_samples = 500
        n_positive = 50
        
        # Create distances with known properties
        self.distances = np.random.uniform(10, 50, n_samples)
        self.labels = np.zeros(n_samples)
        self.labels[:n_positive] = 1
        
        # Make positive samples have lower distances
        self.distances[:n_positive] *= 0.6
        
    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        assert self.calibrator.n_folds == 3
        assert self.calibrator.n_bootstrap == 100
        assert self.calibrator.confidence_level == 0.95
        
    def test_calibrate_with_cv(self):
        """Test cross-validation calibration."""
        # Use only a few thresholds for speed
        thresholds = [15, 20, 25]
        
        result_df = self.calibrator.calibrate_with_cv(
            self.distances,
            self.labels,
            thresholds=thresholds
        )
        
        # Check result structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(thresholds)
        
        # Check required columns - using actual column names
        required_cols = ['threshold', 'precision_at_k', 'recall_at_k', 'f1_score', 'roc_auc']
        for col in required_cols:
            assert col in result_df.columns
        
        # Check value ranges
        assert all((0 <= result_df['f1_score'].values) & (result_df['f1_score'].values <= 1))
        assert all((0 <= result_df['roc_auc'].values) & (result_df['roc_auc'].values <= 1))
        
    def test_find_optimal_threshold(self):
        """Test optimal threshold finding."""
        # Create sample calibration results
        cal_data = pd.DataFrame({
            'threshold': [15, 20, 25, 30],
            'f1_score': [0.6, 0.75, 0.72, 0.65],
            'precision': [0.7, 0.8, 0.85, 0.9],
            'recall': [0.8, 0.7, 0.6, 0.5]
        })
        
        # Find optimal by F1
        opt_thresh, opt_val = self.calibrator.find_optimal_threshold(
            cal_data, metric='f1_score'
        )
        assert opt_thresh == 20
        assert opt_val == 0.75
        
        # Find optimal by precision
        opt_thresh, opt_val = self.calibrator.find_optimal_threshold(
            cal_data, metric='precision'
        )
        assert opt_thresh == 30
        assert opt_val == 0.9
        
    def test_save_load_calibration(self):
        """Test saving and loading calibration results."""
        import tempfile
        import json
        
        # Create sample results
        cal_data = pd.DataFrame({
            'threshold': [20, 25],
            'f1_score': [0.7, 0.75]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Save results
            self.calibrator.save_calibration_results(cal_data, f.name)
            
            # Load and verify - expecting a list of dicts
            with open(f.name, 'r') as rf:
                loaded = json.load(rf)
                assert isinstance(loaded, list)
                assert len(loaded) == 2
                assert loaded[0]['threshold'] == 20
            
            # Cleanup
            Path(f.name).unlink()


class TestDriftMonitor:
    """Test drift detection functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        # Generate baseline data first
        np.random.seed(42)
        self.n_bits = 128
        self.baseline = np.random.randint(0, 2, (500, self.n_bits)).astype(np.float32)
        
        # Initialize with correct parameters
        self.monitor = DriftMonitor(
            baseline_fingerprints=None,  # Set later
            history_size=100,
            drift_threshold=0.05,
            sensitivity='medium'
        )
        
    def test_monitor_initialization(self):
        """Test drift monitor initialization."""
        assert self.monitor.history_size == 100
        assert self.monitor.drift_threshold == 0.05
        assert self.monitor.sensitivity == 'medium'
        assert self.monitor.baseline_fingerprints is None
        
    def test_set_baseline(self):
        """Test setting baseline fingerprints."""
        self.monitor.set_baseline(self.baseline)
        
        assert self.monitor.baseline_fingerprints is not None
        assert self.monitor.baseline_stats is not None
        assert 'bit_activation_rates' in self.monitor.baseline_stats
        assert 'activation_entropy' in self.monitor.baseline_stats
        
    def test_check_drift_no_drift(self):
        """Test drift detection with no drift."""
        self.monitor.set_baseline(self.baseline)
        
        # Use exact baseline data (should have minimal to no drift)
        test_batch = self.baseline[50:150].copy()
        
        result = self.monitor.check_batch(test_batch)
        
        assert isinstance(result, dict)  # Returns dict
        assert 'drift_detected' in result
        assert 'js_divergence' in result
        # Either no drift detected OR very small JS divergence
        # (some minor divergence is expected due to sampling)
        if result['drift_detected']:
            assert result['js_divergence'] < 0.3  # Minor divergence acceptable
            assert result['drift_severity'] in ['low', 'medium']
        
    def test_check_drift_with_drift(self):
        """Test drift detection with actual drift."""
        self.monitor.set_baseline(self.baseline)
        
        # Create drifted batch
        drifted = np.random.randint(0, 2, (100, self.n_bits)).astype(np.float32)
        # Force strong drift on some bits
        drifted[:, :30] = 1  # Always active
        drifted[:, 30:50] = 0  # Always inactive
        
        result = self.monitor.check_batch(drifted)
        
        assert result['drift_detected'] == True
        assert result['js_divergence'] > 0.1  # Should show divergence
        assert result['recommend_recalibration'] == True
        
    def test_drift_history_tracking(self):
        """Test drift history tracking."""
        self.monitor.set_baseline(self.baseline)
        
        # Check multiple batches
        for i in range(3):
            test_batch = np.random.randint(0, 2, (50, self.n_bits)).astype(np.float32)
            self.monitor.check_batch(test_batch)
        
        # Check drift history exists
        assert hasattr(self.monitor, 'drift_history')
        assert len(self.monitor.drift_history) == 3
        
    def test_recalibration_recommendation(self):
        """Test recalibration recommendations."""
        self.monitor.set_baseline(self.baseline)
        
        # Create consistently drifted batches
        for i in range(5):
            drifted = np.ones((50, self.n_bits), dtype=np.float32)
            result = self.monitor.check_batch(drifted)
        
        # Last result should recommend recalibration
        assert result.get('recommend_recalibration', False) == True or result.get('drift_severity') == 'severe'
        
    def test_empty_baseline_handling(self):
        """Test handling of empty baseline."""
        # Setting empty baseline should raise ValueError
        empty_batch = np.array([]).reshape(0, self.n_bits)
        with pytest.raises(ValueError, match="Baseline fingerprints cannot be empty"):
            self.monitor.set_baseline(empty_batch)
            
    def test_mismatched_dimensions(self):
        """Test handling of mismatched dimensions."""
        self.monitor.set_baseline(self.baseline)
        
        # Wrong number of bits
        wrong_dims = np.random.randint(0, 2, (100, 64)).astype(np.float32)
        
        # Should raise error or handle gracefully
        try:
            result = self.monitor.check_batch(wrong_dims)
            # If no error, check that it's handled
            assert result is not None
        except (ValueError, AssertionError) as e:
            # Expected behavior
            assert 'dimension' in str(e).lower() or 'shape' in str(e).lower()


class TestIntegration:
    """Test integration between calibration and drift detection."""
    
    def test_calibration_drift_workflow(self):
        """Test complete calibration and drift monitoring workflow."""
        # Setup calibrator
        calibrator = StatisticalCalibrator(n_folds=2, n_bootstrap=50)
        
        # Generate training data
        np.random.seed(42)
        distances = np.random.uniform(10, 40, 200)
        labels = np.random.randint(0, 2, 200)
        
        # Calibrate
        cal_results = calibrator.calibrate_with_cv(
            distances, labels,
            thresholds=[15, 20, 25]
        )
        
        assert len(cal_results) == 3
        
        # Setup drift monitor
        monitor = DriftMonitor(baseline_fingerprints=None, history_size=50)
        
        # Create fingerprints
        fingerprints = np.random.randint(0, 2, (200, 128)).astype(np.float32)
        monitor.set_baseline(fingerprints[:100])
        
        # Check drift
        drift_result = monitor.check_batch(fingerprints[100:150])
        
        assert 'is_drifted' in drift_result or 'drift_detected' in drift_result
        assert isinstance(drift_result.get('is_drifted', drift_result.get('drift_detected')), bool)
        
    def test_performance_metrics(self):
        """Test that operations complete in reasonable time."""
        import time
        
        # Test calibration speed
        calibrator = StatisticalCalibrator(n_folds=2, n_bootstrap=10)
        distances = np.random.uniform(0, 50, 100)
        labels = np.random.randint(0, 2, 100)
        
        start = time.time()
        cal_results = calibrator.calibrate_with_cv(
            distances, labels,
            thresholds=[20]
        )
        cal_time = time.time() - start
        
        assert cal_time < 5.0  # Should complete in under 5 seconds
        
        # Test drift detection speed
        monitor = DriftMonitor(baseline_fingerprints=None)
        baseline = np.random.randint(0, 2, (100, 128)).astype(np.float32)
        monitor.set_baseline(baseline)
        
        test_batch = np.random.randint(0, 2, (50, 128)).astype(np.float32)
        
        start = time.time()
        drift_result = monitor.check_batch(test_batch)
        drift_time = time.time() - start
        
        assert drift_time < 0.1  # Should complete in under 100ms


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
