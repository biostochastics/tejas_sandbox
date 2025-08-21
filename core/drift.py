"""
Model Drift Detection for Tejas Binary Fingerprints
===================================================

Monitors statistical properties of binary fingerprints to detect model drift
and recommend recalibration. Tracks bit activation rates, entropy, and 
statistical divergence from baseline distributions.

Key Features:
- Real-time drift detection on incoming batches
- Statistical tests for distribution changes
- Configurable drift thresholds and sensitivity
- Automatic recalibration recommendations
- Comprehensive drift metrics and reporting
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import deque
import time
import json
from pathlib import Path

try:
    import scipy.stats as stats
    from scipy.spatial.distance import jensenshannon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    jensenshannon = None
    stats = None
    
    # Fallback implementation for Jensen-Shannon divergence
    def jensenshannon_fallback(p, q):
        """Pure Python Jensen-Shannon divergence implementation."""
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate average distribution
        m = 0.5 * (p + q)
        
        # Calculate KL divergences
        def kl_div(a, b):
            return np.sum(a * np.log2(a / b))
        
        # JS divergence is average of KL divergences
        js_div = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        
        # Return square root for consistency with scipy
        return np.sqrt(js_div)
    
    # Use fallback if scipy not available
    if jensenshannon is None:
        jensenshannon = jensenshannon_fallback

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Container for drift detection metrics."""
    timestamp: float
    batch_size: int
    
    # Bit activation statistics
    bit_activation_rates: np.ndarray
    activation_entropy: float
    activation_mean: float
    activation_std: float
    
    # Distribution comparison with baseline
    js_divergence: float  # Jensen-Shannon divergence
    ks_statistic: float   # Kolmogorov-Smirnov test
    ks_pvalue: float
    
    # Population drift indicators  
    mean_hamming_distance: float
    hamming_std: float
    
    # Drift flags
    is_drifted: bool
    drift_severity: str  # 'none', 'low', 'moderate', 'severe'
    
    # Recommendations
    recommend_recalibration: bool
    confidence_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, handling numpy arrays."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(result['bit_activation_rates'], np.ndarray):
            activation_rates_list = result['bit_activation_rates'].tolist()
            result['bit_activation_rates'] = activation_rates_list
        else:
            activation_rates_list = result['bit_activation_rates']
        
        # Add batch_stats for backward compatibility (reference, not copy)
        result['batch_stats'] = {
            'activation_rates': activation_rates_list,  # Reference the same list
            'activation_entropy': result['activation_entropy'],
            'activation_mean': result['activation_mean'],
            'activation_std': result['activation_std']
        }
        return result


class DriftMonitor:
    """
    Drift Detection and Monitoring for TEJAS V2.

    ⚠️ DISCLAIMER: Experimental research code - not validated for production.

    Provides real-time monitoring of fingerprint distribution drift using
    statistical methods to detect when recalibration is needed.

    Attribution: Drift detection methodology based on standard statistical
    tests and divergence measures. Implementation is original work for the
    TEJAS sandbox project.
    """
    
    def __init__(self,
                 baseline_fingerprints: Optional[np.ndarray] = None,
                 history_size: int = 1000,
                 drift_threshold: float = 0.05,
                 js_threshold: Optional[float] = None,
                 entropy_threshold: Optional[float] = None,
                 sensitivity: str = 'medium',
                 min_batch_size: int = 50):
        """
        Initialize drift monitor.
        
        Args:
            baseline_fingerprints: Reference fingerprints for comparison
            history_size: Number of recent batches to keep in memory
            drift_threshold: P-value threshold for drift detection
            sensitivity: Drift sensitivity ('low', 'medium', 'high')
            min_batch_size: Minimum batch size for reliable drift detection
        """
        self.baseline_fingerprints = baseline_fingerprints
        self.history_size = history_size
        self.drift_threshold = drift_threshold
        self.sensitivity = sensitivity
        self.min_batch_size = min_batch_size
        
        # Internal state
        self.baseline_stats = None
        self.history = deque(maxlen=history_size)
        self.drift_history = []
        self.total_batches_processed = 0
        
        # Sensitivity configuration
        sensitivity_config = {
            'low': {'js_threshold': 0.3, 'activation_threshold': 0.2, 'recal_threshold': 0.4},
            'medium': {'js_threshold': 0.2, 'activation_threshold': 0.15, 'recal_threshold': 0.3},
            'high': {'js_threshold': 0.1, 'activation_threshold': 0.1, 'recal_threshold': 0.2}
        }
        
        self.config = sensitivity_config.get(sensitivity, sensitivity_config['medium'])
        
        # Override with provided thresholds if specified
        if js_threshold is not None:
            self.config['js_threshold'] = js_threshold
        if entropy_threshold is not None:
            self.config['activation_threshold'] = entropy_threshold
        
        # Compute baseline statistics if provided
        if baseline_fingerprints is not None:
            self._compute_baseline_stats()
        
        logger.info(f"Initialized DriftMonitor:")
        logger.info(f"  Sensitivity: {sensitivity}")
        logger.info(f"  Drift threshold: {drift_threshold}")
        logger.info(f"  Min batch size: {min_batch_size}")
        logger.info(f"  Baseline: {'Set' if baseline_fingerprints is not None else 'None'}")
    
    def set_baseline(self, baseline_fingerprints: np.ndarray) -> None:
        """
        Set baseline fingerprints for drift comparison.
        
        Args:
            baseline_fingerprints: Binary fingerprint matrix to use as baseline
        """
        if baseline_fingerprints is None or baseline_fingerprints.size == 0:
            raise ValueError("Baseline fingerprints cannot be empty")
        
        # Convert PyTorch tensor to numpy if needed
        if hasattr(baseline_fingerprints, 'numpy'):
            baseline_fingerprints = baseline_fingerprints.numpy()
        
        self.baseline_fingerprints = baseline_fingerprints
        self._compute_baseline_stats()
    
    def _compute_baseline_stats(self) -> None:
        """Compute statistical properties of baseline fingerprints."""
        if self.baseline_fingerprints is None:
            return
        
        fp = self.baseline_fingerprints
        timestamp = time.time()
        
        # Bit activation rates
        activation_rates = np.mean(fp, axis=0)
        
        # Activation entropy (measure of bit balance)
        # H = -sum(p * log(p) + (1-p) * log(1-p)) for each bit
        eps = 1e-10  # Avoid log(0)
        p = np.clip(activation_rates, eps, 1-eps)
        bit_entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p))
        total_entropy = np.mean(bit_entropy)
        
        # Note: Covariance matrix computation removed (was unused)
        # Could be added back if needed for advanced drift detection
        
        # Hamming distance statistics (pairwise distances in sample)
        if len(fp) > 1:
            # Sample for efficiency if large
            if len(fp) > 1000:
                sample_idx = np.random.choice(len(fp), 1000, replace=False)
                fp_sample = fp[sample_idx]
            else:
                fp_sample = fp
            
            # Compute pairwise Hamming distances
            n_sample = len(fp_sample)
            hamming_distances = []
            for i in range(min(n_sample, 500)):  # Limit for efficiency
                for j in range(i+1, min(n_sample, i+100)):
                    dist = np.sum(fp_sample[i] != fp_sample[j])
                    hamming_distances.append(dist)
            
            mean_hamming = np.mean(hamming_distances) if hamming_distances else 0
            std_hamming = np.std(hamming_distances) if hamming_distances else 0
        else:
            mean_hamming = std_hamming = 0
        
        self.baseline_stats = {
            'bit_activation_rates': activation_rates,
            'bit_entropy': total_entropy,
            'overall_entropy': total_entropy,
            'activation_entropy': total_entropy,
            'activation_mean': np.mean(activation_rates),
            'activation_std': np.std(activation_rates),
            'mean_hamming_distance': mean_hamming,
            'hamming_std': std_hamming,
            'n_samples': len(fp),
            'timestamp': timestamp
        }
        
        logger.info(f"Baseline stats computed:")
        logger.info(f"  Activation entropy: {total_entropy:.3f}")
        logger.info(f"  Mean activation rate: {np.mean(activation_rates):.3f}")
        logger.info(f"  Mean Hamming distance: {mean_hamming:.1f}")
    
    def check_batch(self, fingerprints_batch: np.ndarray) -> DriftMetrics:
        """
        Check a batch of fingerprints for drift.
        
        Args:
            fingerprints_batch: Batch of binary fingerprints to analyze
        
        Returns:
            DriftMetrics with drift analysis results
        """
        # Convert PyTorch tensor to numpy if needed
        if hasattr(fingerprints_batch, 'numpy'):
            fingerprints_batch = fingerprints_batch.numpy()
        
        self.total_batches_processed += 1
        timestamp = time.time()
        
        # Input validation
        if fingerprints_batch.size == 0:
            logger.warning("Empty batch provided to drift monitor")
            return self._create_empty_metrics(timestamp)
        
        if len(fingerprints_batch) < self.min_batch_size:
            logger.warning(f"Batch size {len(fingerprints_batch)} below minimum {self.min_batch_size}")
        
        # Compute current batch statistics
        batch_stats = self._compute_batch_stats(fingerprints_batch)
        
        # Compare with baseline if available
        if self.baseline_stats is not None:
            drift_metrics = self._detect_drift(batch_stats, timestamp, len(fingerprints_batch))
        else:
            # No baseline - just collect statistics
            drift_metrics = DriftMetrics(
                timestamp=timestamp,
                batch_size=len(fingerprints_batch),
                bit_activation_rates=batch_stats['activation_rates'],
                activation_entropy=batch_stats['activation_entropy'],
                activation_mean=batch_stats['activation_mean'],
                activation_std=batch_stats['activation_std'],
                js_divergence=0.0,
                ks_statistic=0.0,
                ks_pvalue=1.0,
                mean_hamming_distance=batch_stats['mean_hamming_distance'],
                hamming_std=batch_stats['hamming_std'],
                is_drifted=False,
                drift_severity='none',
                recommend_recalibration=False,
                confidence_score=0.0
            )
        
        # Update history
        self.history.append(drift_metrics)
        self.drift_history.append(drift_metrics)
        
        # Log drift detection
        if drift_metrics.is_drifted:
            logger.warning(f"Drift detected! Severity: {drift_metrics.drift_severity}")
            logger.warning(f"  JS divergence: {drift_metrics.js_divergence:.3f}")
            logger.warning(f"  KS p-value: {drift_metrics.ks_pvalue:.3f}")
            if drift_metrics.recommend_recalibration:
                logger.warning("  RECOMMENDATION: Recalibrate model")
        
        # Return as dictionary for compatibility
        recommendation_reason = ""
        if drift_metrics.recommend_recalibration:
            if drift_metrics.drift_severity == 'severe':
                recommendation_reason = "Severe drift detected"
            elif drift_metrics.js_divergence > self.config.get('recal_threshold', 0.1):
                recommendation_reason = f"JS divergence ({drift_metrics.js_divergence:.3f}) exceeds threshold"
            elif drift_metrics.ks_pvalue < self.drift_threshold / 2:
                recommendation_reason = f"KS p-value ({drift_metrics.ks_pvalue:.3f}) indicates significant drift"
            else:
                recommendation_reason = "Moderate drift accumulated over time"
        
        return {
            'drift_detected': drift_metrics.is_drifted,
            'drift_severity': drift_metrics.drift_severity,
            'recommend_recalibration': drift_metrics.recommend_recalibration,
            'recommendation_reason': recommendation_reason,
            'js_divergence': drift_metrics.js_divergence,
            'ks_statistic': drift_metrics.ks_statistic,
            'ks_pvalue': drift_metrics.ks_pvalue,
            'confidence_score': drift_metrics.confidence_score,
            'batch_size': drift_metrics.batch_size,
            'timestamp': drift_metrics.timestamp,
            'activation_mean': drift_metrics.activation_mean,
            'activation_std': drift_metrics.activation_std,
            'mean_hamming_distance': drift_metrics.mean_hamming_distance,
            'hamming_std': drift_metrics.hamming_std,
            'batch_stats': {
                'activation_rates': drift_metrics.bit_activation_rates.tolist() if isinstance(drift_metrics.bit_activation_rates, np.ndarray) else drift_metrics.bit_activation_rates,
                'activation_entropy': drift_metrics.activation_entropy,
                'activation_mean': drift_metrics.activation_mean,
                'activation_std': drift_metrics.activation_std
            }
        }
    
    def _compute_batch_stats(self, fingerprints: np.ndarray) -> Dict:
        """Compute statistical properties of a fingerprint batch."""
        # Bit activation rates
        activation_rates = np.mean(fingerprints, axis=0)
        
        # Activation entropy
        eps = 1e-10
        p = np.clip(activation_rates, eps, 1-eps)
        bit_entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p))
        total_entropy = np.mean(bit_entropy)
        
        # Hamming distance statistics (sample for efficiency)
        if len(fingerprints) > 1:
            n_sample = min(len(fingerprints), 100)
            sample_idx = np.random.choice(len(fingerprints), n_sample, replace=False)
            fp_sample = fingerprints[sample_idx]
            
            hamming_distances = []
            for i in range(n_sample):
                for j in range(i+1, min(n_sample, i+20)):
                    dist = np.sum(fp_sample[i] != fp_sample[j])
                    hamming_distances.append(dist)
            
            mean_hamming = np.mean(hamming_distances) if hamming_distances else 0
            std_hamming = np.std(hamming_distances) if hamming_distances else 0
        else:
            mean_hamming = std_hamming = 0
        
        return {
            'activation_rates': activation_rates,
            'activation_entropy': total_entropy,
            'activation_mean': np.mean(activation_rates),
            'activation_std': np.std(activation_rates),
            'mean_hamming_distance': mean_hamming,
            'hamming_std': std_hamming
        }
    
    def _detect_drift(self, batch_stats: Dict, timestamp: float, batch_size: int) -> DriftMetrics:
        """Detect drift by comparing batch statistics with baseline."""
        baseline = self.baseline_stats
        
        # Jensen-Shannon divergence for activation rate distributions
        try:
            # Create histograms for comparison
            baseline_hist, _ = np.histogram(baseline['bit_activation_rates'], bins=20, range=(0, 1))
            batch_hist, _ = np.histogram(batch_stats['activation_rates'], bins=20, range=(0, 1))
            
            # Normalize to probabilities
            baseline_hist = baseline_hist / (baseline_hist.sum() + 1e-10)
            batch_hist = batch_hist / (batch_hist.sum() + 1e-10)
            
            # Use jensenshannon (either scipy or fallback)
            js_div = jensenshannon(baseline_hist, batch_hist) ** 2  # Square for JS divergence
        except Exception as e:
            logger.warning(f"Failed to compute JS divergence: {e}")
            js_div = 0.0
        
        # Kolmogorov-Smirnov test for distribution comparison
        if HAS_SCIPY and stats is not None:
            try:
                ks_stat, ks_pval = stats.ks_2samp(
                    baseline['bit_activation_rates'], 
                    batch_stats['activation_rates']
                )
            except Exception as e:
                logger.warning(f"Failed to compute KS test: {e}")
                ks_stat = 0.0
                ks_pval = 1.0
        else:
            # Simple fallback: use maximum difference between CDFs
            try:
                # Sort both arrays
                baseline_sorted = np.sort(baseline['bit_activation_rates'])
                batch_sorted = np.sort(batch_stats['activation_rates'])
                
                # Compute empirical CDFs
                n1 = len(baseline_sorted)
                n2 = len(batch_sorted)
                
                # Combine and sort all values
                all_values = np.concatenate([baseline_sorted, batch_sorted])
                all_values = np.unique(all_values)
                
                # Compute maximum difference between CDFs
                max_diff = 0.0
                for val in all_values:
                    cdf1 = np.sum(baseline_sorted <= val) / n1
                    cdf2 = np.sum(batch_sorted <= val) / n2
                    max_diff = max(max_diff, abs(cdf1 - cdf2))
                
                ks_stat = max_diff
                # Approximate p-value (simplified)
                ks_pval = np.exp(-2 * n1 * n2 * ks_stat**2 / (n1 + n2))
            except:
                ks_stat = 0.0
                ks_pval = 1.0
        
        # Drift detection logic
        drift_indicators = []
        
        # 1. Jensen-Shannon divergence threshold
        if js_div > self.config['js_threshold']:
            drift_indicators.append('js_divergence')
        
        # 2. Statistical significance (KS test)
        if ks_pval < self.drift_threshold:
            drift_indicators.append('ks_test')
        
        # 3. Activation entropy change
        entropy_change = abs(batch_stats['activation_entropy'] - baseline['activation_entropy'])
        if entropy_change > self.config['activation_threshold']:
            drift_indicators.append('entropy_change')
        
        # 4. Mean activation rate change
        mean_change = abs(batch_stats['activation_mean'] - baseline['activation_mean'])
        if mean_change > self.config['activation_threshold']:
            drift_indicators.append('mean_activation')
        
        # 5. Hamming distance distribution change
        hamming_change = abs(batch_stats['mean_hamming_distance'] - baseline['mean_hamming_distance'])
        expected_hamming = baseline['mean_hamming_distance']
        if expected_hamming > 0 and hamming_change / expected_hamming > 0.2:
            drift_indicators.append('hamming_distribution')
        
        # Determine drift severity and recommendations
        n_indicators = len(drift_indicators)
        is_drifted = n_indicators > 0
        
        if n_indicators == 0:
            severity = 'none'
            confidence = 0.0
        elif n_indicators == 1:
            severity = 'low'
            confidence = 0.3
        elif n_indicators == 2:
            severity = 'moderate' 
            confidence = 0.6
        else:
            severity = 'severe'
            confidence = 0.9
        
        # Recalibration recommendation
        recommend_recal = (
            severity in ['moderate', 'severe'] or 
            js_div > self.config['recal_threshold'] or
            ks_pval < self.drift_threshold / 2
        )
        
        return DriftMetrics(
            timestamp=timestamp,
            batch_size=batch_size,  # Use actual batch size passed as parameter
            bit_activation_rates=batch_stats['activation_rates'],
            activation_entropy=batch_stats['activation_entropy'],
            activation_mean=batch_stats['activation_mean'],
            activation_std=batch_stats['activation_std'],
            js_divergence=js_div,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            mean_hamming_distance=batch_stats['mean_hamming_distance'],
            hamming_std=batch_stats['hamming_std'],
            is_drifted=is_drifted,
            drift_severity=severity,
            recommend_recalibration=recommend_recal,
            confidence_score=confidence
        )
    
    def _create_empty_metrics(self, timestamp: float) -> DriftMetrics:
        """Create empty drift metrics for invalid inputs."""
        return DriftMetrics(
            timestamp=timestamp,
            batch_size=0,
            bit_activation_rates=np.array([]),
            activation_entropy=0.0,
            activation_mean=0.0,
            activation_std=0.0,
            js_divergence=0.0,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            mean_hamming_distance=0.0,
            hamming_std=0.0,
            is_drifted=False,
            drift_severity='none',
            recommend_recalibration=False,
            confidence_score=0.0
        )
    
    def get_drift_history(self) -> List[Dict]:
        """Get the drift detection history as a list of dictionaries."""
        return [m.to_dict() for m in self.drift_history]
    
    def get_drift_summary(self, window_size: int = 10) -> Dict:
        """Get summary of recent drift detection results (alias for get_summary)."""
        return self.get_summary(window_size)
    
    def get_summary(self, window_size: int = 10) -> Dict:
        """Get summary of recent drift detection results."""
        if not self.drift_history:
            return {'no_data': True}
        
        recent_history = self.drift_history[-window_size:]
        
        # Count drift occurrences
        drift_counts = {
            'none': sum(1 for m in recent_history if m.drift_severity == 'none'),
            'low': sum(1 for m in recent_history if m.drift_severity == 'low'),
            'moderate': sum(1 for m in recent_history if m.drift_severity == 'moderate'),
            'severe': sum(1 for m in recent_history if m.drift_severity == 'severe')
        }
        
        # Recent metrics
        latest = recent_history[-1]
        
        # Identify drift episodes (consecutive drift detections)
        drift_episodes = []
        current_episode = None
        for i, m in enumerate(recent_history):
            if m.is_drifted:
                if current_episode is None:
                    current_episode = {'start': i, 'end': i, 'severities': [m.drift_severity]}
                else:
                    current_episode['end'] = i
                    current_episode['severities'].append(m.drift_severity)
            elif current_episode is not None:
                drift_episodes.append(current_episode)
                current_episode = None
        if current_episode is not None:
            drift_episodes.append(current_episode)
        
        # Calculate severity distribution
        total_drift = sum(drift_counts[k] for k in ['low', 'moderate', 'severe'])
        severity_distribution = {}
        if total_drift > 0:
            severity_distribution = {
                'low': drift_counts['low'] / total_drift,
                'moderate': drift_counts['moderate'] / total_drift,
                'severe': drift_counts['severe'] / total_drift
            }
        
        return {
            'total_batches': self.total_batches_processed,
            'window_size': len(recent_history),
            'drift_counts': drift_counts,
            'drift_rate': sum(drift_counts[k] for k in ['low', 'moderate', 'severe']) / len(recent_history),
            'drift_episodes': drift_episodes,
            'severity_distribution': severity_distribution,
            'latest_metrics': latest.to_dict(),
            'avg_js_divergence': np.mean([m.js_divergence for m in recent_history]),
            'avg_confidence': np.mean([m.confidence_score for m in recent_history]),
            'recalibration_recommended': latest.recommend_recalibration
        }
    
    def recommend_recalibration(self) -> Tuple[bool, str]:
        """
        Get recalibration recommendation based on recent drift history.
        
        Returns:
            (should_recalibrate, reason) tuple
        """
        if not self.drift_history:
            return False, "No drift history available"
        
        recent_metrics = self.drift_history[-10:]  # Last 10 batches
        
        # Count high-severity drifts
        high_severity_count = sum(1 for m in recent_metrics if m.drift_severity == 'severe')
        
        # Count recalibration recommendations
        recal_recommendations = sum(1 for m in recent_metrics if m.recommend_recalibration)
        
        # Decision logic
        if high_severity_count >= 3:
            return True, f"Multiple high-severity drifts detected ({high_severity_count}/10)"
        
        if recal_recommendations >= 5:
            return True, f"Frequent recalibration recommendations ({recal_recommendations}/10)"
        
        # Check latest metrics
        latest = recent_metrics[-1]
        if latest.drift_severity == 'severe' and latest.confidence_score > 0.8:
            return True, f"High-confidence drift detected (confidence: {latest.confidence_score:.2f})"
        
        return False, "No recalibration needed"
    
    def save_drift_history(self, filepath: Union[str, Path]) -> None:
        """Save drift detection history to file."""
        import os
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        # Convert numpy arrays to lists for JSON serialization
        baseline_stats_json = {}
        if self.baseline_stats:
            for key, value in self.baseline_stats.items():
                if isinstance(value, np.ndarray):
                    baseline_stats_json[key] = value.tolist()
                else:
                    baseline_stats_json[key] = value
        
        history_data = {
            'total_batches': self.total_batches_processed,
            'config': self.config,
            'baseline_stats': baseline_stats_json,
            'drift_history': [m.to_dict() for m in self.drift_history]
        }
        
        with open(filepath, 'w') as f:
            os.chmod(filepath, 0o600)  # Restricted file permissions
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Drift history saved to {filepath}")


if __name__ == "__main__":
    # Demo of drift detection
    print("Tejas Drift Detection Demo")
    print("=" * 50)
    
    # Generate synthetic baseline fingerprints
    np.random.seed(42)
    n_baseline = 1000
    n_bits = 128
    
    # Baseline: balanced bit activation (around 0.5)
    baseline_fp = np.random.binomial(1, 0.5, (n_baseline, n_bits)).astype(np.uint8)
    
    print(f"Generated baseline: {baseline_fp.shape}")
    print(f"Baseline activation rate: {np.mean(baseline_fp):.3f}")
    
    # Initialize drift monitor
    monitor = DriftMonitor(
        baseline_fingerprints=baseline_fp,
        sensitivity='medium',
        drift_threshold=0.05
    )
    
    # Test normal batch (no drift)
    print("\n1. Testing normal batch (no drift expected)...")
    normal_batch = np.random.binomial(1, 0.5, (200, n_bits)).astype(np.uint8)
    metrics_normal = monitor.check_batch(normal_batch)
    
    print(f"   Drift detected: {metrics_normal.is_drifted}")
    print(f"   Severity: {metrics_normal.drift_severity}")
    print(f"   JS divergence: {metrics_normal.js_divergence:.3f}")
    
    # Test drifted batch (higher activation rate)
    print("\n2. Testing drifted batch (higher activation rate)...")
    drifted_batch = np.random.binomial(1, 0.7, (200, n_bits)).astype(np.uint8)
    metrics_drifted = monitor.check_batch(drifted_batch)
    
    print(f"   Drift detected: {metrics_drifted.is_drifted}")
    print(f"   Severity: {metrics_drifted.drift_severity}")
    print(f"   JS divergence: {metrics_drifted.js_divergence:.3f}")
    print(f"   Recommend recalibration: {metrics_drifted.recommend_recalibration}")
    
    # Get summary
    print("\n3. Drift summary:")
    summary = monitor.get_drift_summary()
    print(f"   Total batches processed: {summary['total_batches']}")
    print(f"   Drift rate: {summary['drift_rate']:.2f}")
    print(f"   Average JS divergence: {summary['avg_js_divergence']:.3f}")
    
    # Recalibration recommendation
    should_recal, reason = monitor.recommend_recalibration()
    print(f"\n4. Recalibration recommendation: {should_recal}")
    print(f"   Reason: {reason}")
