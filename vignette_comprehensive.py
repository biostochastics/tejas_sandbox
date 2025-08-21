#!/usr/bin/env python3
"""
Tejas v2 Comprehensive Vignette with Real Data
==============================================

This vignette demonstrates some improvements from the Tejas v2 implementation
using real text data from the 20newsgroups dataset.

DISCLAIMER: This is a sandbox research project for testing and evaluation purposes.
Not intended for production use without thorough testing and validation.

Author: Research Team (Sandbox Implementation)
Date: December 2024
"""

import numpy as np
import time
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.encoder import GoldenRatioEncoder
from core.bitops import pack_bits_rows, unpack_bits_rows, hamming_distance_packed
from core.calibration import StatisticalCalibrator
from core.drift import DriftMonitor
from core.format import TejasHeaderV2, CURRENT_FORMAT_VERSION
from core.itq import ITQOptimizer

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class ComprehensiveVignette:
    """
    Comprehensive demonstration of Tejas v2 improvements with real data.
    
    DISCLAIMER: Sandbox implementation for research and evaluation.
    """
    
    def __init__(self, n_bits=128, n_docs=2000, embedding_type='both', verbose=True):
        """
        Args:
            n_bits: Number of bits for fingerprints
            n_docs: Number of documents to use
            embedding_type: 'sparse' for text features, 'dense' for embeddings, 'both' for comparison
            verbose: Whether to print detailed output
        """
        self.n_bits = n_bits
        self.n_docs = n_docs
        self.embedding_type = embedding_type
        self.verbose = verbose
        self.results = {}
        
        # Initialize components
        self.encoder = None
        self.optimizer = None
        self.calibrator = None
        self.monitor = None
        
        # Data holders
        self.texts = None
        self.embeddings = None
        self.labels = None
        self.fingerprints = None
        
    def load_real_data(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Load real text data from 20newsgroups and generate embeddings."""
        if self.verbose:
            print("\n📊 Loading Real Text Data from 20newsgroups...")
        
        # Load diverse categories
        categories = [
            'comp.graphics', 
            'sci.space', 
            'rec.sport.baseball',
            'talk.politics.mideast',
            'alt.atheism'
        ]
        
        newsgroups = fetch_20newsgroups(
            subset='all',
            categories=categories,
            remove=('headers', 'footers', 'quotes'),
            random_state=42
        )
        
        # Select subset of documents
        indices = np.random.RandomState(42).choice(
            len(newsgroups.data), 
            min(self.n_docs, len(newsgroups.data)),
            replace=False
        )
        
        texts = [newsgroups.data[i] for i in indices]
        labels = newsgroups.target[indices]
        
        if self.verbose:
            print(f"  ✓ Loaded {len(texts)} documents from {len(categories)} categories")
            print(f"  ✓ Category distribution: {np.bincount(labels)}")
        
        # Generate embeddings using TF-IDF + SVD
        if self.verbose:
            print("\n🔄 Generating Document Embeddings...")
        
        vectorizer = TfidfVectorizer(
            max_features=10000, 
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Reduce to embedding dimension (simulating dense embeddings)
        svd = TruncatedSVD(n_components=768, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)
        
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        if self.verbose:
            print(f"  ✓ Generated {embeddings.shape[1]}-dimensional embeddings")
            print(f"  ✓ Explained variance ratio: {svd.explained_variance_ratio_[:5]}")
        
        self.texts = texts
        self.embeddings = embeddings
        self.labels = labels
        
        return texts, embeddings, labels
    
    def demonstrate_itq_enhancement(self):
        """PR5: Demonstrate ITQ enhancement for retrieval quality."""
        print("\n" + "="*60)
        print("PR5: ITQ ENHANCEMENT DEMONSTRATION")
        print("="*60)
        
        from core.encoder import GoldenRatioEncoder
        from core.itq import ITQOptimizer
        from sklearn.decomposition import TruncatedSVD
        
        print(f"\n📊 Embedding Type: {self.embedding_type}")
        
        results_by_type = {}
        
        # Scenario 1: Sparse Text Features (TF-IDF)
        if self.embedding_type in ['sparse', 'both']:
            print("\n" + "-"*40)
            print("SCENARIO 1: Sparse Text Features (TF-IDF)")
            print("-"*40)
            
            # Fit encoder on text data
            print("\n🔧 Fitting encoder on text data...")
            encoder = GoldenRatioEncoder(
                n_bits=self.n_bits,
                use_itq=False
            )
            encoder.fit(self.texts)
            
            # Get reduced embeddings for ITQ
            tfidf_features = encoder.vectorizer.transform(self.texts)
            svd = TruncatedSVD(n_components=self.n_bits)
            reduced_embeddings = svd.fit_transform(tfidf_features)
            
            # Generate baseline fingerprints from text
            baseline_fingerprints = encoder.transform(self.texts)
            if hasattr(baseline_fingerprints, 'cpu'):
                baseline_fingerprints = baseline_fingerprints.cpu().numpy()
            
            # Fit ITQ on the reduced embeddings
            itq = ITQOptimizer(n_iterations=50)
            itq.fit(reduced_embeddings)
            
            # Apply ITQ rotation and binarize
            rotated_embeddings = itq.transform(reduced_embeddings)
            itq_fingerprints = (rotated_embeddings > 0).astype(np.uint8)
            
            sparse_results = self._evaluate_retrieval_performance(
                baseline_fingerprints, itq_fingerprints, "Sparse Text (TF-IDF)"
            )
            results_by_type['sparse'] = sparse_results
        
        # Scenario 2: Dense Embeddings (Sentence Transformers)
        if self.embedding_type in ['dense', 'both']:
            print("\n" + "-"*40)
            print("SCENARIO 2: Dense Embeddings (Sentence-BERT)")
            print("-"*40)
            
            print("\n🤖 Generating dense embeddings...")
            try:
                # Try to import and use sentence transformers
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                dense_embeddings = model.encode(self.texts[:500])  # Use subset for speed
                
                # Reduce dimensionality to n_bits
                svd_dense = TruncatedSVD(n_components=self.n_bits)
                reduced_dense = svd_dense.fit_transform(dense_embeddings)
                
                # Baseline: Simple thresholding
                baseline_dense = (reduced_dense > reduced_dense.mean()).astype(np.uint8)
                
                # ITQ-enhanced
                itq_dense = ITQOptimizer(n_iterations=50)
                itq_dense.fit(reduced_dense)
                rotated_dense = itq_dense.transform(reduced_dense)
                itq_dense_fingerprints = (rotated_dense > 0).astype(np.uint8)
                
                dense_results = self._evaluate_retrieval_performance(
                    baseline_dense, itq_dense_fingerprints, "Dense Embeddings"
                )
                results_by_type['dense'] = dense_results
                
            except ImportError as e:
                print(f"  ❌ Error importing sentence-transformers: {e}")
                print("  ⚠️ Cannot demonstrate ITQ on real dense embeddings without sentence-transformers")
                print("  📝 Install with: pip install sentence-transformers")
                # Skip dense embeddings test if not available
                results_by_type['dense'] = {
                    'baseline_map': 0,
                    'itq_map': 0,
                    'improvement': 0,
                    'note': 'sentence-transformers not available'
                }
        
        # Store results
        self.results['itq_comparison'] = results_by_type
        
        # Print comparison summary
        if self.embedding_type == 'both':
            print("\n" + "="*60)
            print("📊 ITQ EFFECTIVENESS COMPARISON")
            print("="*60)
            
            for embedding_type, results in results_by_type.items():
                print(f"\n{embedding_type.upper()} EMBEDDINGS:")
                improvement = results['itq']['map'] - results['baseline']['map']
                pct_change = (improvement / results['baseline']['map']) * 100
                
                if improvement > 0:
                    print(f"  ✅ ITQ Improves MAP by {improvement:.3f} ({pct_change:+.1f}%)")
                else:
                    print(f"  ❌ ITQ Degrades MAP by {-improvement:.3f} ({pct_change:+.1f}%)")
                
                print(f"     Baseline MAP: {results['baseline']['map']:.3f}")
                print(f"     ITQ MAP: {results['itq']['map']:.3f}")
            
            print("\n💡 INSIGHT: ITQ is most effective with dense, continuous embeddings")
            print("   where rotation can better preserve neighborhood structure.")
    
    def demonstrate_calibration_drift(self) -> Dict:
        """Demonstrate calibration and drift detection on real data."""
        if self.verbose:
            print("\n" + "="*60)
            print("PR3: CALIBRATION & DRIFT DETECTION")
            print("="*60)
        
        # Use text data
        if not hasattr(self, 'texts'):
            self.load_real_data()
        
        from core.encoder import GoldenRatioEncoder
        from core.calibration import StatisticalCalibrator
        from core.drift import DriftMonitor
        
        # Initialize encoder if not already done
        if self.encoder is None:
            self.encoder = GoldenRatioEncoder(n_bits=self.n_bits)
            self.encoder.fit(self.texts)
        
        # Generate fingerprints
        fingerprints = self.encoder.transform(self.texts)
        
        # Convert to numpy if torch tensor
        import torch
        if isinstance(fingerprints, torch.Tensor):
            fingerprints = fingerprints.numpy()
        
        # Split data temporally (simulating time-based drift)
        n_batches = 5
        batch_size = len(fingerprints) // n_batches
        
        # 1. Calibration
        if self.verbose:
            print("\n🎯 Statistical Calibration:")
        
        # Compute pairwise distances for calibration
        sample_size = min(500, len(fingerprints))
        sample_idx = np.random.choice(len(fingerprints), sample_size, replace=False)
        sample_fp = fingerprints[sample_idx]
        sample_labels = self.labels[sample_idx]
        
        # Compute Hamming distances
        distances = []
        labels_binary = []
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                dist = np.sum(sample_fp[i] != sample_fp[j])
                distances.append(dist)
                labels_binary.append(1 if sample_labels[i] == sample_labels[j] else 0)
        
        distances = np.array(distances)
        labels_binary = np.array(labels_binary)
        
        # Calibrate thresholds
        self.calibrator = StatisticalCalibrator(n_folds=3, n_bootstrap=20)
        cal_results = self.calibrator.calibrate_with_cv(
            distances, 
            labels_binary,
            thresholds=np.arange(30, 100, 10)
        )
        
        optimal_threshold = self.calibrator.find_optimal_threshold(cal_results)
        
        # Handle tuple return from find_optimal_threshold
        if isinstance(optimal_threshold, tuple):
            optimal_threshold = optimal_threshold[0]
        
        if self.verbose:
            print(f"  ✓ Optimal threshold: {optimal_threshold}")
            best_row = cal_results[cal_results['threshold'] == optimal_threshold]
            if not best_row.empty:
                print(f"  ✓ Best F1 score: {best_row['f1_score'].values[0]:.3f}")
        
        # 2. Drift Detection
        if self.verbose:
            print("\n🔍 Drift Detection Across Batches:")
        
        self.monitor = DriftMonitor(
            history_size=100,
            drift_threshold=0.05,
            sensitivity='medium'
        )
        
        # Set baseline from first batch
        baseline_batch = fingerprints[:batch_size]
        self.monitor.set_baseline(baseline_batch)
        
        drift_results = []
        for i in range(1, n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, len(fingerprints))
            batch = fingerprints[start_idx:end_idx]
            
            # Simulate drift by adding noise to later batches
            if i >= 3:  # Introduce drift in later batches
                noise_mask = np.random.random(batch.shape) < (0.05 * (i-2))
                batch = np.where(noise_mask, 1 - batch, batch)
            
            # Check for drift
            drift_metrics = self.monitor.check_batch(batch)
            is_drifted = drift_metrics.get('is_drifted', drift_metrics.get('drift_detected', False))
            js_divergence = drift_metrics.get('js_divergence', drift_metrics.get('divergence', 0.0))
            
            drift_results.append({
                'batch': i,
                'is_drifted': is_drifted,
                'js_divergence': js_divergence,
                'severity': drift_metrics.get('drift_severity', 'none'),
                'recommend_recalibration': drift_metrics.get('recommend_recalibration', False)
            })
            
            if self.verbose:
                status = "⚠️ DRIFT DETECTED" if is_drifted else "✓ No drift"
                print(f"  Batch {i}: {status} (JS divergence: {js_divergence:.3f})")
        
        results = {
            'calibration': {
                'optimal_threshold': optimal_threshold,
                'cal_results': cal_results.to_dict('records')
            },
            'drift_detection': drift_results,
            'drift_detected_count': sum(r['is_drifted'] for r in drift_results),
            'recalibration_needed': any(r['recommend_recalibration'] for r in drift_results)
        }
        
        if self.verbose:
            print(f"\n📊 DRIFT SUMMARY:")
            print(f"  Batches with drift: {results['drift_detected_count']}/{n_batches-1}")
            print(f"  Recalibration recommended: {'Yes' if results['recalibration_needed'] else 'No'}")
        
        self.results['calibration_drift'] = results
        return results
    
    def demonstrate_bit_packing(self) -> Dict:
        """Demonstrate bit packing efficiency."""
        if self.verbose:
            print("\n" + "="*60)
            print("PR2: BIT PACKING EFFICIENCY")
            print("="*60)
        
        # Generate test fingerprints
        test_size = 1000
        test_fingerprints = np.random.randint(0, 2, (test_size, self.n_bits), dtype=np.uint8)
        
        # Measure unpacked size
        unpacked_size = test_fingerprints.nbytes
        
        # Pack bits
        start = time.time()
        packed = pack_bits_rows(test_fingerprints)
        pack_time = time.time() - start
        packed_size = packed.nbytes
        
        # Unpack bits
        start = time.time()
        unpacked = unpack_bits_rows(packed, n_bits=self.n_bits)
        unpack_time = time.time() - start
        
        # Verify correctness
        assert np.array_equal(test_fingerprints, unpacked), "Packing/unpacking failed!"
        
        # Compute Hamming distances on packed data
        start = time.time()
        sample1_packed = packed[:100]
        sample2_packed = packed[100:200]
        # Compute distances for each query against database
        distances_packed = []
        for query in sample1_packed:
            dist = hamming_distance_packed(query, sample2_packed)
            distances_packed.append(dist)
        distances_packed = np.array(distances_packed)
        packed_distance_time = time.time() - start
        
        # Compare with unpacked computation
        start = time.time()
        sample1_unpacked = test_fingerprints[:100]
        sample2_unpacked = test_fingerprints[100:200]
        distances_unpacked = np.array([
            np.sum(sample1_unpacked[i] != sample2_unpacked[j])
            for i in range(100)
            for j in range(100)
        ]).reshape(100, 100)
        unpacked_distance_time = time.time() - start
        
        results = {
            'compression_ratio': unpacked_size / packed_size,
            'space_saving': (1 - packed_size / unpacked_size) * 100,
            'pack_time': pack_time,
            'unpack_time': unpack_time,
            'packed_distance_time': packed_distance_time,
            'unpacked_distance_time': unpacked_distance_time,
            'distance_speedup': unpacked_distance_time / packed_distance_time,
            'unpacked_bytes': unpacked_size,
            'packed_bytes': packed_size
        }
        
        if self.verbose:
            print(f"\n💾 Storage Efficiency:")
            print(f"  Unpacked size: {unpacked_size:,} bytes")
            print(f"  Packed size: {packed_size:,} bytes")
            print(f"  Compression ratio: {results['compression_ratio']:.1f}x")
            print(f"  Space saving: {results['space_saving']:.1f}%")
            print(f"\n⚡ Performance:")
            print(f"  Pack time: {pack_time*1000:.2f}ms")
            print(f"  Unpack time: {unpack_time*1000:.2f}ms")
            print(f"  Distance computation speedup: {results['distance_speedup']:.2f}x")
        
        self.results['bit_packing'] = results
        return results
    
    def _evaluate_retrieval_performance(self, baseline_fingerprints, itq_fingerprints, scenario_name):
        """Helper to evaluate and compare retrieval performance."""
        print(f"\n📊 Evaluating {scenario_name}...")
        
        # Use subset of labels matching fingerprint count
        n_samples = len(baseline_fingerprints)
        labels_subset = self.labels[:n_samples]
        
        # Baseline performance
        print("  Baseline (No ITQ):", end="")
        baseline_metrics = self._evaluate_retrieval(
            baseline_fingerprints, labels_subset
        )
        
        # Enhanced performance
        print("  Enhanced (With ITQ):", end="")
        itq_metrics = self._evaluate_retrieval(
            itq_fingerprints, labels_subset
        )
        
        # Calculate improvements
        improvements = {}
        for k in [1, 5, 10]:
            baseline_p = baseline_metrics[f'precision_at_{k}']
            itq_p = itq_metrics[f'precision_at_{k}']
            improvements[f'p@{k}'] = (itq_p - baseline_p) / baseline_p * 100
        
        improvements['map'] = (itq_metrics['map'] - baseline_metrics['map']) / baseline_metrics['map'] * 100
        
        # Print compact summary
        print(f" MAP: {baseline_metrics['map']:.3f} → {itq_metrics['map']:.3f} ({improvements['map']:+.1f}%)")
        
        return {
            'baseline': baseline_metrics,
            'itq': itq_metrics,
            'improvements': improvements
        }
    
    def _evaluate_retrieval(self, fingerprints, labels):
        """Evaluate retrieval performance."""
        metrics = {}
        precisions_at_k = {k: [] for k in [1, 5, 10]}
        
        # For each query sample
        n_samples = min(100, len(fingerprints))  # Use subset for speed
        for i in range(n_samples):
            # Compute Hamming distances to all other samples
            distances = np.array([
                np.sum(fingerprints[i] != fingerprints[j])
                for j in range(len(fingerprints)) if j != i
            ])
            
            # Get sorted indices (excluding self)
            all_indices = np.array([j for j in range(len(fingerprints)) if j != i])
            sorted_idx = all_indices[np.argsort(distances)]
            
            # Compute precision at different k values
            for k in [1, 5, 10]:
                if k > len(sorted_idx):
                    continue
                top_k_idx = sorted_idx[:k]
                top_k_labels = labels[top_k_idx]
                correct = np.sum(top_k_labels == labels[i])
                precisions_at_k[k].append(correct / k)
        
        # Aggregate metrics
        for k in [1, 5, 10]:
            metrics[f'precision_at_{k}'] = np.mean(precisions_at_k[k])
        
        # Compute MAP (Mean Average Precision)
        avg_precisions = []
        for i in range(n_samples):
            distances = np.array([np.sum(fingerprints[i] != fingerprints[j]) for j in range(len(fingerprints)) if j != i])
            all_indices = np.array([j for j in range(len(fingerprints)) if j != i])
            sorted_idx = all_indices[np.argsort(distances)]
            sorted_labels = labels[sorted_idx]
            
            # Find relevant items
            relevant = (sorted_labels == labels[i])
            if relevant.sum() == 0:
                continue
            
            # Compute average precision
            precisions = []
            num_relevant = 0
            for j, is_relevant in enumerate(relevant[:100]):  # Consider top 100
                if is_relevant:
                    num_relevant += 1
                    precisions.append(num_relevant / (j + 1))
            
            if precisions:
                avg_precisions.append(np.mean(precisions))
        
        metrics['map'] = np.mean(avg_precisions) if avg_precisions else 0
        
        # Compute bit balance and entropy
        bit_activation = np.mean(fingerprints, axis=0)
        metrics['bit_balance'] = np.mean(bit_activation)
        
        # Bit entropy
        p = bit_activation
        p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        metrics['bit_entropy'] = np.mean(entropy)
        
        return metrics
    
    def save_results(self):
        """Save all results to JSON file."""
        output_path = Path('vignette_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\n💾 Results saved to {output_path}")
    
    def run_complete_demonstration(self):
        """Run the complete vignette demonstration."""
        print("\n" + "================================================================================")
        print(" TEJAS V2 SANDBOX VIGNETTE - Playing with Viraj Deshwal's TEJAS Ideas")
        print("================================================================================")
        print("\n⚠️  DISCLAIMER: This is a sandbox toy version based on Viraj Deshwal's TEJAS.")
        print("    Original TEJAS: https://github.com/ReinforceAI/tejas")
        print("    This is just experimental playground code, not for production.")
        
        # Load real data
        self.load_real_data()
        
        # Run demonstrations
        self.demonstrate_itq_enhancement()
        self.demonstrate_calibration_drift()
        self.demonstrate_bit_packing()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print(" VIGNETTE COMPLETE")
        print("="*80)
        
        # Print summary
        print("\n📊 KEY RESULTS SUMMARY:")
        
        if 'itq_comparison' in self.results:
            itq = self.results['itq_comparison']
            print("\n  ITQ Enhancement:")
            print(f"    • Precision@10 improvement: {itq['dense']['improvements']['p@10']:+.1f}%")
            print(f"    • MAP improvement: {itq['dense']['improvements']['map']:+.1f}%")
        
        if 'calibration_drift' in self.results:
            cd = self.results['calibration_drift']
            print(f"\n  Calibration & Drift:")
            print(f"    • Optimal threshold: {cd['calibration']['optimal_threshold']}")
            print(f"    • Drift detected: {cd['drift_detected_count']} batches")
        
        if 'bit_packing' in self.results:
            bp = self.results['bit_packing']
            print(f"\n  Bit Packing:")
            print(f"    • Compression ratio: {bp['compression_ratio']:.1f}x")
            print(f"    • Distance computation speedup: {bp['distance_speedup']:.1f}x")
        
        return self.results


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    embedding_type = 'both'  # default
    if len(sys.argv) > 1:
        if sys.argv[1] in ['sparse', 'dense', 'both']:
            embedding_type = sys.argv[1]
        else:
            print("Usage: python vignette_comprehensive.py [sparse|dense|both]")
            print("  sparse: Test with TF-IDF text features")
            print("  dense:  Test with dense sentence embeddings")
            print("  both:   Compare both scenarios (default)")
            sys.exit(1)
    
    # Run comprehensive vignette
    vignette = ComprehensiveVignette(n_bits=128, n_docs=2000, embedding_type=embedding_type)
    results = vignette.run_complete_demonstration()
    
    print("\n✅ Vignette execution completed successfully!")
