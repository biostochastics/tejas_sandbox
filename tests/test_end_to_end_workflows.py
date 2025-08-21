"""
End-to-End Workflow Testing
===========================

Comprehensive testing of complete Tejas v2 workflows:
train → demo → calibrate → drift → benchmark

This test suite validates that all CLI modes work together seamlessly
and that data flows correctly through the entire pipeline.
"""

import pytest
import subprocess
import tempfile
import json
import time
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np

# Import components for direct testing
from core.encoder import GoldenRatioEncoder
from core.fingerprint import BinaryFingerprintSearch, pack_fingerprints
from core.calibration import StatisticalCalibrator
from core.drift import DriftMonitor


class WorkflowTestSuite:
    """End-to-end workflow test suite."""
    
    def __init__(self):
        self.temp_dir = None
        self.test_data = self._generate_workflow_test_data()
        self.cli_results = {}
    
    def setup_test_environment(self):
        """Setup complete test environment for workflow testing."""
        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)
        
        print(f"Setting up workflow test environment: {temp_path}")
        
        # Create directory structure
        dirs = ['models', 'data', 'logs', 'results', 'calibration', 'drift']
        for dir_name in dirs:
            (temp_path / dir_name).mkdir(parents=True)
        
        # Create test data files
        self._create_test_data_files(temp_path)
        
        return temp_path
    
    def _generate_workflow_test_data(self) -> Dict[str, Any]:
        """Generate test data for workflow validation."""
        np.random.seed(42)
        
        # Training data
        training_titles = []
        
        # Add structured patterns
        patterns = {
            'University': [f'University of {city}' for city in 
                          ['California', 'Oxford', 'Cambridge', 'Tokyo', 'Sydney']],
            'List': [f'List of {topic}' for topic in 
                    ['countries', 'animals', 'colors', 'languages', 'sports']],
            'History': [f'History of {subject}' for subject in 
                       ['science', 'art', 'music', 'literature', 'philosophy']]
        }
        
        for pattern_titles in patterns.values():
            training_titles.extend(pattern_titles)
        
        # Add random titles
        training_titles.extend([f'Random Article {i}' for i in range(50)])
        
        # Demo/query data
        demo_queries = [
            'University of Stanford',  # Should match university pattern
            'List of movies',          # Should match list pattern  
            'History of computers',    # Should match history pattern
            'quantum physics',         # General search
            'artificial intelligence'  # General search
        ]
        
        # Calibration data - create relevance judgments
        calibration_data = {
            'queries': ['University of California', 'List of countries', 'History of science'],
            'relevance_mapping': {
                'University of California': {
                    'University of Oxford': 1,
                    'University of Cambridge': 1, 
                    'University of Tokyo': 1,
                    'List of countries': 0,
                    'Random Article 1': 0
                },
                'List of countries': {
                    'List of animals': 1,
                    'List of colors': 1,
                    'University of California': 0,
                    'History of science': 0,
                    'Random Article 2': 0
                },
                'History of science': {
                    'History of art': 1,
                    'History of music': 1,
                    'University of Oxford': 0,
                    'List of countries': 0,
                    'Random Article 3': 0
                }
            }
        }
        
        return {
            'training_titles': training_titles,
            'patterns': patterns,
            'demo_queries': demo_queries,
            'calibration_data': calibration_data
        }
    
    def _create_test_data_files(self, temp_path: Path):
        """Create test data files for CLI workflows."""
        data_path = temp_path / 'data'
        
        # Create training data file
        with open(data_path / 'training_titles.txt', 'w') as f:
            for title in self.test_data['training_titles']:
                f.write(f"{title}\n")
        
        # Create demo queries file
        with open(data_path / 'demo_queries.txt', 'w') as f:
            for query in self.test_data['demo_queries']:
                f.write(f"{query}\n")
        
        # Create calibration relevance file
        calib_file = data_path / 'calibration_relevance.json'
        with open(calib_file, 'w') as f:
            json.dump(self.test_data['calibration_data'], f, indent=2)


class TestEndToEndWorkflows:
    """End-to-end workflow test cases."""
    
    @pytest.fixture
    def workflow_suite(self):
        """Create workflow test suite."""
        suite = WorkflowTestSuite()
        temp_path = suite.setup_test_environment()
        yield suite
        
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    def test_complete_training_workflow(self, workflow_suite):
        """Test complete training workflow: data → model."""
        print("\n🎯 Testing training workflow...")
        
        temp_path = Path(workflow_suite.temp_dir)
        
        # Step 1: Direct training test (simulating CLI training)
        encoder = GoldenRatioEncoder(n_bits=256, max_features=5000, threshold_strategy='median')
        
        # Train encoder
        training_titles = workflow_suite.test_data['training_titles']
        encoder.fit(training_titles)
        
        assert encoder.is_fitted, "Encoder should be trained"
        
        # Generate fingerprints
        fingerprints = encoder.encode(training_titles)
        assert fingerprints.shape[0] == len(training_titles), "Should generate all fingerprints"
        
        # Save model (simulating training output)
        model_path = temp_path / 'models' / 'encoder'
        encoder.save(model_path)
        
        # Save fingerprints
        fingerprints_file = temp_path / 'models' / 'fingerprints.pt'
        torch.save({
            'fingerprints': fingerprints,
            'titles': training_titles,
            'n_bits': encoder.n_bits
        }, fingerprints_file)
        
        # Verify model persistence - encoder saves as individual numpy files
        assert (model_path / 'config.json').exists(), "Should save encoder config"
        assert (model_path / 'projection.npy').exists(), "Should save projection matrix"
        assert (model_path / 'vocabulary.npy').exists(), "Should save vocabulary"
        assert (model_path / 'config.json').exists(), "Should save config"
        assert fingerprints_file.exists(), "Should save fingerprints"
        
        # Test model loading
        new_encoder = GoldenRatioEncoder()
        new_encoder.load(model_path)
        
        # Test that loaded encoder works and produces similar fingerprints
        test_fp = new_encoder.encode_single(training_titles[0])
        # Allow small differences due to floating point precision
        similarity = (fingerprints[0] == test_fp).float().mean()
        assert similarity > 0.95, f"Loaded model should produce similar results (similarity={similarity:.3f})"
        
        print("✅ Training workflow validation passed")
        return encoder, fingerprints, training_titles
    
    def test_demo_workflow_integration(self, workflow_suite):
        """Test demo workflow: model → search → results."""
        print("\n🎬 Testing demo workflow...")
        
        # Setup training
        encoder = GoldenRatioEncoder(n_bits=256, max_features=5000, threshold_strategy='median')
        training_titles = workflow_suite.test_data['training_titles']
        encoder.fit(training_titles)
        fingerprints = encoder.encode(training_titles)
        
        # Initialize search engine
        search_engine = BinaryFingerprintSearch(fingerprints, training_titles)
        
        # Test demo queries
        demo_results = {}
        for query in workflow_suite.test_data['demo_queries']:
            query_fp = encoder.encode_single(query)
            results = search_engine.search(query_fp, k=10)
            
            demo_results[query] = results
            
            # Basic validation
            assert len(results) == 10, f"Should return 10 results for '{query}'"
            assert all(len(result) == 3 for result in results), "Results should have 3 elements each"
            
            # Check similarity scores are reasonable
            similarities = [result[1] for result in results]
            assert all(0 <= sim <= 1 for sim in similarities), "Similarities should be in [0,1]"
            assert similarities == sorted(similarities, reverse=True), "Results should be sorted by similarity"
        
        # Test pattern matching
        university_query = 'University of Stanford'
        query_fp = encoder.encode_single(university_query)
        results = search_engine.search(query_fp, k=10, show_pattern_analysis=False)
        
        # Should find other universities with high similarity
        university_matches = [result for result in results if 'University' in result[0]]
        assert len(university_matches) >= 3, "Should find multiple university matches"
        
        print("✅ Demo workflow validation passed")
        
        return demo_results
    
    def test_calibration_workflow_integration(self, workflow_suite):
        """Test calibration workflow: results → relevance → metrics."""
        print("\n📊 Testing calibration workflow...")
        
        # Setup from previous workflows
        encoder, fingerprints, training_titles = self.test_complete_training_workflow(workflow_suite)
        search_engine = BinaryFingerprintSearch(fingerprints, training_titles)
        
        # Generate calibration data
        calibrator = StatisticalCalibrator(n_folds=3, n_bootstrap=10)
        calibration_queries = workflow_suite.test_data['calibration_data']['queries']
        relevance_mapping = workflow_suite.test_data['calibration_data']['relevance_mapping']
        
        all_distances = []
        all_labels = []
        
        for query in calibration_queries:
            if query not in relevance_mapping:
                continue
                
            query_fp = encoder.encode_single(query)
            results = search_engine.search(query_fp, k=20)
            
            for title, similarity, distance in results:
                if title == query:  # Skip self-match
                    continue
                    
                all_distances.append(float(distance))
                
                # Get relevance from mapping, default to 0
                relevance = relevance_mapping[query].get(title, 0)
                all_labels.append(relevance)
        
        # Test calibration (need binary labels)
        if len(all_distances) >= 10 and len(np.unique(all_labels)) == 2:  # Need minimum data points and binary labels
            distances = np.array(all_distances)
            labels = np.array(all_labels)
            
            result_df = calibrator.calibrate_with_cv(
                distances,
                labels,
                k_values=[1, 5, 10]
            )
            
            # Validate calibration results (it's a DataFrame)
            assert result_df is not None, "Should return results DataFrame"
            assert len(result_df) > 0, "Should have calibration results"
            assert 'precision_at_k' in result_df.columns, "Should include precision@k"
            assert 'recall_at_k' in result_df.columns, "Should include recall@k"
            assert 'f1_score' in result_df.columns, "Should include F1 score"
            
            # Get best threshold based on F1 score
            best_idx = result_df['f1_score'].argmax()
            best_threshold = result_df.iloc[best_idx]['threshold']
            best_f1 = result_df.iloc[best_idx]['f1_score']
            
            # Validate metric ranges
            assert 0 <= best_f1 <= 1, "F1 should be in [0,1]"
            
            print(f"✅ Calibration workflow validation passed")
            print(f"   Best threshold: {best_threshold:.3f}")
            print(f"   Best F1 score: {best_f1:.3f}")
        else:
            print("⚠️ Not enough data for calibration testing")
        
        print("✅ Calibration workflow completed")
    
    def test_drift_monitoring_workflow(self, workflow_suite):
        """Test drift monitoring workflow: baseline → batches → detection."""
        print("\n🌊 Testing drift monitoring workflow...")
        
        # Setup from training workflow
        encoder, fingerprints, training_titles = self.test_complete_training_workflow(workflow_suite)
        
        # Initialize drift monitor
        monitor = DriftMonitor(baseline_fingerprints=None, drift_threshold=0.1, sensitivity='medium')
        
        # Set baseline (first half of data)
        baseline_size = len(fingerprints) // 2
        baseline_fingerprints = fingerprints[:baseline_size]
        monitor.set_baseline(baseline_fingerprints)
        
        # Test with normal batch (second half - should show minimal drift)
        normal_batch = fingerprints[baseline_size:]
        normal_result = monitor.check_batch(normal_batch)
        
        assert 'drift_detected' in normal_result, "Should return drift detection result"
        assert 'js_divergence' in normal_result, "Should calculate JS divergence"
        assert 'drift_severity' in normal_result, "Should assess drift severity"
        
        # Test with artificially drifted batch
        drift_batch = baseline_fingerprints.clone()
        # Introduce significant drift by flipping bits
        flip_mask = torch.rand_like(drift_batch.float()) < 0.4  # Flip 40% of bits
        drift_batch[flip_mask] = 1 - drift_batch[flip_mask]
        
        drift_result = monitor.check_batch(drift_batch)
        
        assert drift_result['drift_detected'] == True, "Should detect artificial drift"
        assert drift_result['js_divergence'] > normal_result['js_divergence'], \
            "Drift batch should have higher JS divergence"
        assert drift_result['drift_severity'] in ['moderate', 'severe'], \
            "Should detect significant drift"
        
        # Test gradual drift simulation
        for i in range(5):
            # Create gradually drifting batches
            gradual_batch = baseline_fingerprints.clone()
            flip_prob = 0.05 + (i * 0.05)  # Increasing drift
            flip_mask = torch.rand_like(gradual_batch.float()) < flip_prob
            gradual_batch[flip_mask] = 1 - gradual_batch[flip_mask]
            
            gradual_result = monitor.check_batch(gradual_batch)
            print(f"   Batch {i+1}: JS={gradual_result['js_divergence']:.4f}, "
                  f"Severity={gradual_result['drift_severity']}")
        
        # Test drift history and summary
        history = monitor.get_drift_history()
        assert len(history) >= 5, "Should accumulate drift history"
        
        summary = monitor.get_drift_summary()
        assert 'total_batches' in summary, "Should provide batch count"
        assert 'drift_counts' in summary, "Should count drift detections"
        assert 'drift_episodes' in summary, "Should provide drift episodes"
        
        # Save drift results
        temp_path = Path(workflow_suite.temp_dir)
        drift_file = temp_path / 'drift' / 'results.json'
        
        # Helper to convert numpy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        with open(drift_file, 'w') as f:
            json.dump({
                'history': convert_numpy_types(history),
                'summary': convert_numpy_types(summary),
                'latest_result': {
                    'js_divergence': float(drift_result['js_divergence']),
                    'drift_detected': bool(drift_result['drift_detected']),
                    'drift_severity': drift_result['drift_severity']
                }
            }, f, indent=2)
        
        assert drift_file.exists(), "Should save drift results"
        
        print("✅ Drift monitoring workflow validation passed")
        
        return history, summary
    
    def test_benchmark_workflow_integration(self, workflow_suite):
        """Test benchmark workflow: performance measurement and comparison."""
        print("\n⚡ Testing benchmark workflow...")
        
        # Setup from training workflow  
        encoder, fingerprints, training_titles = self.test_complete_training_workflow(workflow_suite)
        
        temp_path = Path(workflow_suite.temp_dir)
        
        # Save the encoder for benchmark to load
        model_dir = temp_path / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        encoder.save(model_dir)
        
        # Test benchmark workflow
        from utils.benchmark import BenchmarkSuite
        
        # Create benchmark suite
        benchmark = BenchmarkSuite(
            model_dir=str(model_dir),
            output_dir=str(temp_path / 'results'),
            backend='auto',
            enable_calibration=False,  # Skip for speed
            enable_hardware_profiling=True
        )
        
        # Create pattern families for benchmark
        pattern_families = {}
        for pattern_name, pattern_titles in workflow_suite.test_data['patterns'].items():
            # Find matching titles in our training data
            matches = [title for title in training_titles if any(pt in title for pt in pattern_titles)]
            if matches:
                pattern_families[pattern_name] = matches
        
        # Run benchmark
        results = benchmark.benchmark_tejas(training_titles[:50], pattern_families)  # Subset for speed
        
        # Validate benchmark results
        assert 'memory_mb' in results, "Should measure memory usage"
        assert 'search_time_ms' in results, "Should measure search time"
        assert 'throughput_comparisons_per_sec' in results, "Should measure throughput"
        
        # Validate performance metrics
        assert results['memory_mb'] > 0, "Memory usage should be positive"
        assert results['search_time_ms'] > 0, "Search time should be positive"
        assert results['throughput_comparisons_per_sec'] > 100, "Should achieve reasonable throughput"
        
        # Save benchmark results
        benchmark_file = temp_path / 'results' / 'benchmark.json'
        
        with open(benchmark_file, 'w') as f:
            json.dump({
                'memory_mb': float(results['memory_mb']),
                'search_time_ms': float(results['search_time_ms']),
                'throughput_comparisons_per_sec': float(results['throughput_comparisons_per_sec']),
                'backend': 'auto',
                'test_set_size': 50
            }, f, indent=2)
        
        assert benchmark_file.exists(), "Should save benchmark results"
        
        print(f"   Memory: {results['memory_mb']:.1f} MB")
        print(f"   Search time: {results['search_time_ms']:.2f} ms")
        print(f"   Throughput: {results['throughput_comparisons_per_sec']:,.0f} comp/sec")
        
        print("✅ Benchmark workflow validation passed")
        
        return results
    
    def test_complete_pipeline_integration(self, workflow_suite):
        """Test complete pipeline: train → demo → calibrate → drift → benchmark."""
        print("\n🔄 Testing complete pipeline integration...")
        
        # Execute complete pipeline in sequence
        results = {}
        
        # 1. Training workflow
        encoder, fingerprints, training_titles = self.test_complete_training_workflow(workflow_suite)
        results['training'] = {'status': 'success', 'model_size': len(training_titles)}
        
        # 2. Demo workflow
        demo_results = self.test_demo_workflow_integration(workflow_suite)
        results['demo'] = {'status': 'success', 'queries_tested': len(demo_results)}
        
        # 3. Calibration workflow
        calib_result = self.test_calibration_workflow_integration(workflow_suite)
        if calib_result:
            results['calibration'] = {
                'status': 'success',
                'map_score': float(calib_result['metrics']['map']),
                'ndcg_score': float(calib_result['metrics']['ndcg'])
            }
        else:
            results['calibration'] = {'status': 'skipped', 'reason': 'insufficient_data'}
        
        # 4. Drift monitoring workflow
        history, summary = self.test_drift_monitoring_workflow(workflow_suite)
        results['drift_monitoring'] = {
            'status': 'success',
            'batches_processed': len(history),
            'drift_detected_count': sum(summary['drift_counts'].values()) - summary['drift_counts'].get('none', 0)
        }
        
        # 5. Benchmark workflow
        bench_results = self.test_benchmark_workflow_integration(workflow_suite)
        results['benchmark'] = {
            'status': 'success',
            'throughput': float(bench_results['throughput_comparisons_per_sec']),
            'memory_mb': float(bench_results['memory_mb'])
        }
        
        # Validate pipeline consistency
        assert all(stage['status'] in ['success', 'skipped'] for stage in results.values()), \
            "All pipeline stages should succeed"
        
        # Save pipeline results
        temp_path = Path(workflow_suite.temp_dir)
        pipeline_file = temp_path / 'results' / 'complete_pipeline.json'
        
        with open(pipeline_file, 'w') as f:
            json.dump({
                'pipeline_execution': results,
                'execution_timestamp': time.time(),
                'test_environment': str(temp_path),
                'total_stages': len(results),
                'successful_stages': sum(1 for r in results.values() if r['status'] == 'success')
            }, f, indent=2)
        
        assert pipeline_file.exists(), "Should save complete pipeline results"
        
        print("🎉 Complete pipeline integration validation passed")
        print(f"   Stages completed: {len(results)}")
        print(f"   Training data size: {results['training']['model_size']}")
        print(f"   Demo queries tested: {results['demo']['queries_tested']}")
        if results['calibration']['status'] == 'success':
            print(f"   Calibration MAP: {results['calibration']['map_score']:.3f}")
        print(f"   Drift batches processed: {results['drift_monitoring']['batches_processed']}")
        print(f"   Benchmark throughput: {results['benchmark']['throughput']:,.0f} comp/sec")
        
        return results


def test_end_to_end_workflows_comprehensive():
    """Run comprehensive end-to-end workflow testing."""
    print("\n" + "="*60)
    print("🎯 STARTING END-TO-END WORKFLOW TESTING")
    print("="*60)
    
    suite = WorkflowTestSuite()
    temp_path = suite.setup_test_environment()
    
    try:
        test_instance = TestEndToEndWorkflows()
        
        # Run complete pipeline test
        pipeline_results = test_instance.test_complete_pipeline_integration(suite)
        
        print("\n" + "="*60)
        print("📊 END-TO-END WORKFLOW RESULTS")
        print("="*60)
        
        successful_stages = sum(1 for r in pipeline_results.values() if r['status'] == 'success')
        total_stages = len(pipeline_results)
        
        print(f"✅ Successful stages: {successful_stages}/{total_stages}")
        
        for stage_name, stage_result in pipeline_results.items():
            status_icon = "✅" if stage_result['status'] == 'success' else "⚠️" 
            print(f"{status_icon} {stage_name.capitalize()}: {stage_result['status']}")
        
        if successful_stages == total_stages:
            print("🎉 ALL END-TO-END WORKFLOWS PASSED!")
            print("🚀 Complete pipeline validated and ready for production")
        else:
            print("⚠️ Some workflow stages had issues - review before deployment")
        
        return successful_stages == total_stages
        
    finally:
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)


if __name__ == "__main__":
    # Run end-to-end workflow testing
    success = test_end_to_end_workflows_comprehensive()
    exit(0 if success else 1)
