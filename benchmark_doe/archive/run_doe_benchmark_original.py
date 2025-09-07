#!/usr/bin/env python3
"""
DOE Benchmark Runner - Actual pipeline comparison
This script runs the actual DOE experiments using available encoders.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

# Add paths for imports
current_dir = Path(__file__).parent
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir))
# Also add the benchmark_doe directory itself
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import DOE framework components directly
try:
    from benchmark_doe.core.compatibility import CompatibilityValidator
    from benchmark_doe.core.safe_evaluator import SafeEvaluator
    from benchmark_doe.core.dataset_loader import load_benchmark_dataset
    from benchmark_doe.core.encoder_factory import EncoderFactory
    from benchmark_doe.core.resource_guard import ResourceGuard, run_with_limits
    from benchmark_doe.core.utils import safe_divide, validate_dataframe
except ImportError:
    # Try relative imports if running as script
    from core.compatibility import CompatibilityValidator
    from core.safe_evaluator import SafeEvaluator
    from core.dataset_loader import load_benchmark_dataset
    from core.encoder_factory import EncoderFactory
    from core.resource_guard import ResourceGuard, run_with_limits
    from core.utils import safe_divide, validate_dataframe

def create_doe_design(include_datasets=False):
    """Create a simplified DOE design for testing.
    
    Args:
        include_datasets: If True, include different dataset types in the design
    """
    experiments = []
    
    # Define factors and levels
    pipelines = ['original_tejas', 'fused_char', 'fused_byte']
    n_bits_levels = [128, 256, 512]
    batch_sizes = [100, 500, 1000]
    
    # Add dataset types if requested
    if include_datasets:
        dataset_types = ['wikipedia', 'msmarco', 'beir']  # All three dataset types
    else:
        dataset_types = ['wikipedia']  # Default to Wikipedia only
    
    # Create factorial design (simplified)
    exp_id = 0
    for dataset_type in dataset_types:
        for pipeline in pipelines:
            for n_bits in n_bits_levels:
                for batch_size in batch_sizes:
                    config = {
                        'experiment_id': f'exp_{exp_id:04d}',
                        'pipeline_architecture': pipeline,
                        'n_bits': n_bits,
                        'batch_size': batch_size,
                        'dataset_type': dataset_type,
                    }
                
                    # Add pipeline-specific defaults
                    if pipeline == 'original_tejas':
                        config.update({
                            'tokenizer': 'char_ngram',
                            'backend': 'numpy',
                            'svd_method': 'truncated',
                            'use_numba': False,
                            'use_simd': False
                        })
                    elif pipeline == 'fused_char':
                        config.update({
                            'tokenizer': 'char_ngram',
                            'backend': 'numpy',
                            'svd_method': 'randomized',
                            'use_numba': True,
                            'use_simd': False
                        })
                    elif pipeline == 'fused_byte':
                        config.update({
                            'tokenizer': 'byte_bpe',
                            'backend': 'numpy',
                            'svd_method': 'randomized',
                            'use_numba': True,
                            'use_simd': False
                        })
                    
                    experiments.append(config)
                    exp_id += 1
    
    return experiments


def run_single_experiment(config):
    """Run a single experiment with the given configuration."""
    result = {
        'experiment_id': config['experiment_id'],
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending'
    }
    
    try:
        # Validate configuration
        validator = CompatibilityValidator()
        is_valid, issues = validator.validate_configuration(config)
        
        if not is_valid and len(issues['hard']) > 0:
            # Try to auto-fix
            config = validator.fix_configuration(config)
            is_valid, issues = validator.validate_configuration(config)
            
            if not is_valid:
                result['status'] = 'invalid_config'
                result['error'] = str(issues['hard'])
                return result
        
        # Create encoder safely using factory
        pipeline = config['pipeline_architecture']
        
        try:
            encoder_config = {
                'n_bits': config['n_bits'],
                'batch_size': config.get('batch_size', 1000),
                'energy_threshold': config.get('energy_threshold', 0.95),
                'use_itq': config.get('use_itq', False),
                'random_state': config.get('random_state', 42)  # Add seed propagation
            }
            
            # Add pipeline-specific config
            if pipeline == 'fused_byte':
                encoder_config['vocab_size'] = config.get('vocab_size', 1000)
            
            encoder = EncoderFactory.create_encoder(pipeline, encoder_config)
            
        except (ValueError, ImportError) as e:
            result['status'] = 'encoder_creation_failed'
            result['error'] = str(e)
            return result
        
        # Load real benchmark data based on config
        dataset_type = config.get('dataset_type', 'wikipedia')
        
        # Determine dataset size/subset based on batch size and dataset type
        if dataset_type == 'wikipedia':
            if config['batch_size'] <= 100:
                size = "10k"
            elif config['batch_size'] <= 500:
                size = "50k"
            else:
                size = "125k"
            
            documents, queries, relevance = load_benchmark_dataset(
                dataset_type=dataset_type,
                size=size,
                sample=True,
                n_docs=config['batch_size'],
                n_queries=min(100, config['batch_size'] // 10)
            )
        elif dataset_type == 'msmarco':
            # MS MARCO uses 'dev' subset
            documents, queries, relevance = load_benchmark_dataset(
                dataset_type=dataset_type,
                subset="dev",
                sample=True,
                n_docs=config['batch_size'],
                n_queries=min(100, config['batch_size'] // 10),
                max_docs=config['batch_size'] * 2  # Load more docs for sampling
            )
        elif dataset_type == 'beir':
            # BEIR uses specific dataset names (e.g., 'scifact')
            documents, queries, relevance = load_benchmark_dataset(
                dataset_type=dataset_type,
                dataset_name="scifact",  # Can be made configurable later
                sample=True,
                n_docs=config['batch_size'],
                n_queries=min(100, config['batch_size'] // 10)
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        n_docs = len(documents)
        
        # Measure encoding performance
        start_time = time.perf_counter()
        
        # Fit encoder
        fit_start = time.perf_counter()
        encoder.fit(documents)
        fit_time = time.perf_counter() - fit_start
        
        # Transform documents
        encode_start = time.perf_counter()
        doc_codes = encoder.transform(documents)
        encode_time = time.perf_counter() - encode_start
        
        # Transform queries
        query_start = time.perf_counter()
        query_codes = []
        for query in queries:
            code = encoder.transform([query])
            query_codes.append(code[0] if len(code) > 0 else None)
        query_time = time.perf_counter() - query_start
        
        total_time = time.perf_counter() - start_time
        
        # Calculate metrics
        result.update({
            'status': 'success',
            'metrics': {
                'fit_time': fit_time,
                'encode_time': encode_time,
                'query_time': query_time,
                'total_time': total_time,
                'docs_per_second': safe_divide(n_docs, encode_time, default=0),
                'queries_per_second': safe_divide(len(queries), query_time, default=0),
                'avg_query_latency_ms': safe_divide(query_time * 1000, len(queries), default=0),
                'n_documents': n_docs,
                'n_queries': len(queries),
                'code_shape': doc_codes.shape if hasattr(doc_codes, 'shape') else str(type(doc_codes)),
            }
        })
        
        # Memory estimate (simplified)
        import sys
        if hasattr(doc_codes, 'nbytes'):
            result['metrics']['index_size_mb'] = doc_codes.nbytes / (1024 * 1024)
        else:
            result['metrics']['index_size_mb'] = sys.getsizeof(doc_codes) / (1024 * 1024)
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
    
    return result


def run_single_experiment_safe(config):
    """
    Run a single experiment with resource limits.
    
    This wrapper adds timeout and memory protection to prevent
    resource exhaustion attacks.
    """
    guard = ResourceGuard(
        timeout_seconds=config.get('timeout', 300),
        memory_limit_mb=config.get('memory_limit_mb', 2048)
    )
    
    try:
        return guard.run_with_limits(
            run_single_experiment,
            args=(config,),
            description=f"Experiment {config.get('experiment_id', 'unknown')}"
        )
    except Exception as e:
        return {
            'experiment_id': config.get('experiment_id', 'unknown'),
            'status': 'resource_limit_exceeded',
            'error': str(e)
        }


def run_doe_benchmark(n_runs=1, include_datasets=False, seed=None):
    """Run the complete DOE benchmark.
    
    Args:
        n_runs: Number of times to run each experiment (for median calculation)
        include_datasets: If True, test with multiple dataset types
        seed: Random seed for reproducibility (None = generate seed)
    """
    # Import and set global seed for reproducibility
    try:
        from core.reproducibility import set_global_seed
        actual_seed = set_global_seed(seed)
        print(f"\n🔧 Random seed set to: {actual_seed}")
    except ImportError:
        print("⚠️ Reproducibility module not found, running without seed control")
        actual_seed = None
    print("\n" + "="*60)
    print("DOE BENCHMARK - PIPELINE COMPARISON")
    print("="*60)
    
    if n_runs > 1:
        print(f"\n🔄 Running {n_runs} repetitions per experiment for median calculation")
    
    # Create experimental design
    experiments = create_doe_design(include_datasets=include_datasets)
    print(f"\n📊 Created {len(experiments)} experiments")
    print(f"   Pipelines: original_tejas, fused_char, fused_byte")
    print(f"   n_bits: 128, 256, 512")
    print(f"   batch_sizes: 100, 500, 1000")
    if include_datasets:
        print(f"   datasets: wikipedia, msmarco, beir (scifact)")
    
    # Create output directory
    output_dir = Path("./benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run experiments
    results = []
    successful = 0
    failed = 0
    
    print("\n🚀 Running experiments...")
    print("-" * 40)
    
    for i, config in enumerate(experiments):
        exp_id = config['experiment_id']
        pipeline = config['pipeline_architecture']
        n_bits = config['n_bits']
        batch_size = config['batch_size']
        dataset_type = config.get('dataset_type', 'wikipedia')
        
        print(f"\n[{i+1}/{len(experiments)}] {exp_id}: {pipeline} | n_bits={n_bits} | batch={batch_size} | dataset={dataset_type}")
        
        # Run experiment multiple times if requested
        if n_runs > 1:
            run_results = []
            for run_idx in range(n_runs):
                print(f"  Run {run_idx + 1}/{n_runs}...", end="")
                single_result = run_single_experiment_safe(config)
                run_results.append(single_result)
                if single_result['status'] == 'success':
                    print(" ✓")
                else:
                    print(f" ✗ ({single_result.get('error', 'Unknown')[:30]}...)")
            
            # Aggregate results from multiple runs
            result = aggregate_multiple_runs(config, run_results)
        else:
            result = run_single_experiment_safe(config)
        
        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
            metrics = result['metrics']
            if n_runs > 1:
                print(f"  ✅ Success (median of {n_runs} runs): {metrics['docs_per_second']:.1f} docs/sec, "
                      f"{metrics['avg_query_latency_ms']:.2f}ms/query")
                if 'std_docs_per_second' in metrics:
                    print(f"     Std Dev: ±{metrics['std_docs_per_second']:.1f} docs/sec, "
                          f"±{metrics['std_avg_query_latency_ms']:.2f}ms/query")
            else:
                print(f"  ✅ Success: {metrics['docs_per_second']:.1f} docs/sec, "
                      f"{metrics['avg_query_latency_ms']:.2f}ms/query")
        else:
            failed += 1
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Initialize manifest tracker
    try:
        from core.manifest import ManifestTracker
        tracker = ManifestTracker.get_instance()
        tracker.set_output_dir(output_dir)
        tracker.set_experiment_id(f"doe_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tracker.set_configuration({
            'n_runs': n_runs,
            'include_datasets': include_datasets,
            'seed': actual_seed if 'actual_seed' in locals() else None,
            'n_experiments': len(experiments)
        })
    except ImportError:
        tracker = None
        print("⚠️ Manifest module not found, outputs won't be tracked")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"doe_results_{timestamp}.json"
    
    # Add schema version to results
    results_with_metadata = {
        'schema_version': '1.0',
        'seed_used': actual_seed if 'actual_seed' in locals() else None,
        'timestamp': timestamp,
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2, default=str)
    
    # Track in manifest
    if tracker:
        tracker.add_output(results_file, 'results_json')
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"✅ Successful: {successful}/{len(experiments)}")
    print(f"❌ Failed: {failed}/{len(experiments)}")
    print(f"📁 Results saved to: {results_file}")
    
    # Analyze results
    if successful > 0:
        analyze_results(results)
    
    # Save manifest if tracker is available
    if tracker:
        manifest_path = tracker.save_manifest()
        print(f"📋 Manifest saved to: {manifest_path}")
        
        # Validate outputs
        validation = tracker.validate_outputs()
        if validation['missing']:
            print(f"⚠️ Missing outputs: {len(validation['missing'])}")
        if validation['checksum_mismatch']:
            print(f"⚠️ Checksum mismatches: {len(validation['checksum_mismatch'])}")
    
    return results


def analyze_results(results):
    """Perform basic analysis on results."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Group by pipeline
    pipelines = {}
    for result in results:
        if result['status'] != 'success':
            continue
        
        pipeline = result['config']['pipeline_architecture']
        if pipeline not in pipelines:
            pipelines[pipeline] = []
        pipelines[pipeline].append(result)
    
    # Calculate statistics per pipeline
    for pipeline, pipeline_results in pipelines.items():
        print(f"\n📈 {pipeline.upper()}:")
        
        if not pipeline_results:
            print("  No successful runs")
            continue
        
        # Extract metrics
        docs_per_sec = [r['metrics']['docs_per_second'] for r in pipeline_results]
        query_latencies = [r['metrics']['avg_query_latency_ms'] for r in pipeline_results]
        total_times = [r['metrics']['total_time'] for r in pipeline_results]
        
        print(f"  Throughput (docs/sec):")
        print(f"    Mean: {np.mean(docs_per_sec):.1f}")
        print(f"    Min:  {np.min(docs_per_sec):.1f}")
        print(f"    Max:  {np.max(docs_per_sec):.1f}")
        
        print(f"  Query Latency (ms):")
        print(f"    Mean: {np.mean(query_latencies):.2f}")
        print(f"    Min:  {np.min(query_latencies):.2f}")
        print(f"    Max:  {np.max(query_latencies):.2f}")
        
        print(f"  Total Time (sec):")
        print(f"    Mean: {np.mean(total_times):.3f}")
    
    # Find best configurations
    print("\n🏆 BEST CONFIGURATIONS:")
    
    # Best throughput
    best_throughput = max(
        (r for r in results if r['status'] == 'success'),
        key=lambda r: r['metrics']['docs_per_second'],
        default=None
    )
    
    if best_throughput:
        config = best_throughput['config']
        metrics = best_throughput['metrics']
        print(f"\n  Highest Throughput: {metrics['docs_per_second']:.1f} docs/sec")
        print(f"    Pipeline: {config['pipeline_architecture']}")
        print(f"    n_bits: {config['n_bits']}, batch_size: {config['batch_size']}")
    
    # Best latency
    best_latency = min(
        (r for r in results if r['status'] == 'success'),
        key=lambda r: r['metrics']['avg_query_latency_ms'],
        default=None
    )
    
    if best_latency:
        config = best_latency['config']
        metrics = best_latency['metrics']
        print(f"\n  Lowest Latency: {metrics['avg_query_latency_ms']:.2f}ms")
        print(f"    Pipeline: {config['pipeline_architecture']}")
        print(f"    n_bits: {config['n_bits']}, batch_size: {config['batch_size']}")


def aggregate_multiple_runs(config, run_results):
    """Aggregate results from multiple runs of the same experiment.
    
    Args:
        config: Experiment configuration
        run_results: List of results from multiple runs
        
    Returns:
        Aggregated result with median, mean, and std of metrics
    """
    # Filter successful runs
    successful_runs = [r for r in run_results if r['status'] == 'success']
    
    if not successful_runs:
        # If all runs failed, return the first failure
        return run_results[0]
    
    # Extract metrics from successful runs
    metrics_lists = {}
    for run in successful_runs:
        for key, value in run['metrics'].items():
            if isinstance(value, (int, float)):
                if key not in metrics_lists:
                    metrics_lists[key] = []
                metrics_lists[key].append(value)
    
    # Calculate median, mean, and std for each metric
    aggregated_metrics = {}
    for key, values in metrics_lists.items():
        values_array = np.array(values)
        aggregated_metrics[key] = np.median(values_array)  # Use median as primary value
        aggregated_metrics[f'mean_{key}'] = np.mean(values_array)
        aggregated_metrics[f'std_{key}'] = np.std(values_array)
        aggregated_metrics[f'min_{key}'] = np.min(values_array)
        aggregated_metrics[f'max_{key}'] = np.max(values_array)
    
    # Include non-numeric metrics from the first successful run
    for key, value in successful_runs[0]['metrics'].items():
        if not isinstance(value, (int, float)):
            aggregated_metrics[key] = value
    
    # Create aggregated result with preserved individual run data
    result = {
        'experiment_id': config['experiment_id'],
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'status': 'success',
        'metrics': aggregated_metrics,
        'n_runs': len(run_results),
        'n_successful': len(successful_runs),
        'aggregation_method': 'median',
        'schema_version': '1.0',  # Add schema version for future compatibility
        'individual_runs': successful_runs,  # PRESERVE ALL RUN DATA
        'failed_runs': [r for r in run_results if r['status'] != 'success']  # Track failures too
    }
    
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DOE Benchmark")
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs per experiment (default: 1)')
    parser.add_argument('--datasets', action='store_true',
                        help='Include multiple dataset types')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: auto-generate)')
    args = parser.parse_args()
    
    try:
        results = run_doe_benchmark(
            n_runs=args.runs, 
            include_datasets=args.datasets,
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        traceback.print_exc()