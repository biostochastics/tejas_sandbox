#!/usr/bin/env python3
"""
Quick test harness for DOE benchmark validation
Tests all 7 pipelines with minimal data to ensure functionality
"""

import sys
import os
from pathlib import Path
import json
import time
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_doe.run_doe_benchmark import EnhancedBenchmarkRunner
from benchmark_doe.core.encoder_factory import EncoderFactory


def test_pipeline(pipeline_config, runner):
    """Test a single pipeline configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {pipeline_config['pipeline_name']}")
    print(f"{'='*60}")
    
    # Create minimal test configuration
    config = {
        'experiment_id': f"test_{pipeline_config['pipeline_name']}",
        'pipeline': pipeline_config['pipeline'],
        'pipeline_type': pipeline_config['pipeline_type'],
        'pipeline_name': pipeline_config['pipeline_name'],
        'dataset': 'wikipedia',
        'dataset_size': '10k',  # Use smallest size for quick test
        'n_bits': 128,  # Smaller for faster test
        'batch_size': 100,  # Smaller batch
        'seed': 42,
        'timeout': 60,  # 1 minute timeout per test
        **pipeline_config.get('config', {})
    }
    
    try:
        start_time = time.time()
        result = runner.run_single_experiment_safe(config)
        elapsed = time.time() - start_time
        
        if result['status'] == 'success':
            metrics = result.get('metrics', {})
            print(f"✓ SUCCESS in {elapsed:.1f}s")
            print(f"  Encoding speed: {metrics.get('docs_per_second', 0):.0f} docs/s")
            print(f"  Query latency: {metrics.get('search_latency_p50', 0):.2f} ms")
            print(f"  Peak memory: {metrics.get('peak_memory_mb', 0):.1f} MB")
            print(f"  NDCG@10: {metrics.get('ndcg_at_10', 0):.3f}")
            return True
        else:
            print(f"✗ FAILED: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run quick tests for all pipelines."""
    print("DOE BENCHMARK QUICK TEST")
    print("Testing all 7 pipelines with minimal data")
    
    # Initialize runner
    runner = EnhancedBenchmarkRunner()
    
    # Define all 7 pipelines to test
    pipelines = [
        {
            'pipeline': 'original_tejas',
            'pipeline_type': 'original_tejas',
            'pipeline_name': 'TEJAS-Original',
            'config': {
                'svd_method': 'truncated',
                'backend': 'numpy'
            }
        },
        {
            'pipeline': 'goldenratio',
            'pipeline_type': 'goldenratio',
            'pipeline_name': 'TEJAS-GoldenRatio',
            'config': {
                'svd_method': 'randomized',
                'backend': 'numpy'
            }
        },
        {
            'pipeline': 'fused_char',
            'pipeline_type': 'fused_char',
            'pipeline_name': 'TEJAS-FusedChar',
            'config': {
                'backend': 'numpy',
                'use_simd': False  # Disable for quick test
            }
        },
        {
            'pipeline': 'fused_byte',
            'pipeline_type': 'fused_byte',
            'pipeline_name': 'TEJAS-FusedByte',
            'config': {
                'backend': 'numpy',
                'tokenizer': 'byte_bpe'
            }
        },
        {
            'pipeline': 'optimized_fused',
            'pipeline_type': 'optimized_fused',
            'pipeline_name': 'TEJAS-Optimized',
            'config': {
                'backend': 'numba',
                'use_numba': True,
                'use_simd': True
            }
        },
        {
            'pipeline': 'bert_mini',
            'pipeline_type': 'bert_mini',
            'pipeline_name': 'BERT-MiniLM',
            'config': {
                'model_name': 'all-MiniLM-L6-v2',
                'embedding_dim': 384,
                'batch_size': 32,
                'device': 'cpu'
            }
        },
        {
            'pipeline': 'bert_base',
            'pipeline_type': 'bert_base',
            'pipeline_name': 'BERT-MPNet',
            'config': {
                'model_name': 'all-mpnet-base-v2',
                'embedding_dim': 768,
                'batch_size': 32,
                'device': 'cpu'
            }
        }
    ]
    
    # Track results
    results = []
    passed = 0
    failed = 0
    
    for pipeline in pipelines:
        success = test_pipeline(pipeline, runner)
        results.append({
            'pipeline': pipeline['pipeline_name'],
            'success': success
        })
        if success:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{len(pipelines)}")
    print(f"Failed: {failed}/{len(pipelines)}")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['pipeline']}")
    
    # Save results
    output_file = Path("benchmark_results") / "quick_test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': results,
            'summary': {
                'total': len(pipelines),
                'passed': passed,
                'failed': failed
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)