#!/usr/bin/env python3
"""
Complete benchmarks across all pipelines and datasets
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_doe.run_doe_benchmark import EnhancedBenchmarkRunner
from benchmark_doe.core.enhanced_metrics import compute_confidence_intervals

class CompleteBenchmarks:
    """Run complete benchmarks across all pipelines and datasets."""
    
    # All TEJAS pipelines (goldenratio removed)
    ALL_PIPELINES = [
        'original_tejas',  # Golden ratio subsampling + sklearn
        'fused_char',      # Fused pipeline with char ngrams
        'fused_byte',      # Fused pipeline with byte BPE
        'optimized_fused'  # Optimized with SIMD + Numba
    ]
    
    # All datasets
    ALL_DATASETS = [
        'wikipedia',
        # 'msmarco',  # Uncomment when ready
        # 'beir'      # Uncomment when ready
    ]
    
    def __init__(self, output_dir: str = None):
        """Initialize benchmark runner."""
        self.output_dir = Path(output_dir or "benchmark_results/complete")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runner = EnhancedBenchmarkRunner(output_dir=str(self.output_dir))
        self.results = []
        
    def run_benchmarks(self, n_runs: int = 3, dataset_size: str = '125k'):
        """Run benchmarks for all configurations."""
        
        total = len(self.ALL_PIPELINES) * len(self.ALL_DATASETS) * n_runs
        count = 0
        
        print("=" * 80)
        print("COMPLETE BENCHMARKS - ALL PIPELINES & DATASETS")
        print("=" * 80)
        print(f"Pipelines: {self.ALL_PIPELINES}")
        print(f"Datasets: {self.ALL_DATASETS}")
        print(f"Runs per config: {n_runs}")
        print(f"Dataset size: {dataset_size}")
        print(f"Total experiments: {total}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Results by pipeline
        pipeline_results = {}
        
        for pipeline in self.ALL_PIPELINES:
            pipeline_results[pipeline] = {}
            print(f"\n{'='*60}")
            print(f"PIPELINE: {pipeline.upper()}")
            print(f"{'='*60}")
            
            for dataset in self.ALL_DATASETS:
                dataset_runs = []
                print(f"\nDataset: {dataset}")
                print("-" * 40)
                
                for run_idx in range(n_runs):
                    count += 1
                    
                    # Configure experiment
                    config = {
                        'experiment_id': f'{pipeline}_{dataset}_run{run_idx+1}',
                        'pipeline': pipeline,
                        'dataset': dataset,
                        'n_bits': 256,
                        'batch_size': 128,
                        'dataset_size': dataset_size if dataset == 'wikipedia' else None,
                        'seed': 42 + run_idx
                    }
                    
                    # Special config for fused_byte
                    if pipeline == 'fused_byte':
                        config['tokenizer'] = 'byte_bpe'
                    
                    print(f"[{count}/{total}] Run {run_idx+1}: ", end='', flush=True)
                    
                    try:
                        exp_start = time.time()
                        result = self.runner.run_single_experiment_safe(config)
                        elapsed = time.time() - exp_start
                        
                        if result['status'] == 'success':
                            m = result.get('metrics', {})
                            
                            # Store result
                            exp_data = {
                                'pipeline': pipeline,
                                'dataset': dataset,
                                'run': run_idx + 1,
                                'elapsed': elapsed,
                                # Performance metrics
                                'speed': m.get('docs_per_second', 0),
                                'latency_p50': m.get('search_latency_p50', 0),
                                'memory_mb': m.get('peak_memory_mb', 0),
                                # IR metrics
                                'ndcg_10': m.get('ndcg_at_10', 0),
                                'precision_1': m.get('precision_at_1', 0),
                                'recall_10': m.get('recall_at_10', 0),
                                'success_1': m.get('success_at_1', 0),
                                'mrr': m.get('mrr', 0)
                            }
                            
                            dataset_runs.append(exp_data)
                            self.results.append(exp_data)
                            
                            print(f"✓ {elapsed:.1f}s (Speed: {exp_data['speed']:.0f} d/s, "
                                  f"NDCG: {exp_data['ndcg_10']:.3f})")
                            
                        else:
                            print(f"✗ Failed: {result.get('error', 'Unknown')}")
                            
                    except Exception as e:
                        print(f"✗ Error: {e}")
                
                # Calculate statistics for this dataset
                if dataset_runs:
                    stats = self._calculate_stats(dataset_runs)
                    pipeline_results[pipeline][dataset] = stats
                    self._print_dataset_summary(stats)
        
        # Total time
        total_time = time.time() - start_time
        
        # Save and report
        self._save_results(pipeline_results)
        self._print_final_summary(pipeline_results, total_time)
        
    def _calculate_stats(self, runs):
        """Calculate statistics for a set of runs."""
        metrics = ['speed', 'latency_p50', 'ndcg_10', 'precision_1', 
                  'recall_10', 'success_1', 'memory_mb', 'mrr']
        
        stats = {}
        for metric in metrics:
            values = [r[metric] for r in runs if metric in r]
            if values:
                median, lower, upper = compute_confidence_intervals(values)
                stats[metric] = {
                    'median': median,
                    'ci_lower': lower, 
                    'ci_upper': upper,
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        return stats
    
    def _print_dataset_summary(self, stats):
        """Print summary statistics for a dataset."""
        print(f"\n  Summary:")
        print(f"    Speed: {stats.get('speed', {}).get('median', 0):.0f} docs/s "
              f"[{stats.get('speed', {}).get('ci_lower', 0):.0f}, "
              f"{stats.get('speed', {}).get('ci_upper', 0):.0f}]")
        print(f"    Latency: {stats.get('latency_p50', {}).get('median', 0):.2f} ms")
        print(f"    NDCG@10: {stats.get('ndcg_10', {}).get('median', 0):.4f}")
        print(f"    P@1: {stats.get('precision_1', {}).get('median', 0):.4f}")
        print(f"    MRR: {stats.get('mrr', {}).get('median', 0):.4f}")
        print(f"    Memory: {stats.get('memory_mb', {}).get('median', 0):.1f} MB")
    
    def _save_results(self, pipeline_results):
        """Save results to files."""
        # Save raw results
        json_path = self.output_dir / f"complete_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'results': self.results,
                'summary': pipeline_results,
                'timestamp': self.timestamp
            }, f, indent=2)
        
        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = self.output_dir / f"complete_results_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def _print_final_summary(self, pipeline_results, total_time):
        """Print final summary table."""
        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        
        # Create comparison table
        print("\n| Pipeline | Speed (d/s) | Latency (ms) | NDCG@10 | P@1 | MRR | Memory |")
        print("|----------|-------------|--------------|---------|-----|-----|--------|")
        
        for pipeline in self.ALL_PIPELINES:
            if pipeline in pipeline_results:
                for dataset, stats in pipeline_results[pipeline].items():
                    speed = stats.get('speed', {}).get('median', 0)
                    latency = stats.get('latency_p50', {}).get('median', 0)
                    ndcg = stats.get('ndcg_10', {}).get('median', 0)
                    p1 = stats.get('precision_1', {}).get('median', 0)
                    mrr = stats.get('mrr', {}).get('median', 0)
                    mem = stats.get('memory_mb', {}).get('median', 0)
                    
                    print(f"| {pipeline:<16} | {speed:>11.0f} | {latency:>12.2f} | "
                          f"{ndcg:>7.4f} | {p1:>3.2f} | {mrr:>3.2f} | {mem:>6.0f} |")
        
        print("\n" + "=" * 80)
        print(f"Total time: {total_time/60:.1f} minutes")
        print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--size', default='125k', help='Dataset size for Wikipedia')
    parser.add_argument('--output', default='benchmark_results/complete')
    args = parser.parse_args()
    
    benchmark = CompleteBenchmarks(output_dir=args.output)
    benchmark.run_benchmarks(n_runs=args.runs, dataset_size=args.size)