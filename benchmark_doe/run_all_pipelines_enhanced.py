#!/usr/bin/env python3
"""
Run all pipelines with enhanced metrics
Comprehensive benchmark across all TEJAS pipelines and datasets
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_doe.run_doe_benchmark import EnhancedBenchmarkRunner
from benchmark_doe.core.enhanced_metrics import compute_confidence_intervals

class AllPipelinesBenchmark:
    """Run all pipelines with enhanced metrics."""
    
    # Define all available pipelines
    ALL_PIPELINES = [
        'original_tejas',  # Golden ratio subsampling + sklearn
        'fused_char',      # Fused pipeline with char ngrams
        'fused_byte',      # Fused pipeline with byte BPE
        'optimized_fused'  # Optimized with SIMD + Numba
    ]
    
    # Define all datasets (simplified for now)
    ALL_DATASETS = [
        'wikipedia'
        # 'msmarco',  # Uncomment when ready
        # 'beir'      # Uncomment when ready
    ]
    
    def __init__(self, output_dir: str = None):
        """Initialize benchmark runner."""
        self.output_dir = Path(output_dir or "benchmark_results/all_pipelines")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runner = EnhancedBenchmarkRunner(output_dir=str(self.output_dir))
        self.results = []
        self.summary = {}
        
    def run_all_benchmarks(self, n_runs: int = 3, quick_mode: bool = False):
        """Run benchmarks for all pipelines and datasets."""
        
        pipelines = self.ALL_PIPELINES[:2] if quick_mode else self.ALL_PIPELINES
        datasets = self.ALL_DATASETS
        
        total_experiments = len(pipelines) * len(datasets) * n_runs
        experiment_count = 0
        
        print("=" * 80)
        print("COMPREHENSIVE PIPELINE BENCHMARKS WITH ENHANCED METRICS")
        print("=" * 80)
        print(f"Pipelines: {pipelines}")
        print(f"Datasets: {datasets}")
        print(f"Runs per config: {n_runs}")
        print(f"Total experiments: {total_experiments}")
        print("=" * 80)
        
        start_time = time.time()
        
        for pipeline in pipelines:
            pipeline_results = {}
            
            for dataset in datasets:
                dataset_results = []
                
                print(f"\n{'='*60}")
                print(f"Pipeline: {pipeline.upper()} | Dataset: {dataset.upper()}")
                print(f"{'='*60}")
                
                for run_idx in range(n_runs):
                    experiment_count += 1
                    
                    # Configure experiment
                    config = {
                        'experiment_id': f'{pipeline}_{dataset}_run{run_idx+1}',
                        'pipeline': pipeline,
                        'dataset': dataset,
                        'n_bits': 256,
                        'batch_size': 128,
                        'dataset_size': '10k' if quick_mode else '125k',
                        'seed': 42 + run_idx
                    }
                    
                    # Special handling for fused_byte pipeline
                    if pipeline == 'fused_byte':
                        config['tokenizer'] = 'byte_bpe'
                    
                    print(f"\n[{experiment_count}/{total_experiments}] {config['experiment_id']}")
                    exp_start = time.time()
                    
                    try:
                        # Run experiment
                        result = self.runner.run_single_experiment_safe(config)
                        elapsed = time.time() - exp_start
                        
                        if result['status'] == 'success':
                            metrics = result.get('metrics', {})
                            
                            # Extract key metrics
                            exp_result = {
                                'pipeline': pipeline,
                                'dataset': dataset,
                                'run': run_idx + 1,
                                'status': 'success',
                                'elapsed_time': elapsed,
                                
                                # Performance
                                'encoding_speed': metrics.get('docs_per_second', 0),
                                'query_latency_p50': metrics.get('search_latency_p50', 0),
                                'query_latency_p95': metrics.get('search_latency_p95', 0),
                                'memory_mb': metrics.get('peak_memory_mb', 0),
                                
                                # Core IR metrics
                                'ndcg_at_10': metrics.get('ndcg_at_10', 0),
                                'map_at_10': metrics.get('map_at_10', 0),
                                'mrr': metrics.get('mrr', metrics.get('mrr_at_10', 0)),
                                
                                # Enhanced metrics
                                'precision_at_1': metrics.get('precision_at_1', 0),
                                'precision_at_5': metrics.get('precision_at_5', 0),
                                'recall_at_10': metrics.get('recall_at_10', 0),
                                'success_at_1': metrics.get('success_at_1', 0),
                                'success_at_5': metrics.get('success_at_5', 0),
                            }
                            
                            dataset_results.append(exp_result)
                            self.results.append(exp_result)
                            
                            # Print summary
                            print(f"  ✓ Success in {elapsed:.1f}s")
                            print(f"    Speed: {exp_result['encoding_speed']:.0f} docs/s")
                            print(f"    Latency P50: {exp_result['query_latency_p50']:.2f} ms")
                            print(f"    NDCG@10: {exp_result['ndcg_at_10']:.4f}")
                            print(f"    Memory: {exp_result['memory_mb']:.1f} MB")
                            
                        else:
                            print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
                            self.results.append({
                                'pipeline': pipeline,
                                'dataset': dataset,
                                'run': run_idx + 1,
                                'status': 'failed',
                                'error': str(result.get('error', 'Unknown'))
                            })
                            
                    except Exception as e:
                        print(f"  ✗ Exception: {e}")
                        self.results.append({
                            'pipeline': pipeline,
                            'dataset': dataset,
                            'run': run_idx + 1,
                            'status': 'error',
                            'error': str(e)
                        })
                
                # Calculate statistics for this pipeline-dataset pair
                if dataset_results:
                    stats = self._calculate_statistics(dataset_results)
                    pipeline_results[dataset] = stats
                    self._print_statistics(pipeline, dataset, stats)
            
            self.summary[pipeline] = pipeline_results
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}")
        
        # Save all results
        self.save_results()
        self.generate_final_report()
        
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics with confidence intervals."""
        
        stats = {}
        metrics = ['encoding_speed', 'query_latency_p50', 'ndcg_at_10', 
                  'precision_at_1', 'recall_at_10', 'success_at_1', 'memory_mb']
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and r[metric] is not None]
            if values:
                median, lower, upper = compute_confidence_intervals(values)
                stats[metric] = {
                    'median': median,
                    'ci_lower': lower,
                    'ci_upper': upper,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n_samples': len(values)
                }
        
        return stats
    
    def _print_statistics(self, pipeline: str, dataset: str, stats: Dict):
        """Print formatted statistics."""
        
        print(f"\nStatistics Summary:")
        print(f"  Encoding Speed: {stats.get('encoding_speed', {}).get('median', 0):.0f} docs/s "
              f"[{stats.get('encoding_speed', {}).get('ci_lower', 0):.0f}, "
              f"{stats.get('encoding_speed', {}).get('ci_upper', 0):.0f}]")
        print(f"  Query Latency: {stats.get('query_latency_p50', {}).get('median', 0):.2f} ms "
              f"[{stats.get('query_latency_p50', {}).get('ci_lower', 0):.2f}, "
              f"{stats.get('query_latency_p50', {}).get('ci_upper', 0):.2f}]")
        print(f"  NDCG@10: {stats.get('ndcg_at_10', {}).get('median', 0):.4f} "
              f"[{stats.get('ndcg_at_10', {}).get('ci_lower', 0):.4f}, "
              f"{stats.get('ndcg_at_10', {}).get('ci_upper', 0):.4f}]")
        print(f"  Memory: {stats.get('memory_mb', {}).get('median', 0):.1f} MB")
    
    def save_results(self):
        """Save results in multiple formats."""
        
        # Save raw results as JSON
        json_path = self.output_dir / f"all_pipelines_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'raw_results': self.results,
                'summary': self.summary,
                'timestamp': self.timestamp
            }, f, indent=2)
        
        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = self.output_dir / f"all_pipelines_results_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FINAL BENCHMARK REPORT - ALL PIPELINES WITH ENHANCED METRICS")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Create comparison table
        report_lines.append("\nPERFORMANCE COMPARISON TABLE")
        report_lines.append("-" * 80)
        
        # Header
        report_lines.append(f"{'Pipeline':<20} {'Speed (docs/s)':<15} {'Latency (ms)':<15} "
                          f"{'NDCG@10':<12} {'Memory (MB)':<12}")
        report_lines.append("-" * 80)
        
        # Results for each pipeline
        for pipeline in self.ALL_PIPELINES:
            if pipeline in self.summary:
                for dataset, stats in self.summary[pipeline].items():
                    if stats:
                        speed = stats.get('encoding_speed', {}).get('median', 0)
                        latency = stats.get('query_latency_p50', {}).get('median', 0)
                        ndcg = stats.get('ndcg_at_10', {}).get('median', 0)
                        memory = stats.get('memory_mb', {}).get('median', 0)
                        
                        report_lines.append(
                            f"{pipeline:<20} {speed:<15.0f} {latency:<15.2f} "
                            f"{ndcg:<12.4f} {memory:<12.1f}"
                        )
        
        report_lines.append("-" * 80)
        
        # Enhanced metrics section
        report_lines.append("\nENHANCED METRICS SUMMARY")
        report_lines.append("-" * 80)
        
        for pipeline in self.ALL_PIPELINES:
            if pipeline in self.summary:
                report_lines.append(f"\n{pipeline.upper()}:")
                for dataset, stats in self.summary[pipeline].items():
                    if stats:
                        p1 = stats.get('precision_at_1', {}).get('median', 0)
                        r10 = stats.get('recall_at_10', {}).get('median', 0)
                        s1 = stats.get('success_at_1', {}).get('median', 0)
                        
                        report_lines.append(f"  {dataset}: P@1={p1:.4f}, R@10={r10:.4f}, S@1={s1:.4f}")
        
        # Write report
        report_text = "\n".join(report_lines)
        report_path = self.output_dir / f"final_report_{self.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Print summary to console
        print("\n" + report_text)
        
        return report_text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run all pipelines with enhanced metrics')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per configuration')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode - fewer pipelines and smaller dataset')
    parser.add_argument('--output', default='benchmark_results/all_pipelines',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = AllPipelinesBenchmark(output_dir=args.output)
    benchmark.run_all_benchmarks(n_runs=args.runs, quick_mode=args.quick)


if __name__ == '__main__':
    main()