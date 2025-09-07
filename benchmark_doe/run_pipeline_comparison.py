#!/usr/bin/env python3
"""
Pipeline Comparison: All pipelines across all datasets with 10 runs each
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

class PipelineComparison:
    """Compare all pipelines across all datasets."""
    
    ALL_PIPELINES = ['original_tejas', 'fused_char', 'fused_byte', 'optimized_fused']
    ALL_DATASETS = ['wikipedia', 'msmarco', 'beir']
    
    def __init__(self, output_dir: str = None):
        """Initialize pipeline comparison."""
        self.output_dir = Path(output_dir or "benchmark_results/pipeline_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runner = EnhancedBenchmarkRunner(output_dir=str(self.output_dir))
        self.results = []
        
    def run_comparison(self, pipelines: List[str] = None, datasets: List[str] = None, 
                       n_runs: int = 10):
        """Run complete pipeline comparison."""
        
        pipelines = pipelines or self.ALL_PIPELINES
        datasets = datasets or self.ALL_DATASETS
        
        print("="*80)
        print("PIPELINE COMPARISON - ALL DATASETS")
        print("="*80)
        print(f"Pipelines: {pipelines}")
        print(f"Datasets: {datasets}")
        print(f"Runs per configuration: {n_runs}")
        
        start_time = time.time()
        
        for dataset in datasets:
            print(f"\n{'='*70}")
            print(f"DATASET: {dataset.upper()}")
            print(f"{'='*70}")
            
            for pipeline in pipelines:
                print(f"\n--- Pipeline: {pipeline} ---")
                
                for run_idx in range(n_runs):
                    # Configure experiment
                    config = {
                        'experiment_id': f'{pipeline}_{dataset}_run{run_idx+1}',
                        'pipeline': pipeline,
                        'dataset': dataset,
                        'n_bits': 256,
                        'batch_size': 128,
                        'seed': 42 + run_idx
                    }
                    
                    # Dataset-specific configuration
                    if dataset == 'wikipedia':
                        config['dataset_size'] = '125k'
                    elif dataset == 'beir':
                        config['beir_dataset'] = 'scifact'
                    elif dataset == 'msmarco':
                        config['dataset_subset'] = 'dev'
                    
                    # Pipeline-specific configuration
                    if pipeline == 'fused_byte':
                        config['tokenizer'] = 'byte_bpe'
                    
                    print(f"  Run {run_idx+1:2d}/{n_runs}: ", end='', flush=True)
                    
                    try:
                        start = time.time()
                        result = self.runner.run_single_experiment_safe(config)
                        elapsed = time.time() - start
                        
                        if result['status'] == 'success':
                            m = result.get('metrics', {})
                            
                            exp_data = {
                                'pipeline': pipeline,
                                'dataset': dataset,
                                'run': run_idx + 1,
                                'elapsed': elapsed,
                                # Performance metrics
                                'speed': m.get('docs_per_second', 0),
                                'latency_p50': m.get('search_latency_p50', 0),
                                'latency_p95': m.get('search_latency_p95', 0),
                                'memory_mb': m.get('peak_memory_mb', 0),
                                # Enhanced IR metrics
                                'ndcg_at_10': m.get('ndcg_at_10', 0),
                                'precision_at_1': m.get('precision_at_1', 0),
                                'precision_at_5': m.get('precision_at_5', 0),
                                'recall_at_10': m.get('recall_at_10', 0),
                                'recall_at_100': m.get('recall_at_100', 0),
                                'success_at_1': m.get('success_at_1', 0),
                                'success_at_10': m.get('success_at_10', 0),
                                'mrr': m.get('mrr', 0),
                                'map_at_10': m.get('map_at_10', 0)
                            }
                            
                            self.results.append(exp_data)
                            print(f"✓ Speed: {exp_data['speed']:6.0f} d/s, NDCG: {exp_data['ndcg_at_10']:.4f}")
                            
                        else:
                            print(f"✗ Failed: {result.get('error', 'Unknown')[:30]}")
                            
                    except Exception as e:
                        print(f"✗ Error: {str(e)[:30]}")
        
        total_time = time.time() - start_time
        
        # Save and report
        self._save_results()
        self._generate_report(total_time)
    
    def _save_results(self):
        """Save all results to files."""
        
        # Save raw results as JSON
        json_path = self.output_dir / f"pipeline_comparison_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'results': self.results,
                'timestamp': self.timestamp,
                'summary': self._compute_summary()
            }, f, indent=2)
        
        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = self.output_dir / f"pipeline_comparison_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def _compute_summary(self):
        """Compute summary statistics for each pipeline-dataset combination."""
        
        summary = {}
        
        for pipeline in self.ALL_PIPELINES:
            summary[pipeline] = {}
            
            for dataset in self.ALL_DATASETS:
                dataset_results = [r for r in self.results 
                                  if r['pipeline'] == pipeline and r['dataset'] == dataset]
                
                if dataset_results:
                    metrics = {}
                    for metric in ['speed', 'latency_p50', 'ndcg_at_10', 'precision_at_1', 
                                  'recall_at_10', 'success_at_1', 'memory_mb', 'mrr']:
                        values = [r.get(metric, 0) for r in dataset_results]
                        if values:
                            median, lower, upper = compute_confidence_intervals(values)
                            metrics[metric] = {
                                'median': median,
                                'ci_lower': lower,
                                'ci_upper': upper,
                                'mean': np.mean(values),
                                'std': np.std(values)
                            }
                    summary[pipeline][dataset] = metrics
        
        return summary
    
    def _generate_report(self, total_time: float):
        """Generate final report."""
        
        print("\n" + "="*80)
        print("PIPELINE COMPARISON REPORT")
        print("="*80)
        
        summary = self._compute_summary()
        
        # Create comparison table for each dataset
        for dataset in self.ALL_DATASETS:
            print(f"\n{dataset.upper()} Dataset:")
            print("-" * 70)
            print("| Pipeline         | Speed (d/s) | Latency (ms) | NDCG@10 | P@1  | Memory |")
            print("|------------------|-------------|--------------|---------|------|--------|")
            
            for pipeline in self.ALL_PIPELINES:
                if pipeline in summary and dataset in summary[pipeline]:
                    s = summary[pipeline][dataset]
                    speed = s.get('speed', {}).get('median', 0)
                    latency = s.get('latency_p50', {}).get('median', 0)
                    ndcg = s.get('ndcg_at_10', {}).get('median', 0)
                    p1 = s.get('precision_at_1', {}).get('median', 0)
                    mem = s.get('memory_mb', {}).get('median', 0)
                    
                    print(f"| {pipeline:<16} | {speed:>11.0f} | {latency:>12.2f} | "
                          f"{ndcg:>7.4f} | {p1:>4.2f} | {mem:>6.0f} |")
        
        # Best pipeline analysis
        print("\n" + "="*60)
        print("BEST PIPELINE BY METRIC:")
        print("-" * 60)
        
        for dataset in self.ALL_DATASETS:
            print(f"\n{dataset.upper()}:")
            
            # Find best for each metric
            best_speed = max(self.ALL_PIPELINES, 
                           key=lambda p: summary.get(p, {}).get(dataset, {}).get('speed', {}).get('median', 0))
            best_ndcg = max(self.ALL_PIPELINES,
                          key=lambda p: summary.get(p, {}).get(dataset, {}).get('ndcg_at_10', {}).get('median', 0))
            
            print(f"  Best Speed: {best_speed}")
            print(f"  Best NDCG:  {best_ndcg}")
        
        print(f"\nTotal experiments: {len(self.results)}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Pipeline comparison across all datasets')
    parser.add_argument('--pipelines', nargs='+',
                       default=['original_tejas', 'fused_char', 'fused_byte', 'optimized_fused'],
                       help='Pipelines to compare')
    parser.add_argument('--datasets', nargs='+',
                       default=['wikipedia', 'msmarco', 'beir'],
                       help='Datasets to test on')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs per configuration (default: 10)')
    parser.add_argument('--output', 
                       default='benchmark_results/pipeline_comparison',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = PipelineComparison(output_dir=args.output)
    comparator.run_comparison(pipelines=args.pipelines, datasets=args.datasets, n_runs=args.runs)


if __name__ == '__main__':
    main()