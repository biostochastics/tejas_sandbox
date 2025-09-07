#!/usr/bin/env python3
"""
Simplified TEJAS vs BERT Comparison Runner
Runs 7 pipelines (5 TEJAS + 2 BERT) × 3 datasets × 10 runs = 210 experiments
Computes median and 95% confidence intervals for fair comparison
"""

import sys
import os
import json
import yaml
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from local benchmark_doe directory
from benchmark_doe.run_doe_benchmark import EnhancedBenchmarkRunner
from benchmark_doe.core.encoder_factory import EncoderFactory


class TEJASvsBERTComparison:
    """Simplified comparison runner for TEJAS vs BERT benchmarks."""
    
    def __init__(self, config_path: str = None):
        """Initialize comparison runner."""
        self.config_path = config_path or "configs/tejas_vs_bert_comparison.yaml"
        self.config = self.load_config()
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir = Path(self.config['output']['results_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize benchmark runner
        self.runner = EnhancedBenchmarkRunner(output_dir=str(self.output_dir))
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(__file__).parent / self.config_path
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_single_experiment(self, pipeline: Dict, dataset: str, run_idx: int) -> Dict:
        """Run a single experiment and return results."""
        
        # Create experiment configuration
        config = {
            'experiment_id': f"{pipeline['name']}_{dataset}_{run_idx+1}",
            'pipeline': pipeline['type'],
            'pipeline_type': pipeline['type'],
            'pipeline_name': pipeline['name'],
            'dataset': dataset,
            'run_number': run_idx + 1,
            'seed': self.config['benchmark']['random_seeds'][run_idx],
            **pipeline.get('config', {})
        }
        
        # Add n_bits only for TEJAS pipelines
        if not pipeline['type'].startswith('bert'):
            config['n_bits'] = config.get('n_bits', 256)
        
        try:
            # Run experiment
            start_time = time.time()
            result = self.runner.run_single_experiment_safe(config)
            elapsed_time = time.time() - start_time
            
            # Extract metrics
            if result['status'] == 'success':
                metrics = result.get('metrics', {})
                return {
                    'pipeline': pipeline['name'],
                    'dataset': dataset,
                    'run': run_idx + 1,
                    'encoding_speed': metrics.get('docs_per_second', 0),
                    'query_latency_p50': metrics.get('search_latency_p50', 0),
                    'query_latency_p95': metrics.get('search_latency_p95', 0),
                    'peak_memory_mb': metrics.get('peak_memory_mb', 0),
                    'ndcg_at_10': metrics.get('ndcg_at_10', 0),
                    'map_at_10': metrics.get('map_at_10', 0),
                    'mrr_at_10': metrics.get('mrr_at_10', 0),
                    'recall_at_100': metrics.get('recall_at_100', 0),
                    'index_size_mb': metrics.get('index_size_mb', 0),
                    'status': 'success',
                    'elapsed_time': elapsed_time
                }
            else:
                return {
                    'pipeline': pipeline['name'],
                    'dataset': dataset,
                    'run': run_idx + 1,
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            return {
                'pipeline': pipeline['name'],
                'dataset': dataset,
                'run': run_idx + 1,
                'status': 'error',
                'error': str(e)
            }
    
    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute median and 95% confidence intervals."""
        if not values:
            return {'median': 0, 'ci_lower': 0, 'ci_upper': 0}
        
        values = np.array(values)
        median = np.median(values)
        
        # Compute 95% CI using percentile method
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        
        return {
            'median': median,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    def run_comparison(self):
        """Run the complete comparison benchmark."""
        
        pipelines = self.config['pipelines']
        datasets = self.config['datasets']
        n_runs = self.config['benchmark']['n_runs']
        
        total_experiments = len(pipelines) * len(datasets) * n_runs
        
        print("="*80)
        print("TEJAS vs BERT HEAD-TO-HEAD COMPARISON")
        print("="*80)
        print(f"Pipelines: {len(pipelines)} ({', '.join(p['name'] for p in pipelines)})")
        print(f"Datasets: {len(datasets)} ({', '.join(d['name'] for d in datasets)})")
        print(f"Runs per configuration: {n_runs}")
        print(f"Total experiments: {total_experiments}")
        print("="*80)
        
        completed = 0
        start_time = time.time()
        
        # Run all experiments
        for pipeline in pipelines:
            for dataset_config in datasets:
                dataset = dataset_config['name']
                print(f"\n--- {pipeline['name']} on {dataset} ---")
                
                for run_idx in range(n_runs):
                    # Run experiment
                    result = self.run_single_experiment(pipeline, dataset, run_idx)
                    self.results.append(result)
                    
                    completed += 1
                    
                    # Progress update
                    status = "✓" if result['status'] == 'success' else "✗"
                    print(f"  Run {run_idx+1}/{n_runs}: {status} ", end="")
                    
                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = (total_experiments - completed) / rate if rate > 0 else 0
                        print(f"\n  Progress: {completed}/{total_experiments} ({100*completed/total_experiments:.1f}%) - ETA: {remaining/60:.1f} min")
                        self.save_checkpoint()
                
                print()  # New line after runs
        
        # Save final results
        self.save_results()
        self.generate_report()
    
    def save_checkpoint(self):
        """Save intermediate results."""
        checkpoint_file = self.output_dir / f"checkpoint_{self.timestamp}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def save_results(self):
        """Save final results in multiple formats."""
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save as JSON
        json_file = self.output_dir / f"results_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'config': self.config,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save as CSV
        csv_file = self.output_dir / f"results_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {json_file}")
        print(f"  - {csv_file}")
    
    def generate_report(self):
        """Generate comparison report with median and 95% CI."""
        
        df = pd.DataFrame(self.results)
        successful_df = df[df['status'] == 'success']
        
        if successful_df.empty:
            print("\nNo successful experiments to report")
            return
        
        print("\n" + "="*80)
        print("COMPARISON RESULTS (Median [95% CI])")
        print("="*80)
        
        # Aggregate by pipeline and dataset
        metrics = ['encoding_speed', 'query_latency_p50', 'peak_memory_mb', 'ndcg_at_10', 'map_at_10', 'mrr_at_10']
        
        for dataset in self.config['datasets']:
            dataset_name = dataset['name']
            dataset_df = successful_df[successful_df['dataset'] == dataset_name]
            
            if dataset_df.empty:
                continue
            
            print(f"\n### Dataset: {dataset_name.upper()}")
            print("-"*70)
            
            # Create comparison table
            table_data = []
            for pipeline in self.config['pipelines']:
                pipeline_name = pipeline['name']
                pipeline_df = dataset_df[dataset_df['pipeline'] == pipeline_name]
                
                if pipeline_df.empty:
                    continue
                
                row = [pipeline_name]
                for metric in metrics:
                    values = pipeline_df[metric].values
                    stats = self.compute_statistics(values)
                    
                    # Format based on metric type
                    if metric == 'encoding_speed':
                        row.append(f"{stats['median']:.0f} [{stats['ci_lower']:.0f}, {stats['ci_upper']:.0f}]")
                    elif metric == 'query_latency_p50':
                        row.append(f"{stats['median']:.2f} [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}]")
                    elif metric == 'peak_memory_mb':
                        row.append(f"{stats['median']:.1f} [{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]")
                    elif metric in ['ndcg_at_10', 'map_at_10', 'mrr_at_10']:
                        row.append(f"{stats['median']:.3f} [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
                
                table_data.append(row)
            
            # Print table
            headers = ['Pipeline', 'Speed (docs/s)', 'Latency (ms)', 'Memory (MB)', 'NDCG@10', 'MAP@10', 'MRR@10']
            col_widths = [15, 20, 20, 20, 12, 12, 12]
            
            # Print headers
            header_line = "|"
            for header, width in zip(headers, col_widths):
                header_line += f" {header:<{width-2}} |"
            print(header_line)
            print("|" + "-"*(sum(col_widths) + len(col_widths) - 1) + "|")
            
            # Print data
            for row in table_data:
                data_line = "|"
                for cell, width in zip(row, col_widths):
                    data_line += f" {cell:<{width-2}} |"
                print(data_line)
        
        # Statistical significance tests
        self.perform_statistical_tests(successful_df)
        
        print("\n" + "="*80)
    
    def perform_statistical_tests(self, df: pd.DataFrame):
        """Perform statistical significance tests between pipelines."""
        
        print("\n### STATISTICAL SIGNIFICANCE TESTS")
        print("-"*70)
        
        # Compare TEJAS Optimized vs BERT models
        tejas_opt = df[df['pipeline'] == 'TEJAS-Optimized']
        bert_mini = df[df['pipeline'] == 'BERT-MiniLM']
        bert_mpnet = df[df['pipeline'] == 'BERT-MPNet']
        
        if not tejas_opt.empty and not bert_mini.empty:
            for metric in ['encoding_speed', 'ndcg_at_10']:
                tejas_values = tejas_opt[metric].values
                bert_values = bert_mini[metric].values
                
                if len(tejas_values) > 0 and len(bert_values) > 0:
                    # Mann-Whitney U test
                    statistic, p_value = stats.mannwhitneyu(tejas_values, bert_values, alternative='two-sided')
                    
                    print(f"\nTEJAS-Optimized vs BERT-MiniLM ({metric}):")
                    print(f"  Mann-Whitney U: {statistic:.2f}, p-value: {p_value:.4f}")
                    if p_value < 0.05:
                        print(f"  Result: Statistically significant difference (p < 0.05)")
                    else:
                        print(f"  Result: No significant difference (p >= 0.05)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run TEJAS vs BERT comparison")
    parser.add_argument('--config', type=str, 
                       default='configs/tejas_vs_bert_comparison.yaml',
                       help='Configuration file path')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 2 runs only')
    
    args = parser.parse_args()
    
    # Modify config for quick test if requested
    if args.quick:
        comparison = TEJASvsBERTComparison(args.config)
        comparison.config['benchmark']['n_runs'] = 2
        comparison.config['benchmark']['random_seeds'] = [42, 123]
        print("Running quick test with 2 runs only...")
    else:
        comparison = TEJASvsBERTComparison(args.config)
    
    # Run comparison
    comparison.run_comparison()


if __name__ == "__main__":
    main()