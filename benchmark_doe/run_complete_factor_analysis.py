#!/usr/bin/env python3
"""
Complete Factor Analysis: All Datasets, All Factors, 10 Runs
Tests individual factors across Wikipedia, MS MARCO, and BEIR datasets
Includes specific test for fused_char without bit packing
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
from benchmark_doe.core.enhanced_metrics import compute_confidence_intervals, statistical_significance_test

class CompleteFactorAnalysis:
    """Complete factor analysis across all datasets with 10 runs."""
    
    # Define all factors to test
    ALL_FACTORS = {
        'n_bits': {
            'values': [64, 128, 256, 512],
            'pipelines': ['original_tejas', 'fused_char', 'fused_byte', 'optimized_fused'],
            'description': 'Number of bits for binary encoding'
        },
        'bit_packing': {
            'values': [False, True],
            'pipelines': ['fused_char'],  # Specifically test fused_char with/without bit packing
            'description': 'Bit packing for memory efficiency'
        },
        'use_simd': {
            'values': [False, True],
            'pipelines': ['fused_char', 'optimized_fused'],
            'description': 'SIMD acceleration'
        },
        'use_numba': {
            'values': [False, True],
            'pipelines': ['optimized_fused'],
            'description': 'Numba JIT compilation'
        },
        'batch_size': {
            'values': [64, 128, 256, 512],
            'pipelines': ['all'],
            'description': 'Processing batch size'
        },
        'energy_threshold': {
            'values': [0.85, 0.90, 0.95, 0.99],
            'pipelines': ['all'],
            'description': 'SVD energy threshold'
        }
    }
    
    # All datasets to test
    ALL_DATASETS = ['wikipedia', 'msmarco', 'beir']
    
    def __init__(self, output_dir: str = None):
        """Initialize complete factor analysis."""
        self.output_dir = Path(output_dir or "benchmark_results/complete_factor_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runner = EnhancedBenchmarkRunner(output_dir=str(self.output_dir))
        self.results = []
        
    def test_factor_on_dataset(self, factor_name: str, dataset: str, pipeline: str, 
                               n_runs: int = 10):
        """Test a single factor on a specific dataset with 10 runs."""
        
        factor_info = self.ALL_FACTORS[factor_name]
        factor_results = []
        
        print(f"\n{'='*70}")
        print(f"FACTOR: {factor_name} | DATASET: {dataset} | PIPELINE: {pipeline}")
        print(f"{'='*70}")
        print(f"Description: {factor_info['description']}")
        print(f"Testing values: {factor_info['values']}")
        print(f"Number of runs: {n_runs}")
        
        for value in factor_info['values']:
            print(f"\n--- {factor_name} = {value} ---")
            value_results = []
            
            for run_idx in range(n_runs):
                # Configure experiment
                config = {
                    'experiment_id': f'{factor_name}_{value}_{dataset}_{pipeline}_run{run_idx+1}',
                    'pipeline': pipeline,
                    'dataset': dataset,
                    'n_bits': 256,  # Default unless testing n_bits
                    'batch_size': 128,  # Default unless testing batch_size
                    'seed': 42 + run_idx,
                    factor_name: value  # Set the factor being tested
                }
                
                # Dataset-specific configuration
                if dataset == 'wikipedia':
                    config['dataset_size'] = '125k'
                elif dataset == 'beir':
                    config['beir_dataset'] = 'scifact'
                elif dataset == 'msmarco':
                    config['dataset_subset'] = 'dev'
                
                # Special handling for pipelines
                if pipeline == 'fused_byte':
                    config['tokenizer'] = 'byte_bpe'
                
                print(f"  Run {run_idx+1:2d}/10: ", end='', flush=True)
                
                try:
                    start = time.time()
                    result = self.runner.run_single_experiment_safe(config)
                    elapsed = time.time() - start
                    
                    if result['status'] == 'success':
                        m = result.get('metrics', {})
                        
                        exp_data = {
                            'factor': factor_name,
                            'value': value,
                            'dataset': dataset,
                            'pipeline': pipeline,
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
                        
                        value_results.append(exp_data)
                        self.results.append(exp_data)
                        
                        print(f"✓ Speed: {exp_data['speed']:6.0f} d/s, NDCG: {exp_data['ndcg_at_10']:.4f}")
                        
                    else:
                        print(f"✗ Failed: {result.get('error', 'Unknown')[:30]}")
                        
                except Exception as e:
                    print(f"✗ Error: {str(e)[:30]}")
            
            # Calculate statistics for this value
            if value_results:
                stats = self._calculate_statistics(value_results)
                factor_results.append({
                    'factor': factor_name,
                    'value': value,
                    'dataset': dataset,
                    'pipeline': pipeline,
                    'stats': stats,
                    'n_runs': len(value_results)
                })
        
        return factor_results
    
    def test_fused_char_bit_packing(self, n_runs: int = 10):
        """Specifically test fused_char with and without bit packing across all datasets."""
        
        print("\n" + "="*80)
        print("SPECIAL TEST: fused_char WITH/WITHOUT BIT PACKING")
        print("="*80)
        
        results = []
        
        for dataset in self.ALL_DATASETS:
            print(f"\nDataset: {dataset}")
            dataset_results = self.test_factor_on_dataset(
                'bit_packing', dataset, 'fused_char', n_runs=n_runs
            )
            results.extend(dataset_results)
        
        # Analyze bit packing impact
        self._analyze_bit_packing_impact(results)
        
        return results
    
    def run_complete_analysis(self, factors: List[str] = None, n_runs: int = 10):
        """Run complete factor analysis across all datasets."""
        
        factors = factors or ['n_bits', 'bit_packing', 'use_simd', 'batch_size']
        
        print("="*80)
        print("COMPLETE FACTOR ANALYSIS - ALL DATASETS, 10 RUNS")
        print("="*80)
        print(f"Factors: {factors}")
        print(f"Datasets: {self.ALL_DATASETS}")
        print(f"Runs per configuration: {n_runs}")
        
        start_time = time.time()
        
        # Test each factor
        for factor in factors:
            factor_info = self.ALL_FACTORS.get(factor)
            if not factor_info:
                print(f"Unknown factor: {factor}")
                continue
            
            # Determine which pipelines to test
            pipelines = factor_info['pipelines']
            if pipelines == ['all']:
                pipelines = ['original_tejas', 'fused_char', 'fused_byte', 'optimized_fused']
            
            for pipeline in pipelines:
                for dataset in self.ALL_DATASETS:
                    try:
                        self.test_factor_on_dataset(factor, dataset, pipeline, n_runs=n_runs)
                    except Exception as e:
                        print(f"Error testing {factor} on {dataset} with {pipeline}: {e}")
        
        # Special test for fused_char bit packing
        if 'bit_packing' in factors:
            self.test_fused_char_bit_packing(n_runs=n_runs)
        
        total_time = time.time() - start_time
        
        # Save and report
        self._save_results()
        self._generate_report(total_time)
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics with confidence intervals."""
        
        metrics = ['speed', 'latency_p50', 'ndcg_at_10', 'precision_at_1', 
                  'recall_at_10', 'success_at_1', 'mrr', 'memory_mb']
        
        stats = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                median, lower, upper = compute_confidence_intervals(values)
                stats[metric] = {
                    'median': median,
                    'ci_lower': lower,
                    'ci_upper': upper,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n': len(values)
                }
        return stats
    
    def _analyze_bit_packing_impact(self, results: List[Dict]):
        """Analyze specific impact of bit packing on fused_char."""
        
        print("\n" + "="*60)
        print("BIT PACKING IMPACT ANALYSIS (fused_char)")
        print("="*60)
        
        # Group by dataset
        datasets = set(r['dataset'] for r in results)
        
        for dataset in datasets:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            
            # Get with and without bit packing
            without_bp = next((r for r in dataset_results if not r['value']), None)
            with_bp = next((r for r in dataset_results if r['value']), None)
            
            if without_bp and with_bp:
                print(f"\n{dataset.upper()}:")
                print("  Without bit packing:")
                print(f"    Speed: {without_bp['stats']['speed']['median']:.0f} d/s")
                print(f"    NDCG: {without_bp['stats']['ndcg_at_10']['median']:.4f}")
                print(f"    Memory: {without_bp['stats']['memory_mb']['median']:.1f} MB")
                
                print("  With bit packing:")
                print(f"    Speed: {with_bp['stats']['speed']['median']:.0f} d/s")
                print(f"    NDCG: {with_bp['stats']['ndcg_at_10']['median']:.4f}")
                print(f"    Memory: {with_bp['stats']['memory_mb']['median']:.1f} MB")
                
                # Calculate impact
                speed_impact = ((with_bp['stats']['speed']['median'] - 
                               without_bp['stats']['speed']['median']) / 
                               without_bp['stats']['speed']['median'] * 100)
                mem_impact = ((with_bp['stats']['memory_mb']['median'] - 
                             without_bp['stats']['memory_mb']['median']) / 
                             without_bp['stats']['memory_mb']['median'] * 100) if without_bp['stats']['memory_mb']['median'] > 0 else 0
                
                print(f"  Impact:")
                print(f"    Speed change: {speed_impact:+.1f}%")
                print(f"    Memory change: {mem_impact:+.1f}%")
    
    def _save_results(self):
        """Save all results to files."""
        
        # Save raw results as JSON
        json_path = self.output_dir / f"complete_factor_analysis_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = self.output_dir / f"complete_factor_analysis_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            # Also save summary statistics
            summary = []
            for r in self.results:
                if isinstance(r, dict) and 'stats' in r:
                    summary.append({
                        'factor': r.get('factor'),
                        'value': r.get('value'),
                        'dataset': r.get('dataset'),
                        'pipeline': r.get('pipeline'),
                        'speed_median': r['stats'].get('speed', {}).get('median', 0),
                        'ndcg_median': r['stats'].get('ndcg_at_10', {}).get('median', 0),
                        'p1_median': r['stats'].get('precision_at_1', {}).get('median', 0),
                        'memory_median': r['stats'].get('memory_mb', {}).get('median', 0)
                    })
            
            if summary:
                summary_df = pd.DataFrame(summary)
                summary_path = self.output_dir / f"factor_summary_{self.timestamp}.csv"
                summary_df.to_csv(summary_path, index=False)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def _generate_report(self, total_time: float):
        """Generate final report."""
        
        print("\n" + "="*80)
        print("COMPLETE FACTOR ANALYSIS REPORT")
        print("="*80)
        print(f"Total experiments: {len(self.results)}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Complete factor analysis')
    parser.add_argument('--factors', nargs='+',
                       help='Factors to analyze (default: all)')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs per configuration (default: 10)')
    parser.add_argument('--output', 
                       default='benchmark_results/complete_factor_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run complete analysis
    analyzer = CompleteFactorAnalysis(output_dir=args.output)
    analyzer.run_complete_analysis(factors=args.factors, n_runs=args.runs)


if __name__ == '__main__':
    main()