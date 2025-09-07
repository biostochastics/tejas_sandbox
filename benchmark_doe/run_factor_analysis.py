#!/usr/bin/env python3
"""
Simplified Factor Analysis Runner for DOE Benchmarking
Tests individual factors and their interactions systematically
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
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import itertools
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_doe.run_doe_benchmark import EnhancedBenchmarkRunner
from benchmark_doe.core.factors import FactorRegistry


class FactorAnalysis:
    """Simplified factor analysis for TEJAS optimization parameters."""
    
    def __init__(self, output_dir: str = "./benchmark_results/factor_analysis"):
        """Initialize factor analysis runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.runner = EnhancedBenchmarkRunner(output_dir=str(self.output_dir))
        self.registry = FactorRegistry()
        self.results = []
        
        # Base configuration
        self.base_config = {
            'pipeline': 'optimized_fused',
            'pipeline_type': 'optimized_fused',
            'dataset': 'wikipedia',
            'n_bits': 256,
            'batch_size': 1000,
            'backend': 'numpy',
            'tokenizer': 'char_ngram',
            'svd_method': 'randomized',
            'use_simd': False,
            'use_numba': False,
            'use_itq': False,
            'use_reranker': False,
            'bit_packing': False,
            'downsample_ratio': 1.0,
            'energy_threshold': 0.95
        }
    
    def test_single_factor(self, factor_name: str, values: List[Any], n_runs: int = 10):
        """
        Test impact of a single factor on performance.
        
        Args:
            factor_name: Name of factor to test
            values: List of values to test for the factor
            n_runs: Number of runs per configuration
        """
        print(f"\n{'='*60}")
        print(f"SINGLE FACTOR ANALYSIS: {factor_name}")
        print(f"{'='*60}")
        print(f"Testing values: {values}")
        print(f"Runs per value: {n_runs}")
        print(f"Base configuration: {self.base_config['pipeline']}")
        
        results = []
        
        for value in values:
            print(f"\n--- Testing {factor_name}={value} ---")
            
            # Create configuration
            config = self.base_config.copy()
            config[factor_name] = value
            
            # Validate configuration
            is_valid, issues = self.registry.validate_configuration(config, strict=False)
            if not is_valid and len(issues.get('error', [])) > 0:
                print(f"  ⚠ Invalid configuration: {issues['error']}")
                continue
            
            # Run experiments
            run_results = []
            for run_idx in range(n_runs):
                config['experiment_id'] = f"{factor_name}_{value}_{run_idx+1}"
                config['run_number'] = run_idx + 1
                config['seed'] = 42 + run_idx
                
                result = self.runner.run_single_experiment_safe(config)
                
                if result['status'] == 'success':
                    metrics = result.get('metrics', {})
                    run_results.append({
                        'encoding_speed': metrics.get('docs_per_second', 0),
                        'query_latency': metrics.get('search_latency_p50', 0),
                        'memory_mb': metrics.get('peak_memory_mb', 0),
                        'ndcg': metrics.get('ndcg_at_10', 0)
                    })
                    print(f"  Run {run_idx+1}: ✓", end=" ")
                else:
                    print(f"  Run {run_idx+1}: ✗", end=" ")
            
            print()  # New line
            
            # Compute statistics
            if run_results:
                stats = self._compute_statistics(run_results)
                stats['factor'] = factor_name
                stats['value'] = value
                stats['n_runs'] = len(run_results)
                results.append(stats)
        
        # Save and report results
        self._save_factor_results(factor_name, results)
        self._report_factor_analysis(factor_name, results)
        
        return results
    
    def test_interaction(self, factor1: str, factor2: str, 
                        values1: List[Any], values2: List[Any], 
                        n_runs: int = 5):
        """
        Test interaction between two factors.
        
        Args:
            factor1, factor2: Names of factors to test
            values1, values2: Values to test for each factor
            n_runs: Number of runs per configuration
        """
        print(f"\n{'='*60}")
        print(f"INTERACTION ANALYSIS: {factor1} × {factor2}")
        print(f"{'='*60}")
        print(f"{factor1} values: {values1}")
        print(f"{factor2} values: {values2}")
        print(f"Configurations: {len(values1) * len(values2)}")
        print(f"Total experiments: {len(values1) * len(values2) * n_runs}")
        
        results = []
        
        for val1, val2 in itertools.product(values1, values2):
            print(f"\n--- Testing {factor1}={val1}, {factor2}={val2} ---")
            
            # Create configuration
            config = self.base_config.copy()
            config[factor1] = val1
            config[factor2] = val2
            
            # Validate configuration
            is_valid, issues = self.registry.validate_configuration(config, strict=False)
            if not is_valid and len(issues.get('error', [])) > 0:
                print(f"  ⚠ Invalid combination: {issues['error']}")
                continue
            
            # Run experiments
            run_results = []
            for run_idx in range(n_runs):
                config['experiment_id'] = f"{factor_name}_{value}_{run_idx+1}"
                config['run_number'] = run_idx + 1
                config['seed'] = 42 + run_idx
                
                result = self.runner.run_single_experiment_safe(config)
                
                if result['status'] == 'success':
                    metrics = result.get('metrics', {})
                    run_results.append({
                        'encoding_speed': metrics.get('docs_per_second', 0),
                        'query_latency': metrics.get('search_latency_p50', 0),
                        'memory_mb': metrics.get('peak_memory_mb', 0),
                        'ndcg': metrics.get('ndcg_at_10', 0)
                    })
                    print(f"  Run {run_idx+1}: ✓", end=" ")
                else:
                    print(f"  Run {run_idx+1}: ✗", end=" ")
            
            print()  # New line
            
            # Compute statistics
            if run_results:
                stats = self._compute_statistics(run_results)
                stats[factor1] = val1
                stats[factor2] = val2
                stats['n_runs'] = len(run_results)
                results.append(stats)
        
        # Save and report results
        self._save_interaction_results(factor1, factor2, results)
        self._report_interaction_analysis(factor1, factor2, results)
        
        return results
    
    def _compute_statistics(self, run_results: List[Dict]) -> Dict:
        """Compute median and 95% CI for metrics."""
        stats = {}
        
        for metric in ['encoding_speed', 'query_latency', 'memory_mb', 'ndcg']:
            values = [r[metric] for r in run_results if metric in r]
            if values:
                stats[f'{metric}_median'] = np.median(values)
                stats[f'{metric}_ci_lower'] = np.percentile(values, 2.5)
                stats[f'{metric}_ci_upper'] = np.percentile(values, 97.5)
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values)
        
        return stats
    
    def _save_factor_results(self, factor_name: str, results: List[Dict]):
        """Save factor analysis results."""
        output_file = self.output_dir / f"factor_{factor_name}_{self.timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'analysis_type': 'single_factor',
                'factor': factor_name,
                'base_config': self.base_config,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Also save as CSV for easy analysis
        df = pd.DataFrame(results)
        csv_file = self.output_dir / f"factor_{factor_name}_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {output_file}")
        print(f"  - {csv_file}")
    
    def _save_interaction_results(self, factor1: str, factor2: str, results: List[Dict]):
        """Save interaction analysis results."""
        output_file = self.output_dir / f"interaction_{factor1}_{factor2}_{self.timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'analysis_type': 'interaction',
                'factors': [factor1, factor2],
                'base_config': self.base_config,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Also save as CSV
        df = pd.DataFrame(results)
        csv_file = self.output_dir / f"interaction_{factor1}_{factor2}_{self.timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {output_file}")
        print(f"  - {csv_file}")
    
    def _report_factor_analysis(self, factor_name: str, results: List[Dict]):
        """Report single factor analysis results."""
        print(f"\n{'='*60}")
        print(f"FACTOR ANALYSIS RESULTS: {factor_name}")
        print(f"{'='*60}")
        
        # Create table
        print(f"\n| {factor_name:<12} | Speed (docs/s) | Latency (ms) | Memory (MB) | NDCG |")
        print("|" + "-"*14 + "|" + "-"*16 + "|" + "-"*14 + "|" + "-"*13 + "|" + "-"*6 + "|")
        
        for result in results:
            value = str(result['value'])
            speed = f"{result['encoding_speed_median']:.0f}"
            latency = f"{result['query_latency_median']:.2f}"
            memory = f"{result['memory_mb_median']:.1f}"
            ndcg = f"{result['ndcg_median']:.3f}"
            
            print(f"| {value:<12} | {speed:>14} | {latency:>12} | {memory:>11} | {ndcg:>4} |")
        
        # Calculate impact
        if len(results) >= 2:
            print(f"\n### Impact Analysis")
            baseline = results[0]
            for result in results[1:]:
                print(f"\n{factor_name}={result['value']} vs {baseline['value']}:")
                
                speed_change = (result['encoding_speed_median'] - baseline['encoding_speed_median']) / baseline['encoding_speed_median'] * 100
                memory_change = (result['memory_mb_median'] - baseline['memory_mb_median']) / baseline['memory_mb_median'] * 100
                ndcg_change = (result['ndcg_median'] - baseline['ndcg_median']) / baseline['ndcg_median'] * 100
                
                print(f"  Speed: {speed_change:+.1f}%")
                print(f"  Memory: {memory_change:+.1f}%")
                print(f"  NDCG: {ndcg_change:+.1f}%")
    
    def _report_interaction_analysis(self, factor1: str, factor2: str, results: List[Dict]):
        """Report interaction analysis results."""
        print(f"\n{'='*60}")
        print(f"INTERACTION RESULTS: {factor1} × {factor2}")
        print(f"{'='*60}")
        
        # Create pivot table for encoding speed
        df = pd.DataFrame(results)
        if not df.empty:
            pivot = df.pivot_table(
                values='encoding_speed_median',
                index=factor1,
                columns=factor2,
                aggfunc='first'
            )
            
            print("\n### Encoding Speed (docs/s)")
            print(pivot.to_string())
            
            # Check for interaction effect
            print("\n### Interaction Effect")
            # Simple check: if lines would cross when plotted, there's interaction
            values1 = df[factor1].unique()
            values2 = df[factor2].unique()
            
            if len(values1) >= 2 and len(values2) >= 2:
                # Calculate slopes for each level of factor2
                for val2 in values2[:2]:  # Just check first two levels
                    subset = df[df[factor2] == val2].sort_values(factor1)
                    if len(subset) >= 2:
                        slope = (subset.iloc[-1]['encoding_speed_median'] - 
                                subset.iloc[0]['encoding_speed_median'])
                        print(f"  Effect of {factor1} when {factor2}={val2}: {slope:.0f} docs/s")
                
                print("\nSignificant interaction if effects differ substantially.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run factor analysis for TEJAS optimization")
    
    # Single factor analysis
    parser.add_argument('--factor', type=str, help='Single factor to analyze')
    parser.add_argument('--values', type=str, help='Comma-separated values to test')
    
    # Interaction analysis
    parser.add_argument('--factors', type=str, help='Two factors for interaction (comma-separated)')
    parser.add_argument('--interaction', action='store_true', help='Run interaction analysis')
    
    # Common options
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per configuration')
    parser.add_argument('--output', type=str, default='./benchmark_results/factor_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FactorAnalysis(output_dir=args.output)
    
    if args.factor and args.values:
        # Single factor analysis
        values = []
        for v in args.values.split(','):
            # Try to convert to appropriate type
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    # Keep as string, but handle boolean
                    if v.lower() in ['true', 'false']:
                        values.append(v.lower() == 'true')
                    else:
                        values.append(v)
        
        analyzer.test_single_factor(args.factor, values, args.runs)
        
    elif args.factors and args.interaction:
        # Interaction analysis
        factors = args.factors.split(',')
        if len(factors) != 2:
            print("Error: Interaction analysis requires exactly 2 factors")
            sys.exit(1)
        
        # Default values for common factors
        default_values = {
            'use_simd': [False, True],
            'bit_packing': [False, True],
            'use_numba': [False, True],
            'use_itq': [False, True],
            'n_bits': [128, 256, 512],
            'batch_size': [500, 1000, 2000],
            'downsample_ratio': [0.5, 0.75, 1.0],
            'energy_threshold': [0.90, 0.95, 0.99]
        }
        
        values1 = default_values.get(factors[0], [False, True])
        values2 = default_values.get(factors[1], [False, True])
        
        analyzer.test_interaction(factors[0], factors[1], values1, values2, args.runs)
        
    else:
        # Run default analysis
        print("Running default factor analysis...")
        
        # Test n_bits impact
        analyzer.test_single_factor('n_bits', [64, 128, 256, 512], args.runs)
        
        # Test SIMD and bit packing interaction
        analyzer.test_interaction('use_simd', 'bit_packing', 
                                 [False, True], [False, True], args.runs)


if __name__ == "__main__":
    main()