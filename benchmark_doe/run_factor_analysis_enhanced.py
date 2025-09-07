#!/usr/bin/env python3
"""
Enhanced Factor Analysis with Comprehensive Metrics
Tests individual factors and their impact on performance using enhanced IR metrics
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
import itertools
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_doe.run_doe_benchmark import EnhancedBenchmarkRunner
from benchmark_doe.core.enhanced_metrics import compute_confidence_intervals, statistical_significance_test

class EnhancedFactorAnalysis:
    """Factor analysis with enhanced metrics for TEJAS optimization parameters."""
    
    # Define factors to test
    FACTORS = {
        'n_bits': {
            'values': [64, 128, 256, 512],
            'type': 'ordinal',
            'description': 'Number of bits for binary encoding'
        },
        'use_simd': {
            'values': [False, True],
            'type': 'binary',
            'description': 'SIMD acceleration',
            'applicable_to': ['fused_char', 'fused_byte', 'optimized_fused']
        },
        'use_numba': {
            'values': [False, True],
            'type': 'binary', 
            'description': 'Numba JIT compilation',
            'applicable_to': ['optimized_fused']
        },
        'bit_packing': {
            'values': [False, True],
            'type': 'binary',
            'description': 'Pack bits for memory efficiency',
            'applicable_to': ['fused_char', 'optimized_fused']
        },
        'batch_size': {
            'values': [128, 256, 512, 1024],
            'type': 'ordinal',
            'description': 'Processing batch size'
        },
        'energy_threshold': {
            'values': [0.80, 0.85, 0.90, 0.95, 0.99],
            'type': 'continuous',
            'description': 'SVD energy threshold'
        },
        'downsample_ratio': {
            'values': [0.1, 0.25, 0.5, 0.75, 1.0],
            'type': 'continuous',
            'description': 'Data downsampling ratio',
            'applicable_to': ['optimized_fused']
        }
    }
    
    def __init__(self, output_dir: str = None):
        """Initialize enhanced factor analysis."""
        self.output_dir = Path(output_dir or "benchmark_results/factor_analysis_enhanced")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.runner = EnhancedBenchmarkRunner(output_dir=str(self.output_dir))
        self.results = []
        
    def test_single_factor(self, factor_name: str, pipeline: str = 'optimized_fused', 
                          n_runs: int = 3, dataset_size: str = '10k'):
        """Test impact of a single factor with enhanced metrics."""
        
        if factor_name not in self.FACTORS:
            raise ValueError(f"Unknown factor: {factor_name}")
        
        factor_info = self.FACTORS[factor_name]
        
        # Check if factor is applicable to pipeline
        if 'applicable_to' in factor_info:
            if pipeline not in factor_info['applicable_to']:
                print(f"Factor {factor_name} not applicable to {pipeline}")
                return None
        
        print(f"\n{'='*70}")
        print(f"FACTOR ANALYSIS: {factor_name}")
        print(f"{'='*70}")
        print(f"Pipeline: {pipeline}")
        print(f"Description: {factor_info['description']}")
        print(f"Testing values: {factor_info['values']}")
        print(f"Runs per value: {n_runs}")
        
        factor_results = []
        
        for value in factor_info['values']:
            print(f"\n--- Testing {factor_name}={value} ---")
            value_results = []
            
            for run_idx in range(n_runs):
                # Create configuration
                config = {
                    'experiment_id': f'{factor_name}_{value}_run{run_idx+1}',
                    'pipeline': pipeline,
                    'dataset': 'wikipedia',
                    'dataset_size': dataset_size,
                    'n_bits': 256,  # Default
                    'batch_size': 128,  # Default
                    'seed': 42 + run_idx
                }
                
                # Set the factor value
                config[factor_name] = value
                
                # Special handling for certain factors
                if pipeline == 'fused_byte':
                    config['tokenizer'] = 'byte_bpe'
                
                print(f"  Run {run_idx+1}: ", end='', flush=True)
                
                try:
                    start = time.time()
                    result = self.runner.run_single_experiment_safe(config)
                    elapsed = time.time() - start
                    
                    if result['status'] == 'success':
                        m = result.get('metrics', {})
                        
                        exp_data = {
                            'factor': factor_name,
                            'value': value,
                            'run': run_idx + 1,
                            'elapsed': elapsed,
                            # Performance metrics
                            'speed': m.get('docs_per_second', 0),
                            'latency_p50': m.get('search_latency_p50', 0),
                            'memory_mb': m.get('peak_memory_mb', 0),
                            # Enhanced IR metrics
                            'ndcg_10': m.get('ndcg_at_10', 0),
                            'precision_1': m.get('precision_at_1', 0),
                            'precision_5': m.get('precision_at_5', 0),
                            'recall_10': m.get('recall_at_10', 0),
                            'success_1': m.get('success_at_1', 0),
                            'mrr': m.get('mrr', 0)
                        }
                        
                        value_results.append(exp_data)
                        self.results.append(exp_data)
                        
                        print(f"✓ Speed: {exp_data['speed']:.0f} d/s, NDCG: {exp_data['ndcg_10']:.4f}")
                        
                    else:
                        print(f"✗ Failed: {result.get('error', 'Unknown')}")
                        
                except Exception as e:
                    print(f"✗ Error: {e}")
            
            # Calculate statistics for this value
            if value_results:
                stats = self._calculate_statistics(value_results)
                factor_results.append({
                    'value': value,
                    'stats': stats,
                    'n_runs': len(value_results)
                })
        
        # Analyze factor impact
        if factor_results:
            self._analyze_factor_impact(factor_name, factor_results)
        
        return factor_results
    
    def test_interaction(self, factor1: str, factor2: str, pipeline: str = 'optimized_fused',
                        n_runs: int = 2, dataset_size: str = '10k'):
        """Test interaction between two factors."""
        
        print(f"\n{'='*70}")
        print(f"INTERACTION ANALYSIS: {factor1} × {factor2}")
        print(f"{'='*70}")
        
        factor1_info = self.FACTORS[factor1]
        factor2_info = self.FACTORS[factor2]
        
        # Test all combinations
        interaction_results = []
        
        for v1 in factor1_info['values'][:2]:  # Limit to first 2 values for speed
            for v2 in factor2_info['values'][:2]:
                print(f"\nTesting {factor1}={v1}, {factor2}={v2}")
                
                combo_results = []
                for run_idx in range(n_runs):
                    config = {
                        'experiment_id': f'{factor1}_{v1}_{factor2}_{v2}_run{run_idx+1}',
                        'pipeline': pipeline,
                        'dataset': 'wikipedia',
                        'dataset_size': dataset_size,
                        'n_bits': 256,
                        'batch_size': 128,
                        'seed': 42 + run_idx,
                        factor1: v1,
                        factor2: v2
                    }
                    
                    result = self.runner.run_single_experiment_safe(config)
                    
                    if result['status'] == 'success':
                        m = result.get('metrics', {})
                        combo_results.append({
                            'factor1': factor1, 'value1': v1,
                            'factor2': factor2, 'value2': v2,
                            'speed': m.get('docs_per_second', 0),
                            'ndcg': m.get('ndcg_at_10', 0),
                            'precision_1': m.get('precision_at_1', 0),
                            'success_1': m.get('success_at_1', 0)
                        })
                
                if combo_results:
                    stats = self._calculate_statistics(combo_results)
                    interaction_results.append({
                        'combination': f'{factor1}={v1}, {factor2}={v2}',
                        'stats': stats
                    })
        
        # Analyze interaction effects
        if interaction_results:
            self._analyze_interaction(factor1, factor2, interaction_results)
        
        return interaction_results
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics with confidence intervals."""
        
        metrics = ['speed', 'latency_p50', 'ndcg_10', 'precision_1', 
                  'recall_10', 'success_1', 'mrr', 'memory_mb']
        
        stats = {}
        for metric in metrics:
            if metric in results[0]:
                values = [r[metric] for r in results if metric in r]
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
    
    def _analyze_factor_impact(self, factor_name: str, results: List[Dict]):
        """Analyze and report factor impact."""
        
        print(f"\n{'='*50}")
        print(f"FACTOR IMPACT ANALYSIS: {factor_name}")
        print(f"{'='*50}")
        
        # Create comparison table
        print("\n| Value | Speed (d/s) | NDCG@10 | P@1 | Success@1 | Memory |")
        print("|-------|-------------|---------|-----|-----------|--------|")
        
        baseline = results[0]['stats'] if results else None
        
        for result in results:
            value = result['value']
            stats = result['stats']
            
            speed = stats.get('speed', {}).get('median', 0)
            ndcg = stats.get('ndcg_10', {}).get('median', 0) 
            p1 = stats.get('precision_1', {}).get('median', 0)
            s1 = stats.get('success_1', {}).get('median', 0)
            mem = stats.get('memory_mb', {}).get('median', 0)
            
            # Calculate relative change from baseline
            if baseline and result != results[0]:
                speed_change = ((speed - baseline['speed']['median']) / 
                               baseline['speed']['median'] * 100) if baseline['speed']['median'] > 0 else 0
                ndcg_change = ((ndcg - baseline['ndcg_10']['median']) / 
                              baseline['ndcg_10']['median'] * 100) if baseline['ndcg_10']['median'] > 0 else 0
                
                print(f"| {value:<5} | {speed:>11.0f} ({speed_change:+.1f}%) | "
                      f"{ndcg:>7.4f} ({ndcg_change:+.1f}%) | {p1:>3.2f} | {s1:>9.2f} | {mem:>6.0f} |")
            else:
                print(f"| {value:<5} | {speed:>11.0f} | {ndcg:>7.4f} | {p1:>3.2f} | {s1:>9.2f} | {mem:>6.0f} |")
        
        # Statistical significance test
        if len(results) == 2:
            # Binary factor - test significance
            values1 = [r['ndcg_10'] for r in self.results if r['factor'] == factor_name and r['value'] == results[0]['value']]
            values2 = [r['ndcg_10'] for r in self.results if r['factor'] == factor_name and r['value'] == results[1]['value']]
            
            if values1 and values2:
                p_value, is_significant = statistical_significance_test(values1, values2)
                print(f"\nStatistical significance (NDCG): p-value = {p_value:.4f}")
                if is_significant:
                    print("✓ Difference is statistically significant (p < 0.05)")
                else:
                    print("✗ Difference is NOT statistically significant")
    
    def _analyze_interaction(self, factor1: str, factor2: str, results: List[Dict]):
        """Analyze interaction effects between factors."""
        
        print(f"\n{'='*50}")
        print(f"INTERACTION EFFECTS")
        print(f"{'='*50}")
        
        for result in results:
            combo = result['combination']
            stats = result['stats']
            
            speed = stats.get('speed', {}).get('median', 0)
            ndcg = stats.get('ndcg', {}).get('median', 0)
            
            print(f"{combo}: Speed={speed:.0f} d/s, NDCG={ndcg:.4f}")
    
    def save_results(self):
        """Save factor analysis results."""
        
        # Save as JSON
        json_path = self.output_dir / f"factor_analysis_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_path = self.output_dir / f"factor_analysis_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to {self.output_dir}")
    
    def run_complete_analysis(self, factors: List[str] = None, pipelines: List[str] = None):
        """Run complete factor analysis for specified factors and pipelines."""
        
        factors = factors or ['n_bits', 'use_simd', 'batch_size']
        pipelines = pipelines or ['optimized_fused']
        
        print("="*70)
        print("COMPLETE FACTOR ANALYSIS WITH ENHANCED METRICS")
        print("="*70)
        print(f"Factors: {factors}")
        print(f"Pipelines: {pipelines}")
        
        for pipeline in pipelines:
            print(f"\n{'='*60}")
            print(f"PIPELINE: {pipeline.upper()}")
            print(f"{'='*60}")
            
            for factor in factors:
                # Check applicability
                factor_info = self.FACTORS.get(factor, {})
                if 'applicable_to' in factor_info:
                    if pipeline not in factor_info['applicable_to']:
                        print(f"\nSkipping {factor} (not applicable to {pipeline})")
                        continue
                
                self.test_single_factor(factor, pipeline=pipeline, n_runs=3)
        
        self.save_results()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced factor analysis for TEJAS')
    parser.add_argument('--factors', nargs='+', 
                       choices=list(EnhancedFactorAnalysis.FACTORS.keys()),
                       help='Factors to analyze')
    parser.add_argument('--pipelines', nargs='+',
                       choices=['original_tejas', 'fused_char', 'fused_byte', 'optimized_fused'],
                       default=['optimized_fused'],
                       help='Pipelines to test')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per configuration')
    parser.add_argument('--size', default='10k',
                       help='Dataset size for testing')
    parser.add_argument('--output', default='benchmark_results/factor_analysis_enhanced',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = EnhancedFactorAnalysis(output_dir=args.output)
    analyzer.run_complete_analysis(factors=args.factors, pipelines=args.pipelines)


if __name__ == '__main__':
    main()