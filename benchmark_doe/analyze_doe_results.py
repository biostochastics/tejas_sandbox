#!/usr/bin/env python3
"""
Analyze DOE benchmark results with comprehensive statistical analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.doe_analysis import DOEAnalyzer
from core.utils import safe_divide

def load_latest_results():
    """Load the most recent benchmark results."""
    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        raise FileNotFoundError("No benchmark_results directory found")
    
    # Find latest results file
    result_files = list(results_dir.glob("doe_results_*.json"))
    if not result_files:
        raise FileNotFoundError("No results files found")
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file) as f:
        return json.load(f)

def prepare_dataframe(results):
    """Convert results to DataFrame for analysis."""
    rows = []
    # Handle both dict with 'results' key and direct list formats
    if isinstance(results, dict) and 'results' in results:
        results_list = results['results']
    else:
        results_list = results
    
    for result in results_list:
        if result['status'] == 'success':
            row = {
                'experiment_id': result['experiment_id'],
                'pipeline': result['config']['pipeline_architecture'],
                'n_bits': result['config']['n_bits'],
                'batch_size': result['config']['batch_size'],
                'throughput': result['metrics']['docs_per_second'],
                'latency': result['metrics']['avg_query_latency_ms'],
                'total_time': result['metrics']['total_time'],
                'index_size_mb': result['metrics'].get('index_size_mb', 0)
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add derived metrics
    df['efficiency'] = safe_divide(df['throughput'], df['index_size_mb'], default=0)
    df['speed_score'] = safe_divide(1000, df['latency'], default=0)  # Inverse of latency
    
    return df

def analyze_main_effects(df, csv_dir=None):
    """Perform main effects analysis.
    
    Args:
        df: Results DataFrame
        csv_dir: Directory for CSV exports (None = no export)
    """
    print("\n" + "="*60)
    print("MAIN EFFECTS ANALYSIS")
    print("="*60)
    
    factors = ['pipeline', 'n_bits', 'batch_size']
    responses = ['throughput', 'latency', 'efficiency']
    
    analyzer = DOEAnalyzer(df, factors, responses)
    
    all_effects = []
    for response in responses:
        print(f"\n📊 Main Effects on {response.upper()}:")
        effects = analyzer.compute_main_effects(response)
        if not effects.empty:
            effects['response'] = response
            all_effects.append(effects)
            print(effects[['factor', 'effect', 'p_value']].to_string(index=False))
    
    # Export to CSV if directory provided
    if csv_dir and all_effects:
        effects_df = pd.concat(all_effects, ignore_index=True)
        effects_csv = csv_dir / f"main_effects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        effects_df.to_csv(effects_csv, index=False, encoding='utf-8')
        print(f"\n📁 Main effects exported to: {effects_csv}")

def analyze_interactions(df):
    """Analyze factor interactions."""
    print("\n" + "="*60)
    print("INTERACTION ANALYSIS")
    print("="*60)
    
    # Focus on numeric factors for interactions
    numeric_factors = ['n_bits', 'batch_size']
    
    # Convert to numeric for interaction analysis
    df_numeric = df.copy()
    
    analyzer = DOEAnalyzer(df_numeric, numeric_factors, ['throughput', 'latency'])
    
    for response in ['throughput', 'latency']:
        print(f"\n🔄 Interactions affecting {response.upper()}:")
        interactions = analyzer.compute_interactions(response, significance_level=0.1)
        if not interactions.empty:
            print(interactions[['interaction', 'effect', 'p_value', 'significant']].to_string(index=False))

def find_optimal_configs(df):
    """Find optimal configurations for different objectives."""
    print("\n" + "="*60)
    print("OPTIMAL CONFIGURATIONS")
    print("="*60)
    
    # Maximum throughput
    max_throughput = df.loc[df['throughput'].idxmax()]
    print(f"\n🚀 Maximum Throughput: {max_throughput['throughput']:.1f} docs/sec")
    print(f"   Pipeline: {max_throughput['pipeline']}")
    print(f"   n_bits: {max_throughput['n_bits']}, batch_size: {max_throughput['batch_size']}")
    
    # Minimum latency
    min_latency = df.loc[df['latency'].idxmin()]
    print(f"\n⚡ Minimum Latency: {min_latency['latency']:.2f} ms")
    print(f"   Pipeline: {min_latency['pipeline']}")
    print(f"   n_bits: {min_latency['n_bits']}, batch_size: {min_latency['batch_size']}")
    
    # Best efficiency (throughput per MB)
    if df['efficiency'].max() > 0:
        max_efficiency = df.loc[df['efficiency'].idxmax()]
        print(f"\n💡 Best Efficiency: {max_efficiency['efficiency']:.1f} docs/sec/MB")
        print(f"   Pipeline: {max_efficiency['pipeline']}")
        print(f"   n_bits: {max_efficiency['n_bits']}, batch_size: {max_efficiency['batch_size']}")
    
    # Balanced configuration (Pareto optimal)
    # Normalize metrics
    df['norm_throughput'] = df['throughput'] / df['throughput'].max()
    df['norm_latency'] = 1 - (df['latency'] / df['latency'].max())  # Inverted for minimization
    df['balance_score'] = (df['norm_throughput'] + df['norm_latency']) / 2
    
    balanced = df.loc[df['balance_score'].idxmax()]
    print(f"\n⚖️ Best Balanced (Throughput + Latency):")
    print(f"   Pipeline: {balanced['pipeline']}")
    print(f"   n_bits: {balanced['n_bits']}, batch_size: {balanced['batch_size']}")
    print(f"   Throughput: {balanced['throughput']:.1f} docs/sec")
    print(f"   Latency: {balanced['latency']:.2f} ms")
    print(f"   Balance Score: {balanced['balance_score']:.3f}")

def analyze_by_pipeline(df):
    """Detailed analysis by pipeline."""
    print("\n" + "="*60)
    print("PIPELINE COMPARISON")
    print("="*60)
    
    pipelines = df['pipeline'].unique()
    
    comparison = []
    for pipeline in pipelines:
        pipeline_data = df[df['pipeline'] == pipeline]
        stats = {
            'Pipeline': pipeline,
            'Experiments': len(pipeline_data),
            'Avg Throughput': pipeline_data['throughput'].mean(),
            'Std Throughput': pipeline_data['throughput'].std(),
            'Avg Latency': pipeline_data['latency'].mean(),
            'Std Latency': pipeline_data['latency'].std(),
            'Min Memory': pipeline_data['index_size_mb'].min(),
            'Max Memory': pipeline_data['index_size_mb'].max()
        }
        comparison.append(stats)
    
    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False, float_format='%.2f'))

def analyze_scaling(df):
    """Analyze how performance scales with factors."""
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)
    
    # Throughput scaling with batch size
    print("\n📈 Throughput Scaling with Batch Size:")
    for pipeline in df['pipeline'].unique():
        pipeline_data = df[df['pipeline'] == pipeline]
        batch_scaling = pipeline_data.groupby('batch_size')['throughput'].mean()
        print(f"\n  {pipeline}:")
        for batch, throughput in batch_scaling.items():
            print(f"    Batch {batch}: {throughput:.1f} docs/sec")
    
    # Latency scaling with n_bits
    print("\n📊 Latency Scaling with n_bits:")
    for pipeline in df['pipeline'].unique():
        pipeline_data = df[df['pipeline'] == pipeline]
        bits_scaling = pipeline_data.groupby('n_bits')['latency'].mean()
        print(f"\n  {pipeline}:")
        for bits, latency in bits_scaling.items():
            print(f"    {bits} bits: {latency:.2f} ms")

def generate_recommendations(df):
    """Generate recommendations based on analysis."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Best overall pipeline
    pipeline_scores = df.groupby('pipeline').agg({
        'throughput': 'mean',
        'latency': 'mean',
        'efficiency': 'mean'
    })
    
    # Normalize and combine scores
    for col in pipeline_scores.columns:
        if col == 'latency':  # Lower is better
            pipeline_scores[f'{col}_score'] = 1 / pipeline_scores[col]
        else:  # Higher is better
            pipeline_scores[f'{col}_score'] = pipeline_scores[col]
    
    score_cols = [col for col in pipeline_scores.columns if '_score' in col]
    pipeline_scores['overall_score'] = pipeline_scores[score_cols].mean(axis=1)
    
    best_pipeline = pipeline_scores['overall_score'].idxmax()
    
    print(f"\n🏆 Recommended Pipeline: {best_pipeline}")
    print(f"   Best overall balance of throughput, latency, and efficiency")
    
    # Specific recommendations
    print("\n📋 Use Case Recommendations:")
    
    # High throughput requirement
    high_throughput = df.nlargest(3, 'throughput')
    print("\n  For Maximum Throughput:")
    for _, row in high_throughput.iterrows():
        print(f"    • {row['pipeline']} with batch_size={row['batch_size']}, n_bits={row['n_bits']}")
    
    # Low latency requirement
    low_latency = df.nsmallest(3, 'latency')
    print("\n  For Minimum Latency:")
    for _, row in low_latency.iterrows():
        print(f"    • {row['pipeline']} with batch_size={row['batch_size']}, n_bits={row['n_bits']}")
    
    # Memory constrained
    if df['index_size_mb'].max() > 0:
        memory_efficient = df[df['efficiency'] > df['efficiency'].quantile(0.75)]
        memory_efficient = memory_efficient.nsmallest(3, 'index_size_mb')
        print("\n  For Memory-Constrained Environments:")
        for _, row in memory_efficient.iterrows():
            print(f"    • {row['pipeline']} with n_bits={row['n_bits']} ({row['index_size_mb']:.1f} MB)")

def main(export_csv=False):
    """Run comprehensive DOE analysis.
    
    Args:
        export_csv: If True, export all DataFrames to CSV files
    """
    print("\n" + "="*60)
    print("DOE BENCHMARK ANALYSIS REPORT")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load results
    results = load_latest_results()
    df = prepare_dataframe(results)
    
    # Create CSV export directory if needed
    csv_dir = None
    if export_csv:
        csv_dir = Path("benchmark_results/csv_exports")
        csv_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export main results DataFrame
        main_csv = csv_dir / f"doe_results_{timestamp}.csv"
        df.to_csv(main_csv, index=False, encoding='utf-8')
        print(f"📁 Main results exported to: {main_csv}")
    
    print(f"\n📊 Analyzed {len(df)} successful experiments")
    print(f"   Pipelines: {', '.join(df['pipeline'].unique())}")
    print(f"   n_bits: {sorted(df['n_bits'].unique())}")
    print(f"   batch_sizes: {sorted(df['batch_size'].unique())}")
    
    # Run analyses (pass csv_dir for exports if enabled)
    analyze_by_pipeline(df)
    analyze_main_effects(df, csv_dir=csv_dir)
    analyze_interactions(df)
    analyze_scaling(df)
    find_optimal_configs(df)
    generate_recommendations(df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze DOE Benchmark Results")
    parser.add_argument('--export-csv', action='store_true',
                        help='Export DataFrames to CSV files')
    args = parser.parse_args()
    
    main(export_csv=args.export_csv)