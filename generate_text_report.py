#!/usr/bin/env python3
"""
Generate comprehensive text-based benchmark report from pipeline comparison data.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_benchmark_data():
    """Load the pipeline comparison benchmark data."""
    json_path = Path("benchmark_results/comprehensive_20250827_001131/pipeline_comparison/pipeline_comparison_20250827_001132.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data['results']

def calculate_statistics(values):
    """Calculate mean, std, min, max for a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }

def analyze_pipeline_performance(results):
    """Analyze performance by pipeline and dataset."""
    analysis = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for result in results:
        pipeline = result['pipeline']
        dataset = result['dataset']
        
        # Collect metrics
        analysis[dataset][pipeline]['speed'].append(result['speed'])
        analysis[dataset][pipeline]['ndcg_at_10'].append(result['ndcg_at_10'])
        analysis[dataset][pipeline]['precision_at_1'].append(result['precision_at_1'])
        analysis[dataset][pipeline]['memory_mb'].append(result['memory_mb'])
        analysis[dataset][pipeline]['latency_p50'].append(result['latency_p50'])
        analysis[dataset][pipeline]['latency_p95'].append(result['latency_p95'])
    
    return analysis

def generate_report(results):
    """Generate comprehensive text report."""
    analysis = analyze_pipeline_performance(results)
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE PIPELINE BENCHMARK REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-"*40)
    report.append("Benchmarked 4 pipeline implementations across 3 datasets:")
    report.append("• original_tejas: Golden ratio + sklearn SVD (mean energy threshold)")
    report.append("• fused_char: Character-level fused encoder (95% cumulative energy)")
    report.append("• fused_byte: Byte-level fused encoder (95% cumulative energy)")
    report.append("• optimized_fused: Optimized fused encoder with caching")
    report.append("")
    
    # Per-dataset analysis
    for dataset in ['wikipedia', 'openwebtext', 'arxiv']:
        if dataset not in analysis:
            continue
            
        report.append("="*80)
        report.append(f"{dataset.upper()} DATASET RESULTS")
        report.append("="*80)
        report.append("")
        
        # Speed comparison
        report.append("PROCESSING SPEED (documents/second)")
        report.append("-"*40)
        
        speed_rankings = []
        for pipeline in analysis[dataset]:
            stats = calculate_statistics(analysis[dataset][pipeline]['speed'])
            speed_rankings.append((pipeline, stats['mean']))
            report.append(f"{pipeline:20} {stats['mean']:8.0f} ± {stats['std']:6.0f} (min: {stats['min']:.0f}, max: {stats['max']:.0f})")
        
        # Sort and show rankings
        speed_rankings.sort(key=lambda x: x[1], reverse=True)
        report.append("")
        report.append("Speed Ranking:")
        for rank, (pipeline, speed) in enumerate(speed_rankings, 1):
            if rank == 1:
                baseline = speed
                report.append(f"  {rank}. {pipeline}: {speed:.0f} docs/sec (baseline)")
            else:
                improvement = ((speed - baseline) / baseline) * 100
                report.append(f"  {rank}. {pipeline}: {speed:.0f} docs/sec ({improvement:+.1f}% vs baseline)")
        report.append("")
        
        # Accuracy comparison
        report.append("ACCURACY METRICS")
        report.append("-"*40)
        
        # NDCG@10
        report.append("NDCG@10:")
        ndcg_rankings = []
        for pipeline in analysis[dataset]:
            stats = calculate_statistics(analysis[dataset][pipeline]['ndcg_at_10'])
            ndcg_rankings.append((pipeline, stats['mean']))
            report.append(f"  {pipeline:20} {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        ndcg_rankings.sort(key=lambda x: x[1], reverse=True)
        report.append("")
        
        # Precision@1
        report.append("Precision@1:")
        p1_rankings = []
        for pipeline in analysis[dataset]:
            stats = calculate_statistics(analysis[dataset][pipeline]['precision_at_1'])
            p1_rankings.append((pipeline, stats['mean']))
            report.append(f"  {pipeline:20} {stats['mean']:.3f} ± {stats['std']:.3f}")
        report.append("")
        
        # Resource usage
        report.append("RESOURCE USAGE")
        report.append("-"*40)
        
        # Memory
        report.append("Memory (MB):")
        memory_rankings = []
        for pipeline in analysis[dataset]:
            stats = calculate_statistics(analysis[dataset][pipeline]['memory_mb'])
            memory_rankings.append((pipeline, stats['mean']))
            report.append(f"  {pipeline:20} {stats['mean']:6.0f} ± {stats['std']:5.0f}")
        
        memory_rankings.sort(key=lambda x: x[1])  # Lower is better
        report.append("")
        
        # Latency
        report.append("Latency P50/P95 (ms):")
        for pipeline in analysis[dataset]:
            p50_stats = calculate_statistics(analysis[dataset][pipeline]['latency_p50'])
            p95_stats = calculate_statistics(analysis[dataset][pipeline]['latency_p95'])
            report.append(f"  {pipeline:20} P50: {p50_stats['mean']:5.1f}ms, P95: {p95_stats['mean']:5.1f}ms")
        report.append("")
        
        # Trade-off analysis
        report.append("TRADE-OFF ANALYSIS")
        report.append("-"*40)
        
        efficiency_scores = []
        for pipeline in analysis[dataset]:
            speed_mean = calculate_statistics(analysis[dataset][pipeline]['speed'])['mean']
            ndcg_mean = calculate_statistics(analysis[dataset][pipeline]['ndcg_at_10'])['mean']
            efficiency = speed_mean * ndcg_mean
            efficiency_scores.append((pipeline, speed_mean, ndcg_mean, efficiency))
        
        efficiency_scores.sort(key=lambda x: x[3], reverse=True)
        
        report.append("Efficiency Score (Speed × NDCG):")
        for rank, (pipeline, speed, ndcg, score) in enumerate(efficiency_scores, 1):
            report.append(f"  {rank}. {pipeline:20} Score: {score:8.0f} (Speed: {speed:.0f}, NDCG: {ndcg:.4f})")
        report.append("")
    
    # Overall insights
    report.append("="*80)
    report.append("KEY INSIGHTS")
    report.append("="*80)
    report.append("")
    
    report.append("1. ENERGY THRESHOLD IMPACT:")
    report.append("   • original_tejas: Mean energy threshold → ~64 components → Faster")
    report.append("   • Fused encoders: 95% cumulative energy → ~256 components → More accurate")
    report.append("")
    
    report.append("2. PERFORMANCE PATTERNS:")
    
    # Calculate overall averages across all datasets
    overall_speed = defaultdict(list)
    overall_ndcg = defaultdict(list)
    
    for dataset in analysis:
        for pipeline in analysis[dataset]:
            overall_speed[pipeline].extend(analysis[dataset][pipeline]['speed'])
            overall_ndcg[pipeline].extend(analysis[dataset][pipeline]['ndcg_at_10'])
    
    report.append("   Overall Average Performance:")
    for pipeline in overall_speed:
        speed_mean = np.mean(overall_speed[pipeline])
        ndcg_mean = np.mean(overall_ndcg[pipeline])
        report.append(f"   • {pipeline}: {speed_mean:.0f} docs/sec, NDCG: {ndcg_mean:.4f}")
    report.append("")
    
    report.append("3. RECOMMENDATIONS:")
    report.append("   • Maximum Speed: Use original_tejas")
    report.append("   • Balanced Performance: Use optimized_fused")
    report.append("   • Maximum Accuracy: Use fused_char or fused_byte")
    report.append("   • Production Use: optimized_fused offers best efficiency")
    report.append("")
    
    report.append("4. DATASET-SPECIFIC OBSERVATIONS:")
    
    # Find best performer per dataset
    for dataset in analysis:
        speed_winner = max(analysis[dataset].keys(), 
                          key=lambda p: np.mean(analysis[dataset][p]['speed']))
        ndcg_winner = max(analysis[dataset].keys(),
                         key=lambda p: np.mean(analysis[dataset][p]['ndcg_at_10']))
        report.append(f"   • {dataset.capitalize()}: Fastest={speed_winner}, Most Accurate={ndcg_winner}")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    return "\n".join(report)

def save_csv_summary(results):
    """Save summary statistics as CSV."""
    analysis = analyze_pipeline_performance(results)
    
    csv_lines = []
    csv_lines.append("Dataset,Pipeline,Speed_Mean,Speed_Std,NDCG_Mean,NDCG_Std,P@1_Mean,Memory_Mean,Latency_P50,Latency_P95")
    
    for dataset in analysis:
        for pipeline in analysis[dataset]:
            speed_stats = calculate_statistics(analysis[dataset][pipeline]['speed'])
            ndcg_stats = calculate_statistics(analysis[dataset][pipeline]['ndcg_at_10'])
            p1_stats = calculate_statistics(analysis[dataset][pipeline]['precision_at_1'])
            mem_stats = calculate_statistics(analysis[dataset][pipeline]['memory_mb'])
            p50_stats = calculate_statistics(analysis[dataset][pipeline]['latency_p50'])
            p95_stats = calculate_statistics(analysis[dataset][pipeline]['latency_p95'])
            
            csv_lines.append(f"{dataset},{pipeline},{speed_stats['mean']:.1f},{speed_stats['std']:.1f},"
                           f"{ndcg_stats['mean']:.4f},{ndcg_stats['std']:.4f},{p1_stats['mean']:.3f},"
                           f"{mem_stats['mean']:.0f},{p50_stats['mean']:.2f},{p95_stats['mean']:.2f}")
    
    return "\n".join(csv_lines)

def main():
    """Generate complete benchmark report."""
    print("Loading benchmark data...")
    results = load_benchmark_data()
    
    print(f"Loaded {len(results)} experiment results")
    
    # Create output directory
    output_dir = Path("benchmark_results/comprehensive_20250827_001131/report")
    output_dir.mkdir(exist_ok=True)
    
    # Generate text report
    print("Generating comprehensive text report...")
    report = generate_report(results)
    
    # Save report
    report_path = output_dir / "benchmark_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save CSV summary
    print("Generating CSV summary...")
    csv_summary = save_csv_summary(results)
    csv_path = output_dir / "summary_statistics.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_summary)
    
    # Print report to console
    print("\n" + report)
    
    print(f"\nFiles saved to: {output_dir}")
    print(f"  - benchmark_report.txt")
    print(f"  - summary_statistics.csv")

if __name__ == "__main__":
    main()