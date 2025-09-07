#!/usr/bin/env python3
"""
Generate comprehensive benchmark report with figures from pipeline comparison data.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_benchmark_data():
    """Load the pipeline comparison benchmark data."""
    json_path = Path("benchmark_results/comprehensive_20250827_001131/pipeline_comparison/pipeline_comparison_20250827_001132.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data['results'])

def generate_performance_summary(df):
    """Generate performance summary statistics."""
    summary = df.groupby(['pipeline', 'dataset']).agg({
        'speed': ['mean', 'std'],
        'ndcg_at_10': ['mean', 'std'],
        'precision_at_1': ['mean', 'std'],
        'memory_mb': ['mean', 'std'],
        'latency_p50': ['mean', 'std'],
        'latency_p95': ['mean', 'std']
    }).round(2)
    
    return summary

def create_speed_comparison_figure(df):
    """Create speed comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = ['wikipedia', 'openwebtext', 'arxiv']
    
    for idx, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset]
        
        # Calculate mean and std for each pipeline
        stats = data.groupby('pipeline').agg({
            'speed': ['mean', 'std']
        }).round(0)
        
        pipelines = stats.index
        means = stats['speed']['mean'].values
        stds = stats['speed']['std'].values
        
        # Create bar plot
        bars = axes[idx].bar(range(len(pipelines)), means, yerr=stds, capsize=5)
        axes[idx].set_xticks(range(len(pipelines)))
        axes[idx].set_xticklabels(pipelines, rotation=45, ha='right')
        axes[idx].set_ylabel('Documents per Second')
        axes[idx].set_title(f'{dataset.capitalize()} Dataset')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i],
                          f'{mean:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Processing Speed Comparison Across Pipelines', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_accuracy_comparison_figure(df):
    """Create accuracy metrics comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    datasets = ['wikipedia', 'openwebtext', 'arxiv']
    metrics = [
        ('ndcg_at_10', 'NDCG@10'),
        ('precision_at_1', 'Precision@1')
    ]
    
    for row, (metric, metric_name) in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            data = df[df['dataset'] == dataset]
            
            # Create box plot
            pipeline_data = []
            pipeline_labels = []
            
            for pipeline in data['pipeline'].unique():
                pipeline_values = data[data['pipeline'] == pipeline][metric].values
                pipeline_data.append(pipeline_values)
                pipeline_labels.append(pipeline)
            
            bp = axes[row, col].boxplot(pipeline_data, labels=pipeline_labels,
                                        patch_artist=True, showmeans=True)
            
            # Color boxes
            colors = sns.color_palette("husl", len(pipeline_labels))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[row, col].set_xticklabels(pipeline_labels, rotation=45, ha='right')
            axes[row, col].set_ylabel(metric_name)
            axes[row, col].set_title(f'{dataset.capitalize()} - {metric_name}')
            axes[row, col].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Accuracy Metrics Comparison Across Pipelines', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_speed_vs_accuracy_scatter(df):
    """Create speed vs accuracy scatter plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = ['wikipedia', 'openwebtext', 'arxiv']
    
    for idx, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset]
        
        # Get unique pipelines
        pipelines = data['pipeline'].unique()
        colors = sns.color_palette("husl", len(pipelines))
        
        for pipeline, color in zip(pipelines, colors):
            pipeline_data = data[data['pipeline'] == pipeline]
            axes[idx].scatter(pipeline_data['speed'], 
                            pipeline_data['ndcg_at_10'],
                            label=pipeline, s=100, alpha=0.7, color=color)
            
            # Add mean point with larger marker
            mean_speed = pipeline_data['speed'].mean()
            mean_ndcg = pipeline_data['ndcg_at_10'].mean()
            axes[idx].scatter(mean_speed, mean_ndcg, 
                            s=200, color=color, edgecolors='black', 
                            linewidth=2, marker='D')
        
        axes[idx].set_xlabel('Processing Speed (docs/sec)')
        axes[idx].set_ylabel('NDCG@10')
        axes[idx].set_title(f'{dataset.capitalize()} Dataset')
        axes[idx].legend(loc='best', fontsize=9)
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Speed vs Accuracy Trade-off (diamonds show means)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_memory_comparison_figure(df):
    """Create memory usage comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for grouped bar plot
    datasets = df['dataset'].unique()
    pipelines = df['pipeline'].unique()
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, pipeline in enumerate(pipelines):
        means = []
        stds = []
        for dataset in datasets:
            data = df[(df['dataset'] == dataset) & (df['pipeline'] == pipeline)]
            means.append(data['memory_mb'].mean())
            stds.append(data['memory_mb'].std())
        
        offset = width * (i - len(pipelines)/2 + 0.5)
        bars = ax.bar(x + offset, means, width, label=pipeline, yerr=stds, capsize=3)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Comparison Across Pipelines and Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_latency_comparison_figure(df):
    """Create latency comparison violin plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for violin plots
    metrics = [('latency_p50', 'P50 Latency (ms)'), ('latency_p95', 'P95 Latency (ms)')]
    
    for idx, (metric, title) in enumerate(metrics):
        # Combine all datasets for overall comparison
        plot_data = []
        labels = []
        
        for pipeline in df['pipeline'].unique():
            pipeline_data = df[df['pipeline'] == pipeline][metric].values
            plot_data.append(pipeline_data)
            labels.append(pipeline)
        
        parts = axes[idx].violinplot(plot_data, positions=range(len(labels)),
                                     showmeans=True, showmedians=True)
        
        # Customize colors
        colors = sns.color_palette("husl", len(labels))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        axes[idx].set_xticks(range(len(labels)))
        axes[idx].set_xticklabels(labels, rotation=45, ha='right')
        axes[idx].set_ylabel(title)
        axes[idx].set_title(f'{title} Distribution Across All Datasets')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Latency Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def generate_markdown_report(df, summary):
    """Generate markdown report with key findings."""
    
    report = []
    report.append("# Comprehensive Pipeline Benchmark Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## Executive Summary\n")
    report.append("This report presents comprehensive benchmark results comparing four pipeline implementations:")
    report.append("1. **original_tejas**: Golden ratio subsampling + sklearn SVD (mean energy threshold)")
    report.append("2. **fused_char**: Character-level fused encoder (95% cumulative energy)")
    report.append("3. **fused_byte**: Byte-level fused encoder (95% cumulative energy)")
    report.append("4. **optimized_fused**: Optimized fused encoder with caching (95% cumulative energy)\n")
    
    report.append("## Key Findings\n")
    
    # Calculate overall winners
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        
        report.append(f"\n### {dataset.capitalize()} Dataset")
        
        # Speed winner
        speed_stats = data.groupby('pipeline')['speed'].mean().sort_values(ascending=False)
        report.append(f"- **Fastest Pipeline**: {speed_stats.index[0]} ({speed_stats.values[0]:.0f} docs/sec)")
        
        # Accuracy winner
        ndcg_stats = data.groupby('pipeline')['ndcg_at_10'].mean().sort_values(ascending=False)
        report.append(f"- **Most Accurate**: {ndcg_stats.index[0]} (NDCG@10: {ndcg_stats.values[0]:.4f})")
        
        # Memory efficiency
        memory_stats = data.groupby('pipeline')['memory_mb'].mean().sort_values()
        report.append(f"- **Most Memory Efficient**: {memory_stats.index[0]} ({memory_stats.values[0]:.0f} MB)")
    
    report.append("\n## Performance Summary Table\n")
    report.append("### Processing Speed (docs/sec) - Mean ± Std")
    report.append("| Pipeline | Wikipedia | OpenWebText | ArXiv |")
    report.append("|----------|-----------|-------------|--------|")
    
    for pipeline in df['pipeline'].unique():
        row = f"| {pipeline} |"
        for dataset in ['wikipedia', 'openwebtext', 'arxiv']:
            data = df[(df['pipeline'] == pipeline) & (df['dataset'] == dataset)]
            if not data.empty:
                mean = data['speed'].mean()
                std = data['speed'].std()
                row += f" {mean:.0f} ± {std:.0f} |"
            else:
                row += " N/A |"
        report.append(row)
    
    report.append("\n### NDCG@10 - Mean ± Std")
    report.append("| Pipeline | Wikipedia | OpenWebText | ArXiv |")
    report.append("|----------|-----------|-------------|--------|")
    
    for pipeline in df['pipeline'].unique():
        row = f"| {pipeline} |"
        for dataset in ['wikipedia', 'openwebtext', 'arxiv']:
            data = df[(df['pipeline'] == pipeline) & (df['dataset'] == dataset)]
            if not data.empty:
                mean = data['ndcg_at_10'].mean()
                std = data['ndcg_at_10'].std()
                row += f" {mean:.4f} ± {std:.4f} |"
            else:
                row += " N/A |"
        report.append(row)
    
    report.append("\n## Trade-off Analysis\n")
    report.append("### Speed vs Accuracy Trade-off")
    
    # Calculate efficiency scores (speed * accuracy)
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        report.append(f"\n**{dataset.capitalize()}:**")
        
        efficiency = []
        for pipeline in data['pipeline'].unique():
            pipeline_data = data[data['pipeline'] == pipeline]
            speed = pipeline_data['speed'].mean()
            ndcg = pipeline_data['ndcg_at_10'].mean()
            eff_score = speed * ndcg
            efficiency.append((pipeline, speed, ndcg, eff_score))
        
        efficiency.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (pipeline, speed, ndcg, score) in enumerate(efficiency, 1):
            report.append(f"{rank}. {pipeline}: Efficiency Score = {score:.0f} (Speed: {speed:.0f}, NDCG: {ndcg:.4f})")
    
    report.append("\n## Energy Threshold Impact\n")
    report.append("The key difference between original_tejas and fused encoders:")
    report.append("- **original_tejas**: Uses mean energy threshold → ~64 components")
    report.append("- **fused encoders**: Use 95% cumulative energy → ~256 components")
    report.append("\nThis explains the speed/accuracy trade-off observed in results.")
    
    report.append("\n## Recommendations\n")
    report.append("1. **For Maximum Speed**: Use original_tejas (mean energy threshold)")
    report.append("2. **For Balanced Performance**: Use optimized_fused (good speed with better accuracy)")
    report.append("3. **For Maximum Accuracy**: Use fused_char or fused_byte (95% cumulative energy)")
    report.append("4. **For Production**: Consider optimized_fused for best overall efficiency")
    
    return "\n".join(report)

def main():
    """Generate complete benchmark report."""
    print("Loading benchmark data...")
    df = load_benchmark_data()
    
    print("Generating performance summary...")
    summary = generate_performance_summary(df)
    
    # Create output directory
    output_dir = Path("benchmark_results/comprehensive_20250827_001131/report")
    output_dir.mkdir(exist_ok=True)
    
    # Generate figures
    print("Creating speed comparison figure...")
    fig1 = create_speed_comparison_figure(df)
    fig1.savefig(output_dir / "speed_comparison.png", dpi=300, bbox_inches='tight')
    
    print("Creating accuracy comparison figure...")
    fig2 = create_accuracy_comparison_figure(df)
    fig2.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
    
    print("Creating speed vs accuracy scatter plot...")
    fig3 = create_speed_vs_accuracy_scatter(df)
    fig3.savefig(output_dir / "speed_vs_accuracy.png", dpi=300, bbox_inches='tight')
    
    print("Creating memory usage comparison...")
    fig4 = create_memory_comparison_figure(df)
    fig4.savefig(output_dir / "memory_comparison.png", dpi=300, bbox_inches='tight')
    
    print("Creating latency comparison...")
    fig5 = create_latency_comparison_figure(df)
    fig5.savefig(output_dir / "latency_comparison.png", dpi=300, bbox_inches='tight')
    
    # Generate markdown report
    print("Generating markdown report...")
    report = generate_markdown_report(df, summary)
    
    report_path = output_dir / "benchmark_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save summary statistics
    summary.to_csv(output_dir / "summary_statistics.csv")
    
    print(f"\nReport generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - benchmark_report.md")
    print(f"  - speed_comparison.png")
    print(f"  - accuracy_comparison.png")
    print(f"  - speed_vs_accuracy.png")
    print(f"  - memory_comparison.png")
    print(f"  - latency_comparison.png")
    print(f"  - summary_statistics.csv")
    
    # Display key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS FROM PIPELINE COMPARISON")
    print("="*60)
    
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        print(f"\n{dataset.upper()} DATASET:")
        
        speed_stats = data.groupby('pipeline')['speed'].mean().sort_values(ascending=False)
        ndcg_stats = data.groupby('pipeline')['ndcg_at_10'].mean().sort_values(ascending=False)
        
        print(f"  Fastest: {speed_stats.index[0]} ({speed_stats.values[0]:.0f} docs/sec)")
        print(f"  Most Accurate: {ndcg_stats.index[0]} (NDCG@10: {ndcg_stats.values[0]:.4f})")
        
        # Speed improvement over baseline
        baseline_speed = speed_stats['original_tejas']
        for pipeline in speed_stats.index:
            if pipeline != 'original_tejas':
                improvement = ((speed_stats[pipeline] - baseline_speed) / baseline_speed) * 100
                print(f"  {pipeline}: {improvement:+.1f}% speed vs baseline")

if __name__ == "__main__":
    main()