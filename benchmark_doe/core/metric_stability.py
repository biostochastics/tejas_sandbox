#!/usr/bin/env python3
"""
Cross-Scale Metric Stability Validation Module

This module validates that IR metrics remain meaningful and discriminative
at different scales, particularly important when scaling from 10k to 1M+ documents.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    """Container for metric stability analysis results."""
    scale: int
    metric: str
    snr: float  # Signal-to-noise ratio
    between_variance: float
    within_variance: float
    coverage: float  # Percentage of queries with judgments
    avg_judgments_per_query: float
    discriminative: bool  # Whether metric is discriminative at this scale
    recommended_alternatives: List[str]  # Alternative metrics if not discriminative


class MetricStabilityValidator:
    """
    Validates metric stability and discriminative power across scales.
    
    Key concepts:
    - Signal-to-Noise Ratio (SNR): between-system variance / within-system variance
    - Coverage: percentage of queries with relevance judgments in top-k
    - Discriminative power: ability to distinguish between systems
    """
    
    def __init__(self, 
                 min_snr: float = 2.0,
                 min_coverage: float = 0.5,
                 min_queries: int = 50):
        """
        Initialize validator with thresholds.
        
        Args:
            min_snr: Minimum SNR to consider metric discriminative
            min_coverage: Minimum judgment coverage required
            min_queries: Minimum queries needed for reliable analysis
        """
        self.min_snr = min_snr
        self.min_coverage = min_coverage
        self.min_queries = min_queries
    
    def compute_snr(self, 
                   df: pd.DataFrame,
                   metric_col: str,
                   system_col: str = 'pipeline',
                   query_col: str = 'query_id') -> Tuple[float, float, float]:
        """
        Compute signal-to-noise ratio for a metric.
        
        SNR = between-system variance / within-system variance
        Higher SNR means better ability to distinguish systems.
        
        Returns:
            snr: Signal-to-noise ratio
            between_var: Between-system variance
            within_var: Within-system variance
        """
        # Compute mean performance per system
        system_means = df.groupby(system_col)[metric_col].mean()
        
        # Between-system variance (signal)
        grand_mean = df[metric_col].mean()
        between_var = np.var(system_means)
        
        # Within-system variance (noise) - average variance across queries
        within_vars = []
        for system in df[system_col].unique():
            system_df = df[df[system_col] == system]
            if len(system_df) > 1:
                # Variance across queries for this system
                query_scores = system_df.groupby(query_col)[metric_col].mean()
                within_vars.append(np.var(query_scores))
        
        within_var = np.mean(within_vars) if within_vars else 0.0
        
        # Compute SNR
        snr = between_var / (within_var + 1e-10)  # Add epsilon to avoid division by zero
        
        return snr, between_var, within_var
    
    def check_judgment_coverage(self,
                               df: pd.DataFrame,
                               k: int = 10,
                               relevant_col: str = 'num_relevant_at_k',
                               query_col: str = 'query_id') -> Tuple[float, float]:
        """
        Check coverage of relevance judgments.
        
        Returns:
            coverage: Fraction of queries with at least one relevant doc in top-k
            avg_judgments: Average number of relevant docs per query
        """
        query_stats = df.groupby(query_col)[relevant_col].agg(['sum', 'count'])
        
        # Coverage: queries with at least one relevant document
        queries_with_relevant = (query_stats['sum'] > 0).sum()
        total_queries = len(query_stats)
        coverage = queries_with_relevant / total_queries if total_queries > 0 else 0
        
        # Average judgments per query
        avg_judgments = query_stats['sum'].mean()
        
        return coverage, avg_judgments
    
    def validate_metric_at_scale(self,
                                df: pd.DataFrame,
                                scale: int,
                                metric: str,
                                system_col: str = 'pipeline') -> StabilityReport:
        """
        Validate a specific metric at a specific scale.
        
        Args:
            df: DataFrame with evaluation results
            scale: Data scale (number of documents)
            metric: Metric name (e.g., 'ndcg_at_10')
            system_col: Column identifying different systems
            
        Returns:
            StabilityReport with validation results
        """
        # Filter to specific scale
        scale_df = df[df['scale_n'] == scale] if 'scale_n' in df.columns else df
        
        if len(scale_df) < self.min_queries:
            logger.warning(f"Only {len(scale_df)} queries at scale {scale}, "
                          f"need {self.min_queries} for reliable analysis")
        
        # Compute SNR
        snr, between_var, within_var = self.compute_snr(scale_df, metric, system_col)
        
        # Check coverage (if available)
        coverage = 1.0  # Default if not available
        avg_judgments = 0.0
        
        if 'num_relevant_at_k' in scale_df.columns:
            coverage, avg_judgments = self.check_judgment_coverage(scale_df)
        
        # Determine if metric is discriminative
        discriminative = (snr >= self.min_snr) and (coverage >= self.min_coverage)
        
        # Recommend alternatives if not discriminative
        alternatives = []
        if not discriminative:
            if coverage < self.min_coverage:
                alternatives.append('bpref')  # Works with incomplete judgments
                alternatives.append('statAP')  # Statistical AP
            if snr < self.min_snr:
                alternatives.append('increase_queries')  # Need more queries
                alternatives.append('use_larger_k')  # Use deeper cutoff
        
        return StabilityReport(
            scale=scale,
            metric=metric,
            snr=snr,
            between_variance=between_var,
            within_variance=within_var,
            coverage=coverage,
            avg_judgments_per_query=avg_judgments,
            discriminative=discriminative,
            recommended_alternatives=alternatives
        )
    
    def validate_across_scales(self,
                              df: pd.DataFrame,
                              metrics: List[str],
                              scales: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Validate multiple metrics across multiple scales.
        
        Args:
            df: DataFrame with evaluation results
            metrics: List of metric names to validate
            scales: List of scales (auto-detect if None)
            
        Returns:
            DataFrame with validation results
        """
        if scales is None:
            if 'scale_n' in df.columns:
                scales = sorted(df['scale_n'].unique())
            else:
                scales = [len(df)]  # Single scale
        
        results = []
        
        for scale in scales:
            for metric in metrics:
                if metric not in df.columns:
                    logger.warning(f"Metric {metric} not found in data")
                    continue
                
                try:
                    report = self.validate_metric_at_scale(df, scale, metric)
                    results.append({
                        'scale': report.scale,
                        'metric': report.metric,
                        'snr': report.snr,
                        'coverage': report.coverage,
                        'discriminative': report.discriminative,
                        'alternatives': ', '.join(report.recommended_alternatives)
                    })
                except Exception as e:
                    logger.error(f"Failed to validate {metric} at scale {scale}: {e}")
        
        return pd.DataFrame(results)
    
    def compute_effect_sizes(self,
                            df: pd.DataFrame,
                            metric: str,
                            system1: str,
                            system2: str,
                            query_col: str = 'query_id') -> Dict[str, float]:
        """
        Compute various effect size measures between two systems.
        
        Returns:
            Dictionary with Cohen's d, Cliff's delta, and other effect sizes
        """
        # Get per-query scores for each system
        scores1 = df[df['pipeline'] == system1].groupby(query_col)[metric].mean().values
        scores2 = df[df['pipeline'] == system2].groupby(query_col)[metric].mean().values
        
        # Ensure equal length (only paired queries)
        min_len = min(len(scores1), len(scores2))
        scores1 = scores1[:min_len]
        scores2 = scores2[:min_len]
        
        # Cohen's d
        diff = scores1 - scores2
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
        
        # Cliff's delta (non-parametric effect size)
        comparisons = []
        for s1 in scores1:
            for s2 in scores2:
                if s1 > s2:
                    comparisons.append(1)
                elif s1 < s2:
                    comparisons.append(-1)
                else:
                    comparisons.append(0)
        cliffs_delta = np.mean(comparisons) if comparisons else 0
        
        # Percentage improvement
        mean1 = np.mean(scores1)
        mean2 = np.mean(scores2)
        percent_improvement = ((mean1 - mean2) / (mean2 + 1e-10)) * 100
        
        return {
            'cohens_d': cohens_d,
            'cliffs_delta': cliffs_delta,
            'percent_improvement': percent_improvement,
            'mean_difference': mean1 - mean2
        }
    
    def check_metric_monotonicity(self,
                                 df: pd.DataFrame,
                                 metric: str,
                                 scales: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Check if metric shows monotonic trends with scale.
        
        Returns:
            Dictionary with Kendall's tau and trend analysis
        """
        if scales is None:
            scales = sorted(df['scale_n'].unique())
        
        # Compute median metric value at each scale
        scale_medians = []
        for scale in scales:
            scale_df = df[df['scale_n'] == scale]
            if metric in scale_df.columns:
                scale_medians.append(scale_df[metric].median())
            else:
                scale_medians.append(np.nan)
        
        # Remove NaN values
        valid_data = [(s, m) for s, m in zip(scales, scale_medians) if not np.isnan(m)]
        if len(valid_data) < 2:
            return {'monotonic': None, 'kendalls_tau': None, 'trend': 'insufficient_data'}
        
        valid_scales, valid_medians = zip(*valid_data)
        
        # Compute Kendall's tau
        tau, p_value = stats.kendalltau(valid_scales, valid_medians)
        
        # Determine trend
        if p_value < 0.05:
            if tau > 0.3:
                trend = 'increasing'
            elif tau < -0.3:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'no_significant_trend'
        
        return {
            'monotonic': abs(tau) > 0.7,
            'kendalls_tau': tau,
            'p_value': p_value,
            'trend': trend
        }
    
    def recommend_metrics(self,
                         df: pd.DataFrame,
                         scale: int,
                         current_metrics: List[str]) -> List[str]:
        """
        Recommend appropriate metrics for a given scale.
        
        Returns:
            List of recommended metrics
        """
        recommendations = []
        
        # Analyze current metrics
        for metric in current_metrics:
            report = self.validate_metric_at_scale(df, scale, metric)
            
            if report.discriminative:
                recommendations.append(metric)
            else:
                # Add alternatives
                if report.coverage < self.min_coverage:
                    # Low coverage - use metrics that handle incomplete judgments
                    if 'bpref' not in recommendations:
                        recommendations.append('bpref')
                    if 'infAP' not in recommendations:
                        recommendations.append('infAP')
                
                if report.snr < self.min_snr:
                    # Low discriminative power - use more robust metrics
                    if metric.endswith('_at_10'):
                        # Try deeper cutoff
                        deeper = metric.replace('_at_10', '_at_100')
                        if deeper not in recommendations:
                            recommendations.append(deeper)
        
        # Always include success@k for sparse scenarios
        if scale > 100000:  # Large scale
            if 'success_at_1' not in recommendations:
                recommendations.append('success_at_1')
            if 'success_at_10' not in recommendations:
                recommendations.append('success_at_10')
        
        return recommendations


def generate_stability_report(benchmark_results: pd.DataFrame,
                             output_path: Optional[str] = None) -> str:
    """
    Generate comprehensive metric stability report.
    
    Args:
        benchmark_results: DataFrame with benchmark results
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    validator = MetricStabilityValidator()
    
    # Metrics to analyze
    metrics_to_validate = [
        'ndcg_at_10', 'map_at_10', 'recall_at_100',
        'precision_at_1', 'mrr', 'success_at_10'
    ]
    
    # Validate across scales
    validation_df = validator.validate_across_scales(
        benchmark_results, 
        metrics_to_validate
    )
    
    # Generate report
    report_lines = [
        "=" * 80,
        "METRIC STABILITY VALIDATION REPORT",
        "=" * 80,
        "",
        "Summary of metric discriminative power across scales:",
        ""
    ]
    
    # Group by scale
    for scale in validation_df['scale'].unique():
        scale_df = validation_df[validation_df['scale'] == scale]
        
        report_lines.append(f"\nScale: {scale:,} documents")
        report_lines.append("-" * 40)
        
        discriminative = scale_df[scale_df['discriminative'] == True]
        non_discriminative = scale_df[scale_df['discriminative'] == False]
        
        if len(discriminative) > 0:
            report_lines.append(f"✓ Discriminative metrics: {', '.join(discriminative['metric'].values)}")
        
        if len(non_discriminative) > 0:
            report_lines.append(f"✗ Non-discriminative metrics:")
            for _, row in non_discriminative.iterrows():
                report_lines.append(f"  - {row['metric']}: SNR={row['snr']:.2f}, "
                                  f"Coverage={row['coverage']:.2%}")
                if row['alternatives']:
                    report_lines.append(f"    Alternatives: {row['alternatives']}")
    
    # Add recommendations
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "-" * 40
    ])
    
    # Check for scale-dependent issues
    problematic_scales = validation_df[validation_df['discriminative'] == False]['scale'].unique()
    if len(problematic_scales) > 0:
        report_lines.append(f"⚠ Metrics lose discriminative power at scales: {problematic_scales}")
        report_lines.append("  Consider:")
        report_lines.append("  - Using bpref or statAP for sparse judgments")
        report_lines.append("  - Increasing number of queries")
        report_lines.append("  - Using deeper cutoffs (e.g., @100 instead of @10)")
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report