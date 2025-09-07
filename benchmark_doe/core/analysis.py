#!/usr/bin/env python3
"""
Statistical Analysis for DOE Benchmark Results

This module provides tools for analyzing experimental results including
main effects analysis, interaction analysis, response surface modeling,
and Pareto frontier identification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import warnings

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed. Some analyses will be unavailable.")


class DOEAnalyzer:
    """
    Analyzer for Design of Experiments results.
    
    Provides statistical analysis tools including ANOVA, regression,
    and visualization of experimental results.
    """
    
    def __init__(self, results_df: pd.DataFrame, factors: List[str]):
        """
        Initialize the analyzer.
        
        Args:
            results_df: DataFrame with experimental results
            factors: List of factor names to analyze
        """
        self.results = results_df
        self.factors = factors
        self.response_vars = self._identify_response_variables()
        
    def _identify_response_variables(self) -> List[str]:
        """Identify response variables (metrics) in the results."""
        # Common metric patterns
        metric_patterns = [
            'speed', 'latency', 'throughput', 'ndcg', 'mrr', 'recall',
            'memory', 'cpu', 'accuracy', 'f1', 'precision'
        ]
        
        response_vars = []
        for col in self.results.columns:
            if any(pattern in col.lower() for pattern in metric_patterns):
                if col not in self.factors:
                    response_vars.append(col)
        
        return response_vars
    
    def compute_main_effects(
        self,
        response: str,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compute main effects of each factor on a response variable.
        
        Main effect = average change in response when factor changes level.
        
        Args:
            response: Response variable to analyze
            normalize: Whether to normalize effects to [-1, 1]
            
        Returns:
            DataFrame with main effects for each factor
        """
        effects = {}
        
        for factor in self.factors:
            # Get unique levels
            levels = self.results[factor].unique()
            
            if len(levels) == 2:
                # Binary factor: effect = mean(high) - mean(low)
                low_mean = self.results[self.results[factor] == levels[0]][response].mean()
                high_mean = self.results[self.results[factor] == levels[1]][response].mean()
                effect = high_mean - low_mean
                
            else:
                # Multi-level factor: use range or variance
                level_means = []
                for level in levels:
                    level_mean = self.results[self.results[factor] == level][response].mean()
                    level_means.append(level_mean)
                
                # Effect = range of means
                effect = np.max(level_means) - np.min(level_means)
            
            effects[factor] = effect
        
        # Create DataFrame
        effects_df = pd.DataFrame({
            'factor': list(effects.keys()),
            'main_effect': list(effects.values())
        })
        
        # Normalize if requested
        if normalize and len(effects_df) > 0:
            max_effect = effects_df['main_effect'].abs().max()
            if max_effect > 0:
                effects_df['normalized_effect'] = effects_df['main_effect'] / max_effect
        
        # Sort by absolute effect size
        effects_df['abs_effect'] = effects_df['main_effect'].abs()
        effects_df = effects_df.sort_values('abs_effect', ascending=False)
        
        return effects_df
    
    def compute_interactions(
        self,
        response: str,
        max_order: int = 2
    ) -> pd.DataFrame:
        """
        Compute interaction effects between factors.
        
        Args:
            response: Response variable to analyze
            max_order: Maximum interaction order (2 = two-way, 3 = three-way)
            
        Returns:
            DataFrame with interaction effects
        """
        if not HAS_STATSMODELS:
            warnings.warn("statsmodels required for interaction analysis")
            return pd.DataFrame()
        
        # Build formula for linear model
        formula_parts = [response, "~"]
        
        # Add main effects
        formula_parts.append(" + ".join(self.factors))
        
        # Add interactions
        if max_order >= 2:
            from itertools import combinations
            for order in range(2, min(max_order + 1, 4)):  # Limit to 3-way
                for combo in combinations(self.factors, order):
                    formula_parts.append(" + " + ":".join(combo))
        
        formula = " ".join(formula_parts)
        
        try:
            # Fit linear model
            model = ols(formula, data=self.results).fit()
            
            # Extract interaction terms
            interactions = []
            for term, coef in model.params.items():
                if ":" in term:  # Interaction term
                    interactions.append({
                        'interaction': term,
                        'coefficient': coef,
                        'p_value': model.pvalues[term],
                        'significant': model.pvalues[term] < 0.05
                    })
            
            return pd.DataFrame(interactions)
            
        except Exception as e:
            warnings.warn(f"Interaction analysis failed: {e}")
            return pd.DataFrame()
    
    def perform_anova(
        self,
        response: str,
        include_interactions: bool = True
    ) -> pd.DataFrame:
        """
        Perform Analysis of Variance (ANOVA) on the results.
        
        Args:
            response: Response variable to analyze
            include_interactions: Whether to include interaction terms
            
        Returns:
            ANOVA table as DataFrame
        """
        if not HAS_STATSMODELS:
            warnings.warn("statsmodels required for ANOVA")
            return pd.DataFrame()
        
        # Build formula
        formula_parts = [response, "~"]
        formula_parts.append(" + ".join(self.factors))
        
        if include_interactions:
            from itertools import combinations
            for combo in combinations(self.factors, 2):
                formula_parts.append(" + " + ":".join(combo))
        
        formula = " ".join(formula_parts)
        
        try:
            # Fit model and perform ANOVA
            model = ols(formula, data=self.results).fit()
            anova_table = anova_lm(model, typ=2)
            
            # Add effect size (eta squared)
            anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
            
            return anova_table
            
        except Exception as e:
            warnings.warn(f"ANOVA failed: {e}")
            return pd.DataFrame()
    
    def fit_response_surface(
        self,
        response: str,
        factors: Optional[List[str]] = None,
        degree: int = 2
    ) -> Dict[str, Any]:
        """
        Fit a polynomial response surface model.
        
        Args:
            response: Response variable to model
            factors: Subset of factors to include (default: all)
            degree: Polynomial degree (2 = quadratic)
            
        Returns:
            Dictionary with model info and coefficients
        """
        if factors is None:
            factors = self.factors
        
        # Prepare data
        X = self.results[factors].values
        y = self.results[response].values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_poly, y, cv=5, 
                                   scoring='neg_mean_squared_error')
        
        # Get feature names
        feature_names = poly.get_feature_names_out(factors)
        
        # Create coefficient dictionary
        coef_dict = dict(zip(feature_names, model.coef_))
        
        return {
            'model': model,
            'poly_transformer': poly,
            'coefficients': coef_dict,
            'intercept': model.intercept_,
            'r2_score': model.score(X_poly, y),
            'cv_rmse': np.sqrt(-cv_scores.mean()),
            'feature_names': feature_names
        }
    
    def identify_pareto_frontier(
        self,
        objectives: List[str],
        minimize: Optional[List[bool]] = None
    ) -> pd.DataFrame:
        """
        Identify the Pareto frontier for multi-objective optimization.
        
        Args:
            objectives: List of objective (response) variables
            minimize: List of booleans indicating minimize (True) or maximize (False)
            
        Returns:
            DataFrame with Pareto-optimal configurations
        """
        if minimize is None:
            # Default: minimize latency/memory, maximize accuracy
            minimize = []
            for obj in objectives:
                if any(term in obj.lower() for term in ['latency', 'memory', 'time']):
                    minimize.append(True)
                else:
                    minimize.append(False)
        
        # Extract objective values
        obj_values = self.results[objectives].values
        
        # Normalize objectives (flip sign for maximization)
        norm_values = obj_values.copy()
        for i, min_flag in enumerate(minimize):
            if not min_flag:
                norm_values[:, i] = -norm_values[:, i]
        
        # Identify Pareto frontier
        pareto_mask = np.ones(len(norm_values), dtype=bool)
        
        for i in range(len(norm_values)):
            if pareto_mask[i]:
                # Check if any other point dominates this one
                dominates = np.all(norm_values <= norm_values[i], axis=1)
                dominates = dominates & np.any(norm_values < norm_values[i], axis=1)
                dominates[i] = False  # Don't compare with itself
                
                # Mark dominated points
                pareto_mask[dominates] = False
        
        # Return Pareto-optimal configurations
        return self.results[pareto_mask].copy()
    
    def plot_main_effects(
        self,
        response: str,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None
    ):
        """
        Create main effects plot for a response variable.
        
        Args:
            response: Response variable to plot
            figsize: Figure size
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, len(self.factors), figsize=figsize)
        
        if len(self.factors) == 1:
            axes = [axes]
        
        for idx, factor in enumerate(self.factors):
            ax = axes[idx]
            
            # Get factor levels and means
            levels = sorted(self.results[factor].unique())
            means = []
            errors = []
            
            for level in levels:
                level_data = self.results[self.results[factor] == level][response]
                means.append(level_data.mean())
                errors.append(level_data.std() / np.sqrt(len(level_data)))
            
            # Plot
            x_pos = np.arange(len(levels))
            ax.errorbar(x_pos, means, yerr=errors, marker='o', capsize=5)
            
            # Formatting
            ax.set_xlabel(factor)
            ax.set_ylabel(response if idx == 0 else '')
            ax.set_xticks(x_pos)
            
            # Handle different types of levels
            if isinstance(levels[0], bool):
                ax.set_xticklabels(['False', 'True'])
            elif isinstance(levels[0], (int, float)):
                ax.set_xticklabels([f'{l:.2g}' for l in levels])
            else:
                ax.set_xticklabels(levels, rotation=45, ha='right')
            
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Main Effects Plot for {response}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_interaction(
        self,
        factor1: str,
        factor2: str,
        response: str,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[Path] = None
    ):
        """
        Create interaction plot between two factors.
        
        Args:
            factor1: First factor
            factor2: Second factor
            response: Response variable
            figsize: Figure size
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique levels
        levels1 = sorted(self.results[factor1].unique())
        levels2 = sorted(self.results[factor2].unique())
        
        # Plot lines for each level of factor2
        for level2 in levels2:
            means = []
            for level1 in levels1:
                mask = (self.results[factor1] == level1) & (self.results[factor2] == level2)
                mean_val = self.results[mask][response].mean()
                means.append(mean_val)
            
            ax.plot(range(len(levels1)), means, marker='o', label=f'{factor2}={level2}')
        
        # Formatting
        ax.set_xlabel(factor1)
        ax.set_ylabel(response)
        ax.set_xticks(range(len(levels1)))
        ax.set_xticklabels([str(l) for l in levels1])
        ax.legend(title=factor2)
        ax.grid(True, alpha=0.3)
        
        plt.title(f'Interaction Plot: {factor1} x {factor2}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_pareto_frontier(
        self,
        obj1: str,
        obj2: str,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[Path] = None
    ):
        """
        Plot 2D Pareto frontier.
        
        Args:
            obj1: First objective
            obj2: Second objective
            figsize: Figure size
            save_path: Path to save figure
        """
        # Identify Pareto frontier
        pareto_df = self.identify_pareto_frontier([obj1, obj2])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all points
        ax.scatter(self.results[obj1], self.results[obj2], 
                  alpha=0.5, label='All configurations')
        
        # Highlight Pareto frontier
        ax.scatter(pareto_df[obj1], pareto_df[obj2], 
                  color='red', s=100, marker='*', label='Pareto optimal')
        
        # Connect Pareto points
        pareto_sorted = pareto_df.sort_values(obj1)
        ax.plot(pareto_sorted[obj1], pareto_sorted[obj2], 
               'r--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.title('Pareto Frontier')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_report(
        self,
        output_dir: Path,
        responses: Optional[List[str]] = None
    ):
        """
        Generate comprehensive analysis report.
        
        Args:
            output_dir: Directory to save report files
            responses: List of response variables to analyze
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if responses is None:
            responses = self.response_vars[:3]  # Top 3 responses
        
        report_data = {
            'summary': {},
            'main_effects': {},
            'interactions': {},
            'anova': {},
            'pareto': {}
        }
        
        # Analyze each response
        for response in responses:
            # Main effects
            main_effects = self.compute_main_effects(response)
            report_data['main_effects'][response] = main_effects.to_dict()
            
            # Save main effects plot
            self.plot_main_effects(
                response,
                save_path=output_dir / f'main_effects_{response}.png'
            )
            
            # ANOVA
            if HAS_STATSMODELS:
                anova = self.perform_anova(response)
                if not anova.empty:
                    report_data['anova'][response] = anova.to_dict()
            
            # Top interactions
            interactions = self.compute_interactions(response)
            if not interactions.empty:
                top_interactions = interactions.nsmallest(5, 'p_value')
                report_data['interactions'][response] = top_interactions.to_dict()
        
        # Pareto frontier for key objectives
        if len(responses) >= 2:
            pareto = self.identify_pareto_frontier(responses[:2])
            report_data['pareto']['frontier'] = pareto.to_dict()
            
            # Save Pareto plot
            self.plot_pareto_frontier(
                responses[0], responses[1],
                save_path=output_dir / 'pareto_frontier.png'
            )
        
        # Summary statistics
        report_data['summary'] = {
            'n_experiments': len(self.results),
            'n_factors': len(self.factors),
            'n_responses': len(self.response_vars),
            'factors': self.factors,
            'responses': self.response_vars
        }
        
        # Save JSON report
        with open(output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate markdown summary
        self._generate_markdown_report(report_data, output_dir / 'report.md')
    
    def _generate_markdown_report(self, report_data: Dict, filepath: Path):
        """Generate a markdown summary report."""
        lines = [
            "# DOE Analysis Report",
            "",
            "## Summary",
            f"- Experiments: {report_data['summary']['n_experiments']}",
            f"- Factors: {report_data['summary']['n_factors']}",
            f"- Response Variables: {report_data['summary']['n_responses']}",
            "",
            "## Main Effects",
            ""
        ]
        
        for response, effects in report_data['main_effects'].items():
            lines.append(f"### {response}")
            if effects:
                df = pd.DataFrame(effects)
                lines.append(df.to_markdown())
            lines.append("")
        
        lines.extend([
            "## Pareto Frontier",
            "",
            "Configurations that provide the best trade-offs between objectives.",
            "",
            "See `pareto_frontier.png` for visualization.",
            ""
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))