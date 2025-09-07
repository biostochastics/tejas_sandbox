#!/usr/bin/env python3
"""
DOE Statistical Analysis Module

This module provides statistical analysis capabilities for DOE experiments,
including main effects analysis, interaction detection, and response surface modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Import defensive utilities
from .utils import safe_divide, validate_dataframe, ensure_finite


class DOEAnalyzer:
    """
    Statistical analysis for DOE experiments.
    
    Implements:
    - Main effects analysis (Plackett-Burman)
    - Interaction effects detection
    - Response surface methodology
    - ANOVA analysis
    - Pareto frontier identification
    """
    
    def __init__(
        self,
        results_df: pd.DataFrame,
        factors: List[str],
        responses: Optional[List[str]] = None
    ):
        """
        Initialize analyzer with experimental results.
        
        Args:
            results_df: DataFrame with experiment results
            factors: List of factor names
            responses: List of response variables (auto-detected if None)
        """
        self.df = results_df.copy()
        self.factors = factors
        
        # Auto-detect response variables if not provided
        if responses is None:
            self.response_vars = [
                col for col in self.df.columns
                if col not in factors 
                and not col.startswith('_')
                and not col in ['experiment_id', 'timestamp', 'configuration']
            ]
        else:
            self.response_vars = responses
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate input data integrity."""
        # Check for required columns
        missing_factors = [f for f in self.factors if f not in self.df.columns]
        if missing_factors:
            raise ValueError(f"Missing factors in data: {missing_factors}")
        
        # Check for sufficient data
        if len(self.df) < len(self.factors) + 1:
            warnings.warn(
                f"Limited data: {len(self.df)} experiments for {len(self.factors)} factors. "
                "Results may be unreliable."
            )
    
    def compute_main_effects(self, response: str) -> pd.DataFrame:
        """
        Compute main effects for each factor.
        
        Args:
            response: Response variable to analyze
            
        Returns:
            DataFrame with effects for each factor
        """
        if response not in self.df.columns:
            raise ValueError(f"Response '{response}' not found in data")
        
        effects = []
        
        for factor in self.factors:
            if factor not in self.df.columns:
                continue
            
            # Get unique levels
            levels = self.df[factor].dropna().unique()
            
            if len(levels) < 2:
                # Skip factors with only one level
                continue
            
            if len(levels) == 2:
                # Binary factor - compute simple effect
                level_low, level_high = sorted(levels)
                
                low_data = self.df[self.df[factor] == level_low][response].dropna()
                high_data = self.df[self.df[factor] == level_high][response].dropna()
                
                if len(low_data) > 0 and len(high_data) > 0:
                    # Calculate effect size
                    effect = high_data.mean() - low_data.mean()
                    
                    # Calculate standard error
                    n_low, n_high = len(low_data), len(high_data)
                    var_low = low_data.var() if n_low > 1 else 0
                    var_high = high_data.var() if n_high > 1 else 0
                    
                    df_test = n_low + n_high - 2
                    if df_test > 0:
                        pooled_variance = safe_divide(
                            var_low * (n_low - 1) + var_high * (n_high - 1),
                            df_test,
                            default=0
                        )
                        pooled_std = np.sqrt(pooled_variance) if pooled_variance >= 0 else 0
                        se = pooled_std * np.sqrt(1/n_low + 1/n_high) if pooled_std > 0 else 0
                    else:
                        se = 0
                    
                    # Calculate t-statistic and p-value safely
                    t_stat = safe_divide(effect, se, default=0)
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_test)) if df_test > 0 and se > 0 else 1.0
                    
                    effects.append({
                        'factor': factor,
                        'effect': effect,
                        'std_error': se,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'n_low': n_low,
                        'n_high': n_high,
                        'type': 'binary'
                    })
            
            else:
                # Multi-level factor - use ANOVA
                groups = []
                for level in levels:
                    group_data = self.df[self.df[factor] == level][response].dropna()
                    if len(group_data) > 0:
                        groups.append(group_data.values)
                
                if len(groups) >= 2:
                    # Perform one-way ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # Calculate effect size (eta squared)
                    grand_mean = self.df[response].mean()
                    ss_between = sum(
                        len(group) * (np.mean(group) - grand_mean)**2
                        for group in groups
                    )
                    ss_total = np.sum((self.df[response].dropna() - grand_mean)**2)
                    eta_squared = safe_divide(ss_between, ss_total, default=0)
                    
                    effects.append({
                        'factor': factor,
                        'effect': eta_squared,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'n_levels': len(levels),
                        'type': 'multi_level'
                    })
        
        if not effects:
            return pd.DataFrame()
        
        effects_df = pd.DataFrame(effects)
        
        # Add normalized effects for comparison
        if 'effect' in effects_df.columns:
            max_effect = effects_df['effect'].abs().max()
            effects_df['normalized_effect'] = safe_divide(
                effects_df['effect'],
                max_effect,
                default=0
            )
        
        # Sort by p-value (most significant first)
        effects_df = effects_df.sort_values('p_value')
        
        return effects_df
    
    def compute_interactions(
        self,
        response: str,
        max_order: int = 2,
        significance_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Compute interaction effects between factors.
        
        Args:
            response: Response variable to analyze
            max_order: Maximum interaction order (2 for two-way interactions)
            significance_level: P-value threshold for significance
            
        Returns:
            DataFrame with interaction effects
        """
        from itertools import combinations
        
        if response not in self.df.columns:
            raise ValueError(f"Response '{response}' not found in data")
        
        interactions = []
        
        # Focus on two-way interactions for now
        binary_factors = []
        for factor in self.factors:
            if factor in self.df.columns:
                unique_vals = self.df[factor].dropna().unique()
                if len(unique_vals) == 2:
                    binary_factors.append(factor)
        
        # Check all pairs of binary factors
        for f1, f2 in combinations(binary_factors, 2):
            # Encode factors as -1, 1
            levels1 = sorted(self.df[f1].unique())
            levels2 = sorted(self.df[f2].unique())
            
            # Create interaction term
            f1_encoded = self.df[f1].map({levels1[0]: -1, levels1[1]: 1})
            f2_encoded = self.df[f2].map({levels2[0]: -1, levels2[1]: 1})
            interaction_term = f1_encoded * f2_encoded
            
            # Split data by interaction term
            low_interaction = self.df[interaction_term == -1][response].dropna()
            high_interaction = self.df[interaction_term == 1][response].dropna()
            
            if len(low_interaction) > 0 and len(high_interaction) > 0:
                # Calculate interaction effect
                effect = high_interaction.mean() - low_interaction.mean()
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(low_interaction, high_interaction)
                
                interactions.append({
                    'interaction': f"{f1} × {f2}",
                    'factor1': f1,
                    'factor2': f2,
                    'effect': effect,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < significance_level
                })
        
        if not interactions:
            return pd.DataFrame()
        
        return pd.DataFrame(interactions).sort_values('p_value')
    
    def fit_response_surface(
        self,
        response: str,
        factors: Optional[List[str]] = None,
        degree: int = 2
    ) -> Dict[str, Any]:
        """
        Fit a response surface model (polynomial regression).
        
        Args:
            response: Response variable to model
            factors: Factors to include (uses all if None)
            degree: Polynomial degree (2 for quadratic)
            
        Returns:
            Dictionary with model information and statistics
        """
        if response not in self.df.columns:
            raise ValueError(f"Response '{response}' not found in data")
        
        if factors is None:
            factors = self.factors
        
        # Prepare data
        X_data = self.df[factors].dropna()
        y_data = self.df.loc[X_data.index, response]
        
        # Remove any remaining NaN values
        valid_mask = ~y_data.isna()
        X_data = X_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(X_data) < 10:
            warnings.warn("Insufficient data for response surface modeling")
            return {}
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_data)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y_data)
        
        # Compute statistics
        y_pred = model.predict(X_poly)
        r2 = model.score(X_poly, y_data)
        
        # Adjusted R²
        n = len(y_data)
        p = X_poly.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else 0
        
        # RMSE
        rmse = np.sqrt(np.mean((y_data - y_pred)**2))
        
        # Get feature names
        feature_names = poly.get_feature_names_out(factors)
        
        # Extract significant coefficients
        coef_dict = dict(zip(feature_names, model.coef_))
        significant_coefs = {
            k: v for k, v in coef_dict.items()
            if abs(v) > 0.01 * np.std(y_data)  # Simple significance threshold
        }
        
        return {
            'model': model,
            'poly_transformer': poly,
            'coefficients': coef_dict,
            'significant_coefficients': significant_coefs,
            'intercept': model.intercept_,
            'r2_score': r2,
            'adj_r2_score': adj_r2,
            'rmse': rmse,
            'n_samples': n,
            'n_features': p,
            'feature_names': feature_names
        }
    
    def identify_pareto_frontier(
        self,
        objectives: List[str],
        minimize: Optional[List[bool]] = None
    ) -> pd.DataFrame:
        """
        Identify Pareto-optimal configurations.
        
        Args:
            objectives: List of objective functions
            minimize: Whether to minimize each objective (default: all False)
            
        Returns:
            DataFrame with Pareto-optimal configurations
        """
        # Validate objectives
        missing_objectives = [obj for obj in objectives if obj not in self.df.columns]
        if missing_objectives:
            raise ValueError(f"Objectives not found: {missing_objectives}")
        
        if minimize is None:
            minimize = [False] * len(objectives)
        
        # Get objective values
        data = self.df[self.factors + objectives].dropna()
        
        if len(data) == 0:
            return pd.DataFrame()
        
        # Convert to numpy for efficiency
        obj_values = data[objectives].values
        
        # Adjust for minimization/maximization
        adjusted_values = obj_values.copy()
        for i, (obj, min_flag) in enumerate(zip(objectives, minimize)):
            if not min_flag:
                adjusted_values[:, i] = -adjusted_values[:, i]
        
        # Find Pareto frontier
        n_points = len(adjusted_values)
        pareto_mask = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if not pareto_mask[i]:
                continue
            
            # Check if any other point dominates this one
            for j in range(n_points):
                if i == j:
                    continue
                
                # Check domination (all objectives at least as good, at least one strictly better)
                at_least_as_good = np.all(adjusted_values[j] <= adjusted_values[i])
                strictly_better = np.any(adjusted_values[j] < adjusted_values[i])
                
                if at_least_as_good and strictly_better:
                    pareto_mask[i] = False
                    break
        
        # Return Pareto-optimal configurations
        pareto_df = data[pareto_mask].copy()
        
        # Add dominance count (how many solutions this dominates)
        dominance_counts = []
        for i in pareto_df.index:
            count = 0
            for j in data.index:
                if i == j:
                    continue
                values_i = adjusted_values[data.index.get_loc(i)]
                values_j = adjusted_values[data.index.get_loc(j)]
                if np.all(values_i <= values_j) and np.any(values_i < values_j):
                    count += 1
            dominance_counts.append(count)
        
        pareto_df['dominance_count'] = dominance_counts
        
        return pareto_df.sort_values('dominance_count', ascending=False)
    
    def plot_main_effects(
        self,
        response: str,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create main effects plot."""
        effects = self.compute_main_effects(response)
        
        if effects.empty:
            warnings.warn("No effects to plot")
            return go.Figure()
        
        fig = go.Figure()
        
        # Separate binary and multi-level factors
        binary_effects = effects[effects.get('type', '') == 'binary']
        multi_effects = effects[effects.get('type', '') == 'multi_level']
        
        # Plot binary effects
        if not binary_effects.empty:
            fig.add_trace(go.Bar(
                x=binary_effects['factor'],
                y=binary_effects['effect'],
                error_y=dict(
                    type='data',
                    array=binary_effects.get('std_error', 0),
                    visible=True
                ),
                marker_color='steelblue',
                name='Effect Size',
                text=[f"p={p:.3f}" for p in binary_effects['p_value']],
                textposition='outside'
            ))
        
        # Add significance threshold line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f"Main Effects Plot for {response}",
            xaxis_title="Factor",
            yaxis_title="Effect Size",
            showlegend=False,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_interaction(
        self,
        factor1: str,
        factor2: str,
        response: str,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create interaction plot for two factors."""
        if factor1 not in self.df.columns or factor2 not in self.df.columns:
            raise ValueError(f"Factors not found in data")
        
        if response not in self.df.columns:
            raise ValueError(f"Response not found in data")
        
        # Get unique levels
        levels1 = sorted(self.df[factor1].dropna().unique())
        levels2 = sorted(self.df[factor2].dropna().unique())
        
        fig = go.Figure()
        
        # Plot lines for each level of factor2
        for level2 in levels2:
            means = []
            errors = []
            
            for level1 in levels1:
                mask = (self.df[factor1] == level1) & (self.df[factor2] == level2)
                values = self.df[mask][response].dropna()
                
                if len(values) > 0:
                    means.append(values.mean())
                    errors.append(values.std() / np.sqrt(len(values)) if len(values) > 1 else 0)
                else:
                    means.append(None)
                    errors.append(None)
            
            fig.add_trace(go.Scatter(
                x=levels1,
                y=means,
                error_y=dict(type='data', array=errors, visible=True),
                mode='lines+markers',
                name=f"{factor2}={level2}",
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title=f"Interaction Plot: {factor1} × {factor2} on {response}",
            xaxis_title=factor1,
            yaxis_title=response,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_pareto_frontier(
        self,
        objective1: str,
        objective2: str,
        minimize: Tuple[bool, bool] = (False, False),
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot 2D Pareto frontier."""
        # Get Pareto optimal points
        pareto_df = self.identify_pareto_frontier(
            [objective1, objective2],
            list(minimize)
        )
        
        if pareto_df.empty:
            warnings.warn("No Pareto optimal points found")
            return go.Figure()
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot all points
        fig.add_trace(go.Scatter(
            x=self.df[objective1],
            y=self.df[objective2],
            mode='markers',
            name='All configurations',
            marker=dict(color='lightgray', size=8),
            text=[f"Config {i}" for i in range(len(self.df))],
            hovertemplate='%{text}<br>' + 
                         f'{objective1}: %{{x}}<br>' + 
                         f'{objective2}: %{{y}}<extra></extra>'
        ))
        
        # Plot Pareto frontier
        pareto_sorted = pareto_df.sort_values(objective1)
        fig.add_trace(go.Scatter(
            x=pareto_sorted[objective1],
            y=pareto_sorted[objective2],
            mode='lines+markers',
            name='Pareto frontier',
            marker=dict(color='red', size=12),
            line=dict(color='red', dash='dash'),
            text=[f"Optimal {i}" for i in range(len(pareto_sorted))],
            hovertemplate='%{text}<br>' + 
                         f'{objective1}: %{{x}}<br>' + 
                         f'{objective2}: %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Pareto Frontier Analysis",
            xaxis_title=f"{objective1} {'(minimize)' if minimize[0] else '(maximize)'}",
            yaxis_title=f"{objective2} {'(minimize)' if minimize[1] else '(maximize)'}",
            height=500,
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis summary.
        
        Returns:
            Dictionary with analysis results for all responses
        """
        report = {
            'n_experiments': len(self.df),
            'n_factors': len(self.factors),
            'n_responses': len(self.response_vars),
            'factors': self.factors,
            'responses': self.response_vars,
            'analyses': {}
        }
        
        # Analyze each response variable
        for response in self.response_vars:
            if response not in self.df.columns:
                continue
            
            analysis = {
                'mean': self.df[response].mean(),
                'std': self.df[response].std(),
                'min': self.df[response].min(),
                'max': self.df[response].max()
            }
            
            # Main effects
            try:
                effects = self.compute_main_effects(response)
                if not effects.empty:
                    # Get top 3 significant factors
                    significant = effects[effects['p_value'] < 0.05].head(3)
                    analysis['significant_factors'] = significant['factor'].tolist()
                    analysis['main_effects'] = effects.to_dict('records')
            except Exception as e:
                analysis['main_effects_error'] = str(e)
            
            # Interactions
            try:
                interactions = self.compute_interactions(response)
                if not interactions.empty:
                    significant_int = interactions[interactions['p_value'] < 0.05]
                    analysis['significant_interactions'] = significant_int['interaction'].tolist()
            except Exception as e:
                analysis['interactions_error'] = str(e)
            
            # Response surface R²
            try:
                rsm = self.fit_response_surface(response)
                if rsm:
                    analysis['rsm_r2'] = rsm['r2_score']
                    analysis['rsm_adj_r2'] = rsm['adj_r2_score']
            except Exception as e:
                analysis['rsm_error'] = str(e)
            
            report['analyses'][response] = analysis
        
        return report


# Utility function for quick analysis
def analyze_doe_results(
    results_file: str,
    factors: List[str],
    responses: Optional[List[str]] = None
) -> DOEAnalyzer:
    """
    Quick function to load and analyze DOE results.
    
    Args:
        results_file: Path to CSV results file
        factors: List of factor names
        responses: List of response names (optional)
        
    Returns:
        Configured DOEAnalyzer instance
    """
    df = pd.read_csv(results_file)
    return DOEAnalyzer(df, factors, responses)