#!/usr/bin/env python3
"""
Parametric Scaling Analysis Module for DOE Benchmarks

This module provides capabilities to:
1. Fit complexity curves (O(n), O(n log n), O(n^α))
2. Model selection with AICc/BIC
3. Predict performance at untested scales with confidence intervals
4. Validate scaling assumptions
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScalingModel:
    """Container for fitted scaling model."""
    name: str
    function: Callable
    params: np.ndarray
    param_cov: np.ndarray
    rmse: float
    aicc: float
    bic: float
    r_squared: float
    
    def predict(self, n: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict values at new scales with confidence intervals.
        
        Returns:
            predictions: Point predictions
            lower_ci: Lower confidence interval
            upper_ci: Upper confidence interval
        """
        predictions = self.function(n, *self.params)
        
        # Compute prediction intervals using parameter covariance
        # This is a simplified approach - more sophisticated methods exist
        param_std = np.sqrt(np.diag(self.param_cov))
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Propagate uncertainty (simplified)
        uncertainty = z_score * self.rmse * np.sqrt(1 + 1/len(self.params))
        
        lower_ci = predictions - uncertainty
        upper_ci = predictions + uncertainty
        
        return predictions, lower_ci, upper_ci


class ScalingAnalyzer:
    """
    Analyzes scaling behavior of performance metrics.
    
    Fits various complexity models and selects the best one based on
    information criteria (AICc, BIC).
    """
    
    def __init__(self, min_samples: int = 4):
        """
        Initialize scaling analyzer.
        
        Args:
            min_samples: Minimum number of data points required for fitting
        """
        self.min_samples = min_samples
        self.models = {}
        
    @staticmethod
    def _model_linear(n: np.ndarray, a: float, b: float) -> np.ndarray:
        """O(n) model: T(n) = a*n + b"""
        return a * n + b
    
    @staticmethod
    def _model_nlogn(n: np.ndarray, a: float, b: float) -> np.ndarray:
        """O(n log n) model: T(n) = a*n*log(n) + b"""
        # Use log base 2, handle n=1 case
        log_n = np.log2(np.maximum(n, 2))
        return a * n * log_n + b
    
    @staticmethod
    def _model_power(n: np.ndarray, a: float, alpha: float, b: float) -> np.ndarray:
        """O(n^α) model: T(n) = a*n^α + b"""
        return a * np.power(n, alpha) + b
    
    @staticmethod
    def _model_power_log(n: np.ndarray, a: float, alpha: float, beta: float, b: float) -> np.ndarray:
        """O(n^α * log^β n) model: T(n) = a*n^α*log(n)^β + b"""
        log_n = np.log2(np.maximum(n, 2))
        return a * np.power(n, alpha) * np.power(log_n, beta) + b
    
    @staticmethod
    def _model_quadratic(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """O(n²) model: T(n) = a*n² + b*n + c"""
        return a * n**2 + b * n + c
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          n_params: int) -> Dict[str, float]:
        """Calculate goodness-of-fit metrics."""
        n = len(y_true)
        
        # RMSE
        residuals = y_true - y_pred
        sse = np.sum(residuals**2)
        rmse = np.sqrt(sse / n)
        
        # R-squared
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r_squared = 1 - (sse / ss_tot) if ss_tot > 0 else 0
        
        # AICc (corrected AIC for small samples)
        if n > n_params + 1:
            aic = n * np.log(sse/n) + 2 * n_params
            aicc = aic + (2 * n_params * (n_params + 1)) / (n - n_params - 1)
        else:
            aicc = np.inf
        
        # BIC
        bic = n * np.log(sse/n) + n_params * np.log(n)
        
        return {
            'rmse': rmse,
            'r_squared': r_squared,
            'aicc': aicc,
            'bic': bic
        }
    
    def fit_complexity(self, 
                      df: pd.DataFrame,
                      scale_col: str = "scale_n",
                      y_col: str = "latency_ms",
                      models: Optional[List[str]] = None) -> Dict[str, ScalingModel]:
        """
        Fit multiple complexity models to scaling data.
        
        Args:
            df: DataFrame with scaling data
            scale_col: Column name for scale (n)
            y_col: Column name for metric to model
            models: List of models to fit (default: all)
            
        Returns:
            Dictionary of fitted models
        """
        if len(df) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} data points, got {len(df)}")
        
        # Extract data
        n = df[scale_col].values.astype(float)
        y = df[y_col].values.astype(float)
        
        # Check for valid data
        if np.any(n <= 0):
            raise ValueError("Scale values must be positive")
        if np.any(~np.isfinite(y)):
            raise ValueError("Metric values must be finite")
        
        # Define available models
        model_specs = {
            'linear': (self._model_linear, 2, [y[-1]/n[-1], np.median(y)]),
            'nlogn': (self._model_nlogn, 2, [y[-1]/(n[-1]*np.log2(n[-1])), np.median(y)]),
            'power': (self._model_power, 3, [1.0, 1.0, 0.0]),
            'power_log': (self._model_power_log, 4, [1.0, 1.0, 0.0, 0.0]),
            'quadratic': (self._model_quadratic, 3, [1e-6, 1.0, 0.0])
        }
        
        if models is None:
            models = list(model_specs.keys())
        
        fitted_models = {}
        
        for model_name in models:
            if model_name not in model_specs:
                logger.warning(f"Unknown model: {model_name}")
                continue
            
            func, n_params, initial_guess = model_specs[model_name]
            
            try:
                # Fit model
                popt, pcov = curve_fit(
                    func, n, y, 
                    p0=initial_guess,
                    maxfev=10000,
                    bounds=([-np.inf]*n_params, [np.inf]*n_params)
                )
                
                # Calculate predictions and metrics
                y_pred = func(n, *popt)
                metrics = self._calculate_metrics(y, y_pred, n_params)
                
                # Check for monotonicity (performance should generally increase with n)
                if np.all(np.diff(y_pred) >= 0):  # Monotonic increasing
                    fitted_models[model_name] = ScalingModel(
                        name=model_name,
                        function=func,
                        params=popt,
                        param_cov=pcov,
                        rmse=metrics['rmse'],
                        aicc=metrics['aicc'],
                        bic=metrics['bic'],
                        r_squared=metrics['r_squared']
                    )
                else:
                    logger.warning(f"Model {model_name} is not monotonic, skipping")
                    
            except Exception as e:
                logger.warning(f"Failed to fit {model_name}: {e}")
                continue
        
        self.models = fitted_models
        return fitted_models
    
    def select_best_model(self, criterion: str = 'aicc') -> Optional[ScalingModel]:
        """
        Select best model based on information criterion.
        
        Args:
            criterion: 'aicc', 'bic', or 'rmse'
            
        Returns:
            Best model or None if no models fitted
        """
        if not self.models:
            return None
        
        if criterion not in ['aicc', 'bic', 'rmse']:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        # For information criteria, lower is better
        best_model = min(
            self.models.values(),
            key=lambda m: getattr(m, criterion)
        )
        
        return best_model
    
    def predict_at_scales(self, 
                         scales: np.ndarray,
                         model: Optional[str] = None,
                         confidence: float = 0.95) -> pd.DataFrame:
        """
        Predict performance at new scales.
        
        Args:
            scales: Array of scale values to predict
            model: Model name to use (default: best model)
            confidence: Confidence level for intervals
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if model is None:
            selected_model = self.select_best_model()
            if selected_model is None:
                raise ValueError("No models fitted")
        else:
            if model not in self.models:
                raise ValueError(f"Model {model} not found")
            selected_model = self.models[model]
        
        predictions, lower_ci, upper_ci = selected_model.predict(scales, confidence)
        
        return pd.DataFrame({
            'scale': scales,
            'prediction': predictions,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'model': selected_model.name
        })
    
    def validate_extrapolation(self, 
                              new_scale: float,
                              training_scales: np.ndarray) -> Dict[str, any]:
        """
        Check if prediction at new scale is extrapolation.
        
        Returns:
            Dictionary with extrapolation info and warnings
        """
        min_train = np.min(training_scales)
        max_train = np.max(training_scales)
        
        result = {
            'is_extrapolation': new_scale < min_train or new_scale > max_train,
            'training_range': (min_train, max_train),
            'extrapolation_factor': None,
            'warning': None
        }
        
        if result['is_extrapolation']:
            if new_scale < min_train:
                result['extrapolation_factor'] = min_train / new_scale
            else:
                result['extrapolation_factor'] = new_scale / max_train
            
            if result['extrapolation_factor'] > 2:
                result['warning'] = (
                    f"Significant extrapolation: {result['extrapolation_factor']:.1f}x "
                    f"beyond training range. Predictions may be unreliable."
                )
        
        return result
    
    def analyze_complexity_class(self, 
                                 model: Optional[ScalingModel] = None) -> str:
        """
        Determine the complexity class of the best-fitting model.
        
        Returns:
            String description of complexity class
        """
        if model is None:
            model = self.select_best_model()
            if model is None:
                return "Unknown (no models fitted)"
        
        name = model.name
        params = model.params
        
        if name == 'linear':
            return "O(n) - Linear complexity"
        elif name == 'nlogn':
            return "O(n log n) - Linearithmic complexity"
        elif name == 'power':
            alpha = params[1]
            if 0.9 < alpha < 1.1:
                return "O(n) - Approximately linear"
            elif 1.9 < alpha < 2.1:
                return "O(n²) - Quadratic complexity"
            else:
                return f"O(n^{alpha:.2f}) - Power law complexity"
        elif name == 'quadratic':
            return "O(n²) - Quadratic complexity"
        elif name == 'power_log':
            alpha, beta = params[1], params[2]
            return f"O(n^{alpha:.2f} * log^{beta:.2f} n) - Mixed complexity"
        else:
            return f"{name} complexity"


def fit_scaling_curves(benchmark_results: pd.DataFrame,
                      group_by: List[str] = ['pipeline_type', 'backend'],
                      scale_col: str = 'dataset_size',
                      metrics: List[str] = ['encoding_speed', 'search_latency_p50', 'peak_memory_mb']
                      ) -> Dict[str, Dict[str, ScalingModel]]:
    """
    Convenience function to fit scaling curves for multiple configurations and metrics.
    
    Args:
        benchmark_results: DataFrame with benchmark results
        group_by: Columns to group by (e.g., pipeline, backend)
        scale_col: Column containing scale values
        metrics: List of metrics to model
        
    Returns:
        Nested dictionary: group -> metric -> model
    """
    analyzer = ScalingAnalyzer()
    results = {}
    
    for group_key, group_df in benchmark_results.groupby(group_by):
        group_name = '_'.join(map(str, group_key)) if isinstance(group_key, tuple) else str(group_key)
        results[group_name] = {}
        
        for metric in metrics:
            if metric not in group_df.columns:
                logger.warning(f"Metric {metric} not found for group {group_name}")
                continue
            
            try:
                models = analyzer.fit_complexity(group_df, scale_col, metric)
                best_model = analyzer.select_best_model()
                results[group_name][metric] = best_model
                
                logger.info(f"Group {group_name}, Metric {metric}: "
                          f"Best model is {best_model.name} "
                          f"(R²={best_model.r_squared:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to fit {metric} for {group_name}: {e}")
    
    return results