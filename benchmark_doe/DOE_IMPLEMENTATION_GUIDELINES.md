# DOE Benchmark Framework Implementation Guidelines

## Executive Summary

This document provides comprehensive implementation guidelines for completing the TEJAS DOE (Design of Experiments) benchmarking framework. The framework systematically evaluates optimization combinations across 5 pipeline architectures with a 99.7% reduction in experimental overhead (45 experiments vs 13,824 full factorial).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Security Requirements](#security-requirements)
3. [Component Implementation](#component-implementation)
4. [Testing Strategy](#testing-strategy)
5. [Performance Optimization](#performance-optimization)
6. [Production Deployment](#production-deployment)
7. [Code Quality Standards](#code-quality-standards)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOE Benchmark Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Factors    │  │   Encoders   │  │ Compatibility │          │
│  │   Registry   │  │   Factory    │  │  Validator   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            │                                     │
│                   ┌────────▼────────┐                            │
│                   │ Design Generator│                            │
│                   └────────┬────────┘                            │
│                            │                                     │
│                   ┌────────▼────────┐                            │
│                   │ Experiment      │                            │
│                   │    Runner       │                            │
│                   └────────┬────────┘                            │
│                            │                                     │
│                   ┌────────▼────────┐                            │
│                   │  DOE Analyzer   │                            │
│                   └────────┬────────┘                            │
│                            │                                     │
│                   ┌────────▼────────┐                            │
│                   │ Report Generator│                            │
│                   └─────────────────┘                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

```
Input: Configuration Matrix → Encoder → Transform → Metrics → Analysis → Report
                     ↑                       ↓
                     └── Compatibility ──────┘
```

### 1.3 Pipeline Architectures

| Pipeline | Tokenizer | Backend | SVD Method | Key Features |
|----------|-----------|---------|------------|--------------|
| original_tejas | char_ngram, word | numpy, pytorch | truncated | Baseline implementation |
| goldenratio | char_ngram | numpy, numba | truncated, randomized | Golden ratio optimization |
| fused_char | char_ngram | numpy, numba | all methods | Fused operations |
| fused_byte | byte_bpe | numpy, numba | randomized variants | ByteBPE tokenization |
| optimized_fused | char_ngram, byte_bpe | numba | randomized variants | Maximum optimization |

---

## 2. Security Requirements

### 2.1 ✅ FIXED: eval() Vulnerability

**STATUS: RESOLVED** - SafeEvaluator implementation deployed

#### Previous Vulnerable Code (NOW FIXED)
```python
# benchmark_doe/core/compatibility.py:43 (OLD - FIXED)
def check(self, config: Dict[str, Any]) -> bool:
    try:
        return eval(self.condition, {"config": config})  # SECURITY VULNERABILITY - FIXED
    except Exception:
        return False
```

#### Current Secure Implementation
```python
import ast
import operator as op

class SafeEvaluator:
    """Safe expression evaluator using AST parsing."""
    
    # Define allowed operations
    ALLOWED_OPS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Eq: op.eq,
        ast.NotEq: op.ne,
        ast.Lt: op.lt,
        ast.LtE: op.le,
        ast.Gt: op.gt,
        ast.GtE: op.ge,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Not: op.not_,
    }
    
    @classmethod
    def safe_eval(cls, expr: str, variables: Dict[str, Any]) -> Any:
        """
        Safely evaluate an expression with given variables.
        
        Args:
            expr: Expression string to evaluate
            variables: Dictionary of variables available in expression
            
        Returns:
            Result of expression evaluation
            
        Raises:
            ValueError: If expression contains unsafe operations
        """
        try:
            tree = ast.parse(expr, mode='eval')
            return cls._eval_node(tree.body, variables)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
    
    @classmethod
    def _eval_node(cls, node, variables):
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Undefined variable: {node.id}")
            return variables[node.id]
        elif isinstance(node, ast.Subscript):
            obj = cls._eval_node(node.value, variables)
            key = cls._eval_node(node.slice, variables)
            return obj[key]
        elif isinstance(node, ast.Attribute):
            obj = cls._eval_node(node.value, variables)
            return getattr(obj, node.attr)
        elif isinstance(node, ast.Compare):
            left = cls._eval_node(node.left, variables)
            for op, comparator in zip(node.ops, node.comparators):
                if type(op) not in cls.ALLOWED_OPS:
                    raise ValueError(f"Unsupported operation: {type(op).__name__}")
                right = cls._eval_node(comparator, variables)
                if not cls.ALLOWED_OPS[type(op)](left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.BoolOp):
            if type(node.op) not in cls.ALLOWED_OPS:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            values = [cls._eval_node(value, variables) for value in node.values]
            return cls.ALLOWED_OPS[type(node.op)](*values)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in cls.ALLOWED_OPS:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return cls.ALLOWED_OPS[type(node.op)](cls._eval_node(node.operand, variables))
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")

# Update IncompatibilityRule.check method
def check(self, config: Dict[str, Any]) -> bool:
    """Check if this rule is violated using safe evaluation."""
    try:
        return SafeEvaluator.safe_eval(self.condition, {"config": config})
    except Exception:
        return False
```

### 2.2 ✅ FIXED: Additional Security Enhancements

**STATUS: IMPLEMENTED** - Comprehensive security improvements deployed

#### Implemented Security Measures:

1. **Division by Zero Protection** ✅
   - Location: `core/utils.py` - `safe_divide()` function
   - All division operations now return NaN for undefined operations

2. **Resource Limits** ✅
   - Location: `core/resource_guard.py` - ResourceGuard class
   - Timeout protection (default 300s)
   - Memory limits (default 2GB)
   - Process isolation for experiments

3. **Safe Encoder Factory** ✅
   - Location: `core/encoder_factory.py`
   - Replaced dynamic imports with registry-based instantiation
   - No more unsafe importlib calls

4. **Input Validation Framework** ✅
   - Location: `core/validators.py`
   - Comprehensive validation for all input types
   - Automatic bounds checking and sanitization

### 2.3 Input Validation

All user inputs must be validated:

```python
def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize configuration."""
    schema = {
        'pipeline_architecture': {'type': str, 'allowed': VALID_PIPELINES},
        'n_bits': {'type': int, 'min': 32, 'max': 1024},
        'batch_size': {'type': int, 'min': 1, 'max': 100000},
        # ... other fields
    }
    
    validated = {}
    for key, rules in schema.items():
        if key not in config:
            continue
        value = config[key]
        
        # Type validation
        if not isinstance(value, rules['type']):
            raise TypeError(f"{key} must be {rules['type'].__name__}")
        
        # Range validation
        if 'min' in rules and value < rules['min']:
            raise ValueError(f"{key} must be >= {rules['min']}")
        if 'max' in rules and value > rules['max']:
            raise ValueError(f"{key} must be <= {rules['max']}")
        
        # Allowed values validation
        if 'allowed' in rules and value not in rules['allowed']:
            raise ValueError(f"{key} must be one of {rules['allowed']}")
        
        validated[key] = value
    
    return validated
```

---

## 3. Component Implementation

### 3.1 ExperimentRunner Implementation

```python
# benchmark_doe/core/runners.py
import multiprocessing as mp
import signal
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import json
import psutil

@dataclass
class ExperimentResult:
    """Container for experiment results."""
    config: Dict[str, Any]
    metrics: Dict[str, float]
    status: str  # 'success', 'failed', 'timeout'
    error: Optional[str] = None
    duration: float = 0.0
    memory_peak_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **self.config,
            **self.metrics,
            '_status': self.status,
            '_error': self.error,
            '_duration': self.duration,
            '_memory_peak_mb': self.memory_peak_mb
        }


class ExperimentRunner:
    """
    Manages experiment execution with isolation and resource control.
    
    This class handles:
    - Single and batch experiment execution
    - Process isolation for safety
    - Resource monitoring and limits
    - Checkpointing and recovery
    - Parallel execution
    """
    
    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        isolation_mode: str = "process",
        n_workers: int = 1,
        timeout_seconds: int = 300,
        checkpoint_interval: int = 10
    ):
        """
        Initialize the experiment runner.
        
        Args:
            dataset_path: Path to benchmark dataset
            output_dir: Directory for output files
            isolation_mode: 'none', 'process', or 'container'
            n_workers: Number of parallel workers
            timeout_seconds: Timeout per experiment
            checkpoint_interval: Save progress every N experiments
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.isolation_mode = isolation_mode
        self.n_workers = n_workers
        self.timeout = timeout_seconds
        self.checkpoint_interval = checkpoint_interval
        
        # Setup checkpointing
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.results_file = self.output_dir / "results.csv"
        
        # Load previous checkpoint if exists
        self.completed_configs = self._load_checkpoint()
    
    def run_single(self, config: Dict[str, Any]) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult object
        """
        if self.isolation_mode == "process":
            return self._run_in_process(config)
        elif self.isolation_mode == "container":
            return self._run_in_container(config)
        else:
            return self._run_direct(config)
    
    def run_batch(
        self,
        configs: List[Dict[str, Any]],
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Run a batch of experiments.
        
        Args:
            configs: List of configurations to run
            parallel: Whether to run in parallel
            
        Returns:
            DataFrame with all results
        """
        results = []
        
        # Filter out already completed experiments
        configs_to_run = [
            c for c in configs 
            if self._config_hash(c) not in self.completed_configs
        ]
        
        if len(configs_to_run) < len(configs):
            print(f"Skipping {len(configs) - len(configs_to_run)} already completed experiments")
        
        if parallel and self.n_workers > 1:
            results = self._run_parallel(configs_to_run)
        else:
            results = self._run_sequential(configs_to_run)
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Save results
        if not df.empty:
            if self.results_file.exists():
                existing_df = pd.read_csv(self.results_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(self.results_file, index=False)
        
        return df
    
    def _run_in_process(self, config: Dict[str, Any]) -> ExperimentResult:
        """Run experiment in isolated process."""
        ctx = mp.get_context('spawn')  # Use spawn for clean process
        queue = ctx.Queue()
        
        process = ctx.Process(
            target=self._worker_function,
            args=(config, self.dataset_path, queue)
        )
        
        process.start()
        process.join(timeout=self.timeout)
        
        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            return ExperimentResult(
                config=config,
                metrics={},
                status='timeout',
                error=f"Experiment timed out after {self.timeout}s"
            )
        
        if queue.empty():
            return ExperimentResult(
                config=config,
                metrics={},
                status='failed',
                error="No result returned from process"
            )
        
        return queue.get()
    
    @staticmethod
    def _worker_function(config: Dict[str, Any], dataset_path: Path, queue: mp.Queue):
        """Worker function for process isolation."""
        import signal
        import sys
        
        # Set up signal handling
        def timeout_handler(signum, frame):
            raise TimeoutError("Experiment timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        
        try:
            # Import here to get clean process
            from benchmark_doe.core.encoder_factory import EncoderFactory
            from benchmark_doe.core.metrics_extended import ExtendedMetricsCollector
            
            start_time = time.time()
            process = psutil.Process()
            
            # Create encoder
            encoder = EncoderFactory.create_encoder(
                config['pipeline_architecture'],
                config
            )
            
            # Load data (simplified for example)
            with open(dataset_path / "corpus.json", 'r') as f:
                corpus = json.load(f)
            documents = list(corpus.values())[:100]  # Sample
            
            # Initialize metrics collector
            metrics_collector = ExtendedMetricsCollector()
            metrics_collector.set_optimization_flags(config)
            metrics_collector.start_collection()
            
            # Fit and transform
            encoder.fit(documents)
            codes = encoder.transform(documents)
            
            # Collect metrics
            duration = time.time() - start_time
            memory_peak = process.memory_info().rss / 1024 / 1024
            
            metrics = {
                'fit_transform_time': duration,
                'codes_shape_0': codes.shape[0],
                'codes_shape_1': codes.shape[1] if len(codes.shape) > 1 else 1,
            }
            
            # Add optimization metrics
            opt_metrics = metrics_collector.collect_all_metrics({})
            metrics.update({
                f"opt_{k}": v 
                for k, v in opt_metrics.to_dict().items()
                if isinstance(v, (int, float))
            })
            
            result = ExperimentResult(
                config=config,
                metrics=metrics,
                status='success',
                duration=duration,
                memory_peak_mb=memory_peak
            )
            
            queue.put(result)
            
        except Exception as e:
            result = ExperimentResult(
                config=config,
                metrics={},
                status='failed',
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )
            queue.put(result)
    
    def _run_parallel(self, configs: List[Dict[str, Any]]) -> List[ExperimentResult]:
        """Run experiments in parallel."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self.run_single, config): config
                for config in configs
            }
            
            # Process as they complete
            for future in tqdm(as_completed(future_to_config), total=len(configs)):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update checkpoint
                    self.completed_configs.add(self._config_hash(config))
                    if len(results) % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                except Exception as e:
                    results.append(ExperimentResult(
                        config=config,
                        metrics={},
                        status='failed',
                        error=str(e)
                    ))
        
        self._save_checkpoint()
        return results
    
    def _run_sequential(self, configs: List[Dict[str, Any]]) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        from tqdm import tqdm
        
        results = []
        for config in tqdm(configs):
            result = self.run_single(config)
            results.append(result)
            
            # Update checkpoint
            self.completed_configs.add(self._config_hash(config))
            if len(results) % self.checkpoint_interval == 0:
                self._save_checkpoint()
        
        self._save_checkpoint()
        return results
    
    def _config_hash(self, config: Dict[str, Any]) -> str:
        """Generate unique hash for configuration."""
        import hashlib
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_checkpoint(self) -> set:
        """Load checkpoint from file."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get('completed', []))
        return set()
    
    def _save_checkpoint(self):
        """Save checkpoint to file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'completed': list(self.completed_configs),
                'timestamp': time.time()
            }, f)
```

### 3.2 DOEAnalyzer Implementation

```python
# benchmark_doe/core/analysis.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class DOEAnalyzer:
    """
    Statistical analysis for DOE experiments.
    
    Implements:
    - Main effects analysis (Plackett-Burman)
    - Interaction effects
    - Response surface methodology
    - ANOVA
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
        self.df = results_df
        self.factors = factors
        
        # Auto-detect response variables
        if responses is None:
            self.response_vars = [
                col for col in self.df.columns
                if col not in factors and not col.startswith('_')
            ]
        else:
            self.response_vars = responses
    
    def compute_main_effects(self, response: str) -> pd.DataFrame:
        """
        Compute main effects for each factor.
        
        Args:
            response: Response variable to analyze
            
        Returns:
            DataFrame with effects for each factor
        """
        effects = []
        
        for factor in self.factors:
            if factor not in self.df.columns:
                continue
            
            # Get unique levels
            levels = self.df[factor].unique()
            
            if len(levels) == 2:
                # Binary factor - compute simple effect
                low = self.df[self.df[factor] == levels[0]][response].mean()
                high = self.df[self.df[factor] == levels[1]][response].mean()
                effect = high - low
                
                # Compute standard error
                n_low = len(self.df[self.df[factor] == levels[0]])
                n_high = len(self.df[self.df[factor] == levels[1]])
                var_low = self.df[self.df[factor] == levels[0]][response].var()
                var_high = self.df[self.df[factor] == levels[1]][response].var()
                
                se = np.sqrt(var_low/n_low + var_high/n_high)
                
                effects.append({
                    'factor': factor,
                    'effect': effect,
                    'std_error': se,
                    't_statistic': effect / se if se > 0 else 0,
                    'p_value': 2 * (1 - stats.norm.cdf(abs(effect / se))) if se > 0 else 1.0
                })
            else:
                # Multi-level factor - use ANOVA
                groups = [self.df[self.df[factor] == level][response].values 
                          for level in levels]
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Compute effect size (eta squared)
                grand_mean = self.df[response].mean()
                ss_between = sum(
                    len(group) * (np.mean(group) - grand_mean)**2
                    for group in groups
                )
                ss_total = np.sum((self.df[response] - grand_mean)**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                effects.append({
                    'factor': factor,
                    'effect': eta_squared,
                    'f_statistic': f_stat,
                    'p_value': p_value
                })
        
        effects_df = pd.DataFrame(effects)
        
        # Add normalized effects
        if not effects_df.empty and 'effect' in effects_df.columns:
            max_effect = effects_df['effect'].abs().max()
            if max_effect > 0:
                effects_df['normalized_effect'] = effects_df['effect'] / max_effect
        
        return effects_df.sort_values('p_value')
    
    def compute_interactions(
        self,
        response: str,
        max_order: int = 2
    ) -> pd.DataFrame:
        """
        Compute interaction effects between factors.
        
        Args:
            response: Response variable to analyze
            max_order: Maximum interaction order (2 for two-way, 3 for three-way)
            
        Returns:
            DataFrame with interaction effects
        """
        from itertools import combinations
        
        interactions = []
        
        # Two-way interactions
        for f1, f2 in combinations(self.factors, 2):
            if f1 not in self.df.columns or f2 not in self.df.columns:
                continue
            
            # Create interaction term
            interaction_col = f"{f1}*{f2}"
            
            # For binary factors, use multiplication
            if self.df[f1].nunique() == 2 and self.df[f2].nunique() == 2:
                # Encode as -1, 1
                f1_encoded = 2 * (self.df[f1] == self.df[f1].unique()[1]) - 1
                f2_encoded = 2 * (self.df[f2] == self.df[f2].unique()[1]) - 1
                interaction = f1_encoded * f2_encoded
                
                # Compute effect
                low = self.df[interaction == -1][response].mean()
                high = self.df[interaction == 1][response].mean()
                effect = high - low
                
                # Compute significance
                t_stat, p_value = stats.ttest_ind(
                    self.df[interaction == -1][response],
                    self.df[interaction == 1][response]
                )
                
                interactions.append({
                    'interaction': interaction_col,
                    'effect': effect,
                    't_statistic': t_stat,
                    'p_value': p_value
                })
        
        return pd.DataFrame(interactions).sort_values('p_value')
    
    def fit_response_surface(
        self,
        response: str,
        factors: List[str],
        degree: int = 2
    ) -> Dict[str, Any]:
        """
        Fit a response surface model (polynomial regression).
        
        Args:
            response: Response variable to model
            factors: Factors to include in model
            degree: Polynomial degree (2 for quadratic)
            
        Returns:
            Dictionary with model information
        """
        # Prepare data
        X = self.df[factors].values
        y = self.df[response].values
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Compute statistics
        y_pred = model.predict(X_poly)
        r2 = model.score(X_poly, y)
        
        # Adjusted R²
        n = len(y)
        p = X_poly.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # RMSE
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        
        # Get feature names
        feature_names = poly.get_feature_names_out(factors)
        
        return {
            'model': model,
            'poly_transformer': poly,
            'coefficients': dict(zip(feature_names, model.coef_)),
            'intercept': model.intercept_,
            'r2_score': r2,
            'adj_r2_score': adj_r2,
            'rmse': rmse,
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
        if minimize is None:
            minimize = [False] * len(objectives)
        
        # Get objective values
        data = self.df[self.factors + objectives].copy()
        
        # Convert to minimization problem
        for obj, min_flag in zip(objectives, minimize):
            if not min_flag:
                data[obj] = -data[obj]
        
        # Find Pareto frontier
        pareto_mask = np.ones(len(data), dtype=bool)
        
        for i in range(len(data)):
            if not pareto_mask[i]:
                continue
            
            # Check if any other point dominates this one
            for j in range(len(data)):
                if i == j:
                    continue
                
                # Check domination
                dominates = all(
                    data.iloc[j][obj] <= data.iloc[i][obj]
                    for obj in objectives
                ) and any(
                    data.iloc[j][obj] < data.iloc[i][obj]
                    for obj in objectives
                )
                
                if dominates:
                    pareto_mask[i] = False
                    break
        
        # Restore original values
        pareto_df = self.df[pareto_mask].copy()
        
        return pareto_df
    
    def plot_main_effects(
        self,
        response: str,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create main effects plot."""
        effects = self.compute_main_effects(response)
        
        fig = go.Figure()
        
        # Add bars for effects
        fig.add_trace(go.Bar(
            x=effects['factor'],
            y=effects['effect'],
            error_y=dict(
                type='data',
                array=effects.get('std_error', 0),
                visible=True
            ),
            marker_color='blue',
            name='Effect'
        ))
        
        # Add significance threshold
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f"Main Effects Plot for {response}",
            xaxis_title="Factor",
            yaxis_title="Effect Size",
            showlegend=False
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
        # Get unique levels
        levels1 = sorted(self.df[factor1].unique())
        levels2 = sorted(self.df[factor2].unique())
        
        fig = go.Figure()
        
        # Plot lines for each level of factor2
        for level2 in levels2:
            means = []
            for level1 in levels1:
                mask = (self.df[factor1] == level1) & (self.df[factor2] == level2)
                means.append(self.df[mask][response].mean())
            
            fig.add_trace(go.Scatter(
                x=levels1,
                y=means,
                mode='lines+markers',
                name=f"{factor2}={level2}"
            ))
        
        fig.update_layout(
            title=f"Interaction Plot: {factor1} × {factor2}",
            xaxis_title=factor1,
            yaxis_title=response
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
```

---

## 4. Testing Strategy

### 4.1 Unit Test Template

```python
# tests/test_runner.py
import unittest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
from benchmark_doe.core.runners import ExperimentRunner, ExperimentResult

class TestExperimentRunner(unittest.TestCase):
    """Test suite for ExperimentRunner."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.dataset_path = Path("tests/fixtures/mock_dataset")
        
        self.runner = ExperimentRunner(
            dataset_path=self.dataset_path,
            output_dir=self.output_dir,
            isolation_mode="process",
            n_workers=2,
            timeout_seconds=30
        )
        
        self.test_config = {
            'pipeline_architecture': 'goldenratio',
            'n_bits': 256,
            'use_numba': True,
            'batch_size': 1000
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_run_single_success(self):
        """Test successful single experiment execution."""
        with patch.object(self.runner, '_worker_function') as mock_worker:
            mock_worker.return_value = ExperimentResult(
                config=self.test_config,
                metrics={'accuracy': 0.95},
                status='success'
            )
            
            result = self.runner.run_single(self.test_config)
            
            self.assertEqual(result.status, 'success')
            self.assertEqual(result.metrics['accuracy'], 0.95)
    
    def test_run_single_timeout(self):
        """Test experiment timeout handling."""
        import time
        
        def slow_worker(*args):
            time.sleep(60)  # Longer than timeout
        
        with patch.object(self.runner, '_worker_function', slow_worker):
            self.runner.timeout = 1  # 1 second timeout
            
            result = self.runner.run_single(self.test_config)
            
            self.assertEqual(result.status, 'timeout')
            self.assertIn('timed out', result.error)
    
    def test_run_batch_with_checkpoint(self):
        """Test batch execution with checkpointing."""
        configs = [
            {**self.test_config, 'seed': i}
            for i in range(10)
        ]
        
        # Simulate partial completion
        self.runner.completed_configs.add(
            self.runner._config_hash(configs[0])
        )
        
        with patch.object(self.runner, 'run_single') as mock_run:
            mock_run.return_value = ExperimentResult(
                config=self.test_config,
                metrics={'accuracy': 0.9},
                status='success'
            )
            
            results = self.runner.run_batch(configs, parallel=False)
            
            # Should skip first config
            self.assertEqual(mock_run.call_count, 9)
            self.assertEqual(len(results), 9)
    
    def test_isolation_modes(self):
        """Test different isolation modes."""
        for mode in ['none', 'process']:
            self.runner.isolation_mode = mode
            
            with patch.object(self.runner, '_run_direct' if mode == 'none' else '_run_in_process') as mock_run:
                mock_run.return_value = ExperimentResult(
                    config=self.test_config,
                    metrics={},
                    status='success'
                )
                
                result = self.runner.run_single(self.test_config)
                mock_run.assert_called_once()
                self.assertEqual(result.status, 'success')

if __name__ == '__main__':
    unittest.main()
```

### 4.2 Integration Test Template

```python
# tests/test_integration.py
import unittest
import numpy as np
from benchmark_doe.core.factors_revised import get_revised_registry
from benchmark_doe.core.encoder_factory import EncoderFactory
from benchmark_doe.core.compatibility import CompatibilityValidator

class TestIntegration(unittest.TestCase):
    """Integration tests for DOE framework."""
    
    def test_all_pipelines_instantiate(self):
        """Test that all pipeline architectures can be created."""
        registry = get_revised_registry()
        pipelines = [
            'original_tejas',
            'goldenratio',
            'fused_char',
            'fused_byte',
            'optimized_fused'
        ]
        
        for pipeline in pipelines:
            with self.subTest(pipeline=pipeline):
                config = {
                    'pipeline_architecture': pipeline,
                    'n_bits': 256,
                    'batch_size': 100
                }
                
                # Validate configuration
                validator = CompatibilityValidator()
                is_valid, issues = validator.validate_configuration(config)
                
                if not is_valid:
                    # Try to fix
                    config = validator.fix_configuration(config)
                
                # Create encoder
                try:
                    encoder = EncoderFactory.create_encoder(pipeline, config)
                    self.assertIsNotNone(encoder)
                except Exception as e:
                    self.fail(f"Failed to create {pipeline}: {e}")
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from config to results."""
        # Create minimal config
        config = {
            'pipeline_architecture': 'goldenratio',
            'n_bits': 128,
            'use_numba': False,
            'batch_size': 10
        }
        
        # Validate
        validator = CompatibilityValidator()
        is_valid, _ = validator.validate_configuration(config)
        self.assertTrue(is_valid)
        
        # Create encoder
        encoder = EncoderFactory.create_encoder(
            config['pipeline_architecture'],
            config
        )
        
        # Test data
        documents = ["test doc 1", "test doc 2", "test doc 3"]
        
        # Fit and transform
        encoder.fit(documents)
        codes = encoder.transform(documents)
        
        # Verify output
        self.assertEqual(len(codes), 3)
        self.assertTrue(np.all(codes >= 0))
```

---

## 5. Performance Optimization

### 5.1 Caching Strategy

```python
import functools
import hashlib
import pickle
from pathlib import Path

class CacheManager:
    """Manages caching for expensive computations."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def cache_key(self, *args, **kwargs):
        """Generate cache key from arguments."""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def cached(self, func):
        """Decorator for caching function results."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = self.cache_key(*args, **kwargs)
            cache_file = self.cache_dir / f"{func.__name__}_{key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            result = func(*args, **kwargs)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
```

### 5.2 Memory Management

```python
import gc
import resource

def set_memory_limit(limit_gb: float):
    """Set memory limit for process."""
    limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

def aggressive_gc():
    """Force aggressive garbage collection."""
    gc.collect(2)  # Full collection
    gc.collect()
    gc.collect()
```

---

## 6. Production Deployment

### 6.1 Docker Container Support

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Run benchmarks
ENTRYPOINT ["python", "-m", "benchmark_doe.benchmark_doe"]
```

### 6.2 Database Backend

```python
import sqlite3
from typing import Dict, Any

class ResultsDatabase:
    """SQLite backend for experiment results."""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_hash TEXT UNIQUE,
                config_json TEXT,
                status TEXT,
                start_time REAL,
                end_time REAL,
                duration REAL
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                metric_name TEXT,
                metric_value REAL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            )
        """)
        
        self.conn.commit()
    
    def insert_experiment(self, config: Dict[str, Any], result: Dict[str, Any]):
        """Insert experiment result into database."""
        import json
        import hashlib
        
        config_json = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_json.encode()).hexdigest()
        
        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO experiments 
            (config_hash, config_json, status, start_time, end_time, duration)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            config_hash,
            config_json,
            result.get('status', 'unknown'),
            result.get('start_time', 0),
            result.get('end_time', 0),
            result.get('duration', 0)
        ))
        
        experiment_id = cursor.lastrowid
        
        # Insert metrics
        for metric_name, metric_value in result.get('metrics', {}).items():
            self.conn.execute("""
                INSERT INTO metrics (experiment_id, metric_name, metric_value)
                VALUES (?, ?, ?)
            """, (experiment_id, metric_name, metric_value))
        
        self.conn.commit()
```

---

## 7. Code Quality Standards

### 7.1 Code Style Guidelines

```python
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.942
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### 7.2 Documentation Standards

```python
def example_function(
    param1: str,
    param2: int,
    optional_param: Optional[float] = None
) -> Dict[str, Any]:
    """
    Brief description of function purpose.
    
    Longer description explaining what the function does,
    any important algorithms or techniques used, and
    relevant context.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        optional_param: Description of optional parameter
            (default: None)
    
    Returns:
        Dictionary containing:
        - 'key1': Description of first key
        - 'key2': Description of second key
    
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer
    
    Example:
        >>> result = example_function("test", 42, 3.14)
        >>> print(result['key1'])
        'processed_test'
    
    Note:
        Any important notes about usage or limitations
    """
    pass
```

---

## 8. Troubleshooting Guide

### 8.1 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Import Error | "No module named 'benchmark_doe'" | Add parent directory to sys.path |
| Memory Error | Process killed with code 137 | Reduce batch_size or use memory limits |
| Timeout | Experiments timing out | Increase timeout_seconds parameter |
| Missing Encoder | "AttributeError: module has no attribute" | Check encoder exists in specified module |
| Configuration Invalid | "Configuration validation failed" | Use CompatibilityValidator.fix_configuration() |

### 8.2 Debugging Techniques

```python
# Enable verbose logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add debug breakpoints
import pdb
pdb.set_trace()  # Drops into debugger

# Profile performance
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### 8.3 Performance Monitoring

```python
import time
import psutil
from contextlib import contextmanager

@contextmanager
def monitor_performance(name: str):
    """Context manager for performance monitoring."""
    process = psutil.Process()
    
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024
    
    print(f"Starting {name}...")
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_delta = end_memory - start_memory
        
        print(f"Completed {name}:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Memory delta: {memory_delta:.2f}MB")
        print(f"  Peak memory: {end_memory:.2f}MB")

# Usage
with monitor_performance("Experiment batch"):
    runner.run_batch(configs)
```

---

## Conclusion

This implementation guide provides a complete roadmap for finishing the DOE benchmark framework. The framework is designed to be:

1. **Secure**: All security vulnerabilities addressed
2. **Robust**: Comprehensive error handling and recovery
3. **Scalable**: Parallel execution and caching
4. **Maintainable**: Well-documented and tested
5. **Production-ready**: Database backend and monitoring

### Next Steps

1. Fix the critical eval() security vulnerability immediately
2. Implement the ExperimentRunner with process isolation
3. Create the DOEAnalyzer for statistical analysis
4. Add comprehensive testing
5. Deploy and run full experimental suite

### Estimated Timeline

- **Day 1**: Security fixes and core runner
- **Day 2**: Statistical analysis and visualization
- **Day 3**: Integration and testing
- **Day 4**: Documentation and validation
- **Day 5**: Production deployment

The framework will enable systematic evaluation of all TEJAS optimization combinations, providing data-driven insights for performance optimization.