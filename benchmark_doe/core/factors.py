#!/usr/bin/env python3
"""
Factor Definitions and Constraints for DOE Benchmark Framework

This module defines all controllable factors in the TEJAS optimization space,
their levels, and compatibility constraints between factors.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Literal
import numpy as np
import warnings
from collections import defaultdict


class FactorType(Enum):
    """Types of experimental factors."""
    BINARY = auto()      # Two levels: True/False
    CATEGORICAL = auto() # Multiple discrete levels
    CONTINUOUS = auto()  # Continuous range (discretized for DOE)
    ORDINAL = auto()     # Ordered discrete levels


@dataclass
class Factor:
    """
    Definition of an experimental factor.
    
    Attributes:
        name: Unique identifier for the factor
        type: Type of factor (binary, categorical, etc.)
        levels: Possible values the factor can take
        default: Default value for the factor
        description: Human-readable description
        units: Units of measurement (if applicable)
        constraints: Set of constraint functions
    """
    name: str
    type: FactorType
    levels: Union[List[Any], Tuple[float, float]]
    default: Any
    description: str
    units: Optional[str] = None
    constraints: Set[str] = field(default_factory=set)
    
    def validate_level(self, value: Any) -> bool:
        """Check if a value is a valid level for this factor."""
        if self.type in [FactorType.BINARY, FactorType.CATEGORICAL, FactorType.ORDINAL]:
            return value in self.levels
        elif self.type == FactorType.CONTINUOUS:
            min_val, max_val = self.levels
            return min_val <= value <= max_val
        return False
    
    def get_discrete_levels(self, n_levels: int = 3) -> List[Any]:
        """
        Get discrete levels for the factor.
        
        For continuous factors, returns n_levels equally spaced values.
        For discrete factors, returns all levels.
        """
        if self.type == FactorType.CONTINUOUS:
            min_val, max_val = self.levels
            return list(np.linspace(min_val, max_val, n_levels))
        return list(self.levels)


class IncompatibilityType(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Will cause failures
    WARNING = "warning"  # May degrade performance
    INFO = "info"       # Informational only


@dataclass
class ValidationRule:
    """A validation rule for factor combinations."""
    name: str
    check: Callable[[Dict[str, Any]], bool]
    message: str
    severity: IncompatibilityType = IncompatibilityType.ERROR
    auto_fix: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class FactorRegistry:
    """
    Registry of all experimental factors and their constraints.
    
    This class maintains the complete set of factors that can be varied
    in experiments, along with rules for valid combinations.
    Now includes unified validation system replacing compatibility.py.
    """
    
    def __init__(self):
        """Initialize the factor registry with TEJAS-specific factors."""
        self.factors: Dict[str, Factor] = {}
        self.constraints: List[Dict[str, Any]] = []  # Legacy constraints
        self.validation_rules: List[ValidationRule] = []  # New unified validation
        self.pipeline_constraints: Dict[str, Dict[str, List[str]]] = {}  # Pipeline-specific
        self._initialize_factors()
        self._initialize_constraints()
        self._initialize_validation_rules()
        self._initialize_pipeline_constraints()
    
    def _initialize_factors(self):
        """Define all controllable factors in the TEJAS system."""
        
        # Binary factors (on/off optimizations)
        self.add_factor(Factor(
            name="bit_packing",
            type=FactorType.BINARY,
            levels=[True, False],
            default=True,
            description="Whether to pack bits into uint32 arrays"
        ))
        
        self.add_factor(Factor(
            name="use_numba",
            type=FactorType.BINARY,
            levels=[True, False],
            default=False,
            description="Enable Numba JIT compilation"
        ))
        
        self.add_factor(Factor(
            name="use_itq",
            type=FactorType.BINARY,
            levels=[True, False],
            default=False,
            description="Apply Iterative Quantization rotation"
        ))
        
        self.add_factor(Factor(
            name="use_simd",
            type=FactorType.BINARY,
            levels=[True, False],
            default=True,
            description="Use SIMD acceleration for Hamming distance"
        ))
        
        self.add_factor(Factor(
            name="use_reranker",
            type=FactorType.BINARY,
            levels=[True, False],
            default=False,
            description="Apply reranking post-processing"
        ))
        
        # Categorical factors (multiple discrete choices)
        self.add_factor(Factor(
            name="tokenizer",
            type=FactorType.CATEGORICAL,
            levels=["char_ngram", "byte_bpe", "word", "hybrid"],
            default="char_ngram",
            description="Tokenization method"
        ))
        
        self.add_factor(Factor(
            name="svd_method",
            type=FactorType.CATEGORICAL,
            levels=["truncated", "randomized", "randomized_downsampled"],
            default="randomized",
            description="SVD computation method"
        ))
        
        self.add_factor(Factor(
            name="backend",
            type=FactorType.CATEGORICAL,
            levels=["numpy", "pytorch", "numba"],
            default="numpy",
            description="Computational backend"
        ))
        
        self.add_factor(Factor(
            name="pipeline_type",
            type=FactorType.CATEGORICAL,
            levels=["original_tejas", "goldenratio", "fused_char", "fused_byte", "optimized_fused"],
            default="fused_char",
            description="Pipeline architecture"
        ))
        
        self.add_factor(Factor(
            name="itq_variant",
            type=FactorType.CATEGORICAL,
            levels=["none", "standard", "optimized", "similarity_preserving"],
            default="none",
            description="ITQ implementation variant"
        ))
        
        # Ordinal factors (ordered discrete levels)
        self.add_factor(Factor(
            name="n_bits",
            type=FactorType.ORDINAL,
            levels=[64, 128, 256, 512],
            default=256,
            description="Number of binary bits per document"
        ))
        
        # Continuous factors (will be discretized)
        self.add_factor(Factor(
            name="downsample_ratio",
            type=FactorType.CONTINUOUS,
            levels=(0.1, 1.0),
            default=1.0,
            description="Fraction of data to use for SVD (1.0 = no downsampling)"
        ))
        
        self.add_factor(Factor(
            name="energy_threshold",
            type=FactorType.CONTINUOUS,
            levels=(0.80, 0.99),
            default=0.95,
            description="Fraction of variance to preserve in SVD"
        ))
    
    def _initialize_constraints(self):
        """
        Define incompatible factor combinations.
        
        Each constraint is a rule that invalidates certain factor combinations.
        """
        
        # Numba and PyTorch are incompatible
        self.add_constraint({
            "name": "numba_pytorch_conflict",
            "factors": ["use_numba", "backend"],
            "rule": lambda f: not (f["use_numba"] and f["backend"] == "pytorch"),
            "message": "Cannot use Numba JIT with PyTorch backend"
        })
        
        # ITQ requires use_itq to be True
        self.add_constraint({
            "name": "itq_consistency",
            "factors": ["use_itq", "itq_variant"],
            "rule": lambda f: f["use_itq"] or f["itq_variant"] == "none",
            "message": "ITQ variant requires use_itq=True"
        })
        
        # Downsampling only applies to specific SVD methods
        self.add_constraint({
            "name": "downsample_svd_compatibility",
            "factors": ["svd_method", "downsample_ratio"],
            "rule": lambda f: (
                f["svd_method"] == "randomized_downsampled" or 
                f["downsample_ratio"] >= 0.99
            ),
            "message": "Downsampling only works with randomized_downsampled SVD"
        })
        
        # Bit packing requires compatible bit counts
        self.add_constraint({
            "name": "bit_packing_alignment",
            "factors": ["bit_packing", "n_bits"],
            "rule": lambda f: not f["bit_packing"] or f["n_bits"] % 32 == 0,
            "message": "Bit packing requires n_bits divisible by 32"
        })
        
        # Numba backend requires use_numba
        self.add_constraint({
            "name": "numba_backend_consistency",
            "factors": ["backend", "use_numba"],
            "rule": lambda f: f["backend"] != "numba" or f["use_numba"],
            "message": "Numba backend requires use_numba=True"
        })
        
        # SIMD requires bit packing
        self.add_constraint({
            "name": "simd_packing_requirement",
            "factors": ["use_simd", "bit_packing"],
            "rule": lambda f: not f["use_simd"] or f["bit_packing"],
            "message": "SIMD acceleration requires bit packing"
        })
        
        # Modular pipeline doesn't support all optimizations
        self.add_constraint({
            "name": "modular_pipeline_limitations",
            "factors": ["pipeline_type", "use_simd"],
            "rule": lambda f: f["pipeline_type"] != "modular" or not f["use_simd"],
            "message": "Modular pipeline doesn't support SIMD optimization"
        })
    
    def add_factor(self, factor: Factor):
        """Add a factor to the registry."""
        self.factors[factor.name] = factor
    
    def add_constraint(self, constraint: Dict[str, Any]):
        """Add a constraint rule to the registry (legacy)."""
        self.constraints.append(constraint)
    
    def _initialize_validation_rules(self):
        """Initialize unified validation rules replacing compatibility.py."""
        
        # ERROR level rules (will cause crashes)
        self.validation_rules.append(ValidationRule(
            name="pytorch_numba_conflict",
            check=lambda c: not (c.get('backend') == 'pytorch' and c.get('use_numba', False)),
            message="PyTorch backend incompatible with Numba JIT compilation",
            severity=IncompatibilityType.ERROR,
            auto_fix=lambda c: {**c, 'use_numba': False} if c.get('backend') == 'pytorch' else c
        ))
        
        self.validation_rules.append(ValidationRule(
            name="simd_without_packing",
            check=lambda c: not (c.get('use_simd', False) and not c.get('bit_packing', True)),
            message="SIMD acceleration requires bit packing to be enabled",
            severity=IncompatibilityType.ERROR,
            auto_fix=lambda c: {**c, 'bit_packing': True} if c.get('use_simd', False) else c
        ))
        
        self.validation_rules.append(ValidationRule(
            name="bit_packing_alignment",
            check=lambda c: not c.get('bit_packing', False) or c.get('n_bits', 256) % 32 == 0,
            message="Bit packing requires n_bits divisible by 32",
            severity=IncompatibilityType.ERROR
        ))
        
        self.validation_rules.append(ValidationRule(
            name="numba_backend_consistency",
            check=lambda c: c.get('backend') != 'numba' or c.get('use_numba', False),
            message="Numba backend requires use_numba=True",
            severity=IncompatibilityType.ERROR,
            auto_fix=lambda c: {**c, 'use_numba': True} if c.get('backend') == 'numba' else c
        ))
        
        # WARNING level rules (performance degradation)
        self.validation_rules.append(ValidationRule(
            name="itq_high_bits",
            check=lambda c: c.get('itq_variant') == 'none' or c.get('n_bits', 256) <= 512,
            message="ITQ performance degrades significantly with n_bits > 512",
            severity=IncompatibilityType.WARNING
        ))
        
        self.validation_rules.append(ValidationRule(
            name="randomized_svd_high_threshold",
            check=lambda c: not (c.get('svd_method', '').startswith('randomized') and 
                               c.get('energy_threshold', 0.95) > 0.99),
            message="Randomized SVD accuracy degrades with energy_threshold > 0.99",
            severity=IncompatibilityType.WARNING
        ))
        
        self.validation_rules.append(ValidationRule(
            name="byte_bpe_with_itq",
            check=lambda c: not (c.get('tokenizer') == 'byte_bpe' and 
                                c.get('itq_variant') not in ['none', None]),
            message="ByteBPE tokenizer doesn't benefit from ITQ optimization",
            severity=IncompatibilityType.WARNING,
            auto_fix=lambda c: {**c, 'itq_variant': 'none'} if c.get('tokenizer') == 'byte_bpe' else c
        ))
        
        # INFO level rules (recommendations)
        self.validation_rules.append(ValidationRule(
            name="small_bits_without_itq",
            check=lambda c: c.get('n_bits', 256) >= 128 or c.get('itq_variant') != 'none',
            message="Small n_bits (<128) benefit significantly from ITQ optimization",
            severity=IncompatibilityType.INFO
        ))
    
    def _initialize_pipeline_constraints(self):
        """Initialize pipeline-specific constraints."""
        self.pipeline_constraints = {
            'original_tejas': {
                'tokenizers': ['char_ngram', 'word'],
                'backends': ['numpy', 'pytorch'],
                'svd_methods': ['truncated'],
                'unsupported_features': ['use_simd']
            },
            'goldenratio': {
                'tokenizers': ['char_ngram'],
                'backends': ['numpy', 'numba'],
                'svd_methods': ['truncated', 'randomized'],
                'unsupported_features': []
            },
            'fused_char': {
                'tokenizers': ['char_ngram'],
                'backends': ['numpy', 'numba'],
                'svd_methods': ['truncated', 'randomized', 'randomized_downsampled'],
                'unsupported_features': []
            },
            'fused_byte': {
                'tokenizers': ['byte_bpe'],
                'backends': ['numpy'],
                'svd_methods': ['truncated', 'randomized'],
                'unsupported_features': ['use_itq']
            },
            'optimized_fused': {
                'tokenizers': ['char_ngram', 'byte_bpe'],
                'backends': ['numpy', 'numba'],
                'svd_methods': ['truncated', 'randomized', 'randomized_downsampled'],
                'unsupported_features': []
            }
        }
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a new validation rule."""
        self.validation_rules.append(rule)
    
    def validate_configuration(self, config: Dict[str, Any], strict: bool = True) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Unified validation replacing compatibility.py functionality.
        
        Args:
            config: Dictionary of factor settings
            strict: If True, warnings also fail validation
            
        Returns:
            Tuple of (is_valid, dict of issues by severity)
        """
        issues = {
            'error': [],
            'warning': [],
            'info': []
        }
        
        # Check individual factor validity
        for name, value in config.items():
            if name in self.factors:
                factor = self.factors[name]
                if not factor.validate_level(value):
                    issues['error'].append(f"Invalid level {value} for factor {name}")
        
        # Check new unified validation rules
        for rule in self.validation_rules:
            if not rule.check(config):
                if rule.severity == IncompatibilityType.ERROR:
                    issues['error'].append(rule.message)
                elif rule.severity == IncompatibilityType.WARNING:
                    issues['warning'].append(rule.message)
                else:
                    issues['info'].append(rule.message)
        
        # Check pipeline-specific constraints if pipeline is specified
        pipeline = config.get('pipeline_type') or config.get('pipeline')
        if pipeline and pipeline in self.pipeline_constraints:
            constraints = self.pipeline_constraints[pipeline]
            
            # Check tokenizer compatibility
            tokenizer = config.get('tokenizer')
            if tokenizer and tokenizer not in constraints['tokenizers']:
                issues['error'].append(
                    f"Pipeline {pipeline} doesn't support tokenizer {tokenizer}"
                )
            
            # Check backend compatibility
            backend = config.get('backend')
            if backend and backend not in constraints['backends']:
                issues['error'].append(
                    f"Pipeline {pipeline} doesn't support backend {backend}"
                )
            
            # Check SVD method compatibility
            svd_method = config.get('svd_method')
            if svd_method and svd_method not in constraints['svd_methods']:
                issues['warning'].append(
                    f"Pipeline {pipeline} may not fully support SVD method {svd_method}"
                )
            
            # Check unsupported features
            for feature in constraints['unsupported_features']:
                if config.get(feature, False):
                    issues['error'].append(
                        f"Pipeline {pipeline} doesn't support {feature}"
                    )
        
        # Legacy constraint checking (will be phased out)
        for constraint in self.constraints:
            factor_subset = {
                f: config.get(f, self.factors[f].default if f in self.factors else None)
                for f in constraint["factors"]
            }
            if not constraint["rule"](factor_subset):
                issues['warning'].append(constraint["message"])
        
        # Determine overall validity
        is_valid = len(issues['error']) == 0
        if strict:
            is_valid = is_valid and len(issues['warning']) == 0
        
        return is_valid, issues
    
    def auto_fix_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to automatically fix invalid configurations.
        
        Args:
            config: Configuration to fix
            
        Returns:
            Fixed configuration (may still have warnings)
        """
        fixed_config = config.copy()
        is_valid, issues = self.validate_configuration(fixed_config, strict=False)
        
        if is_valid and len(issues['error']) == 0:
            return fixed_config
        
        # Apply auto-fixes from validation rules
        for rule in self.validation_rules:
            if not rule.check(fixed_config) and rule.auto_fix:
                fixed_config = rule.auto_fix(fixed_config)
                warnings.warn(f"Auto-fixed: {rule.message}")
        
        # Revalidate
        is_valid, remaining_issues = self.validate_configuration(fixed_config, strict=False)
        
        if remaining_issues['error']:
            warnings.warn(f"Could not fix all errors: {remaining_issues['error']}")
        
        return fixed_config
    
    def explain_issues(self, config: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of configuration issues.
        
        Args:
            config: Configuration to explain
            
        Returns:
            Formatted explanation string
        """
        is_valid, issues = self.validate_configuration(config, strict=False)
        
        if is_valid and not any(issues.values()):
            return "Configuration is valid with no issues."
        
        explanation = []
        
        if issues['error']:
            explanation.append("ERRORS (will cause failures):")
            for issue in issues['error']:
                explanation.append(f"  - {issue}")
        
        if issues['warning']:
            explanation.append("\nWARNINGS (may impact performance):")
            for issue in issues['warning']:
                explanation.append(f"  - {issue}")
        
        if issues['info']:
            explanation.append("\nRECOMMENDATIONS:")
            for issue in issues['info']:
                explanation.append(f"  - {issue}")
        
        return "\n".join(explanation)
    
    def filter_valid_configurations(
        self, 
        configurations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of configurations to only valid ones.
        
        Args:
            configurations: List of factor configurations
            
        Returns:
            List of valid configurations
        """
        valid_configs = []
        
        for config in configurations:
            is_valid, _ = self.validate_configuration(config)
            if is_valid:
                valid_configs.append(config)
        
        return valid_configs
    
    def get_factor_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        """
        Get min/max bounds for each factor.
        
        Useful for optimization algorithms that need parameter bounds.
        """
        bounds = {}
        
        for name, factor in self.factors.items():
            if factor.type == FactorType.CONTINUOUS:
                bounds[name] = factor.levels
            elif factor.type == FactorType.ORDINAL:
                bounds[name] = (min(factor.levels), max(factor.levels))
            elif factor.type == FactorType.BINARY:
                bounds[name] = (0, 1)  # Encode as 0/1
            else:
                # Categorical: use index bounds
                bounds[name] = (0, len(factor.levels) - 1)
        
        return bounds
    
    def encode_configuration(self, config: Dict[str, Any]) -> np.ndarray:
        """
        Encode a configuration as a numeric vector.
        
        Args:
            config: Dictionary of factor settings
            
        Returns:
            Numeric vector representation
        """
        vector = []
        
        for name, factor in self.factors.items():
            value = config.get(name, factor.default)
            
            if factor.type == FactorType.BINARY:
                vector.append(float(value))
            elif factor.type == FactorType.CONTINUOUS:
                vector.append(value)
            elif factor.type == FactorType.ORDINAL:
                vector.append(value)
            else:
                # Categorical: use index
                vector.append(factor.levels.index(value))
        
        return np.array(vector)
    
    def decode_configuration(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Decode a numeric vector to a configuration.
        
        Args:
            vector: Numeric vector representation
            
        Returns:
            Dictionary of factor settings
        """
        config = {}
        factor_names = list(self.factors.keys())
        
        for i, (name, factor) in enumerate(self.factors.items()):
            value = vector[i]
            
            if factor.type == FactorType.BINARY:
                config[name] = bool(value > 0.5)
            elif factor.type == FactorType.CONTINUOUS:
                config[name] = float(value)
            elif factor.type == FactorType.ORDINAL:
                # Round to nearest valid level
                config[name] = min(factor.levels, key=lambda x: abs(x - value))
            else:
                # Categorical: use index
                idx = int(round(value))
                idx = max(0, min(idx, len(factor.levels) - 1))
                config[name] = factor.levels[idx]
        
        return config


# Convenience functions
def get_default_registry() -> FactorRegistry:
    """Get the default TEJAS factor registry."""
    return FactorRegistry()


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a configuration using the default registry."""
    registry = get_default_registry()
    return registry.validate_configuration(config)