#!/usr/bin/env python3
"""
Compatibility Validator for DOE Benchmark Framework
This module validates factor combinations and ensures experiments
use only compatible configurations.
"""

from typing import Dict, Any, List, Tuple, Set
import warnings
from dataclasses import dataclass
from enum import Enum

# Import safe evaluator to replace eval()
from .safe_evaluator import SafeEvaluator


class IncompatibilityType(Enum):
    """Types of incompatibilities between factors."""
    HARD = "hard"      # Cannot work together (will crash)
    SOFT = "soft"      # Works but degrades performance
    WARNING = "warning" # Works but not recommended


@dataclass
class IncompatibilityRule:
    """Definition of an incompatibility rule."""
    name: str
    type: IncompatibilityType
    factors: List[str]
    condition: str
    message: str
    
    def check(self, config: Dict[str, Any]) -> bool:
        """
        Check if this rule is violated using safe evaluation.
        
        Args:
            config: Configuration to check
            
        Returns:
            True if rule is violated, False otherwise
        """
        # Use SafeEvaluator instead of eval() for security
        try:
            return SafeEvaluator.safe_eval(self.condition, {"config": config})
        except Exception:
            return False


class CompatibilityValidator:
    """
    Validates factor combinations before experiment execution.
    
    This validator ensures that only compatible factor combinations
    are used in experiments, preventing crashes and identifying
    suboptimal configurations.
    """
    
    def __init__(self):
        """Initialize the compatibility validator."""
        self.rules = []
        self._initialize_rules()
        self._pipeline_constraints = self._load_pipeline_constraints()
    
    def _initialize_rules(self):
        """Define all incompatibility rules."""
        
        # HARD incompatibilities (will crash)
        self.rules.append(IncompatibilityRule(
            name="pytorch_numba_conflict",
            type=IncompatibilityType.HARD,
            factors=["backend", "use_numba"],
            condition="config.get('backend') == 'pytorch' and config.get('use_numba', False)",
            message="PyTorch backend incompatible with Numba JIT compilation"
        ))
        
        self.rules.append(IncompatibilityRule(
            name="simd_without_packing",
            type=IncompatibilityType.HARD,
            factors=["use_simd", "bit_packing"],
            condition="config.get('use_simd', False) and not config.get('bit_packing', True)",
            message="SIMD acceleration requires bit packing to be enabled"
        ))
        
        self.rules.append(IncompatibilityRule(
            name="adaptive_svd_without_regularization",
            type=IncompatibilityType.HARD,
            factors=["svd_method", "use_adaptive_regularization"],
            condition="config.get('svd_method') == 'randomized_adaptive' and not config.get('use_adaptive_regularization', True)",
            message="Adaptive SVD requires adaptive regularization to be enabled"
        ))
        
        # SOFT incompatibilities (performance degradation)
        self.rules.append(IncompatibilityRule(
            name="itq_high_bits",
            type=IncompatibilityType.SOFT,
            factors=["itq_method", "n_bits"],
            condition="config.get('itq_method') != 'none' and config.get('n_bits', 256) > 512",
            message="ITQ performance degrades with n_bits > 512"
        ))
        
        self.rules.append(IncompatibilityRule(
            name="randomized_svd_high_threshold",
            type=IncompatibilityType.SOFT,
            factors=["svd_method", "energy_threshold"],
            condition="config.get('svd_method', '').startswith('randomized') and config.get('energy_threshold', 0.95) > 0.99",
            message="Randomized SVD accuracy degrades with energy_threshold > 0.99"
        ))
        
        self.rules.append(IncompatibilityRule(
            name="byte_bpe_with_itq",
            type=IncompatibilityType.SOFT,
            factors=["tokenizer", "itq_method"],
            condition="config.get('tokenizer') == 'byte_bpe' and config.get('itq_method') != 'none'",
            message="ByteBPE tokenizer doesn't benefit from ITQ optimization"
        ))
        
        # WARNING incompatibilities (not recommended)
        self.rules.append(IncompatibilityRule(
            name="memory_profiler_overhead",
            type=IncompatibilityType.WARNING,
            factors=["use_memory_profiler", "batch_size"],
            condition="config.get('use_memory_profiler', False) and config.get('batch_size', 10000) > 50000",
            message="Memory profiler adds significant overhead with large batch sizes"
        ))
        
        self.rules.append(IncompatibilityRule(
            name="small_bits_without_itq",
            type=IncompatibilityType.WARNING,
            factors=["n_bits", "itq_method"],
            condition="config.get('n_bits', 256) < 128 and config.get('itq_method') == 'none'",
            message="Small n_bits (<128) benefit significantly from ITQ optimization"
        ))
    
    def _load_pipeline_constraints(self) -> Dict[str, Dict[str, List[str]]]:
        """Load pipeline-specific constraints."""
        return {
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
                'svd_methods': ['truncated', 'randomized', 'randomized_adaptive'],
                'unsupported_features': []
            },
            'fused_byte': {
                'tokenizers': ['byte_bpe'],
                'backends': ['numpy', 'numba'],
                'svd_methods': ['randomized', 'randomized_adaptive'],
                'unsupported_features': ['itq_method']  # ITQ doesn't work well
            },
            'optimized_fused': {
                'tokenizers': ['char_ngram', 'byte_bpe'],
                'backends': ['numba'],  # Only Numba for optimized version
                'svd_methods': ['randomized', 'randomized_adaptive'],
                'unsupported_features': []
            }
        }
    
    def validate_configuration(
        self,
        config: Dict[str, Any],
        strict: bool = True
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate a configuration against all rules.
        
        Args:
            config: Configuration to validate
            strict: If True, soft incompatibilities also fail validation
            
        Returns:
            Tuple of (is_valid, dict_of_issues_by_type)
        """
        issues = {
            'hard': [],
            'soft': [],
            'warning': []
        }
        
        # Check general incompatibility rules
        for rule in self.rules:
            if rule.check(config):
                issues[rule.type.value].append(rule.message)
        
        # Check pipeline-specific constraints
        pipeline = config.get('pipeline_architecture')
        if pipeline and pipeline in self._pipeline_constraints:
            constraints = self._pipeline_constraints[pipeline]
            
            # Check tokenizer compatibility
            tokenizer = config.get('tokenizer')
            if tokenizer and tokenizer not in constraints['tokenizers']:
                issues['hard'].append(
                    f"Tokenizer '{tokenizer}' not compatible with {pipeline} pipeline. "
                    f"Valid options: {constraints['tokenizers']}"
                )
            
            # Check backend compatibility
            backend = config.get('backend')
            if backend and backend not in constraints['backends']:
                issues['hard'].append(
                    f"Backend '{backend}' not compatible with {pipeline} pipeline. "
                    f"Valid options: {constraints['backends']}"
                )
            
            # Check SVD method compatibility
            svd_method = config.get('svd_method')
            if svd_method and svd_method not in constraints['svd_methods']:
                issues['hard'].append(
                    f"SVD method '{svd_method}' not compatible with {pipeline} pipeline. "
                    f"Valid options: {constraints['svd_methods']}"
                )
            
            # Check unsupported features
            for feature in constraints['unsupported_features']:
                if config.get(feature):
                    issues['hard'].append(
                        f"Feature '{feature}' not supported by {pipeline} pipeline"
                    )
        
        # Determine overall validity
        is_valid = len(issues['hard']) == 0
        if strict:
            is_valid = is_valid and len(issues['soft']) == 0
        
        return is_valid, issues
    
    def fix_configuration(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attempt to fix an invalid configuration.
        
        Args:
            config: Configuration to fix
            
        Returns:
            Fixed configuration (may still have warnings)
        """
        fixed_config = config.copy()
        is_valid, issues = self.validate_configuration(fixed_config, strict=True)
        
        if is_valid:
            return fixed_config
        
        # Fix hard incompatibilities
        for issue in issues['hard']:
            if "PyTorch backend incompatible with Numba" in issue:
                fixed_config['use_numba'] = False
                warnings.warn("Disabled Numba JIT due to PyTorch backend")
            
            elif "SIMD acceleration requires bit packing" in issue:
                fixed_config['bit_packing'] = True
                warnings.warn("Enabled bit packing for SIMD acceleration")
            
            elif "Adaptive SVD requires adaptive regularization" in issue:
                fixed_config['use_adaptive_regularization'] = True
                warnings.warn("Enabled adaptive regularization for adaptive SVD")
        
        # Fix soft incompatibilities
        for issue in issues['soft']:
            if "ITQ performance degrades" in issue:
                fixed_config['itq_method'] = 'none'
                warnings.warn("Disabled ITQ due to high n_bits")
            
            elif "Randomized SVD accuracy degrades" in issue:
                fixed_config['energy_threshold'] = 0.99
                warnings.warn("Reduced energy threshold for randomized SVD")
            
            elif "ByteBPE tokenizer doesn't benefit from ITQ" in issue:
                fixed_config['itq_method'] = 'none'
                warnings.warn("Disabled ITQ for ByteBPE tokenizer")
        
        # Revalidate
        is_valid, remaining_issues = self.validate_configuration(fixed_config, strict=True)
        
        if not is_valid:
            warnings.warn(f"Could not fix all issues: {remaining_issues}")
        
        return fixed_config
    
    def filter_valid_configurations(
        self,
        configurations: List[Dict[str, Any]],
        strict: bool = True,
        auto_fix: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of configurations to only valid ones.
        
        Args:
            configurations: List of configurations to filter
            strict: If True, soft incompatibilities also fail validation
            auto_fix: If True, attempt to fix invalid configurations
            
        Returns:
            List of valid configurations
        """
        valid_configs = []
        
        for config in configurations:
            is_valid, issues = self.validate_configuration(config, strict)
            
            if is_valid:
                valid_configs.append(config)
            elif auto_fix:
                fixed_config = self.fix_configuration(config)
                is_valid, _ = self.validate_configuration(fixed_config, strict)
                if is_valid:
                    valid_configs.append(fixed_config)
                    warnings.warn(f"Auto-fixed configuration: {config} -> {fixed_config}")
        
        return valid_configs
    
    def get_compatible_values(
        self,
        factor: str,
        partial_config: Dict[str, Any]
    ) -> List[Any]:
        """
        Get compatible values for a factor given partial configuration.
        
        Args:
            factor: Factor name to get compatible values for
            partial_config: Partial configuration already set
            
        Returns:
            List of compatible values for the factor
        """
        pipeline = partial_config.get('pipeline_architecture')
        
        if not pipeline or pipeline not in self._pipeline_constraints:
            return []  # Cannot determine without pipeline
        
        constraints = self._pipeline_constraints[pipeline]
        
        if factor == 'tokenizer':
            return constraints['tokenizers']
        elif factor == 'backend':
            return constraints['backends']
        elif factor == 'svd_method':
            return constraints['svd_methods']
        else:
            return []  # No specific constraints
    
    def explain_incompatibility(
        self,
        config: Dict[str, Any]
    ) -> str:
        """
        Generate a detailed explanation of configuration issues.
        
        Args:
            config: Configuration to explain
            
        Returns:
            Human-readable explanation string
        """
        is_valid, issues = self.validate_configuration(config, strict=False)
        
        if is_valid and not issues['warning']:
            return "Configuration is valid with no issues."
        
        explanation = []
        
        if issues['hard']:
            explanation.append("CRITICAL ISSUES (will cause failures):")
            for issue in issues['hard']:
                explanation.append(f"  - {issue}")
        
        if issues['soft']:
            explanation.append("\nPERFORMANCE ISSUES (will degrade performance):")
            for issue in issues['soft']:
                explanation.append(f"  - {issue}")
        
        if issues['warning']:
            explanation.append("\nWARNINGS (not recommended):")
            for issue in issues['warning']:
                explanation.append(f"  - {issue}")
        
        return "\n".join(explanation)


# Convenience functions
def validate_config(config: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
    """Validate a configuration using default validator."""
    validator = CompatibilityValidator()
    return validator.validate_configuration(config)


def fix_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to fix an invalid configuration."""
    validator = CompatibilityValidator()
    return validator.fix_configuration(config)