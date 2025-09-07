#!/usr/bin/env python3
"""
Encoder Factory for DOE Benchmark Framework
This module creates encoder instances based on pipeline architecture type,
handling all the configuration nuances and import requirements.
"""

import importlib
import sys
import os
from typing import Dict, Any, Optional
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class EncoderFactory:
    """
    Factory for creating encoder instances based on pipeline type.
    
    This factory handles the complexity of different encoder implementations,
    their specific requirements, and configuration parameters.
    """
    
    ENCODER_CONFIGS = {
        'original_tejas': {
            'module': 'core.encoder',
            'class': 'GoldenRatioEncoder',  # Uses golden ratio subsampling + sklearn
            'defaults': {
                'use_sklearn': True,
                'n_bits': 256,
                'energy_threshold': 0.95,
                'expansion_strategy': 'cyclic',
                'use_golden_ratio_sampling': True  # This is the key feature
            },
            'requires': ['sklearn']
        },
        'fused_char': {
            'module': 'core.fused_encoder_v2',
            'class': 'FusedPipelineEncoder',
            'defaults': {
                'n_bits': 256,
                'tokenizer_type': 'char_ngram',
                'ngram_range': (3, 5),
                'max_features': 10000,
                'use_itq': False,
                'energy_threshold': 0.95,
                'expansion_strategy': 'cyclic',
                'batch_size': 10000
            },
            'requires': []
        },
        'fused_byte': {
            'module': 'core.fused_encoder_v2_bytebpe',
            'class': 'FusedPipelineEncoder',
            'defaults': {
                'n_bits': 256,
                'tokenizer_type': 'byte_bpe',
                'vocab_size': 1000,
                'use_itq': False,  # ByteBPE typically doesn't work well with ITQ
                'energy_threshold': 0.95,
                'expansion_strategy': 'cyclic',
                'batch_size': 10000
            },
            'requires': []
        },
        'optimized_fused': {
            'module': 'core.fused_encoder_v2_optimized',
            'class': 'OptimizedFusedEncoder',
            'defaults': {
                'n_bits': 256,
                'tokenizer_type': 'char_ngram',
                'use_simd': True,
                'use_numba': True,
                'use_itq': False,
                'energy_threshold': 0.95,
                'expansion_strategy': 'cyclic',
                'batch_size': 10000
            },
            'requires': ['numba']
        },
        'optimized_fused_nopack': {
            'module': 'core.fused_encoder_nopack',
            'class': 'OptimizedFusedEncoderNoPack',
            'defaults': {
                'n_bits': 256,
                'tokenizer': 'char',
                'ngram_range': (3, 5),
                'max_features': 10000,
                'energy_threshold': 0.95,
                'use_cache': True,
                'cache_size': 10000,
                'batch_size': 512
            },
            'requires': []
        },
        # BERT baselines for comparison
        'bert_mini': {
            'module': 'benchmark_doe.core.bert_encoder',
            'function': 'create_bert_encoder',  # Uses function instead of class
            'defaults': {
                'model_name': 'all-MiniLM-L6-v2',
                'device': 'cpu',
                'batch_size': 32,
                'pipeline_type': 'bert_mini'
            },
            'requires': ['sentence_transformers']
        },
        'bert_base': {
            'module': 'benchmark_doe.core.bert_encoder',
            'function': 'create_bert_encoder',  # Uses function instead of class
            'defaults': {
                'model_name': 'all-mpnet-base-v2',
                'device': 'cpu',
                'batch_size': 32,
                'pipeline_type': 'bert_base'
            },
            'requires': ['sentence_transformers']
        }
    }
    
    @classmethod
    def create_encoder(
        cls,
        pipeline_type: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Create an encoder instance for the specified pipeline type.
        
        Args:
            pipeline_type: Type of pipeline architecture
            config: Optional configuration overrides
            
        Returns:
            Encoder instance configured and ready to use
            
        Raises:
            ValueError: If pipeline type is unknown
            ImportError: If required dependencies are missing
        """
        if pipeline_type not in cls.ENCODER_CONFIGS:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}. "
                           f"Valid options: {list(cls.ENCODER_CONFIGS.keys())}")
        
        encoder_info = cls.ENCODER_CONFIGS[pipeline_type]
        
        # Check required dependencies
        cls._check_dependencies(encoder_info.get('requires', []))
        
        # Merge default config with user config
        final_config = encoder_info['defaults'].copy()
        if config:
            # Apply optimization flags
            cls._apply_optimization_flags(final_config, config)
            # Apply other configurations
            final_config.update(config)
        
        # Check if this is a function-based encoder (like BERT)
        if 'function' in encoder_info:
            # Import and call the function
            if pipeline_type in ['bert_mini', 'bert_base']:
                from benchmark_doe.core.bert_encoder import create_bert_encoder
                encoder = create_bert_encoder(pipeline_type, final_config)
            else:
                raise ImportError(f"Unknown function-based encoder: {pipeline_type}")
        else:
            # Get encoder class from registry (safer than dynamic import)
            encoder_class = cls._get_encoder_class(encoder_info)
            if encoder_class is None:
                raise ImportError(f"Failed to load encoder {pipeline_type}")
            
            # Special handling for different encoder types
            encoder = cls._create_encoder_instance(
                pipeline_type,
                encoder_class,
                final_config
            )
        
        return encoder
    
    @classmethod
    def _get_encoder_class(cls, encoder_info: Dict[str, Any]):
        """
        Safely get encoder class from pre-imported modules.
        
        This avoids dynamic imports by using a registry of known encoders.
        """
        # Pre-import known encoder classes (safer than dynamic import)
        encoder_registry = {}
        
        try:
            from core.encoder import GoldenRatioEncoder
            encoder_registry['core.encoder.GoldenRatioEncoder'] = GoldenRatioEncoder
        except ImportError:
            pass
        
        try:
            from core.fused_encoder_v2 import FusedPipelineEncoder
            encoder_registry['core.fused_encoder_v2.FusedPipelineEncoder'] = FusedPipelineEncoder
        except ImportError:
            pass
        
        try:
            from core.fused_encoder_v2_bytebpe import FusedPipelineEncoder as ByteBPEEncoder
            encoder_registry['core.fused_encoder_v2_bytebpe.FusedPipelineEncoder'] = ByteBPEEncoder
        except ImportError:
            pass
        
        try:
            from core.fused_encoder_v2_optimized import OptimizedFusedEncoder
            encoder_registry['core.fused_encoder_v2_optimized.OptimizedFusedEncoder'] = OptimizedFusedEncoder
        except ImportError:
            pass
        
        # Construct registry key
        key = f"{encoder_info['module']}.{encoder_info['class']}"
        
        return encoder_registry.get(key)
    
    @classmethod
    def _check_dependencies(cls, required: list):
        """Check if required dependencies are available."""
        for dep in required:
            try:
                importlib.import_module(dep)
            except ImportError:
                warnings.warn(f"Optional dependency '{dep}' not available. "
                            f"Some features may be limited.")
    
    @classmethod
    def _apply_optimization_flags(
        cls,
        config: Dict[str, Any],
        optimization_config: Dict[str, Any]
    ):
        """
        Apply optimization flags to encoder configuration.
        
        Maps DOE optimization flags to encoder-specific parameters.
        """
        flag_mapping = {
            'use_fast_cache': 'enable_cache',
            'use_safe_entropy': 'safe_entropy',
            'use_fast_hamming': 'fast_hamming',
            'use_simd': 'use_simd',
            'use_memory_profiler': 'profile_memory',
            'use_adaptive_regularization': 'adaptive_regularization',
            'bit_packing': 'pack_bits'
        }
        
        for doe_flag, encoder_param in flag_mapping.items():
            if doe_flag in optimization_config:
                config[encoder_param] = optimization_config[doe_flag]
        
        # Handle ITQ configuration
        if 'itq_method' in optimization_config:
            itq_method = optimization_config['itq_method']
            if itq_method == 'none':
                config['use_itq'] = False
            else:
                config['use_itq'] = True
                config['itq_variant'] = itq_method
    
    @classmethod
    def _create_encoder_instance(
        cls,
        pipeline_type: str,
        encoder_class: type,
        config: Dict[str, Any]
    ):
        """
        Create encoder instance with pipeline-specific handling.
        
        Args:
            pipeline_type: Type of pipeline
            encoder_class: Encoder class to instantiate
            config: Configuration dictionary
            
        Returns:
            Configured encoder instance
        """
        # Remove non-constructor parameters
        constructor_params = cls._filter_constructor_params(
            encoder_class,
            config
        )
        
        # Create instance
        encoder = encoder_class(**constructor_params)
        
        # Apply post-construction configuration
        cls._configure_encoder_post_init(encoder, pipeline_type, config)
        
        return encoder
    
    @classmethod
    def _filter_constructor_params(
        cls,
        encoder_class: type,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter configuration to only include valid constructor parameters.
        
        Args:
            encoder_class: Encoder class
            config: Full configuration dictionary
            
        Returns:
            Filtered configuration for constructor
        """
        # Get class name to determine which parameters to use
        class_name = encoder_class.__name__
        
        # Define encoder-specific parameters
        if class_name == 'GoldenRatioEncoder':
            # GoldenRatioEncoder only accepts these parameters
            valid_params = {'n_bits', 'max_features', 'device'}
        elif class_name == 'FusedPipelineEncoder':
            # FusedPipelineEncoder actual parameters from __init__
            valid_params = {
                'n_bits', 'max_features', 'use_itq', 'n_iterations_itq',
                'batch_size', 'energy_threshold', 'center_data',
                'expansion_strategy', 'dtype', 'random_state',
                'svd_n_oversamples', 'svd_n_iter'
            }
        elif class_name == 'OptimizedFusedEncoder':
            # OptimizedFusedEncoder parameters
            valid_params = {
                'n_bits', 'energy_threshold', 'expansion_strategy',
                'use_itq', 'batch_size', 'max_features', 'random_state',
                'verbose', 'n_jobs', 'memory_limit_gb', 'vocab_size'
            }
        else:
            # Default set of common parameters
            valid_params = {
                'n_bits', 'energy_threshold', 'expansion_strategy',
                'use_itq', 'batch_size', 'max_features', 'random_state',
                'verbose', 'n_jobs', 'memory_limit_gb', 'device',
                'vocab_size'
            }
        
        # Filter to valid parameters for this encoder
        constructor_params = {}
        for key, value in config.items():
            if key in valid_params:
                constructor_params[key] = value
        
        return constructor_params
    
    @classmethod
    def _configure_encoder_post_init(
        cls,
        encoder: Any,
        pipeline_type: str,
        config: Dict[str, Any]
    ):
        """
        Apply post-initialization configuration to encoder.
        
        Args:
            encoder: Encoder instance
            pipeline_type: Type of pipeline
            config: Configuration dictionary
        """
        # Set optimization flags as attributes if encoder supports them
        optimization_attrs = [
            'enable_cache', 'safe_entropy', 'fast_hamming',
            'use_simd', 'profile_memory', 'adaptive_regularization'
        ]
        
        for attr in optimization_attrs:
            if attr in config and hasattr(encoder, attr):
                setattr(encoder, attr, config[attr])
        
        # Pipeline-specific configuration
        if pipeline_type == 'optimized_fused' and hasattr(encoder, 'enable_optimizations'):
            encoder.enable_optimizations()
        
        if pipeline_type == 'fused_byte' and hasattr(encoder, 'configure_byte_bpe'):
            encoder.configure_byte_bpe(vocab_size=config.get('vocab_size', 1000))
    
    @classmethod
    def get_encoder_info(cls, pipeline_type: str) -> Dict[str, Any]:
        """
        Get information about an encoder type.
        
        Args:
            pipeline_type: Type of pipeline
            
        Returns:
            Dictionary with encoder information
        """
        if pipeline_type not in cls.ENCODER_CONFIGS:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        return cls.ENCODER_CONFIGS[pipeline_type].copy()
    
    @classmethod
    def list_available_encoders(cls) -> list:
        """
        List all available encoder types.
        
        Returns:
            List of available encoder type names
        """
        return list(cls.ENCODER_CONFIGS.keys())
    
    @classmethod
    def validate_encoder_config(
        cls,
        pipeline_type: str,
        config: Dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate encoder configuration.
        
        Args:
            pipeline_type: Type of pipeline
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if pipeline_type not in cls.ENCODER_CONFIGS:
            issues.append(f"Unknown pipeline type: {pipeline_type}")
            return False, issues
        
        encoder_info = cls.ENCODER_CONFIGS[pipeline_type]
        
        # Check incompatible settings
        if pipeline_type == 'fused_byte' and config.get('use_itq', False):
            issues.append("ByteBPE encoder doesn't work well with ITQ")
        
        if pipeline_type == 'original_tejas' and config.get('use_simd', False):
            issues.append("Original TEJAS doesn't support SIMD acceleration")
        
        if config.get('backend') == 'pytorch' and config.get('use_numba', False):
            issues.append("Cannot use PyTorch backend with Numba JIT")
        
        return len(issues) == 0, issues


# Convenience function
def create_encoder(pipeline_type: str, **kwargs):
    """
    Convenience function to create an encoder.
    
    Args:
        pipeline_type: Type of pipeline architecture
        **kwargs: Configuration parameters
        
    Returns:
        Configured encoder instance
    """
    return EncoderFactory.create_encoder(pipeline_type, kwargs)