#!/usr/bin/env python3
"""
Input Validation Module for DOE Framework

This module provides comprehensive input validation to prevent
invalid data from causing runtime errors or security issues.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class FactorValidator:
    """Validator for experimental factors."""
    
    @staticmethod
    def validate_continuous(
        value: float,
        min_val: float,
        max_val: float,
        name: str = "value"
    ) -> float:
        """
        Validate a continuous factor value.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error messages
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be numeric, got {type(value)}")
        
        if not np.isfinite(value):
            raise ValidationError(f"{name} must be finite, got {value}")
        
        if value < min_val:
            warnings.warn(
                f"{name} {value} below minimum {min_val}, clamping",
                UserWarning
            )
            return min_val
        
        if value > max_val:
            warnings.warn(
                f"{name} {value} above maximum {max_val}, clamping",
                UserWarning
            )
            return max_val
        
        return value
    
    @staticmethod
    def validate_ordinal(
        value: Any,
        levels: List[Any],
        name: str = "value"
    ) -> Any:
        """
        Validate an ordinal factor value.
        
        Args:
            value: Value to validate
            levels: Allowed levels
            name: Name for error messages
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If value not in levels
        """
        if value not in levels:
            raise ValidationError(
                f"{name} must be one of {levels}, got {value}"
            )
        return value
    
    @staticmethod
    def validate_binary(
        value: Any,
        name: str = "value"
    ) -> bool:
        """
        Validate a binary factor value.
        
        Args:
            value: Value to validate
            name: Name for error messages
            
        Returns:
            Boolean value
        """
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        
        try:
            return bool(value)
        except:
            raise ValidationError(
                f"{name} must be convertible to boolean, got {value}"
            )
    
    @staticmethod
    def validate_categorical(
        value: Any,
        categories: List[str],
        name: str = "value"
    ) -> str:
        """
        Validate a categorical factor value.
        
        Args:
            value: Value to validate
            categories: Allowed categories
            name: Name for error messages
            
        Returns:
            Validated category
            
        Raises:
            ValidationError: If value not in categories
        """
        value = str(value)
        if value not in categories:
            # Try case-insensitive match
            for cat in categories:
                if value.lower() == cat.lower():
                    return cat
            
            raise ValidationError(
                f"{name} must be one of {categories}, got {value}"
            )
        return value


class DataValidator:
    """Validator for data structures."""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        min_rows: int = 1,
        required_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        name: str = "DataFrame"
    ) -> pd.DataFrame:
        """
        Validate a pandas DataFrame comprehensively.
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum required rows
            required_columns: Columns that must exist
            numeric_columns: Columns that must be numeric
            name: Name for error messages
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValidationError: If validation fails
        """
        if df is None:
            raise ValidationError(f"{name} cannot be None")
        
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(
                f"{name} must be a pandas DataFrame, got {type(df)}"
            )
        
        if df.empty:
            raise ValidationError(f"{name} is empty")
        
        if len(df) < min_rows:
            raise ValidationError(
                f"{name} has {len(df)} rows, minimum {min_rows} required"
            )
        
        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValidationError(
                    f"{name} missing required columns: {missing}"
                )
        
        # Check numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        # Try to convert
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except:
                            raise ValidationError(
                                f"Column {col} must be numeric"
                            )
                    
                    # Check for all NaN
                    if df[col].isna().all():
                        raise ValidationError(
                            f"Column {col} contains only NaN values"
                        )
        
        return df
    
    @staticmethod
    def validate_array(
        arr: np.ndarray,
        expected_dims: Optional[int] = None,
        expected_shape: Optional[tuple] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        dtype: Optional[type] = None,
        name: str = "Array"
    ) -> np.ndarray:
        """
        Validate a numpy array comprehensively.
        
        Args:
            arr: Array to validate
            expected_dims: Expected number of dimensions
            expected_shape: Expected shape (None for any)
            min_size: Minimum total size
            max_size: Maximum total size
            dtype: Expected data type
            name: Name for error messages
            
        Returns:
            Validated array
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(arr, np.ndarray):
            try:
                arr = np.asarray(arr)
            except:
                raise ValidationError(
                    f"{name} must be convertible to numpy array"
                )
        
        # Check dimensions
        if expected_dims is not None:
            if arr.ndim != expected_dims:
                raise ValidationError(
                    f"{name} has {arr.ndim} dimensions, expected {expected_dims}"
                )
        
        # Check shape
        if expected_shape is not None:
            for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
                if expected is not None and actual != expected:
                    raise ValidationError(
                        f"{name} dimension {i} is {actual}, expected {expected}"
                    )
        
        # Check size
        if min_size is not None and arr.size < min_size:
            raise ValidationError(
                f"{name} size {arr.size} below minimum {min_size}"
            )
        
        if max_size is not None and arr.size > max_size:
            raise ValidationError(
                f"{name} size {arr.size} exceeds maximum {max_size}"
            )
        
        # Check dtype
        if dtype is not None:
            if not np.issubdtype(arr.dtype, dtype):
                try:
                    arr = arr.astype(dtype)
                except:
                    raise ValidationError(
                        f"{name} cannot be converted to {dtype}"
                    )
        
        # Check for non-finite values
        if np.issubdtype(arr.dtype, np.floating):
            n_nan = np.sum(np.isnan(arr))
            n_inf = np.sum(np.isinf(arr))
            
            if n_nan > 0 or n_inf > 0:
                warnings.warn(
                    f"{name} contains {n_nan} NaN and {n_inf} Inf values",
                    UserWarning
                )
        
        return arr


class ConfigurationValidator:
    """Validator for experiment configurations."""
    
    @staticmethod
    def validate_experiment_config(
        config: Dict[str, Any],
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate an experiment configuration.
        
        Args:
            config: Configuration dictionary
            required_fields: Fields that must be present
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(config, dict):
            raise ValidationError(
                f"Configuration must be a dictionary, got {type(config)}"
            )
        
        # Check required fields
        if required_fields:
            missing = set(required_fields) - set(config.keys())
            if missing:
                raise ValidationError(
                    f"Configuration missing required fields: {missing}"
                )
        
        # Validate specific fields
        validated = config.copy()
        
        # Validate n_bits
        if 'n_bits' in validated:
            validated['n_bits'] = FactorValidator.validate_continuous(
                validated['n_bits'],
                min_val=32,
                max_val=2048,
                name="n_bits"
            )
            
            # Must be divisible by 32 for bit packing
            if validated.get('bit_packing', True):
                if int(validated['n_bits']) % 32 != 0:
                    warnings.warn(
                        f"n_bits {validated['n_bits']} not divisible by 32, "
                        "adjusting for bit packing",
                        UserWarning
                    )
                    validated['n_bits'] = int(validated['n_bits'] / 32) * 32
        
        # Validate batch_size
        if 'batch_size' in validated:
            validated['batch_size'] = int(
                FactorValidator.validate_continuous(
                    validated['batch_size'],
                    min_val=1,
                    max_val=1000000,
                    name="batch_size"
                )
            )
        
        # Validate memory and timeout limits
        if 'memory_limit_mb' in validated:
            validated['memory_limit_mb'] = int(
                FactorValidator.validate_continuous(
                    validated['memory_limit_mb'],
                    min_val=100,
                    max_val=32768,  # 32GB max
                    name="memory_limit_mb"
                )
            )
        
        if 'timeout' in validated:
            validated['timeout'] = int(
                FactorValidator.validate_continuous(
                    validated['timeout'],
                    min_val=1,
                    max_val=3600,  # 1 hour max
                    name="timeout"
                )
            )
        
        return validated
    
    @staticmethod
    def validate_pipeline_config(
        pipeline_type: str,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate pipeline-specific configuration.
        
        Args:
            pipeline_type: Type of pipeline
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Pipeline-specific validation
        if pipeline_type == 'fused_byte' and config.get('use_itq', False):
            issues.append("ByteBPE pipeline doesn't work well with ITQ")
        
        if pipeline_type == 'original_tejas' and config.get('use_simd', False):
            issues.append("Original TEJAS doesn't support SIMD acceleration")
        
        if config.get('backend') == 'pytorch' and config.get('use_numba', False):
            issues.append("Cannot use PyTorch backend with Numba JIT")
        
        if config.get('use_simd', False) and not config.get('bit_packing', True):
            issues.append("SIMD acceleration requires bit packing")
        
        # Check for incompatible ITQ settings
        if config.get('itq_variant') and config.get('itq_variant') != 'none':
            if not config.get('use_itq', False):
                issues.append(f"ITQ variant '{config['itq_variant']}' requires use_itq=True")
        
        return len(issues) == 0, issues