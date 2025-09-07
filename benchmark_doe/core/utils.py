#!/usr/bin/env python3
"""
Defensive Programming Utilities for DOE Framework

This module provides safe operations and validation helpers to prevent
common runtime errors and improve robustness.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Optional, Union, Tuple


def safe_divide(
    numerator: Union[float, int, np.ndarray],
    denominator: Union[float, int, np.ndarray],
    default: Any = np.nan
) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers or arrays.
    
    Returns `default` (np.nan by default) if the denominator is zero or invalid.
    This preserves statistical integrity by using NaN instead of 0 for undefined
    operations.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return for invalid divisions (default: np.nan)
        
    Returns:
        Result of division or default value for invalid operations
    """
    try:
        # Handle scalar division
        if np.isscalar(denominator):
            if np.isclose(denominator, 0):
                warnings.warn(
                    f"Division by zero encountered: {numerator}/{denominator}",
                    RuntimeWarning,
                    stacklevel=2
                )
                return default
        # Handle array division
        else:
            denominator = np.asarray(denominator)
            mask = np.isclose(denominator, 0)
            if np.any(mask):
                warnings.warn(
                    f"Division by zero in array operation: {np.sum(mask)} zero values",
                    RuntimeWarning,
                    stacklevel=2
                )
                # Create result array with default values where division by zero
                result = np.full_like(denominator, default, dtype=float)
                valid_mask = ~mask
                result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
                return result
        
        # Perform normal division
        result = numerator / denominator
        
        # Check for non-finite results
        if np.isscalar(result):
            if not np.isfinite(result):
                warnings.warn(
                    f"Non-finite result from division: {numerator}/{denominator}",
                    RuntimeWarning,
                    stacklevel=2
                )
                return default
        else:
            if not np.all(np.isfinite(result)):
                warnings.warn(
                    "Non-finite results in division operation",
                    RuntimeWarning,
                    stacklevel=2
                )
                result[~np.isfinite(result)] = default
        
        return result
        
    except (TypeError, ZeroDivisionError, ValueError) as e:
        warnings.warn(
            f"Division failed with inputs: {numerator}/{denominator}: {e}",
            RuntimeWarning,
            stacklevel=2
        )
        return default


def validate_dataframe(
    df: pd.DataFrame,
    min_rows: int = 1,
    required_columns: Optional[list] = None,
    name: str = "DataFrame"
) -> Tuple[bool, str]:
    """
    Validate a pandas DataFrame for common issues.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        required_columns: List of column names that must be present
        name: Name for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, f"{name} is None"
    
    if not isinstance(df, pd.DataFrame):
        return False, f"{name} is not a pandas DataFrame"
    
    if df.empty:
        return False, f"{name} is empty"
    
    if len(df) < min_rows:
        return False, f"{name} has {len(df)} rows, minimum {min_rows} required"
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            return False, f"{name} missing required columns: {missing}"
    
    # Check for all-NaN columns which often indicate errors
    nan_columns = df.columns[df.isna().all()].tolist()
    if nan_columns:
        warnings.warn(
            f"{name} has all-NaN columns: {nan_columns}",
            UserWarning,
            stacklevel=2
        )
    
    return True, ""


def check_array_bounds(
    arr: np.ndarray,
    expected_shape: Optional[tuple] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    name: str = "Array"
) -> Tuple[bool, str]:
    """
    Check array dimensions and bounds.
    
    Args:
        arr: Array to check
        expected_shape: Expected shape tuple (use None for any dimension)
        min_size: Minimum total size
        max_size: Maximum total size
        name: Name for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(arr, np.ndarray):
        return False, f"{name} is not a numpy array"
    
    if expected_shape:
        for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
            if expected is not None and actual != expected:
                return False, (
                    f"{name} dimension {i} is {actual}, expected {expected}"
                )
    
    size = arr.size
    
    if min_size is not None and size < min_size:
        return False, f"{name} size {size} is below minimum {min_size}"
    
    if max_size is not None and size > max_size:
        return False, f"{name} size {size} exceeds maximum {max_size}"
    
    # Check for NaN or Inf values
    if np.any(~np.isfinite(arr)):
        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))
        warnings.warn(
            f"{name} contains {n_nan} NaN and {n_inf} Inf values",
            UserWarning,
            stacklevel=2
        )
    
    return True, ""


def sanitize_input(
    value: Any,
    expected_type: type,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    default: Optional[Any] = None
) -> Any:
    """
    Sanitize and validate input values.
    
    Args:
        value: Input value to sanitize
        expected_type: Expected type
        min_value: Minimum allowed value (numeric types)
        max_value: Maximum allowed value (numeric types)
        default: Default value if validation fails
        
    Returns:
        Sanitized value or default
    """
    # Handle None
    if value is None:
        return default
    
    # Type conversion
    try:
        if expected_type in (int, float):
            value = expected_type(value)
            
            # Bounds checking
            if min_value is not None and value < min_value:
                warnings.warn(
                    f"Value {value} below minimum {min_value}, using minimum",
                    UserWarning,
                    stacklevel=2
                )
                return min_value
                
            if max_value is not None and value > max_value:
                warnings.warn(
                    f"Value {value} above maximum {max_value}, using maximum",
                    UserWarning,
                    stacklevel=2
                )
                return max_value
                
        elif expected_type == str:
            value = str(value).strip()
            if not value and default is not None:
                return default
                
        elif expected_type == bool:
            if isinstance(value, str):
                value = value.lower() in ('true', '1', 'yes', 'on')
            else:
                value = bool(value)
                
        else:
            value = expected_type(value)
            
    except (ValueError, TypeError) as e:
        warnings.warn(
            f"Cannot convert {value} to {expected_type}: {e}, using default",
            UserWarning,
            stacklevel=2
        )
        return default
    
    return value


def ensure_finite(
    arr: np.ndarray,
    replace_nan: float = 0.0,
    replace_inf: float = 1e308
) -> np.ndarray:
    """
    Ensure all values in array are finite.
    
    Args:
        arr: Array to clean
        replace_nan: Value to replace NaN with
        replace_inf: Value to replace Inf with
        
    Returns:
        Array with finite values
    """
    arr = np.asarray(arr)
    
    # Replace NaN
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        warnings.warn(
            f"Replacing {np.sum(nan_mask)} NaN values with {replace_nan}",
            UserWarning,
            stacklevel=2
        )
        arr[nan_mask] = replace_nan
    
    # Replace Inf
    inf_mask = np.isinf(arr)
    if np.any(inf_mask):
        warnings.warn(
            f"Replacing {np.sum(inf_mask)} Inf values with {replace_inf}",
            UserWarning,
            stacklevel=2
        )
        arr[inf_mask] = np.sign(arr[inf_mask]) * replace_inf
    
    return arr


def log_computation_issue(
    operation: str,
    details: str,
    severity: str = "WARNING"
) -> None:
    """
    Log computational issues for debugging.
    
    Args:
        operation: Name of operation that failed
        details: Detailed error information
        severity: Log level (INFO, WARNING, ERROR)
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    if severity == "ERROR":
        logger.error(f"{operation}: {details}")
    elif severity == "WARNING":
        logger.warning(f"{operation}: {details}")
    else:
        logger.info(f"{operation}: {details}")