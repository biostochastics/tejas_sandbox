#!/usr/bin/env python3
"""
DOE Design Generators for Systematic Experimentation

This module provides design matrix generators for various experimental designs
including screening designs, optimization designs, and custom factorial designs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from itertools import product
import warnings

try:
    import pyDOE2
    HAS_PYDOE = True
except ImportError:
    HAS_PYDOE = False
    warnings.warn("pyDOE2 not installed. Some DOE designs will be unavailable.")

from .factors import Factor, FactorRegistry, FactorType


class DesignGenerator:
    """
    Generator for experimental design matrices.
    
    This class creates structured experiment designs that systematically
    explore the factor space while minimizing the number of required runs.
    """
    
    def __init__(self, registry: FactorRegistry):
        """
        Initialize the design generator.
        
        Args:
            registry: Factor registry with definitions and constraints
        """
        self.registry = registry
        self.factors = registry.factors
        
    def generate_full_factorial(self) -> pd.DataFrame:
        """
        Generate a full factorial design.
        
        Warning: This can generate a very large number of experiments!
        
        Returns:
            DataFrame with all possible factor combinations
        """
        # Get discrete levels for each factor
        factor_levels = {}
        for name, factor in self.factors.items():
            if factor.type == FactorType.CONTINUOUS:
                # Use 3 levels for continuous factors by default
                factor_levels[name] = factor.get_discrete_levels(3)
            else:
                factor_levels[name] = factor.levels
        
        # Generate all combinations
        factor_names = list(factor_levels.keys())
        level_lists = [factor_levels[name] for name in factor_names]
        
        all_combinations = list(product(*level_lists))
        
        # Create DataFrame
        design = pd.DataFrame(all_combinations, columns=factor_names)
        
        # Filter valid configurations
        valid_indices = []
        for idx, row in design.iterrows():
            config = row.to_dict()
            is_valid, _ = self.registry.validate_configuration(config)
            if is_valid:
                valid_indices.append(idx)
        
        return design.loc[valid_indices].reset_index(drop=True)
    
    def generate_fractional_factorial(
        self, 
        fraction: int = 2,
        resolution: int = 3
    ) -> pd.DataFrame:
        """
        Generate a fractional factorial design.
        
        Args:
            fraction: Fraction denominator (2^k-p where p=log2(fraction))
            resolution: Design resolution (3, 4, or 5)
            
        Returns:
            DataFrame with fractional factorial design
        """
        if not HAS_PYDOE:
            raise ImportError(
                "pyDOE2 is required for fractional factorial designs. "
                "Install with: pip install pyDOE2"
            )
        
        # Get binary and ordinal factors only
        binary_factors = [
            name for name, factor in self.factors.items()
            if factor.type == FactorType.BINARY
        ]
        
        n_factors = len(binary_factors)
        
        if n_factors < 3:
            # Too few factors for fractional design
            return self.generate_full_factorial()
        
        # Generate fractional factorial for binary factors
        if resolution == 3:
            # Resolution III design
            design_matrix = pyDOE2.fracfact(f"a b c {''.join([chr(100+i) for i in range(n_factors-3)])}")
        else:
            # Higher resolution designs
            from pyDOE2 import ff2n
            n_runs = 2 ** (n_factors - int(np.log2(fraction)))
            design_matrix = ff2n(n_factors)[:n_runs]
        
        # Convert to DataFrame with proper factor names
        design = pd.DataFrame(design_matrix, columns=binary_factors)
        
        # Convert from -1/1 to False/True
        for col in binary_factors:
            design[col] = design[col] > 0
        
        # Add default values for non-binary factors
        for name, factor in self.factors.items():
            if name not in binary_factors:
                design[name] = factor.default
        
        # Filter valid configurations
        valid_indices = []
        for idx, row in design.iterrows():
            config = row.to_dict()
            is_valid, _ = self.registry.validate_configuration(config)
            if is_valid:
                valid_indices.append(idx)
        
        return design.loc[valid_indices].reset_index(drop=True)
    
    def generate_plackett_burman(
        self,
        add_center_points: int = 0
    ) -> pd.DataFrame:
        """
        Generate a Plackett-Burman screening design.
        
        Plackett-Burman designs are very efficient for screening main effects,
        requiring only n+1 runs for n factors (rounded up to multiple of 4).
        
        Args:
            add_center_points: Number of center point replicates to add
            
        Returns:
            DataFrame with Plackett-Burman design
        """
        if not HAS_PYDOE:
            raise ImportError(
                "pyDOE2 is required for Plackett-Burman designs. "
                "Install with: pip install pyDOE2"
            )
        
        # Get all factors that can be varied
        factor_names = list(self.factors.keys())
        n_factors = len(factor_names)
        
        # Generate Plackett-Burman design
        design_matrix = pyDOE2.pbdesign(n_factors)
        
        # Convert to DataFrame
        design = pd.DataFrame(design_matrix, columns=factor_names)
        
        # Map from -1/1 to actual factor levels
        for name, factor in self.factors.items():
            if factor.type == FactorType.BINARY:
                design[name] = design[name] > 0
            elif factor.type == FactorType.CONTINUOUS:
                min_val, max_val = factor.levels
                design[name] = min_val + (design[name] + 1) / 2 * (max_val - min_val)
            elif factor.type == FactorType.ORDINAL:
                levels = factor.levels
                design[name] = design[name].apply(
                    lambda x: levels[0] if x < 0 else levels[-1]
                )
            else:  # Categorical
                levels = factor.levels
                # Use first and last levels for screening
                design[name] = design[name].apply(
                    lambda x: levels[0] if x < 0 else levels[-1]
                )
        
        # Add center points if requested
        if add_center_points > 0:
            center_config = {
                name: factor.default for name, factor in self.factors.items()
            }
            center_df = pd.DataFrame([center_config] * add_center_points)
            design = pd.concat([design, center_df], ignore_index=True)
        
        # Filter valid configurations
        valid_indices = []
        for idx, row in design.iterrows():
            config = row.to_dict()
            is_valid, _ = self.registry.validate_configuration(config)
            if is_valid:
                valid_indices.append(idx)
        
        return design.loc[valid_indices].reset_index(drop=True)
    
    def generate_central_composite(
        self,
        factors: Optional[List[str]] = None,
        alpha: str = "rotatable"
    ) -> pd.DataFrame:
        """
        Generate a Central Composite Design for response surface modeling.
        
        CCD is used for optimization after screening has identified
        important factors.
        
        Args:
            factors: List of factor names to include (default: all continuous)
            alpha: Star point distance ("rotatable", "orthogonal", or numeric)
            
        Returns:
            DataFrame with Central Composite Design
        """
        if not HAS_PYDOE:
            raise ImportError(
                "pyDOE2 is required for Central Composite designs. "
                "Install with: pip install pyDOE2"
            )
        
        # Select factors for CCD (continuous and ordinal)
        if factors is None:
            factors = [
                name for name, factor in self.factors.items()
                if factor.type in [FactorType.CONTINUOUS, FactorType.ORDINAL]
            ]
        
        n_factors = len(factors)
        
        if n_factors < 2:
            raise ValueError("CCD requires at least 2 continuous/ordinal factors")
        
        # Generate CCD
        design_matrix = pyDOE2.ccdesign(
            n_factors, 
            center=(0, 0),  # No center points in factorial, add separately
            alpha=alpha,
            face="ccc"  # Circumscribed design
        )
        
        # Add center points
        center_runs = 2 ** (n_factors - 1) if n_factors <= 5 else 6
        center_matrix = np.zeros((center_runs, n_factors))
        design_matrix = np.vstack([design_matrix, center_matrix])
        
        # Convert to DataFrame
        design = pd.DataFrame(design_matrix, columns=factors)
        
        # Map from coded units to actual factor levels
        for name in factors:
            factor = self.factors[name]
            if factor.type == FactorType.CONTINUOUS:
                min_val, max_val = factor.levels
                center = (min_val + max_val) / 2
                half_range = (max_val - min_val) / 2
                design[name] = center + design[name] * half_range
            elif factor.type == FactorType.ORDINAL:
                levels = np.array(factor.levels)
                min_level = levels.min()
                max_level = levels.max()
                center = (min_level + max_level) / 2
                half_range = (max_level - min_level) / 2
                design[name] = center + design[name] * half_range
                # Round to nearest valid level
                design[name] = design[name].apply(
                    lambda x: min(levels, key=lambda l: abs(l - x))
                )
        
        # Add default values for factors not in CCD
        for name, factor in self.factors.items():
            if name not in factors:
                design[name] = factor.default
        
        # Filter valid configurations
        valid_indices = []
        for idx, row in design.iterrows():
            config = row.to_dict()
            is_valid, _ = self.registry.validate_configuration(config)
            if is_valid:
                valid_indices.append(idx)
        
        return design.loc[valid_indices].reset_index(drop=True)
    
    def generate_box_behnken(
        self,
        factors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a Box-Behnken design for response surface modeling.
        
        Box-Behnken designs are more efficient than CCD for 3+ factors
        and don't include extreme corner points.
        
        Args:
            factors: List of factor names to include
            
        Returns:
            DataFrame with Box-Behnken design
        """
        if not HAS_PYDOE:
            raise ImportError(
                "pyDOE2 is required for Box-Behnken designs. "
                "Install with: pip install pyDOE2"
            )
        
        # Select factors for BBD
        if factors is None:
            factors = [
                name for name, factor in self.factors.items()
                if factor.type in [FactorType.CONTINUOUS, FactorType.ORDINAL]
            ]
        
        n_factors = len(factors)
        
        if n_factors < 3:
            raise ValueError("Box-Behnken requires at least 3 factors")
        
        # Generate Box-Behnken design
        design_matrix = pyDOE2.bbdesign(n_factors, center=3)
        
        # Convert to DataFrame
        design = pd.DataFrame(design_matrix, columns=factors)
        
        # Map from coded units to actual factor levels
        for name in factors:
            factor = self.factors[name]
            if factor.type == FactorType.CONTINUOUS:
                min_val, max_val = factor.levels
                center = (min_val + max_val) / 2
                half_range = (max_val - min_val) / 2
                design[name] = center + design[name] * half_range
            elif factor.type == FactorType.ORDINAL:
                levels = np.array(factor.levels)
                # Map -1, 0, 1 to low, medium, high levels
                if len(levels) >= 3:
                    mapping = {-1: levels[0], 0: levels[len(levels)//2], 1: levels[-1]}
                else:
                    mapping = {-1: levels[0], 0: levels[0], 1: levels[-1]}
                design[name] = design[name].map(mapping)
        
        # Add default values for factors not in BBD
        for name, factor in self.factors.items():
            if name not in factors:
                design[name] = factor.default
        
        # Filter valid configurations
        valid_indices = []
        for idx, row in design.iterrows():
            config = row.to_dict()
            is_valid, _ = self.registry.validate_configuration(config)
            if is_valid:
                valid_indices.append(idx)
        
        return design.loc[valid_indices].reset_index(drop=True)
    
    def generate_latin_hypercube(
        self,
        n_samples: int,
        factors: Optional[List[str]] = None,
        criterion: str = "maximin"
    ) -> pd.DataFrame:
        """
        Generate a Latin Hypercube design for space-filling.
        
        LHS designs provide good coverage of the factor space with
        fewer runs than factorial designs.
        
        Args:
            n_samples: Number of experimental runs
            factors: List of factor names to include
            criterion: Optimization criterion ("center", "maximin", "correlation")
            
        Returns:
            DataFrame with Latin Hypercube design
        """
        if not HAS_PYDOE:
            raise ImportError(
                "pyDOE2 is required for Latin Hypercube designs. "
                "Install with: pip install pyDOE2"
            )
        
        # Select factors
        if factors is None:
            factors = list(self.factors.keys())
        
        n_factors = len(factors)
        
        # Generate Latin Hypercube
        design_matrix = pyDOE2.lhs(n_factors, samples=n_samples, criterion=criterion)
        
        # Convert to DataFrame
        design = pd.DataFrame(design_matrix, columns=factors)
        
        # Map from [0, 1] to actual factor levels
        for name in factors:
            factor = self.factors[name]
            
            if factor.type == FactorType.BINARY:
                design[name] = design[name] > 0.5
            elif factor.type == FactorType.CONTINUOUS:
                min_val, max_val = factor.levels
                design[name] = min_val + design[name] * (max_val - min_val)
            elif factor.type == FactorType.ORDINAL:
                levels = factor.levels
                n_levels = len(levels)
                design[name] = design[name].apply(
                    lambda x: levels[int(x * n_levels) if x < 1 else n_levels - 1]
                )
            else:  # Categorical
                levels = factor.levels
                n_levels = len(levels)
                design[name] = design[name].apply(
                    lambda x: levels[int(x * n_levels) if x < 1 else n_levels - 1]
                )
        
        # Filter valid configurations
        valid_indices = []
        for idx, row in design.iterrows():
            config = row.to_dict()
            is_valid, _ = self.registry.validate_configuration(config)
            if is_valid:
                valid_indices.append(idx)
        
        return design.loc[valid_indices].reset_index(drop=True)
    
    def generate_custom_design(
        self,
        configurations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate a design from custom configurations.
        
        Args:
            configurations: List of factor configurations
            
        Returns:
            DataFrame with validated configurations
        """
        # Create DataFrame
        design = pd.DataFrame(configurations)
        
        # Ensure all factors are present
        for name, factor in self.factors.items():
            if name not in design.columns:
                design[name] = factor.default
        
        # Filter valid configurations
        valid_indices = []
        for idx, row in design.iterrows():
            config = row.to_dict()
            is_valid, _ = self.registry.validate_configuration(config)
            if is_valid:
                valid_indices.append(idx)
            else:
                warnings.warn(f"Configuration {idx} is invalid and will be skipped")
        
        return design.loc[valid_indices].reset_index(drop=True)
    
    def augment_design(
        self,
        design: pd.DataFrame,
        n_additional: int,
        method: str = "random"
    ) -> pd.DataFrame:
        """
        Augment an existing design with additional experiments.
        
        Useful for sequential experimentation and adaptive designs.
        
        Args:
            design: Existing design DataFrame
            n_additional: Number of experiments to add
            method: Method for generating additional points
            
        Returns:
            Augmented design DataFrame
        """
        if method == "random":
            # Random sampling within factor bounds
            new_configs = []
            
            for _ in range(n_additional * 3):  # Generate extra for filtering
                config = {}
                for name, factor in self.factors.items():
                    if factor.type == FactorType.BINARY:
                        config[name] = np.random.choice([True, False])
                    elif factor.type == FactorType.CONTINUOUS:
                        min_val, max_val = factor.levels
                        config[name] = np.random.uniform(min_val, max_val)
                    elif factor.type == FactorType.ORDINAL:
                        config[name] = np.random.choice(factor.levels)
                    else:  # Categorical
                        config[name] = np.random.choice(factor.levels)
                
                # Check if valid and not duplicate
                is_valid, _ = self.registry.validate_configuration(config)
                is_duplicate = any(
                    all(design.iloc[i][k] == v for k, v in config.items())
                    for i in range(len(design))
                )
                
                if is_valid and not is_duplicate:
                    new_configs.append(config)
                    if len(new_configs) >= n_additional:
                        break
        
        elif method == "maximin":
            # Space-filling using maximin distance criterion
            # This would require more sophisticated optimization
            raise NotImplementedError("Maximin augmentation not yet implemented")
        
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        # Create augmented design
        new_df = pd.DataFrame(new_configs)
        augmented = pd.concat([design, new_df], ignore_index=True)
        
        return augmented