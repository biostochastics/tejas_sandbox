#!/usr/bin/env python3
"""
Reproducibility utilities for DOE Framework

This module provides comprehensive seed management and reproducibility features
to ensure experiments can be exactly reproduced.
"""

import os
import random
import logging
import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set random seeds for all available random number generators.
    
    This function ensures reproducibility by setting seeds for:
    - Python's random module
    - NumPy's random module
    - PyTorch (if available)
    - TensorFlow (if available)
    - Scikit-learn (if available)
    
    Args:
        seed: Random seed to use. If None, generates a seed based on current time.
        
    Returns:
        The seed that was used (useful when seed=None)
        
    Notes:
        For complete reproducibility, also set environment variable:
        PYTHONHASHSEED=0 before starting Python
    """
    # Generate seed if not provided
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000000) % 2**32
        logger.info(f"Generated random seed: {seed}")
    else:
        logger.info(f"Using provided seed: {seed}")
    
    # Track which RNGs were successfully seeded
    seeded_rngs = []
    
    # Python random module
    try:
        random.seed(seed)
        seeded_rngs.append("random")
    except Exception as e:
        logger.warning(f"Failed to seed Python random: {e}")
    
    # NumPy
    try:
        np.random.seed(seed)
        seeded_rngs.append("numpy")
    except Exception as e:
        logger.warning(f"Failed to seed NumPy: {e}")
    
    # PyTorch (optional)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seeded_rngs.append("torch")
    except ImportError:
        logger.debug("PyTorch not available, skipping")
    except Exception as e:
        logger.warning(f"Failed to seed PyTorch: {e}")
    
    # TensorFlow (optional)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        seeded_rngs.append("tensorflow")
    except ImportError:
        logger.debug("TensorFlow not available, skipping")
    except Exception as e:
        logger.warning(f"Failed to seed TensorFlow: {e}")
    
    # Scikit-learn (optional) - doesn't have global seed but uses numpy
    try:
        import sklearn
        # sklearn uses numpy's random state, already seeded above
        seeded_rngs.append("sklearn (via numpy)")
    except ImportError:
        logger.debug("Scikit-learn not available, skipping")
    
    # Log summary
    logger.info(f"Successfully seeded RNGs: {', '.join(seeded_rngs)}")
    
    # Set environment variable for hash reproducibility (for next run)
    os.environ['PYTHONHASHSEED'] = str(seed % 2**32)
    
    return seed


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get information about reproducibility settings.
    
    Returns:
        Dictionary containing:
        - python_hash_seed: Current PYTHONHASHSEED value
        - numpy_random_state: Current NumPy random state
        - available_rngs: List of available RNG modules
        - recommendations: List of recommendations for full reproducibility
    """
    info = {
        'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'not set'),
        'available_rngs': [],
        'recommendations': []
    }
    
    # Check available RNGs
    if 'random' in dir():
        info['available_rngs'].append('random')
    
    try:
        import numpy as np
        info['available_rngs'].append('numpy')
        info['numpy_version'] = np.__version__
    except ImportError:
        pass
    
    try:
        import torch
        info['available_rngs'].append('torch')
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        info['available_rngs'].append('tensorflow')
        info['tensorflow_version'] = tf.__version__
    except ImportError:
        pass
    
    # Add recommendations
    if info['python_hash_seed'] == 'not set':
        info['recommendations'].append(
            "Set PYTHONHASHSEED=0 before starting Python for hash reproducibility"
        )
    
    if 'OMP_NUM_THREADS' not in os.environ:
        info['recommendations'].append(
            "Set OMP_NUM_THREADS=1 for deterministic parallel execution"
        )
    
    return info


def create_experiment_hash(config: Dict[str, Any], seed: int) -> str:
    """
    Create a unique hash for an experiment configuration.
    
    Args:
        config: Experiment configuration dictionary
        seed: Random seed used
        
    Returns:
        Hexadecimal hash string
    """
    # Create deterministic string representation
    config_copy = config.copy()
    config_copy['seed'] = seed
    config_str = json.dumps(config_copy, sort_keys=True)
    
    # Generate hash
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class ReproducibilityContext:
    """
    Context manager for reproducible code blocks.
    
    Usage:
        with ReproducibilityContext(seed=42) as ctx:
            # Your reproducible code here
            results = run_experiment()
        print(f"Experiment hash: {ctx.hash}")
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.actual_seed = None
        self.hash = None
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.actual_seed = set_global_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            duration = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Reproducible block completed in {duration:.2f} seconds")
        return False
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration and generate hash."""
        self.hash = create_experiment_hash(config, self.actual_seed)
        return self.hash


# Module initialization logging
logger.info("Reproducibility module loaded")
logger.debug(f"Current reproducibility info: {get_reproducibility_info()}")