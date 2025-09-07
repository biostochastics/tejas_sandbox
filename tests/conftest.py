"""
Shared Test Fixtures
====================

Common fixtures and helpers for test suite with proper dependency handling.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for optional dependencies
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available - some optimizations will be skipped")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy import sparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Register custom pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "smoke: mark test as a smoke test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_numba: skip if numba not available")
    config.addinivalue_line("markers", "requires_torch: skip if torch not available")
    config.addinivalue_line("markers", "requires_scipy: skip if scipy not available")


# Auto-skip tests based on dependencies
def pytest_runtest_setup(item):
    """Skip tests based on dependency markers."""
    for mark in item.iter_markers():
        if mark.name == 'requires_numba' and not HAS_NUMBA:
            pytest.skip("Numba not available")
        elif mark.name == 'requires_torch' and not HAS_TORCH:
            pytest.skip("PyTorch not available")
        elif mark.name == 'requires_scipy' and not HAS_SCIPY:
            pytest.skip("Scipy not available")


from core.itq_optimized import ITQConfig


@pytest.fixture
def sample_texts():
    """Standard sample texts for testing."""
    return [
        "machine learning algorithms",
        "deep learning neural networks",
        "natural language processing",
        "computer vision applications",
        "reinforcement learning agents",
    ]


@pytest.fixture
def binary_fingerprints():
    """Sample binary fingerprints."""
    np.random.seed(42)
    return np.random.randint(0, 2, (10, 128), dtype=np.uint8)


@pytest.fixture
def itq_config():
    """Standard ITQ configuration."""
    return ITQConfig(n_bits=32, max_iterations=10, random_state=42)


@pytest.fixture
def mock_database():
    """Mock fingerprint database."""
    np.random.seed(42)
    fingerprints = np.random.randint(0, 2, (100, 64), dtype=np.uint8)
    titles = [f"Document {i}" for i in range(100)]
    return fingerprints, titles


class APIAdapter:
    """Adapter for handling API changes."""

    @staticmethod
    def create_parallel_search(n_threads=4):
        """Create ParallelSearchOptimized with proper initialization."""
        from core.parallel_search import ParallelSearchOptimized

        # Create dummy data for initialization
        fingerprints = np.random.randint(0, 2, (10, 64), dtype=np.uint8)
        titles = [f"Doc {i}" for i in range(10)]
        return ParallelSearchOptimized(fingerprints, titles, n_threads=n_threads)

    @staticmethod
    def create_itq_optimizer(n_bits=32, max_iterations=10, random_state=42):
        """Create ITQOptimizer with proper config."""
        from core.itq_optimized import ITQOptimizer, ITQConfig

        config = ITQConfig(
            n_bits=n_bits, max_iterations=max_iterations, random_state=random_state
        )
        return ITQOptimizer(config)
