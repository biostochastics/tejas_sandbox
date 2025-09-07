"""
Unified Backend Manager for Tejas
==================================

Centralized backend selection and management system that addresses:
- Fragmented backend architecture
- Performance overhead from repeated selection
- Resource conflicts between modules
- Backward compatibility issues
- Memory management and cleanup

Features:
- Singleton pattern for global backend coordination
- Performance profiling and caching
- Resource semaphores for thread/process management
- Consistent fallback chains with warnings
- Runtime benchmarking and adaptive selection
"""

import sys
import time
import json
import logging
import warnings
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Check for optional backends
HAS_NUMBA = False
HAS_TORCH = False
HAS_CUPY = False

try:
    import numba
    from numba import config as numba_config

    HAS_NUMBA = True
except ImportError:
    pass

try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass

try:
    import cupy as cp

    HAS_CUPY = cp.cuda.is_available() if "cp" in locals() else False
except ImportError:
    pass


@dataclass
class BackendCapabilities:
    """Stores capabilities and performance metrics for a backend."""

    name: str
    available: bool
    device_type: str  # 'cpu', 'cuda', 'mps'
    max_threads: int
    max_memory_gb: float
    supports_float64: bool
    supports_parallel: bool
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


@dataclass
class BackendConfig:
    """Configuration for backend selection and management."""

    preferred_order: List[str] = field(
        default_factory=lambda: ["torch_gpu", "numba", "numpy"]
    )
    auto_benchmark: bool = True
    cache_selection: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    warn_on_fallback: bool = True
    max_concurrent_processes: int = 8
    max_concurrent_threads: int = 16
    memory_limit_gb: float = 8.0
    profile_operations: bool = True
    adaptive_selection: bool = True
    min_size_for_gpu: int = 10000
    min_size_for_parallel: int = 1000


class UnifiedBackendManager:
    """
    Singleton backend manager that provides centralized backend selection
    and resource management for all Tejas modules.
    """

    _instance = None
    _lock = threading.Lock()

    # Class-level resource semaphores
    _process_semaphore = None
    _thread_semaphore = None
    _gpu_semaphore = None

    # Performance cache
    _performance_cache: Dict[str, Dict] = {}
    _selection_cache: Dict[str, Tuple[str, float]] = {}

    def __new__(cls):
        """Enforce singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the backend manager (only once)."""
        if self._initialized:
            return

        self.config = BackendConfig()
        self.backends: Dict[str, BackendCapabilities] = {}

        # Initialize available backends
        self._discover_backends()

        # Initialize resource semaphores
        self._init_resource_limits()

        # Load configuration if exists
        self._load_config()

        # Run initial benchmarks if configured
        if self.config.auto_benchmark:
            self._run_benchmarks()

        self._initialized = True
        logger.info("UnifiedBackendManager initialized")

    def _discover_backends(self):
        """Discover and register available backends."""
        # NumPy (always available)
        self.backends["numpy"] = BackendCapabilities(
            name="numpy",
            available=True,
            device_type="cpu",
            max_threads=mp.cpu_count(),
            max_memory_gb=self._get_available_memory(),
            supports_float64=True,
            supports_parallel=False,
        )

        # Numba
        if HAS_NUMBA:
            self.backends["numba"] = BackendCapabilities(
                name="numba",
                available=True,
                device_type="cpu",
                max_threads=mp.cpu_count(),
                max_memory_gb=self._get_available_memory(),
                supports_float64=True,
                supports_parallel=True,
            )
            logger.info("Numba backend registered")

        # PyTorch (CPU)
        if HAS_TORCH:
            self.backends["torch_cpu"] = BackendCapabilities(
                name="torch_cpu",
                available=True,
                device_type="cpu",
                max_threads=torch.get_num_threads() if HAS_TORCH else 1,
                max_memory_gb=self._get_available_memory(),
                supports_float64=True,
                supports_parallel=True,
            )

            # PyTorch (CUDA)
            if torch.cuda.is_available():
                self.backends["torch_cuda"] = BackendCapabilities(
                    name="torch_cuda",
                    available=True,
                    device_type="cuda",
                    max_threads=1,  # GPU doesn't use CPU threads
                    max_memory_gb=torch.cuda.get_device_properties(0).total_memory
                    / 1e9,
                    supports_float64=True,
                    supports_parallel=True,
                )
                logger.info(f"CUDA backend registered: {torch.cuda.get_device_name(0)}")

            # PyTorch (MPS - Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.backends["torch_mps"] = BackendCapabilities(
                    name="torch_mps",
                    available=True,
                    device_type="mps",
                    max_threads=1,
                    max_memory_gb=self._get_available_memory(),
                    supports_float64=False,  # MPS has limited float64 support
                    supports_parallel=True,
                )
                logger.info("MPS (Apple Silicon) backend registered")

        # CuPy
        if HAS_CUPY:
            self.backends["cupy"] = BackendCapabilities(
                name="cupy",
                available=True,
                device_type="cuda",
                max_threads=1,
                max_memory_gb=cp.cuda.MemoryPool().get_limit() / 1e9 if HAS_CUPY else 0,
                supports_float64=True,
                supports_parallel=True,
            )
            logger.info("CuPy backend registered")

        # Native Python (for bitops)
        python_version = sys.version_info
        if python_version >= (3, 10):
            self.backends["native"] = BackendCapabilities(
                name="native",
                available=True,
                device_type="cpu",
                max_threads=1,  # Python GIL limits to single thread
                max_memory_gb=self._get_available_memory(),
                supports_float64=True,
                supports_parallel=False,
            )

    def _init_resource_limits(self):
        """Initialize resource semaphores for thread/process management."""
        self._process_semaphore = threading.Semaphore(
            self.config.max_concurrent_processes
        )
        self._thread_semaphore = threading.Semaphore(self.config.max_concurrent_threads)
        self._gpu_semaphore = threading.Semaphore(
            1
        )  # Usually want exclusive GPU access

        logger.info(
            f"Resource limits: {self.config.max_concurrent_processes} processes, "
            f"{self.config.max_concurrent_threads} threads"
        )

    def _load_config(self):
        """Load configuration from file if exists."""
        config_paths = [
            Path.home() / ".tejas" / "backend_config.json",
            Path.cwd() / "backend_config.json",
            Path(__file__).parent.parent / "backend_config.json",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)

                    # Update config with loaded values
                    for key, value in config_dict.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)

                    logger.info(f"Loaded backend configuration from {config_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")

    def _get_available_memory(self) -> float:
        """Get available system memory in GB."""
        try:
            import psutil

            return psutil.virtual_memory().available / 1e9
        except ImportError:
            # Fallback estimate
            return 8.0

    def _run_benchmarks(self):
        """Run performance benchmarks for each backend."""
        logger.info("Running backend benchmarks...")

        # Test data for benchmarking
        sizes = [1000, 10000, 100000]

        for backend_name, backend in self.backends.items():
            if not backend.available:
                continue

            scores = {}

            for size in sizes:
                try:
                    # Benchmark matrix operations
                    score = self._benchmark_backend(backend_name, size)
                    scores[f"matrix_{size}"] = score
                except Exception as e:
                    logger.debug(
                        f"Benchmark failed for {backend_name} with size {size}: {e}"
                    )
                    scores[f"matrix_{size}"] = float("inf")

            backend.benchmark_scores = scores
            backend.last_updated = time.time()

        logger.info("Backend benchmarks completed")

    def _benchmark_backend(self, backend_name: str, size: int) -> float:
        """Benchmark a specific backend with given data size."""
        # Create test data
        data = np.random.randn(size, 128).astype(np.float32)

        if backend_name == "numpy":
            start = time.perf_counter()
            _ = np.dot(data.T, data)
            return time.perf_counter() - start

        elif backend_name == "numba" and HAS_NUMBA:

            @numba.jit(nopython=True)
            def numba_dot(a, b):
                return np.dot(a, b)

            # Warmup
            _ = numba_dot(data[:100].T, data[:100])

            start = time.perf_counter()
            _ = numba_dot(data.T, data)
            return time.perf_counter() - start

        elif backend_name.startswith("torch") and HAS_TORCH:
            device = backend_name.split("_")[1] if "_" in backend_name else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                return float("inf")

            data_torch = torch.from_numpy(data).to(device)

            # Warmup
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = torch.mm(data_torch.T, data_torch)
            if device == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - start

        return float("inf")

    def select_backend(
        self,
        operation: str,
        data_size: int,
        data_type: np.dtype = np.float32,
        prefer_gpu: bool = False,
        require_parallel: bool = False,
        memory_estimate_gb: float = 0.0,
    ) -> str:
        """
        Select optimal backend for given operation.

        Args:
            operation: Type of operation ('hamming', 'svd', 'matmul', etc.)
            data_size: Size of data to process
            data_type: Data type for computation
            prefer_gpu: Whether to prefer GPU backends
            require_parallel: Whether parallel execution is required
            memory_estimate_gb: Estimated memory requirement

        Returns:
            Selected backend name
        """
        # Check cache first
        if self.config.cache_selection:
            cache_key = (
                f"{operation}_{data_size}_{data_type}_{prefer_gpu}_{require_parallel}"
            )
            if cache_key in self._selection_cache:
                backend, timestamp = self._selection_cache[cache_key]
                if time.time() - timestamp < self.config.cache_ttl_seconds:
                    logger.debug(f"Using cached backend selection: {backend}")
                    return backend

        # Filter available backends
        candidates = []

        for name, backend in self.backends.items():
            if not backend.available:
                continue

            # Check memory constraints
            if memory_estimate_gb > 0 and memory_estimate_gb > backend.max_memory_gb:
                continue

            # Check parallel requirement
            if require_parallel and not backend.supports_parallel:
                continue

            # Check data type support
            if data_type == np.float64 and not backend.supports_float64:
                continue

            # Check size thresholds
            if prefer_gpu and backend.device_type in ["cuda", "mps"]:
                if data_size < self.config.min_size_for_gpu:
                    continue

            candidates.append(name)

        if not candidates:
            # Fallback to numpy
            if self.config.warn_on_fallback:
                warnings.warn(
                    "No suitable backend found, falling back to numpy", RuntimeWarning
                )
            return "numpy"

        # Select based on preference and benchmarks
        selected = self._select_from_candidates(
            candidates, operation, data_size, prefer_gpu
        )

        # Cache the selection
        if self.config.cache_selection:
            self._selection_cache[cache_key] = (selected, time.time())

        # Log selection
        logger.debug(
            f"Selected backend '{selected}' for {operation} with size {data_size}"
        )

        return selected

    def _select_from_candidates(
        self, candidates: List[str], operation: str, data_size: int, prefer_gpu: bool
    ) -> str:
        """Select best backend from candidates based on benchmarks and preferences."""
        # If adaptive selection is enabled and we have benchmark data
        if self.config.adaptive_selection:
            scored_candidates = []

            for candidate in candidates:
                backend = self.backends[candidate]

                # Find closest benchmark size
                benchmark_key = None
                min_diff = float("inf")

                for key in backend.benchmark_scores:
                    if "matrix" in key:
                        bench_size = int(key.split("_")[1])
                        diff = abs(bench_size - data_size)
                        if diff < min_diff:
                            min_diff = diff
                            benchmark_key = key

                if benchmark_key:
                    score = backend.benchmark_scores[benchmark_key]

                    # Apply preferences
                    if prefer_gpu and backend.device_type in ["cuda", "mps"]:
                        score *= 0.8  # 20% bonus for GPU when preferred

                    scored_candidates.append((candidate, score))

            if scored_candidates:
                # Sort by score (lower is better)
                scored_candidates.sort(key=lambda x: x[1])
                return scored_candidates[0][0]

        # Fall back to preference order
        for preferred in self.config.preferred_order:
            if preferred in candidates:
                return preferred

        # Return first available candidate
        return candidates[0]

    def acquire_resources(self, backend: str, operation: str) -> "ResourceContext":
        """
        Acquire resources for backend operation.

        Returns a context manager that ensures proper resource cleanup.
        """
        return ResourceContext(self, backend, operation)

    def report_performance(
        self,
        backend: str,
        operation: str,
        data_size: int,
        elapsed_time: float,
        success: bool = True,
    ):
        """
        Report performance metrics for adaptive selection.

        Args:
            backend: Backend used
            operation: Operation performed
            data_size: Size of data processed
            elapsed_time: Time taken in seconds
            success: Whether operation succeeded
        """
        if not self.config.profile_operations:
            return

        # Update performance cache
        cache_key = f"{backend}_{operation}_{data_size}"

        if cache_key not in self._performance_cache:
            self._performance_cache[cache_key] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "failures": 0,
            }

        entry = self._performance_cache[cache_key]
        entry["count"] += 1

        if success:
            entry["total_time"] += elapsed_time
            entry["min_time"] = min(entry["min_time"], elapsed_time)
            entry["max_time"] = max(entry["max_time"], elapsed_time)
        else:
            entry["failures"] += 1

        # Update backend benchmark scores if significant data
        if entry["count"] >= 10:
            avg_time = entry["total_time"] / entry["count"]
            backend_obj = self.backends.get(backend)
            if backend_obj:
                benchmark_key = f"{operation}_{data_size}"
                backend_obj.benchmark_scores[benchmark_key] = avg_time
                backend_obj.last_updated = time.time()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of backend manager."""
        return {
            "backends": {
                name: {
                    "available": backend.available,
                    "device_type": backend.device_type,
                    "benchmark_scores": backend.benchmark_scores,
                }
                for name, backend in self.backends.items()
            },
            "config": {
                "preferred_order": self.config.preferred_order,
                "auto_benchmark": self.config.auto_benchmark,
                "cache_selection": self.config.cache_selection,
            },
            "cache_size": len(self._selection_cache),
            "performance_entries": len(self._performance_cache),
        }

    def clear_cache(self):
        """Clear selection and performance caches."""
        self._selection_cache.clear()
        self._performance_cache.clear()
        logger.info("Backend caches cleared")

    def save_config(self, path: Optional[Path] = None):
        """Save current configuration to file."""
        if path is None:
            path = Path.home() / ".tejas" / "backend_config.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "preferred_order": self.config.preferred_order,
            "auto_benchmark": self.config.auto_benchmark,
            "cache_selection": self.config.cache_selection,
            "cache_ttl_seconds": self.config.cache_ttl_seconds,
            "warn_on_fallback": self.config.warn_on_fallback,
            "max_concurrent_processes": self.config.max_concurrent_processes,
            "max_concurrent_threads": self.config.max_concurrent_threads,
            "memory_limit_gb": self.config.memory_limit_gb,
            "profile_operations": self.config.profile_operations,
            "adaptive_selection": self.config.adaptive_selection,
            "min_size_for_gpu": self.config.min_size_for_gpu,
            "min_size_for_parallel": self.config.min_size_for_parallel,
        }

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Backend configuration saved to {path}")


class ResourceContext:
    """Context manager for backend resource acquisition and cleanup."""

    def __init__(self, manager: UnifiedBackendManager, backend: str, operation: str):
        self.manager = manager
        self.backend = backend
        self.operation = operation
        self.backend_obj = manager.backends.get(backend)
        self.semaphores_acquired = []
        self.start_time = None

    def __enter__(self):
        """Acquire necessary resources."""
        if not self.backend_obj:
            return self

        # Acquire appropriate semaphore
        if self.backend_obj.device_type in ["cuda", "mps"]:
            self.manager._gpu_semaphore.acquire()
            self.semaphores_acquired.append("gpu")
        elif self.backend_obj.supports_parallel:
            self.manager._thread_semaphore.acquire()
            self.semaphores_acquired.append("thread")

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources and report performance."""
        # Calculate elapsed time
        if self.start_time:
            elapsed = time.perf_counter() - self.start_time

            # Report performance
            self.manager.report_performance(
                self.backend,
                self.operation,
                0,  # Size would need to be passed in
                elapsed,
                success=(exc_type is None),
            )

        # Release semaphores
        for sem_type in self.semaphores_acquired:
            if sem_type == "gpu":
                self.manager._gpu_semaphore.release()
            elif sem_type == "thread":
                self.manager._thread_semaphore.release()

        # Log any errors
        if exc_type is not None:
            logger.error(
                f"Error in {self.backend} backend for {self.operation}: {exc_val}"
            )

        return False  # Don't suppress exceptions


# Global instance getter
def get_backend_manager() -> UnifiedBackendManager:
    """Get the global backend manager instance."""
    return UnifiedBackendManager()


# Convenience function for backend selection
def select_optimal_backend(operation: str, data_size: int, **kwargs) -> str:
    """
    Convenience function to select optimal backend.

    Args:
        operation: Type of operation
        data_size: Size of data
        **kwargs: Additional parameters for backend selection

    Returns:
        Selected backend name
    """
    manager = get_backend_manager()
    return manager.select_backend(operation, data_size, **kwargs)
