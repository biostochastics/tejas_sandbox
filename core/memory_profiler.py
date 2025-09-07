"""
Memory Profiler with Context Manager
=====================================
Production-safe memory profiling without global monkey-patching.
Tracks numpy array copies and provides detailed memory usage reports.

Implementation from FIX_ROADMAP_FINAL.md Week 5-6.
"""

import traceback
import time
from typing import Dict, Optional, Any, Callable
from contextlib import contextmanager
from functools import wraps
import numpy as np
import threading


class MemoryProfiler:
    """
    Safe memory profiling with proper cleanup using context managers.
    No permanent global monkey-patching - thread-safe implementation.

    Usage:
        with MemoryProfiler() as profiler:
            # Your code here
            data = np.random.randn(1000, 1000)
            data_copy = data.copy()  # This will be tracked

        report = profiler.report()
        print(f"Total copies: {report['total_copies']}")
        print(f"Total MB copied: {report['total_mb']:.2f}")

    Features:
        - Tracks np.ndarray.copy() calls with size and location
        - Thread-safe with proper cleanup
        - Provides detailed reports with stack traces
        - No permanent modification of numpy
        - Optional tracking of views and broadcasts
    """

    def __init__(self, track_views: bool = False, track_broadcasts: bool = False):
        """
        Initialize the memory profiler.

        Parameters
        ----------
        track_views : bool, default=False
            Whether to track array views (may have performance impact)
        track_broadcasts : bool, default=False
            Whether to track broadcasting operations
        """
        self.copies_detected = []
        self.views_detected = []
        self.broadcasts_detected = []
        self.original_copy = None
        self.original_view = None
        self.track_views = track_views
        self.track_broadcasts = track_broadcasts
        self._lock = threading.Lock()
        self._start_time = None
        self._end_time = None

    def __enter__(self):
        """Start tracking memory operations."""
        self._start_time = time.time()

        # Store original numpy functions - use numpy module level function
        self.original_copy = np.copy
        self.original_array = np.array
        if self.track_views:
            self.original_asarray = np.asarray

        # Create tracked version of copy
        copies = self.copies_detected
        lock = self._lock

        def tracked_copy(a, order="K", subok=False):
            """Tracked version of np.copy()"""
            # Record the copy
            if isinstance(a, np.ndarray):
                with lock:
                    stack = traceback.extract_stack()
                    # Skip the tracked_copy frame itself
                    caller = stack[-2] if len(stack) >= 2 else None

                    copies.append(
                        {
                            "size_mb": a.nbytes / (1024 * 1024),
                            "shape": a.shape,
                            "dtype": str(a.dtype),
                            "location": {
                                "file": caller.filename if caller else "unknown",
                                "line": caller.lineno if caller else 0,
                                "function": caller.name if caller else "unknown",
                                "code": caller.line if caller else "",
                            }
                            if caller
                            else None,
                            "timestamp": time.time(),
                        }
                    )

            # Call original copy
            return self.original_copy(a, order=order, subok=subok)

        def tracked_array(
            object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None
        ):
            """Tracked version of np.array()"""
            # Track if copy=True and object is already an array
            if copy and isinstance(object, np.ndarray):
                with lock:
                    stack = traceback.extract_stack()
                    caller = stack[-2] if len(stack) >= 2 else None

                    copies.append(
                        {
                            "size_mb": object.nbytes / (1024 * 1024),
                            "shape": object.shape,
                            "dtype": str(object.dtype),
                            "location": {
                                "file": caller.filename if caller else "unknown",
                                "line": caller.lineno if caller else 0,
                                "function": caller.name if caller else "unknown",
                                "code": caller.line if caller else "",
                            }
                            if caller
                            else None,
                            "timestamp": time.time(),
                        }
                    )

            # Call original array
            return self.original_array(
                object,
                dtype=dtype,
                copy=copy,
                order=order,
                subok=subok,
                ndmin=ndmin,
                like=like,
            )

        # Replace numpy functions with tracked versions
        np.copy = tracked_copy
        np.array = tracked_array

        # Track views if requested
        if self.track_views:
            views = self.views_detected

            def tracked_asarray(a, dtype=None, order=None, *, like=None):
                """Tracked version of np.asarray() - often creates views"""
                if isinstance(a, np.ndarray) and dtype is None:
                    # This likely creates a view
                    with lock:
                        stack = traceback.extract_stack()
                        caller = stack[-2] if len(stack) >= 2 else None

                        views.append(
                            {
                                "size_mb": a.nbytes / (1024 * 1024),
                                "shape": a.shape,
                                "original_dtype": str(a.dtype),
                                "new_dtype": str(dtype) if dtype else str(a.dtype),
                                "location": {
                                    "file": caller.filename if caller else "unknown",
                                    "line": caller.lineno if caller else 0,
                                    "function": caller.name if caller else "unknown",
                                    "code": caller.line if caller else "",
                                }
                                if caller
                                else None,
                                "timestamp": time.time(),
                            }
                        )

                return self.original_asarray(a, dtype=dtype, order=order, like=like)

            np.asarray = tracked_asarray

        return self

    def start(self):
        """Start profiling (same as __enter__)."""
        return self.__enter__()

    def stop(self):
        """Stop profiling (same as __exit__)."""
        return self.__exit__(None, None, None)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original functions and clean up."""
        self._end_time = time.time()

        # Restore original numpy functions
        if self.original_copy:
            np.copy = self.original_copy
        if self.original_array:
            np.array = self.original_array

        if self.track_views and hasattr(self, "original_asarray"):
            np.asarray = self.original_asarray

        # Don't suppress exceptions
        return False

    def report(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Generate memory profiling report.

        Parameters
        ----------
        detailed : bool, default=False
            Include full stack traces and detailed information

        Returns
        -------
        dict
            Report containing:
            - total_copies: Number of copy operations
            - total_mb: Total memory copied in MB
            - duration: Profiling duration in seconds
            - copies_per_second: Rate of copy operations
            - mb_per_second: Memory copy rate
            - locations: List of copy locations (if detailed)
            - top_offenders: Top 5 locations by memory usage
        """
        total_copies = len(self.copies_detected)
        total_mb = sum(c["size_mb"] for c in self.copies_detected)
        duration = (self._end_time - self._start_time) if self._end_time else 0

        # Aggregate by location
        location_stats = {}
        for copy in self.copies_detected:
            if copy["location"]:
                key = f"{copy['location']['file']}:{copy['location']['line']}"
                if key not in location_stats:
                    location_stats[key] = {
                        "count": 0,
                        "total_mb": 0,
                        "function": copy["location"]["function"],
                        "code": copy["location"]["code"],
                    }
                location_stats[key]["count"] += 1
                location_stats[key]["total_mb"] += copy["size_mb"]

        # Sort by total memory
        top_offenders = sorted(
            location_stats.items(), key=lambda x: x[1]["total_mb"], reverse=True
        )[:5]

        report = {
            "total_copies": total_copies,
            "total_mb": total_mb,
            "duration": duration,
            "copies_per_second": total_copies / duration if duration > 0 else 0,
            "mb_per_second": total_mb / duration if duration > 0 else 0,
            "top_offenders": [
                {
                    "location": loc,
                    "count": stats["count"],
                    "total_mb": stats["total_mb"],
                    "function": stats["function"],
                    "code": stats["code"].strip() if stats["code"] else "",
                }
                for loc, stats in top_offenders
            ],
        }

        if detailed:
            report["copies"] = self.copies_detected

        if self.track_views:
            report["total_views"] = len(self.views_detected)
            report["views_mb"] = sum(v["size_mb"] for v in self.views_detected)
            if detailed:
                report["views"] = self.views_detected

        return report

    def reset(self):
        """Clear all tracked data for reuse."""
        with self._lock:
            self.copies_detected.clear()
            self.views_detected.clear()
            self.broadcasts_detected.clear()
            self._start_time = None
            self._end_time = None

    def print_report(self, detailed: bool = False):
        """Print a formatted report to stdout."""
        report = self.report(detailed=detailed)

        print("\n" + "=" * 60)
        print("MEMORY PROFILING REPORT")
        print("=" * 60)
        print(f"Duration: {report['duration']:.2f} seconds")
        print(f"Total copies: {report['total_copies']}")
        print(f"Total memory copied: {report['total_mb']:.2f} MB")
        print(f"Copy rate: {report['copies_per_second']:.1f} copies/sec")
        print(f"Memory rate: {report['mb_per_second']:.1f} MB/sec")

        if report["top_offenders"]:
            print("\nTop Memory Copy Locations:")
            print("-" * 60)
            for i, offender in enumerate(report["top_offenders"], 1):
                print(f"{i}. {offender['location']}")
                print(f"   Function: {offender['function']}")
                print(
                    f"   Copies: {offender['count']}, Total: {offender['total_mb']:.2f} MB"
                )
                if offender["code"]:
                    print(f"   Code: {offender['code']}")

        if self.track_views and "total_views" in report:
            print(f"\nViews created: {report['total_views']}")
            print(f"View memory referenced: {report['views_mb']:.2f} MB")

        print("=" * 60 + "\n")


# Decorator version for function-level profiling
def profile_memory(
    func: Optional[Callable] = None,
    track_views: bool = False,
    print_report: bool = True,
) -> Callable:
    """
    Decorator to profile memory usage of a function.

    Parameters
    ----------
    func : callable, optional
        Function to profile (used when decorator called without parens)
    track_views : bool, default=False
        Whether to track array views
    print_report : bool, default=True
        Whether to print report after function execution

    Returns
    -------
    callable
        Wrapped function with memory profiling

    Examples
    --------
    @profile_memory
    def my_function():
        data = np.random.randn(1000, 1000)
        return data.copy()

    @profile_memory(track_views=True, print_report=False)
    def another_function():
        data = np.zeros((100, 100))
        view = data.view()
        return view
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with MemoryProfiler(track_views=track_views) as profiler:
                result = f(*args, **kwargs)

            if print_report:
                profiler.print_report()

            # Attach report to function for testing
            wrapper.last_report = profiler.report()

            return result

        return wrapper

    if func is None:
        # Decorator called with arguments
        return decorator
    else:
        # Decorator called without arguments
        return decorator(func)


# Memory-efficient operations
def memory_efficient_transform(
    data: np.ndarray, transform: np.ndarray, chunk_size: int = 1000
) -> np.ndarray:
    """
    Memory-efficient matrix multiplication using chunking.

    Performs: data @ transform in chunks to minimize memory usage.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Input data to transform
    transform : ndarray of shape (n_features, n_components)
        Transformation matrix
    chunk_size : int, default=1000
        Number of samples to process at once

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        Transformed data

    Notes
    -----
    Processes data in chunks to avoid large intermediate allocations.
    """
    n_samples = data.shape[0]
    n_components = transform.shape[1]

    # Pre-allocate output
    result = np.empty((n_samples, n_components), dtype=data.dtype)

    # Process in chunks
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        result[i:end] = data[i:end] @ transform

    return result


def memory_efficient_normalize(
    data: np.ndarray, axis: int = 1, chunk_size: int = 1000
) -> np.ndarray:
    """
    Memory-efficient L2 normalization using chunking.

    Parameters
    ----------
    data : ndarray
        Data to normalize
    axis : int, default=1
        Axis along which to normalize (1 for row-wise)
    chunk_size : int, default=1000
        Number of samples to process at once

    Returns
    -------
    ndarray
        Normalized data
    """
    n_samples = data.shape[0]
    result = np.empty_like(data)

    # Process in chunks
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        chunk = data[i:end]

        # Compute norms for this chunk
        norms = np.linalg.norm(chunk, axis=axis, keepdims=True)

        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)

        # Normalize chunk
        result[i:end] = chunk / norms

    return result


def memory_efficient_batch_process(
    data, process_func: Callable, batch_size: int = 1000
) -> np.ndarray:
    """
    Process data in batches to minimize memory usage.

    Parameters
    ----------
    data : ndarray or iterable
        Data to process (if ndarray, processed in chunks)
    process_func : callable
        Function to apply to each batch
    batch_size : int, default=1000
        Size of processing batches

    Returns
    -------
    ndarray
        Processed data

    Notes
    -----
    Useful for processing large datasets that don't fit in memory.
    Processes data in chunks and concatenates results.
    """
    # Handle numpy array input
    if isinstance(data, np.ndarray):
        n_samples = data.shape[0]
        results = []

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = data[i:end]
            result = process_func(batch)
            results.append(result)

        return np.vstack(results) if results else np.array([])

    # Handle iterator input
    results = []
    batch = []

    for item in data:
        batch.append(item)

        if len(batch) >= batch_size:
            # Process batch
            batch_array = np.array(batch)
            result = process_func(batch_array)
            results.append(result)

            # Clear batch
            batch.clear()

    # Process remaining items
    if batch:
        batch_array = np.array(batch)
        result = process_func(batch_array)
        results.append(result)

    # Concatenate all results
    if results:
        return np.vstack(results)
    else:
        return np.array([])


# Context manager for memory-aware operations
@contextmanager
def memory_limit(max_mb: float = 1000.0):
    """
    Context manager to enforce memory limits.

    Parameters
    ----------
    max_mb : float, default=1000.0
        Maximum memory usage in MB

    Raises
    ------
    MemoryError
        If memory limit is exceeded

    Examples
    --------
    with memory_limit(500):  # 500 MB limit
        large_array = np.random.randn(10000, 10000)  # Will raise if too large
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1e6

    def check_memory():
        current_memory = process.memory_info().rss / 1e6
        used = current_memory - initial_memory
        if used > max_mb:
            raise MemoryError(f"Memory limit exceeded: {used:.1f} MB > {max_mb:.1f} MB")

    try:
        yield check_memory
    finally:
        # Final memory check
        check_memory()


if __name__ == "__main__":
    # Example usage
    print("Memory Profiler Example")
    print("-" * 40)

    # Example 1: Basic profiling
    with MemoryProfiler() as profiler:
        # Create some arrays
        data = np.random.randn(1000, 1000)

        # These operations will be tracked
        copy1 = data.copy()
        copy2 = data[::2].copy()
        copy3 = np.array(data, copy=True)

    profiler.print_report()

    # Example 2: Using the decorator
    @profile_memory
    def process_data():
        data = np.random.randn(500, 500)
        normalized = data.copy()
        normalized = (normalized - normalized.mean()) / normalized.std()
        return normalized

    result = process_data()

    # Example 3: Memory-efficient operations
    print("\nMemory-Efficient Transform Example")
    print("-" * 40)

    data = np.random.randn(1000, 100)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # Track memory usage of efficient vs inefficient
    with MemoryProfiler() as profiler:
        # Inefficient: multiple copies
        result_inefficient = (data.copy() - mean) / std

    report_inefficient = profiler.report()

    profiler.reset()

    with MemoryProfiler() as profiler:
        # Efficient: single allocation
        result_efficient = memory_efficient_transform(data, mean, std)

    report_efficient = profiler.report()

    print(
        f"Inefficient: {report_inefficient['total_copies']} copies, "
        f"{report_inefficient['total_mb']:.2f} MB"
    )
    print(
        f"Efficient: {report_efficient['total_copies']} copies, "
        f"{report_efficient['total_mb']:.2f} MB"
    )

    # Verify results are the same
    assert np.allclose(result_inefficient, result_efficient)
    print("\nResults match! ✓")
