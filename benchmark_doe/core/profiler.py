#!/usr/bin/env python3
"""
Memory and Performance Profiling for DOE Benchmarks

This module provides context managers and utilities for accurate measurement
of memory usage, execution time, and other performance metrics during experiments.
"""

import os
import sys
import time
import psutil
import tracemalloc
import resource
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import threading
import queue


@dataclass
class ProfileMetrics:
    """Container for profiling metrics from an experiment run."""
    
    # Timing metrics
    wall_time: float = 0.0
    cpu_time: float = 0.0
    system_time: float = 0.0
    user_time: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0
    memory_allocations: int = 0
    memory_blocks: int = 0
    
    # CPU metrics  
    cpu_percent: float = 0.0
    context_switches: int = 0
    
    # I/O metrics
    io_reads: int = 0
    io_writes: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "timing": {
                "wall_time": self.wall_time,
                "cpu_time": self.cpu_time,
                "system_time": self.system_time,
                "user_time": self.user_time
            },
            "memory": {
                "peak_memory_mb": self.peak_memory_mb,
                "memory_delta_mb": self.memory_delta_mb,
                "peak_rss_mb": self.peak_rss_mb,
                "peak_vms_mb": self.peak_vms_mb,
                "allocations": self.memory_allocations,
                "blocks": self.memory_blocks
            },
            "cpu": {
                "cpu_percent": self.cpu_percent,
                "context_switches": self.context_switches
            },
            "io": {
                "reads": self.io_reads,
                "writes": self.io_writes,
                "read_bytes": self.io_read_bytes,
                "write_bytes": self.io_write_bytes
            },
            "metadata": self.metadata
        }


class MemoryProfiler:
    """
    Context manager for detailed memory profiling.
    
    Tracks memory usage using multiple methods for accuracy:
    - tracemalloc for Python heap tracking
    - psutil for process-level memory
    - resource module for system limits
    """
    
    def __init__(
        self,
        trace_malloc: bool = True,
        monitor_interval: float = 0.1,
        include_children: bool = False
    ):
        """
        Initialize the memory profiler.
        
        Args:
            trace_malloc: Enable tracemalloc for detailed Python memory tracking
            monitor_interval: Interval for memory sampling in seconds
            include_children: Include child processes in measurements
        """
        self.trace_malloc = trace_malloc
        self.monitor_interval = monitor_interval
        self.include_children = include_children
        
        self.process = psutil.Process()
        self.metrics = ProfileMetrics()
        self._monitoring = False
        self._monitor_thread = None
        self._memory_samples = []
        
    def __enter__(self):
        """Enter the profiling context."""
        # Start tracemalloc if requested
        if self.trace_malloc:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            tracemalloc.start()
            self._snapshot_start = tracemalloc.take_snapshot()
        
        # Record initial memory state
        self._start_memory = self.process.memory_info()
        self._start_io = self.process.io_counters() if hasattr(self.process, 'io_counters') else None
        
        # Start continuous monitoring in background thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        # Record start time
        self._start_time = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_rusage = resource.getrusage(resource.RUSAGE_SELF)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the profiling context and compute metrics."""
        # Stop monitoring
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        # Compute timing metrics
        self.metrics.wall_time = time.perf_counter() - self._start_time
        self.metrics.cpu_time = time.process_time() - self._start_cpu
        
        end_rusage = resource.getrusage(resource.RUSAGE_SELF)
        self.metrics.user_time = end_rusage.ru_utime - self._start_rusage.ru_utime
        self.metrics.system_time = end_rusage.ru_stime - self._start_rusage.ru_stime
        
        # Compute memory metrics
        end_memory = self.process.memory_info()
        self.metrics.memory_delta_mb = (end_memory.rss - self._start_memory.rss) / 1024 / 1024
        
        if self._memory_samples:
            peak_rss = max(s["rss"] for s in self._memory_samples)
            peak_vms = max(s["vms"] for s in self._memory_samples)
            self.metrics.peak_rss_mb = peak_rss / 1024 / 1024
            self.metrics.peak_vms_mb = peak_vms / 1024 / 1024
            
            # Average CPU usage
            cpu_samples = [s["cpu"] for s in self._memory_samples if s["cpu"] is not None]
            if cpu_samples:
                self.metrics.cpu_percent = np.mean(cpu_samples)
        
        # Tracemalloc statistics
        if self.trace_malloc and tracemalloc.is_tracing():
            snapshot_end = tracemalloc.take_snapshot()
            stats = snapshot_end.compare_to(self._snapshot_start, 'lineno')
            
            total_size = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
            total_count = sum(stat.count_diff for stat in stats if stat.count_diff > 0)
            
            self.metrics.peak_memory_mb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
            self.metrics.memory_allocations = total_count
            self.metrics.memory_blocks = len(stats)
            
            tracemalloc.stop()
        
        # I/O statistics
        if self._start_io and hasattr(self.process, 'io_counters'):
            end_io = self.process.io_counters()
            self.metrics.io_reads = end_io.read_count - self._start_io.read_count
            self.metrics.io_writes = end_io.write_count - self._start_io.write_count
            self.metrics.io_read_bytes = end_io.read_bytes - self._start_io.read_bytes
            self.metrics.io_write_bytes = end_io.write_bytes - self._start_io.write_bytes
        
        # Context switches
        try:
            self.metrics.context_switches = end_rusage.ru_nvcsw - self._start_rusage.ru_nvcsw
        except AttributeError:
            pass  # Not available on all platforms
        
        return False  # Don't suppress exceptions
    
    def _monitor_memory(self):
        """Background thread for continuous memory monitoring."""
        while self._monitoring:
            try:
                mem_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent(interval=None)
                
                sample = {
                    "timestamp": time.perf_counter() - self._start_time,
                    "rss": mem_info.rss,
                    "vms": mem_info.vms,
                    "cpu": cpu_percent
                }
                
                # Include children if requested
                if self.include_children:
                    children = self.process.children(recursive=True)
                    for child in children:
                        try:
                            child_mem = child.memory_info()
                            sample["rss"] += child_mem.rss
                            sample["vms"] += child_mem.vms
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                
                self._memory_samples.append(sample)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            time.sleep(self.monitor_interval)
    
    def get_memory_timeline(self) -> List[Dict[str, float]]:
        """Get the timeline of memory samples collected during profiling."""
        return self._memory_samples.copy()


class ExecutionProfiler:
    """
    Comprehensive profiler combining memory, CPU, and I/O measurements.
    
    This is a higher-level interface that wraps MemoryProfiler and adds
    additional features like warmup runs and statistical aggregation.
    """
    
    def __init__(
        self,
        warmup_runs: int = 1,
        measure_runs: int = 3,
        cooldown_time: float = 0.5,
        profile_memory: bool = True,
        isolation_mode: str = "process"  # "process", "container", or "none"
    ):
        """
        Initialize the execution profiler.
        
        Args:
            warmup_runs: Number of warmup runs before measurement
            measure_runs: Number of measurement runs
            cooldown_time: Time to wait between runs (seconds)
            profile_memory: Enable detailed memory profiling
            isolation_mode: Level of isolation for measurements
        """
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs
        self.cooldown_time = cooldown_time
        self.profile_memory = profile_memory
        self.isolation_mode = isolation_mode
        
        self.results: List[ProfileMetrics] = []
    
    @contextmanager
    def profile(self, run_label: str = ""):
        """
        Profile a code block with multiple runs and aggregation.
        
        Args:
            run_label: Label for this profiling session
            
        Yields:
            ExecutionContext that can be used to pass data between runs
        """
        context = {"label": run_label, "run_count": 0}
        
        try:
            # Warmup runs (not measured)
            for i in range(self.warmup_runs):
                context["run_count"] = -(self.warmup_runs - i)  # Negative for warmup
                yield context
                time.sleep(self.cooldown_time)
                
                # Force garbage collection between runs
                import gc
                gc.collect()
            
            # Measurement runs
            run_metrics = []
            for i in range(self.measure_runs):
                context["run_count"] = i + 1
                
                if self.profile_memory:
                    profiler = MemoryProfiler()
                    with profiler:
                        yield context
                    run_metrics.append(profiler.metrics)
                else:
                    # Just time it
                    start = time.perf_counter()
                    yield context
                    elapsed = time.perf_counter() - start
                    
                    metrics = ProfileMetrics(wall_time=elapsed)
                    run_metrics.append(metrics)
                
                if i < self.measure_runs - 1:
                    time.sleep(self.cooldown_time)
                    gc.collect()
            
            # Store all results
            self.results = run_metrics
            
        finally:
            pass
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all measurement runs.
        
        Returns:
            Dictionary with mean, std, min, max, median for each metric
        """
        if not self.results:
            return {}
        
        # Extract arrays for each metric
        metrics_arrays = {}
        for metric in self.results:
            for key, value in metric.to_dict().items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        full_key = f"{key}.{subkey}"
                        if full_key not in metrics_arrays:
                            metrics_arrays[full_key] = []
                        metrics_arrays[full_key].append(subvalue)
        
        # Compute statistics
        aggregate = {}
        for key, values in metrics_arrays.items():
            if values and all(isinstance(v, (int, float)) for v in values):
                aggregate[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        return aggregate
    
    def save_results(self, filepath: Path):
        """Save profiling results to a JSON file."""
        results_data = {
            "runs": [m.to_dict() for m in self.results],
            "aggregate": self.get_aggregate_metrics(),
            "config": {
                "warmup_runs": self.warmup_runs,
                "measure_runs": self.measure_runs,
                "isolation_mode": self.isolation_mode
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)


@contextmanager
def isolated_execution(mode: str = "process"):
    """
    Context manager for isolated execution of experiments.
    
    Args:
        mode: Isolation mode - "process", "container", or "none"
        
    Yields:
        Execution context for the isolated environment
    """
    if mode == "container":
        # TODO: Implement Docker container isolation
        # This would spawn a container and execute code inside it
        raise NotImplementedError("Container isolation not yet implemented")
        
    elif mode == "process":
        # Process-level isolation using subprocess
        # The actual experiment would run in a separate process
        # This is handled by the runner module
        yield {"isolation": "process"}
        
    else:
        # No isolation, run in current process
        yield {"isolation": "none"}


def profile_function(func, *args, **kwargs) -> Tuple[Any, ProfileMetrics]:
    """
    Profile a single function call.
    
    Args:
        func: Function to profile
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (function_result, profile_metrics)
    """
    profiler = MemoryProfiler()
    
    with profiler:
        result = func(*args, **kwargs)
    
    return result, profiler.metrics