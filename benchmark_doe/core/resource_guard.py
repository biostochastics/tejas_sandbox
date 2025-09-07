#!/usr/bin/env python3
"""
Resource Management and Protection for DOE Framework

This module provides resource limiting and timeout mechanisms to prevent
DoS attacks and resource exhaustion.
"""

import os
import time
import warnings
import traceback
from multiprocessing import Process, Queue, TimeoutError
from typing import Any, Callable, Dict, Optional, Tuple
from contextlib import contextmanager
import psutil


class ExperimentTimeoutError(Exception):
    """Raised when an experiment exceeds its time limit."""
    pass


class ResourceExhaustedError(Exception):
    """Raised when resource limits are exceeded."""
    pass


def worker_wrapper(
    task_func: Callable,
    args: tuple,
    kwargs: dict,
    result_queue: Queue,
    config: Dict[str, Any]
) -> None:
    """
    Worker function that runs in a separate process.
    
    Args:
        task_func: Function to execute
        args: Positional arguments for task_func
        kwargs: Keyword arguments for task_func
        result_queue: Queue to store results
        config: Configuration including resource limits
    """
    try:
        # Monitor memory usage if limit is set
        if config.get('memory_limit_mb'):
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_limit = config['memory_limit_mb']
        
        # Execute the task
        start_time = time.time()
        result = task_func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Check final memory usage
        if config.get('memory_limit_mb'):
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            
            if memory_used > memory_limit:
                raise ResourceExhaustedError(
                    f"Memory usage {memory_used:.1f}MB exceeded limit {memory_limit}MB"
                )
        
        # Store successful result
        result_queue.put({
            'status': 'success',
            'result': result,
            'execution_time': execution_time,
            'memory_used_mb': memory_used if config.get('memory_limit_mb') else None
        })
        
    except Exception as e:
        # Store error information
        result_queue.put({
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        })


class ResourceGuard:
    """
    Context manager and utility class for resource-limited execution.
    
    This class provides safe execution of potentially resource-intensive
    operations with configurable timeouts and memory limits.
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        memory_limit_mb: Optional[int] = 2048,
        check_interval: float = 1.0
    ):
        """
        Initialize resource guard.
        
        Args:
            timeout_seconds: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB (None for unlimited)
            check_interval: Interval for checking resource usage
        """
        self.timeout = timeout_seconds
        self.memory_limit = memory_limit_mb
        self.check_interval = check_interval
        
    def run_with_limits(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        description: str = "Task"
    ) -> Any:
        """
        Run a function with resource limits in a separate process.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            description: Task description for error messages
            
        Returns:
            Function result
            
        Raises:
            ExperimentTimeoutError: If execution exceeds timeout
            ResourceExhaustedError: If memory limit is exceeded
            Exception: If the function raises an exception
        """
        if kwargs is None:
            kwargs = {}
        
        # Create queue for results
        result_queue = Queue()
        
        # Configuration for worker
        config = {
            'memory_limit_mb': self.memory_limit,
            'timeout_seconds': self.timeout
        }
        
        # Create and start worker process
        process = Process(
            target=worker_wrapper,
            args=(func, args, kwargs, result_queue, config),
            name=f"ResourceGuard-{description}"
        )
        
        process.start()
        
        # Wait for completion with timeout
        process.join(timeout=self.timeout)
        
        # Check if process completed
        if process.is_alive():
            # Timeout exceeded - terminate process
            process.terminate()
            process.join(timeout=5)  # Give it 5 seconds to terminate
            
            if process.is_alive():
                # Force kill if still alive
                process.kill()
                process.join()
            
            raise ExperimentTimeoutError(
                f"{description} exceeded {self.timeout}s timeout"
            )
        
        # Get result from queue
        if result_queue.empty():
            raise RuntimeError(f"{description} completed but produced no result")
        
        result_data = result_queue.get()
        
        # Handle result based on status
        if result_data['status'] == 'error':
            error_msg = (
                f"{description} failed: {result_data['error']}\n"
                f"Type: {result_data['error_type']}"
            )
            if result_data.get('traceback'):
                error_msg += f"\nTraceback:\n{result_data['traceback']}"
            
            # Re-raise specific error types
            if result_data['error_type'] == 'ResourceExhaustedError':
                raise ResourceExhaustedError(error_msg)
            else:
                raise RuntimeError(error_msg)
        
        # Log execution stats
        if result_data.get('execution_time'):
            if result_data['execution_time'] > self.timeout * 0.8:
                warnings.warn(
                    f"{description} used {result_data['execution_time']:.1f}s "
                    f"({result_data['execution_time']/self.timeout*100:.0f}% of timeout)",
                    UserWarning
                )
        
        if result_data.get('memory_used_mb'):
            if self.memory_limit and result_data['memory_used_mb'] > self.memory_limit * 0.8:
                warnings.warn(
                    f"{description} used {result_data['memory_used_mb']:.1f}MB "
                    f"({result_data['memory_used_mb']/self.memory_limit*100:.0f}% of limit)",
                    UserWarning
                )
        
        return result_data['result']
    
    @contextmanager
    def guard(self, description: str = "Operation"):
        """
        Context manager for resource-guarded execution.
        
        Usage:
            guard = ResourceGuard(timeout_seconds=60)
            with guard.guard("Heavy computation"):
                result = expensive_function()
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            yield self
            
        finally:
            # Check resource usage
            elapsed = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            
            # Warn if close to limits
            if elapsed > self.timeout * 0.8:
                warnings.warn(
                    f"{description} took {elapsed:.1f}s "
                    f"({elapsed/self.timeout*100:.0f}% of {self.timeout}s timeout)",
                    UserWarning
                )
            
            if self.memory_limit and memory_used > self.memory_limit * 0.8:
                warnings.warn(
                    f"{description} used {memory_used:.1f}MB "
                    f"({memory_used/self.memory_limit*100:.0f}% of {self.memory_limit}MB limit)",
                    UserWarning
                )


class AdaptiveResourceGuard(ResourceGuard):
    """
    Resource guard that adapts limits based on system resources.
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        memory_fraction: float = 0.5,
        check_interval: float = 1.0
    ):
        """
        Initialize adaptive resource guard.
        
        Args:
            timeout_seconds: Maximum execution time
            memory_fraction: Fraction of available memory to use
            check_interval: Resource check interval
        """
        # Calculate memory limit based on available memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
        memory_limit = int(available_memory * memory_fraction)
        
        super().__init__(
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit,
            check_interval=check_interval
        )
        
        # Store configuration
        self.memory_fraction = memory_fraction
        self.adaptive = True
        
    def adjust_limits(self) -> None:
        """
        Adjust resource limits based on current system state.
        """
        # Update memory limit
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        new_limit = int(available_memory * self.memory_fraction)
        
        if abs(new_limit - self.memory_limit) > 100:  # Significant change
            self.memory_limit = new_limit
            warnings.warn(
                f"Adjusted memory limit to {self.memory_limit}MB "
                f"based on available memory",
                UserWarning
            )


# Convenience functions
def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    timeout: int = 300,
    description: str = "Task"
) -> Any:
    """
    Simple wrapper to run a function with timeout.
    
    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout: Timeout in seconds
        description: Task description
        
    Returns:
        Function result
    """
    guard = ResourceGuard(timeout_seconds=timeout, memory_limit_mb=None)
    return guard.run_with_limits(func, args, kwargs, description)


def run_with_limits(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    timeout: int = 300,
    memory_mb: int = 2048,
    description: str = "Task"
) -> Any:
    """
    Run a function with both timeout and memory limits.
    
    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout: Timeout in seconds
        memory_mb: Memory limit in MB
        description: Task description
        
    Returns:
        Function result
    """
    guard = ResourceGuard(
        timeout_seconds=timeout,
        memory_limit_mb=memory_mb
    )
    return guard.run_with_limits(func, args, kwargs, description)