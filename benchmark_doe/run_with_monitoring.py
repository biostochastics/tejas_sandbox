#!/usr/bin/env python3
"""
Enhanced benchmark runner with monitoring, timeouts, and checkpointing
Wrapper for running DOE benchmarks with better error handling
"""

import sys
import os
import json
import time
import signal
import pickle
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkMonitor:
    """Monitor and manage benchmark execution with checkpointing."""
    
    def __init__(self, checkpoint_dir: str = "./benchmark_results/checkpoints"):
        """Initialize the monitor."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.results = []
        self.state = {}
        
    def save_checkpoint(self, experiment_id: str, state: Dict):
        """Save checkpoint for an experiment."""
        checkpoint_file = self.checkpoint_dir / f"{experiment_id}_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'experiment_id': experiment_id,
                'timestamp': datetime.now().isoformat(),
                'elapsed_seconds': time.time() - self.start_time,
                'state': state,
                'results': self.results
            }, f, indent=2)
        print(f"  Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self, experiment_id: str) -> Optional[Dict]:
        """Load checkpoint if it exists."""
        checkpoint_file = self.checkpoint_dir / f"{experiment_id}_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            print(f"  Checkpoint loaded: {checkpoint_file}")
            return data
        return None
    
    def run_with_timeout(self, func, args, timeout_seconds: int = 300):
        """Run function with timeout using multiprocessing."""
        
        def target_wrapper(queue):
            """Wrapper to capture function result."""
            try:
                result = func(*args)
                queue.put(('success', result))
            except Exception as e:
                queue.put(('error', str(e)))
        
        # Create queue for communication
        queue = mp.Queue()
        
        # Start process
        process = mp.Process(target=target_wrapper, args=(queue,))
        process.start()
        
        # Wait for completion or timeout
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join()
            return {'status': 'timeout', 'error': f'Exceeded {timeout_seconds}s timeout'}
        
        # Get result from queue
        if not queue.empty():
            status, data = queue.get()
            if status == 'success':
                return data
            else:
                return {'status': 'error', 'error': data}
        
        return {'status': 'error', 'error': 'Process completed without result'}


def run_benchmark_with_monitoring(script: str, config_file: str = None, 
                                 quick: bool = False, timeout: int = 300):
    """Run a benchmark script with monitoring and checkpointing."""
    
    monitor = BenchmarkMonitor()
    
    # Prepare command
    cmd = ['python3', script]
    if config_file:
        cmd.extend(['--config', config_file])
    if quick:
        cmd.append('--quick')
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Timeout: {timeout} seconds")
    print(f"Checkpoint dir: {monitor.checkpoint_dir}")
    print("="*60)
    
    try:
        # Run with timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Monitor output in real-time
        start_time = time.time()
        lines_buffer = []
        
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                lines_buffer.append(line)
                
                # Check for completion markers
                if "Results saved to:" in line:
                    result_file = line.split("Results saved to:")[-1].strip()
                    monitor.save_checkpoint(f"complete_{int(time.time())}", {
                        'result_file': result_file,
                        'lines': len(lines_buffer)
                    })
                
                # Check timeout
                if time.time() - start_time > timeout:
                    print(f"\n⚠ TIMEOUT after {timeout} seconds")
                    process.terminate()
                    break
        
        # Wait for process to complete
        return_code = process.wait()
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Execution completed in {elapsed:.1f} seconds")
        print(f"Return code: {return_code}")
        
        # Save final checkpoint
        monitor.save_checkpoint("final", {
            'return_code': return_code,
            'elapsed_seconds': elapsed,
            'total_lines': len(lines_buffer)
        })
        
        return return_code == 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        if 'process' in locals():
            process.terminate()
        monitor.save_checkpoint("interrupted", {
            'reason': 'user_interrupt',
            'elapsed_seconds': time.time() - start_time
        })
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        if 'process' in locals():
            process.terminate()
        monitor.save_checkpoint("error", {
            'error': str(e),
            'elapsed_seconds': time.time() - start_time
        })
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run DOE benchmarks with monitoring and checkpointing"
    )
    
    parser.add_argument(
        '--script', 
        type=str,
        default='benchmark_doe/run_tejas_vs_bert.py',
        help='Benchmark script to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file (if required by script)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test mode'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--test-all',
        action='store_true',
        help='Test all pipelines with quick test'
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        # Run quick test for all pipelines
        success = run_benchmark_with_monitoring(
            'benchmark_doe/run_quick_test.py',
            timeout=args.timeout
        )
    else:
        # Run specified script
        success = run_benchmark_with_monitoring(
            args.script,
            config_file=args.config,
            quick=args.quick,
            timeout=args.timeout
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()