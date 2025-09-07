#!/usr/bin/env python3
"""
Monitor running benchmarks and show progress
"""

import time
import subprocess
import glob
from pathlib import Path
from datetime import datetime

def count_completed_experiments(log_file):
    """Count completed experiments in a log file."""
    if not Path(log_file).exists():
        return 0
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Count successful runs
    success_count = content.count('✓ Speed:')
    return success_count

def check_process(pid):
    """Check if a process is still running."""
    try:
        subprocess.check_output(['ps', '-p', str(pid)])
        return True
    except subprocess.CalledProcessError:
        return False

def monitor():
    """Monitor benchmark progress."""
    
    # Find the latest benchmark directory
    dirs = sorted(glob.glob('benchmark_results/comprehensive_*'))
    if not dirs:
        print("No benchmark runs found.")
        return
    
    latest_dir = dirs[-1]
    print(f"Monitoring benchmarks in: {latest_dir}")
    print("="*60)
    
    # Expected totals
    n_pipelines = 4
    n_datasets = 3  
    n_runs = 10
    total_pipeline_experiments = n_pipelines * n_datasets * n_runs  # 120
    
    # Factor analysis: 4 factors x ~4 values each x 3 datasets x 10 runs
    # Approximate total
    total_factor_experiments = 480  # Rough estimate
    
    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress Update:")
        print("-"*50)
        
        # Check pipeline comparison
        pipeline_log = Path(latest_dir) / 'pipeline_comparison.log'
        if pipeline_log.exists():
            completed = count_completed_experiments(pipeline_log)
            progress = (completed / total_pipeline_experiments) * 100
            print(f"Pipeline Comparison: {completed}/{total_pipeline_experiments} ({progress:.1f}%)")
        else:
            print("Pipeline Comparison: Not started")
        
        # Check factor analysis
        factor_log = Path(latest_dir) / 'factor_analysis.log'
        if factor_log.exists():
            completed = count_completed_experiments(factor_log)
            # Use a rough estimate for total
            progress = (completed / total_factor_experiments) * 100
            print(f"Factor Analysis: {completed} experiments (~{progress:.1f}%)")
        else:
            print("Factor Analysis: Not started")
        
        # Check bit packing test
        bp_log = Path(latest_dir) / 'bit_packing_test.log'
        if bp_log.exists():
            completed = count_completed_experiments(bp_log)
            # Expected: 2 values x 3 datasets x 3 runs = 18
            total_bp = 18
            progress = (completed / total_bp) * 100
            print(f"Bit Packing Test: {completed}/{total_bp} ({progress:.1f}%)")
        else:
            print("Bit Packing Test: Not started")
        
        # Check for completion
        if pipeline_log.exists() and factor_log.exists():
            with open(pipeline_log, 'r') as f:
                if "All benchmarks completed at" in f.read():
                    print("\n✓ All benchmarks completed!")
                    break
        
        # Wait before next update
        time.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    monitor()