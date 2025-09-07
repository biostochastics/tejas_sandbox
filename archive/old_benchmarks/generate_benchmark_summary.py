#!/usr/bin/env python3
"""
Generate summary report from completed benchmark results
"""

import json
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

def generate_summary():
    """Generate summary from available benchmark results."""
    
    # Find latest benchmark directory
    dirs = sorted(glob.glob('benchmark_results/comprehensive_*'))
    if not dirs:
        print("No benchmark results found.")
        return
    
    latest_dir = Path(dirs[-1])
    print(f"Generating summary for: {latest_dir}")
    print("="*80)
    
    # 1. Pipeline Comparison Results
    pipeline_files = list(latest_dir.glob('pipeline_comparison/*.json'))
    if pipeline_files:
        print("\n1. PIPELINE COMPARISON RESULTS")
        print("-"*40)
        
        for pf in pipeline_files:
            with open(pf, 'r') as f:
                data = json.load(f)
            
            if 'summary' in data:
                for pipeline, datasets in data['summary'].items():
                    print(f"\n{pipeline}:")
                    for dataset, metrics in datasets.items():
                        if metrics:
                            speed = metrics.get('speed', {}).get('median', 0)
                            ndcg = metrics.get('ndcg_at_10', {}).get('median', 0)
                            print(f"  {dataset}: Speed={speed:.0f} d/s, NDCG={ndcg:.4f}")
    
    # 2. Factor Analysis Results
    factor_files = list(latest_dir.glob('factor_analysis/*.json'))
    if factor_files:
        print("\n2. FACTOR ANALYSIS RESULTS")
        print("-"*40)
        
        for ff in factor_files:
            with open(ff, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list) and data:
                # Group by factor
                factors = {}
                for item in data:
                    factor = item.get('factor', 'unknown')
                    if factor not in factors:
                        factors[factor] = []
                    factors[factor].append(item)
                
                for factor, results in factors.items():
                    print(f"\n{factor}:")
                    for r in results[:3]:  # Show first 3 values
                        value = r.get('value', 'N/A')
                        speed = r.get('speed', 0)
                        ndcg = r.get('ndcg@10', 0) or r.get('ndcg_10', 0)
                        print(f"  {value}: Speed={speed:.0f} d/s, NDCG={ndcg:.4f}")
    
    # 3. Bit Packing Test Results
    bp_files = list(latest_dir.glob('bit_packing_test/*.json'))
    if not bp_files:
        # Check parent directory
        bp_files = list(Path('benchmark_results/bit_packing_test').glob('*.json'))
    
    if bp_files:
        print("\n3. BIT PACKING TEST RESULTS")
        print("-"*40)
        
        for bf in bp_files:
            with open(bf, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list) and data:
                # Group by dataset and bit_packing value
                datasets = {}
                for item in data:
                    ds = item.get('dataset', 'unknown')
                    if ds not in datasets:
                        datasets[ds] = {'with': [], 'without': []}
                    
                    if item.get('bit_packing'):
                        datasets[ds]['with'].append(item)
                    else:
                        datasets[ds]['without'].append(item)
                
                for dataset, results in datasets.items():
                    print(f"\n{dataset.upper()}:")
                    
                    if results['without']:
                        avg_speed = np.mean([r.get('speed', 0) for r in results['without']])
                        avg_ndcg = np.mean([r.get('ndcg@10', 0) for r in results['without']])
                        print(f"  Without bit packing: Speed={avg_speed:.0f} d/s, NDCG={avg_ndcg:.4f}")
                    
                    if results['with']:
                        avg_speed = np.mean([r.get('speed', 0) for r in results['with']])
                        avg_ndcg = np.mean([r.get('ndcg@10', 0) for r in results['with']])
                        print(f"  With bit packing:    Speed={avg_speed:.0f} d/s, NDCG={avg_ndcg:.4f}")
    
    # 4. Check completion status
    print("\n4. COMPLETION STATUS")
    print("-"*40)
    
    # Check log files for completion
    logs = {
        'Pipeline Comparison': latest_dir / 'pipeline_comparison.log',
        'Factor Analysis': latest_dir / 'factor_analysis.log',
        'Bit Packing Test': latest_dir / 'bit_packing_test.log'
    }
    
    for name, log_file in logs.items():
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Count successful runs
            success_count = content.count('✓ Speed:')
            
            # Check if completed
            if "completed" in content.lower() or "finished" in content.lower():
                status = "✓ Completed"
            elif success_count > 0:
                status = f"⏳ In Progress ({success_count} experiments done)"
            else:
                status = "⏸ Waiting to start"
        else:
            status = "❌ Not started"
        
        print(f"{name}: {status}")
    
    print("\n" + "="*80)
    print(f"Summary generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    generate_summary()