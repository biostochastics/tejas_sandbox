#!/usr/bin/env python3
"""
Quick Multi-Configuration Benchmark
====================================
Tests key configurations with different parameters.
"""

import subprocess
import time
from datetime import datetime

# Quick test configurations
CONFIGS = [
    {"scale": 1000, "bits": 64, "features": 5000, "runs": 2},
    {"scale": 1000, "bits": 128, "features": 5000, "runs": 2},
    {"scale": 1000, "bits": 256, "features": 5000, "runs": 2},
    {"scale": 5000, "bits": 128, "features": 10000, "runs": 2},
]

def parse_output(output):
    """Extract metrics from benchmark output."""
    metrics = {}
    for line in output.split('\n'):
        if '✓' in line and 'Train:' in line:
            # Parse summary line
            parts = line.split(',')
            for part in parts:
                if 'Train:' in part:
                    metrics['train'] = part.split(':')[1].strip().replace('s', '')
                elif 'Encode:' in part:
                    metrics['encode'] = part.split(':')[1].strip().split()[0]
                elif 'Search:' in part:
                    metrics['search'] = part.split(':')[1].strip().split()[0]
                elif 'R@10:' in part:
                    metrics['recall'] = part.split(':')[1].strip().replace('%', '')
                elif 'Peak:' in part:
                    metrics['memory'] = part.split(':')[1].strip().replace('MB', '')
    return metrics

def run_config(config):
    """Run a single configuration."""
    cmd = [
        "python3", "unified_benchmark.py",
        "--scale", str(config["scale"]),
        "--runs", str(config["runs"]),
        "--n-bits", str(config["bits"]),
        "--max-features", str(config["features"])
    ]
    
    print(f"\n{'='*60}")
    print(f"Configuration: {config['scale']} docs, {config['bits']} bits, {config['features']} features")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        metrics = parse_output(result.stdout)
        
        if metrics:
            print(f"✅ Success!")
            print(f"  Train: {metrics.get('train', 'N/A')}s")
            print(f"  Encode: {metrics.get('encode', 'N/A')} docs/s")
            print(f"  Search: {metrics.get('search', 'N/A')} q/s")
            print(f"  Recall@10: {metrics.get('recall', 'N/A')}%")
            print(f"  Memory: {metrics.get('memory', 'N/A')}MB")
            return {**config, **metrics, "success": True}
        else:
            print("⚠️  No metrics found in output")
            return {**config, "success": False}
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout")
        return {**config, "success": False, "error": "timeout"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {**config, "success": False, "error": str(e)}

def main():
    print("\n" + "="*60)
    print("QUICK MULTI-CONFIGURATION BENCHMARK")
    print("="*60)
    print(f"Testing {len(CONFIGS)} configurations")
    
    results = []
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}]", end="")
        result = run_config(config)
        results.append(result)
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    print(f"\n{'Scale':<8} {'Bits':<6} {'Features':<10} {'Train(s)':<10} {'Encode(d/s)':<12} {'Search(q/s)':<12} {'Recall':<8} {'Mem(MB)':<8}")
    print("-" * 80)
    
    for r in results:
        if r.get("success"):
            print(f"{r['scale']:<8} {r['bits']:<6} {r['features']:<10} "
                  f"{r.get('train', 'N/A'):<10} {r.get('encode', 'N/A'):<12} "
                  f"{r.get('search', 'N/A'):<12} {r.get('recall', 'N/A'):<8} "
                  f"{r.get('memory', 'N/A'):<8}")
        else:
            print(f"{r['scale']:<8} {r['bits']:<6} {r['features']:<10} "
                  f"{'FAILED':<10} {'-':<12} {'-':<12} {'-':<8} {'-':<8}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    successful = [r for r in results if r.get("success")]
    if successful:
        # Impact of bits
        bits_64 = [r for r in successful if r['bits'] == 64]
        bits_128 = [r for r in successful if r['bits'] == 128]
        bits_256 = [r for r in successful if r['bits'] == 256]
        
        if bits_64 and bits_128:
            print("\n📊 Impact of bit size (at same scale):")
            if bits_64:
                print(f"  64 bits:  {bits_64[0].get('encode', 'N/A')} docs/s, {bits_64[0].get('recall', 'N/A')}% recall")
            if bits_128:
                print(f"  128 bits: {bits_128[0].get('encode', 'N/A')} docs/s, {bits_128[0].get('recall', 'N/A')}% recall")
            if bits_256:
                print(f"  256 bits: {bits_256[0].get('encode', 'N/A')} docs/s, {bits_256[0].get('recall', 'N/A')}% recall")
        
        # Impact of scale
        scale_1k = [r for r in successful if r['scale'] == 1000]
        scale_5k = [r for r in successful if r['scale'] == 5000]
        
        if scale_1k and scale_5k:
            print("\n📈 Impact of scale (at 128 bits):")
            s1k = next((r for r in scale_1k if r['bits'] == 128), None)
            s5k = next((r for r in scale_5k if r['bits'] == 128), None)
            if s1k:
                print(f"  1K docs:  {s1k.get('train', 'N/A')}s training, {s1k.get('memory', 'N/A')}MB")
            if s5k:
                print(f"  5K docs:  {s5k.get('train', 'N/A')}s training, {s5k.get('memory', 'N/A')}MB")
    
    print("\n✅ Benchmark complete!")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()