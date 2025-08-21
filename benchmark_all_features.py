#!/usr/bin/env python3
"""
Comprehensive Real-World Benchmark of ALL TEJAS Features
=========================================================

Tests every feature combination with real Wikipedia data:
1. Standard SVD vs Randomized SVD
2. Zero threshold vs Median threshold vs ITQ
3. Bit packing enabled/disabled
4. All search backends (numpy, numba, torch)
5. Memory usage and speed for each configuration

Produces a complete performance matrix for documentation.
"""

import numpy as np
import torch
import time
import psutil
import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import traceback

# Import TEJAS modules
from core.encoder import GoldenRatioEncoder
from core.fingerprint import BinaryFingerprintSearch
from core.bitops import pack_bits_rows, unpack_bits_rows
from randomized_svd import RandomizedSVD

def get_memory_mb():
    """Get current memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def load_or_generate_data(n_samples=10000):
    """Load real Wikipedia data or generate realistic synthetic data."""
    print(f"\nPreparing dataset with {n_samples} samples...")
    
    # Try to load real Wikipedia data
    wiki_path = Path("data/wikipedia/wikipedia_en_20231101_titles.pt")
    if wiki_path.exists():
        print("  Loading real Wikipedia titles...")
        try:
            data = torch.load(wiki_path)
            titles = data['titles'][:n_samples] if 'titles' in data else data[:n_samples]
            print(f"  Loaded {len(titles)} real Wikipedia titles")
            return titles
        except Exception as e:
            print(f"  Failed to load Wikipedia data: {e}")
    
    # Generate realistic synthetic titles if no real data
    print("  Generating synthetic titles (no Wikipedia data found)...")
    topics = ["History", "Science", "Art", "Technology", "Geography", "Literature", 
              "Mathematics", "Philosophy", "Music", "Biology", "Physics", "Chemistry"]
    
    titles = []
    for i in range(n_samples):
        topic = np.random.choice(topics)
        suffix = np.random.choice(["Theory", "Introduction", "Overview", "Analysis", 
                                   "Study", "Research", "Development", "Application"])
        year = np.random.randint(1800, 2024) if np.random.random() > 0.5 else ""
        title = f"{topic} {suffix} {year} {i:05d}".strip()
        titles.append(title)
    
    return titles

def benchmark_encoding(titles: List[str], config: Dict) -> Dict:
    """Benchmark a specific encoder configuration."""
    print(f"\n  Testing: {config['name']}")
    print(f"    Config: SVD={config['svd_type']}, Threshold={config['threshold']}, "
          f"Bits={config['n_bits']}, Pack={config['bit_packing']}")
    
    results = {
        'config': config,
        'times': {},
        'memory': {},
        'metrics': {}
    }
    
    # Create encoder
    gc.collect()
    mem_start = get_memory_mb()
    start_time = time.time()
    
    try:
        encoder = GoldenRatioEncoder(
            n_bits=config['n_bits'],
            max_features=config['max_features'],
            threshold_strategy=config['threshold'],
            use_itq=(config['threshold'] == 'itq'),
            itq_iterations=50 if config['threshold'] == 'itq' else 0,
            use_randomized_svd=(config['svd_type'] == 'randomized'),
            svd_n_iter=5,
            svd_n_oversamples=20
        )
        
        # Training phase
        print("    Training encoder...")
        train_start = time.time()
        encoder.fit(titles, memory_limit_gb=10)
        train_time = time.time() - train_start
        
        # Encoding phase
        print("    Encoding documents...")
        encode_start = time.time()
        fingerprints = encoder.encode(titles)
        encode_time = time.time() - encode_start
        
        # Convert torch tensors to numpy if needed
        if hasattr(fingerprints, 'detach'):
            fingerprints = fingerprints.detach().cpu().numpy()
        
        # Ensure we have numpy array for memory calculation
        if hasattr(fingerprints, 'detach'):
            fingerprints_np = fingerprints.detach().cpu().numpy()
        else:
            fingerprints_np = fingerprints
        
        # Memory before packing
        memory_unpacked = fingerprints_np.nbytes / 1e6  # MB
        
        # Bit packing (if enabled)
        if config['bit_packing']:
            print("    Packing bits...")
            pack_start = time.time()
            packed = pack_bits_rows(fingerprints_np, n_bits=config['n_bits'])
            pack_time = time.time() - pack_start
            memory_packed = packed.nbytes / 1e6  # MB
            compression_ratio = memory_unpacked / memory_packed
        else:
            pack_time = 0
            memory_packed = memory_unpacked
            compression_ratio = 1.0
        
        total_time = time.time() - start_time
        peak_memory = get_memory_mb() - mem_start
        
        # Calculate fingerprint statistics
        if config['bit_packing']:
            # Unpack a sample for statistics
            sample_unpacked = unpack_bits_rows(packed[:100], n_bits=config['n_bits'])
            bit_balance = np.mean(sample_unpacked)
            unique_fps = len(np.unique(packed[:1000], axis=0))
        else:
            bit_balance = np.mean(fingerprints_np)
            unique_fps = len(np.unique(fingerprints_np[:1000], axis=0))
        
        # Store results
        results['times'] = {
            'total': total_time,
            'train': train_time,
            'encode': encode_time,
            'pack': pack_time
        }
        
        results['memory'] = {
            'peak_mb': peak_memory,
            'unpacked_mb': memory_unpacked,
            'packed_mb': memory_packed,
            'compression_ratio': compression_ratio
        }
        
        results['metrics'] = {
            'bit_balance': bit_balance,
            'unique_fingerprints': unique_fps,
            'docs_per_second': len(titles) / total_time
        }
        
        # Store fingerprints for search benchmark
        results['fingerprints'] = packed if config['bit_packing'] else fingerprints_np
        results['success'] = True
        
        print(f"    ✓ Time: {total_time:.2f}s, Memory: {peak_memory:.1f}MB, "
              f"Compression: {compression_ratio:.1f}x")
        
    except Exception as e:
        print(f"    ✗ Failed: {str(e)[:100]}")
        results['success'] = False
        results['error'] = str(e)
        results['fingerprints'] = None
    
    gc.collect()
    return results

def benchmark_search(fingerprints: np.ndarray, titles: List[str], 
                    backend: str, packed: bool, n_bits: int) -> Dict:
    """Benchmark search performance with a specific backend."""
    print(f"    Testing search with {backend} backend...")
    
    results = {
        'backend': backend,
        'times': {},
        'metrics': {}
    }
    
    try:
        # Unpack if needed
        if packed:
            search_fps = unpack_bits_rows(fingerprints, n_bits)
        else:
            search_fps = fingerprints
        
        # Build index
        gc.collect()
        start = time.time()
        search = BinaryFingerprintSearch(search_fps, titles, backend=backend)
        build_time = time.time() - start
        
        # Run search queries
        n_queries = min(100, len(fingerprints))
        query_indices = np.random.choice(len(fingerprints), n_queries, replace=False)
        
        search_times = []
        precisions = []
        
        for idx in query_indices:
            query = search_fps[idx]
            
            start = time.time()
            results_list = search.search(query, k=10, show_pattern_analysis=False)
            search_time = time.time() - start
            search_times.append(search_time)
            
            # Calculate precision (query should find itself)
            # Results are tuples (title, similarity, distance)
            found_self = any(r[0] == titles[idx] for r in results_list)
            precisions.append(1.0 if found_self else 0.0)
        
        mean_search_time = np.mean(search_times)
        mean_precision = np.mean(precisions)
        throughput = len(search_fps) / mean_search_time  # docs/sec per query
        
        results['times'] = {
            'build_index': build_time,
            'mean_search_ms': mean_search_time * 1000,
            'total_queries': sum(search_times)
        }
        
        results['metrics'] = {
            'precision_at_10': mean_precision,
            'throughput_docs_per_sec': throughput,
            'queries_per_sec': 1.0 / mean_search_time
        }
        
        results['success'] = True
        print(f"      ✓ Search: {mean_search_time*1000:.2f}ms, "
              f"Throughput: {throughput:.0f} docs/sec, "
              f"Precision@10: {mean_precision:.2f}")
        
    except Exception as e:
        print(f"      ✗ Failed: {str(e)[:100]}")
        results['success'] = False
        results['error'] = str(e)
    
    gc.collect()
    return results

def run_comprehensive_benchmark():
    """Run benchmark on all feature combinations."""
    print("="*70)
    print("COMPREHENSIVE TEJAS FEATURE BENCHMARK")
    print("="*70)
    
    # Load data
    n_samples = 5000  # Adjust based on available memory
    titles = load_or_generate_data(n_samples)
    
    # Define all configurations to test
    configurations = [
        # Baseline
        {'name': 'Baseline (Standard SVD + Zero)', 'svd_type': 'standard', 
         'threshold': 'zero', 'n_bits': 128, 'max_features': 5000, 'bit_packing': False},
        
        # Randomized SVD variations
        {'name': 'Randomized SVD + Zero', 'svd_type': 'randomized', 
         'threshold': 'zero', 'n_bits': 128, 'max_features': 5000, 'bit_packing': False},
        
        {'name': 'Randomized SVD + Median', 'svd_type': 'randomized', 
         'threshold': 'median', 'n_bits': 128, 'max_features': 5000, 'bit_packing': False},
        
        # ITQ variations
        {'name': 'Standard SVD + ITQ', 'svd_type': 'standard', 
         'threshold': 'itq', 'n_bits': 128, 'max_features': 5000, 'bit_packing': False},
        
        {'name': 'Randomized SVD + ITQ', 'svd_type': 'randomized', 
         'threshold': 'itq', 'n_bits': 128, 'max_features': 5000, 'bit_packing': False},
        
        # Bit packing variations
        {'name': 'Randomized + ITQ + Packed', 'svd_type': 'randomized', 
         'threshold': 'itq', 'n_bits': 128, 'max_features': 5000, 'bit_packing': True},
        
        # Different bit sizes
        {'name': '256-bit Randomized + ITQ', 'svd_type': 'randomized', 
         'threshold': 'itq', 'n_bits': 256, 'max_features': 5000, 'bit_packing': False},
        
        # Large feature space
        {'name': 'High-dim (10K features)', 'svd_type': 'randomized', 
         'threshold': 'zero', 'n_bits': 128, 'max_features': 10000, 'bit_packing': False},
    ]
    
    # Run encoding benchmarks
    print("\n" + "="*70)
    print("ENCODING BENCHMARKS")
    print("="*70)
    
    encoding_results = []
    for config in configurations:
        result = benchmark_encoding(titles, config)
        encoding_results.append(result)
        
        # Save fingerprints from best config for search benchmark
        if config['name'] == 'Randomized SVD + ITQ' and result['success']:
            best_fingerprints = result['fingerprints']
            best_config = config
    
    # Run search benchmarks on best configuration
    print("\n" + "="*70)
    print("SEARCH BENCHMARKS")
    print("="*70)
    
    if 'best_fingerprints' in locals() and best_fingerprints is not None:
        print(f"\nUsing configuration: {best_config['name']}")
        
        search_backends = ['numpy', 'numba', 'torch']
        search_results = []
        
        for backend in search_backends:
            result = benchmark_search(
                best_fingerprints, titles, backend, 
                best_config['bit_packing'], best_config['n_bits']
            )
            search_results.append(result)
    else:
        print("  No successful encoding to test search on")
        search_results = []
    
    # Generate summary tables
    print("\n" + "="*70)
    print("SUMMARY TABLES")
    print("="*70)
    
    # Encoding performance table
    print("\n### Encoding Performance\n")
    print("| Configuration | Total Time | Train | Encode | Memory | Compression | Docs/sec |")
    print("|--------------|------------|-------|--------|--------|-------------|----------|")
    
    for result in encoding_results:
        if result['success']:
            config = result['config']['name'][:25]
            times = result['times']
            memory = result['memory']
            metrics = result['metrics']
            
            print(f"| {config:25} | {times['total']:10.2f}s | {times['train']:5.2f}s | "
                  f"{times['encode']:6.2f}s | {memory['peak_mb']:6.1f}MB | "
                  f"{memory['compression_ratio']:11.1f}x | {metrics['docs_per_second']:8.0f} |")
    
    # Search performance table
    if search_results:
        print("\n### Search Performance\n")
        print("| Backend | Build Index | Search Time | Throughput | Precision@10 |")
        print("|---------|-------------|-------------|------------|--------------|")
        
        for result in search_results:
            if result['success']:
                backend = result['backend']
                times = result['times']
                metrics = result['metrics']
                
                print(f"| {backend:7} | {times['build_index']:11.3f}s | "
                      f"{times['mean_search_ms']:11.2f}ms | "
                      f"{metrics['throughput_docs_per_sec']:10.0f} | "
                      f"{metrics['precision_at_10']:12.2f} |")
    
    # Memory comparison table
    print("\n### Memory Usage Comparison\n")
    print("| Configuration | Unpacked (MB) | Packed (MB) | Compression Ratio |")
    print("|--------------|---------------|-------------|-------------------|")
    
    for result in encoding_results:
        if result['success']:
            config = result['config']['name'][:30]
            memory = result['memory']
            print(f"| {config:30} | {memory['unpacked_mb']:13.1f} | "
                  f"{memory['packed_mb']:11.1f} | {memory['compression_ratio']:17.1f}x |")
    
    # Best practices recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS BASED ON BENCHMARKS")
    print("="*70)
    
    # Find best configurations
    if encoding_results:
        fastest = min([r for r in encoding_results if r['success']], 
                     key=lambda x: x['times']['total'])
        most_memory_efficient = min([r for r in encoding_results if r['success']], 
                                   key=lambda x: x['memory']['peak_mb'])
        
        print(f"\n✓ Fastest Encoding: {fastest['config']['name']}")
        print(f"  Time: {fastest['times']['total']:.2f}s")
        
        print(f"\n✓ Most Memory Efficient: {most_memory_efficient['config']['name']}")
        print(f"  Peak Memory: {most_memory_efficient['memory']['peak_mb']:.1f}MB")
    
    if search_results:
        fastest_search = min([r for r in search_results if r['success']], 
                           key=lambda x: x['times']['mean_search_ms'])
        print(f"\n✓ Fastest Search Backend: {fastest_search['backend']}")
        print(f"  Mean Search Time: {fastest_search['times']['mean_search_ms']:.2f}ms")
    
    # Save detailed results to JSON
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': n_samples,
        'encoding_results': encoding_results,
        'search_results': search_results
    }
    
    output_file = 'benchmark_results_complete.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Detailed results saved to {output_file}")
    
    return all_results

def main():
    """Run the complete benchmark suite."""
    try:
        results = run_comprehensive_benchmark()
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)
        print("\nUse these results to update documentation with real, verified numbers.")
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())