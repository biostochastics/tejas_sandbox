#!/usr/bin/env python3
"""
Simple Multi-Configuration Test
================================
Direct testing of encoder configurations without complex benchmark framework.
"""

import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import gc

sys.path.insert(0, str(Path.cwd()))

# Import encoders
from core.encoder import GoldenRatioEncoder

def load_documents(n_docs):
    """Load documents from Wikipedia dataset."""
    data_file = Path("data/wikipedia/wikipedia_10000.txt")
    if not data_file.exists():
        data_file = Path("data/wikipedia/wikipedia_en_20231101_titles.txt")
    
    with open(data_file, 'r') as f:
        docs = [line.strip() for line in f.readlines()[:n_docs] if line.strip()]
    
    return docs[:n_docs]

def test_configuration(n_docs, n_bits, max_features):
    """Test a single configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {n_docs} docs, {n_bits} bits, {max_features} features")
    print(f"{'='*60}")
    
    # Load documents
    docs = load_documents(n_docs)
    print(f"Loaded {len(docs)} documents")
    
    # Create encoder
    encoder = GoldenRatioEncoder(
        n_bits=n_bits,
        max_features=max_features
    )
    
    # Training phase
    gc.collect()
    start = time.time()
    encoder.fit(docs)
    train_time = time.time() - start
    print(f"✓ Training: {train_time:.2f}s")
    
    # Encoding phase - test on subset
    test_docs = docs[:100]
    gc.collect()
    start = time.time()
    fingerprints = encoder.transform(test_docs)
    encode_time = time.time() - start
    encode_speed = len(test_docs) / encode_time
    print(f"✓ Encoding: {encode_speed:.0f} docs/s")
    
    # Search phase - simple similarity search
    # Convert to numpy if torch tensor
    if hasattr(fingerprints, 'numpy'):
        fingerprints_np = fingerprints.numpy()
    else:
        fingerprints_np = fingerprints
    
    query_fp = fingerprints_np[0]
    gc.collect()
    start = time.time()
    
    # Compute Hamming distances
    distances = np.sum(fingerprints_np != query_fp, axis=1)
    top_k = np.argsort(distances)[:10]
    
    search_time = time.time() - start
    search_speed = 1 / search_time
    print(f"✓ Search: {search_speed:.0f} queries/s")
    
    # Memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"✓ Memory: {memory_mb:.0f}MB")
    
    return {
        "n_docs": n_docs,
        "n_bits": n_bits,
        "max_features": max_features,
        "train_time": train_time,
        "encode_speed": encode_speed,
        "search_speed": search_speed,
        "memory_mb": memory_mb
    }

def main():
    """Run multiple configurations."""
    print("\n" + "="*60)
    print("SIMPLE MULTI-CONFIGURATION TEST")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")
    
    # Test configurations
    configs = [
        # Vary bit size
        (1000, 64, 5000),
        (1000, 128, 5000),
        (1000, 256, 5000),
        
        # Vary document count
        (500, 128, 5000),
        (1000, 128, 5000),
        (2000, 128, 5000),
        
        # Vary features
        (1000, 128, 2500),
        (1000, 128, 5000),
        (1000, 128, 10000),
    ]
    
    results = []
    for n_docs, n_bits, max_features in configs:
        try:
            result = test_configuration(n_docs, n_bits, max_features)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "n_docs": n_docs,
                "n_bits": n_bits,
                "max_features": max_features,
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Docs':<6} {'Bits':<6} {'Feats':<8} {'Train(s)':<10} {'Encode(d/s)':<12} {'Search(q/s)':<12} {'Mem(MB)':<8}")
    print("-" * 70)
    
    for r in results:
        if 'error' not in r:
            print(f"{r['n_docs']:<6} {r['n_bits']:<6} {r['max_features']:<8} "
                  f"{r['train_time']:<10.2f} {r['encode_speed']:<12.0f} "
                  f"{r['search_speed']:<12.0f} {r['memory_mb']:<8.0f}")
        else:
            print(f"{r['n_docs']:<6} {r['n_bits']:<6} {r['max_features']:<8} "
                  f"{'FAILED':<10} {'-':<12} {'-':<12} {'-':<8}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Impact of bits
    bits_results = [r for r in results if 'error' not in r and r['n_docs'] == 1000 and r['max_features'] == 5000]
    if len(bits_results) > 1:
        print("\n📊 Impact of bit size (1000 docs, 5000 features):")
        for r in sorted(bits_results, key=lambda x: x['n_bits']):
            print(f"  {r['n_bits']:3} bits: {r['encode_speed']:6.0f} docs/s, {r['memory_mb']:4.0f}MB")
    
    # Impact of scale
    scale_results = [r for r in results if 'error' not in r and r['n_bits'] == 128 and r['max_features'] == 5000]
    if len(scale_results) > 1:
        print("\n📈 Impact of scale (128 bits, 5000 features):")
        for r in sorted(scale_results, key=lambda x: x['n_docs']):
            print(f"  {r['n_docs']:4} docs: {r['train_time']:5.2f}s training, {r['encode_speed']:6.0f} docs/s")
    
    # Impact of features
    feat_results = [r for r in results if 'error' not in r and r['n_docs'] == 1000 and r['n_bits'] == 128]
    if len(feat_results) > 1:
        print("\n🔧 Impact of features (1000 docs, 128 bits):")
        for r in sorted(feat_results, key=lambda x: x['max_features']):
            print(f"  {r['max_features']:5} features: {r['train_time']:5.2f}s training, {r['memory_mb']:4.0f}MB")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()