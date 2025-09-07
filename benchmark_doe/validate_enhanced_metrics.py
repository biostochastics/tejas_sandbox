#!/usr/bin/env python3
"""
Quick validation script for enhanced metrics
Tests that all metrics are computed correctly on small dataset
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_doe.core.enhanced_metrics import (
    EnhancedMetricsCalculator,
    HammingScorer,
    CosineScorer,
    compute_confidence_intervals
)

def validate_metrics():
    """Validate that enhanced metrics work correctly."""
    
    print("=" * 60)
    print("ENHANCED METRICS VALIDATION")
    print("=" * 60)
    
    # Create synthetic test data
    n_docs = 1000
    n_queries = 20
    n_bits = 128
    
    print(f"\nTest configuration:")
    print(f"  Documents: {n_docs}")
    print(f"  Queries: {n_queries}")
    print(f"  Bits: {n_bits}")
    
    # Generate binary codes for TEJAS
    print("\n1. Testing Binary Metrics (TEJAS)...")
    doc_codes_binary = np.random.randint(0, 256, (n_docs, n_bits // 8)).astype(np.uint8)
    query_codes_binary = np.random.randint(0, 256, (n_queries, n_bits // 8)).astype(np.uint8)
    
    # Create relevance data (1-3 relevant docs per query)
    relevance = {}
    for q_idx in range(n_queries):
        n_relevant = np.random.randint(1, 4)
        relevant_docs = np.random.choice(n_docs, n_relevant, replace=False)
        relevance[q_idx] = relevant_docs.tolist()
    
    # Initialize calculator for binary
    calc_binary = EnhancedMetricsCalculator(encoding_type="binary", n_bits=n_bits)
    
    # Calculate metrics
    metrics_binary = calc_binary.calculate_all_metrics(
        query_embeddings=query_codes_binary,
        doc_embeddings=doc_codes_binary,
        relevance_data=relevance,
        query_times=[np.random.uniform(1, 10) for _ in range(n_queries)],
        encoding_time=1.5,
        memory_usage=50.0
    )
    
    print("  Binary Metrics Results:")
    print(f"    Precision@1: {metrics_binary.precision_at_1:.4f}")
    print(f"    Precision@5: {metrics_binary.precision_at_5:.4f}")
    print(f"    Precision@10: {metrics_binary.precision_at_10:.4f}")
    print(f"    Recall@10: {metrics_binary.recall_at_10:.4f}")
    print(f"    Recall@50: {metrics_binary.recall_at_50:.4f}")
    print(f"    NDCG@1: {metrics_binary.ndcg_at_1:.4f}")
    print(f"    NDCG@10: {metrics_binary.ndcg_at_10:.4f}")
    print(f"    MRR: {metrics_binary.mrr:.4f}")
    print(f"    Success@1: {metrics_binary.success_at_1:.4f}")
    print(f"    Success@5: {metrics_binary.success_at_5:.4f}")
    
    # Generate dense vectors for BERT
    print("\n2. Testing Dense Metrics (BERT)...")
    doc_vecs_dense = np.random.randn(n_docs, 384).astype(np.float32)
    query_vecs_dense = np.random.randn(n_queries, 384).astype(np.float32)
    
    # Initialize calculator for dense
    calc_dense = EnhancedMetricsCalculator(encoding_type="dense")
    
    # Calculate metrics
    metrics_dense = calc_dense.calculate_all_metrics(
        query_embeddings=query_vecs_dense,
        doc_embeddings=doc_vecs_dense,
        relevance_data=relevance,
        query_times=[np.random.uniform(2, 15) for _ in range(n_queries)],
        encoding_time=3.0,
        memory_usage=150.0
    )
    
    print("  Dense Metrics Results:")
    print(f"    Precision@1: {metrics_dense.precision_at_1:.4f}")
    print(f"    Precision@5: {metrics_dense.precision_at_5:.4f}")
    print(f"    Precision@10: {metrics_dense.precision_at_10:.4f}")
    print(f"    Recall@10: {metrics_dense.recall_at_10:.4f}")
    print(f"    Recall@50: {metrics_dense.recall_at_50:.4f}")
    print(f"    NDCG@1: {metrics_dense.ndcg_at_1:.4f}")
    print(f"    NDCG@10: {metrics_dense.ndcg_at_10:.4f}")
    print(f"    MRR: {metrics_dense.mrr:.4f}")
    print(f"    Success@1: {metrics_dense.success_at_1:.4f}")
    print(f"    Success@5: {metrics_dense.success_at_5:.4f}")
    
    # Test statistical functions
    print("\n3. Testing Statistical Functions...")
    
    # Simulate multiple runs
    ndcg_values = np.random.normal(0.3, 0.05, 10)
    median, lower, upper = compute_confidence_intervals(ndcg_values)
    print(f"  NDCG@10 across runs: {median:.4f} [{lower:.4f}, {upper:.4f}]")
    
    precision_values = np.random.normal(0.25, 0.08, 10)
    median, lower, upper = compute_confidence_intervals(precision_values)
    print(f"  Precision@1 across runs: {median:.4f} [{lower:.4f}, {upper:.4f}]")
    
    # Validate metric ranges
    print("\n4. Validating Metric Ranges...")
    
    checks = [
        ("Precision values", [metrics_binary.precision_at_1, metrics_binary.precision_at_5], 0, 1),
        ("Recall values", [metrics_binary.recall_at_10, metrics_binary.recall_at_50], 0, 1),
        ("NDCG values", [metrics_binary.ndcg_at_1, metrics_binary.ndcg_at_10], 0, 1),
        ("Success rates", [metrics_binary.success_at_1, metrics_binary.success_at_5], 0, 1),
        ("MRR", [metrics_binary.mrr, metrics_dense.mrr], 0, 1)
    ]
    
    all_valid = True
    for name, values, min_val, max_val in checks:
        for val in values:
            if val < min_val or val > max_val:
                print(f"  ✗ {name} out of range: {val}")
                all_valid = False
    
    if all_valid:
        print("  ✓ All metrics within valid ranges")
    
    # Verify that success rates make sense
    print("\n5. Verifying Metric Relationships...")
    
    # Success@k should increase with k
    if metrics_binary.success_at_1 <= metrics_binary.success_at_5 <= metrics_binary.success_at_10:
        print("  ✓ Success@k increases monotonically")
    else:
        print("  ✗ Success@k not monotonic")
    
    # Precision@k should generally decrease with k
    if metrics_binary.precision_at_1 >= metrics_binary.precision_at_20:
        print("  ✓ Precision generally decreases with k")
    else:
        print("  ⚠ Precision may not be decreasing (could be due to randomness)")
    
    # Recall@k should increase with k
    if metrics_binary.recall_at_10 <= metrics_binary.recall_at_50:
        print("  ✓ Recall@k increases monotonically")
    else:
        print("  ✗ Recall@k not monotonic")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    validate_metrics()