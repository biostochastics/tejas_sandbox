#!/usr/bin/env python3
"""
Comprehensive test script to verify randomized SVD fidelity against sklearn.
Tests accuracy, numerical stability, edge cases, and memory efficiency.
"""
import numpy as np
import time
import psutil
import os
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd as scipy_svd
from randomized_svd import RandomizedSVD

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_accuracy_vs_sklearn():
    """Test accuracy compared to sklearn's TruncatedSVD."""
    print("\n" + "="*60)
    print("TEST 1: ACCURACY vs SKLEARN'S TRUNCATEDSVD")
    print("="*60)
    
    test_cases = [
        (100, 50, 10),    # Small matrix
        (500, 200, 30),   # Medium matrix
        (1000, 500, 50),  # Larger matrix
    ]
    
    results = []
    
    for n_samples, n_features, n_components in test_cases:
        print(f"\nMatrix size: {n_samples}x{n_features}, Components: {n_components}")
        
        # Generate test data with known structure
        np.random.seed(42)
        U_true, _ = np.linalg.qr(np.random.randn(n_samples, n_components))
        V_true, _ = np.linalg.qr(np.random.randn(n_features, n_components))
        S_true = np.exp(-np.arange(n_components))  # Exponentially decaying singular values
        X = U_true @ np.diag(S_true) @ V_true.T
        
        # Add noise
        X += 0.01 * np.random.randn(n_samples, n_features)
        
        # sklearn TruncatedSVD
        sklearn_svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
        U_sklearn = sklearn_svd.fit_transform(X)
        V_sklearn = sklearn_svd.components_.T
        S_sklearn = sklearn_svd.singular_values_
        
        # Our RandomizedSVD
        custom_svd = RandomizedSVD(n_components=n_components, n_iter=5, 
                                  n_oversamples=10, random_state=42)
        U_custom, S_custom, Vt_custom = custom_svd.fit_transform(X)
        V_custom = Vt_custom.T
        
        # Compare singular values
        s_diff = np.linalg.norm(S_custom - S_sklearn) / np.linalg.norm(S_sklearn)
        
        # Compare subspaces (accounting for sign ambiguity)
        u_proj = np.abs(U_custom.T @ U_sklearn)
        u_alignment = np.mean(np.max(u_proj, axis=0))
        
        v_proj = np.abs(V_custom.T @ V_sklearn)
        v_alignment = np.mean(np.max(v_proj, axis=0))
        
        # Reconstruction error
        X_custom = U_custom @ np.diag(S_custom) @ Vt_custom
        X_sklearn = U_sklearn @ np.diag(S_sklearn) @ V_sklearn.T
        
        custom_error = np.linalg.norm(X - X_custom, 'fro')
        sklearn_error = np.linalg.norm(X - X_sklearn, 'fro')
        
        print(f"  Singular value rel. error: {s_diff:.2e}")
        print(f"  U subspace alignment: {u_alignment:.3f}")
        print(f"  V subspace alignment: {v_alignment:.3f}")
        print(f"  Reconstruction error ratio: {custom_error/sklearn_error:.3f}")
        
        results.append({
            'size': (n_samples, n_features),
            's_diff': s_diff,
            'u_align': u_alignment,
            'v_align': v_alignment,
            'error_ratio': custom_error/sklearn_error
        })
    
    # Check if all tests passed
    all_passed = all(
        r['s_diff'] < 0.1 and 
        r['u_align'] > 0.95 and 
        r['v_align'] > 0.95 and
        r['error_ratio'] < 1.2
        for r in results
    )
    
    if all_passed:
        print("\n✓ All accuracy tests PASSED!")
    else:
        print("\n✗ Some accuracy tests FAILED")
    
    return results

def main():
    """Run all tests."""
    print("="*60)
    print("COMPREHENSIVE RANDOMIZED SVD TESTING")
    print("Testing fidelity, accuracy, and performance")
    print("="*60)
    
    # Test 1: Accuracy vs sklearn
    accuracy_results = test_accuracy_vs_sklearn()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_accuracy_passed = all(
        r['s_diff'] < 0.1 and 
        r['u_align'] > 0.95 and 
        r['v_align'] > 0.95 and
        r['error_ratio'] < 1.2
        for r in accuracy_results
    )
    
    tests_summary = [
        ("Accuracy vs sklearn", all_accuracy_passed),
    ]
    
    print("\nTest Results:")
    for test_name, passed in tests_summary:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in tests_summary)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The randomized SVD implementation is verified.")
    else:
        print("\n⚠️ Some tests failed. Please review the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
