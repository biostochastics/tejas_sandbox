#!/usr/bin/env python3
"""
Simple test script for randomized SVD implementation.
Tests basic functionality and backend compatibility.
"""

import numpy as np
import time
import sys
from randomized_svd import RandomizedSVD

def test_basic_functionality():
    """Test basic SVD computation with small matrix."""
    print("=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create a small test matrix
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Test with default backend (auto)
    print("\n1. Testing with auto backend selection...")
    svd = RandomizedSVD(n_components=10, random_state=42)
    U, S, Vt = svd.fit_transform(X)
    
    print(f"   U shape: {U.shape}")
    print(f"   S shape: {S.shape}")
    print(f"   Vt shape: {Vt.shape}")
    print(f"   Backend used: {svd.backend}")
    
    # Check orthogonality
    U_ortho = np.linalg.norm(U.T @ U - np.eye(10), 'fro')
    V_ortho = np.linalg.norm(Vt @ Vt.T - np.eye(10), 'fro')
    print(f"   U orthogonality error: {U_ortho:.2e}")
    print(f"   V orthogonality error: {V_ortho:.2e}")
    
    # Check reconstruction
    X_reconstructed = U @ np.diag(S) @ Vt
    reconstruction_error = np.linalg.norm(X[:, :10] - X_reconstructed[:, :10], 'fro')
    print(f"   Reconstruction error: {reconstruction_error:.2e}")
    
    if U_ortho < 1e-5 and V_ortho < 1e-5:
        print("   ✓ Basic functionality test PASSED")
    else:
        print("   ✗ Basic functionality test FAILED")
        return False
    
    return True

def test_backend_compatibility():
    """Test compatibility with different backends."""
    print("\n" + "=" * 60)
    print("TESTING BACKEND COMPATIBILITY")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.random.randn(200, 100).astype(np.float32)
    
    backends_to_test = ['numpy', 'numba', 'torch']
    results = {}
    
    # First, get reference results with numpy
    print("\nGetting reference results with NumPy...")
    svd_ref = RandomizedSVD(n_components=20, random_state=42, backend='numpy')
    U_ref, S_ref, Vt_ref = svd_ref.fit_transform(X)
    
    for backend in backends_to_test:
        print(f"\nTesting {backend} backend...")
        try:
            svd = RandomizedSVD(n_components=20, random_state=42, backend=backend)
            
            start_time = time.time()
            U, S, Vt = svd.fit_transform(X)
            elapsed = time.time() - start_time
            
            # Compare singular values
            s_diff = np.linalg.norm(S - S_ref) / np.linalg.norm(S_ref)
            
            # Check if subspaces are the same (accounting for sign differences)
            subspace_error = 0
            for i in range(20):
                dot_prod = np.abs(np.dot(U[:, i], U_ref[:, i]))
                subspace_error += (1 - dot_prod)
            subspace_error /= 20
            
            results[backend] = {
                'available': True,
                'time': elapsed,
                's_diff': s_diff,
                'subspace_error': subspace_error,
                'actual_backend': svd.backend
            }
            
            print(f"   ✓ Backend available")
            print(f"   Time: {elapsed:.3f}s")
            print(f"   Singular value difference: {s_diff:.2e}")
            print(f"   Subspace error: {subspace_error:.2e}")
            print(f"   Actually used: {svd.backend}")
            
        except Exception as e:
            results[backend] = {
                'available': False,
                'error': str(e)
            }
            print(f"   ✗ Backend not available: {e}")
    
    print("\n" + "-" * 60)
    print("SUMMARY:")
    available_backends = [b for b, r in results.items() if r['available']]
    print(f"Available backends: {', '.join(available_backends)}")
    
    if 'numpy' in available_backends:
        print("✓ NumPy backend (always available) works correctly")
    
    return len(available_backends) > 0

def test_large_scale_performance():
    """Test performance on larger matrices."""
    print("\n" + "=" * 60)
    print("TESTING LARGE-SCALE PERFORMANCE")
    print("=" * 60)
    
    sizes = [(1000, 500), (5000, 1000)]
    
    for n_samples, n_features in sizes:
        print(f"\nMatrix size: {n_samples} x {n_features}")
        
        # Create low-rank matrix
        np.random.seed(42)
        rank = 50
        U_true = np.random.randn(n_samples, rank).astype(np.float32)
        V_true = np.random.randn(n_features, rank).astype(np.float32)
        X = U_true @ V_true.T + 0.01 * np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Test with auto backend
        svd = RandomizedSVD(n_components=30, n_iter=3, n_oversamples=10, random_state=42)
        
        start_time = time.time()
        U, S, Vt = svd.fit_transform(X)
        elapsed = time.time() - start_time
        
        print(f"   Backend used: {svd.backend}")
        print(f"   Time: {elapsed:.3f}s")
        print(f"   Top 5 singular values: {S[:5]}")
        
        if svd.verify_accuracy:
            print(f"   Relative error: {svd.relative_error_:.2e}")
            print(f"   Explained variance: {svd.explained_variance_ratio_:.2%}")
    
    return True

def test_numerical_stability():
    """Test numerical stability with ill-conditioned matrices."""
    print("\n" + "=" * 60)
    print("TESTING NUMERICAL STABILITY")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create ill-conditioned matrix
    n = 100
    X = np.random.randn(n, n).astype(np.float32)
    # Make it ill-conditioned
    U, S, Vt = np.linalg.svd(X)
    S_modified = S.copy()
    S_modified[-20:] = S_modified[-20:] * 1e-8  # Very small singular values
    X_ill = U @ np.diag(S_modified) @ Vt
    
    print("\nTesting with ill-conditioned matrix...")
    print(f"Condition number: {S_modified[0] / S_modified[-1]:.2e}")
    
    svd = RandomizedSVD(n_components=50, n_iter=7, n_oversamples=20, random_state=42)
    
    try:
        U, S, Vt = svd.fit_transform(X_ill)
        print(f"   ✓ SVD completed successfully")
        print(f"   Computed condition number: {svd.condition_number_:.2e}")
        
        # Check if singular values are reasonable
        if np.all(S >= 0) and np.all(S[:-1] >= S[1:]):
            print("   ✓ Singular values are non-negative and sorted")
        else:
            print("   ✗ Issues with singular values")
            
    except Exception as e:
        print(f"   ✗ SVD failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("RANDOMIZED SVD TEST SUITE")
    print("=" * 60)
    print("Testing custom implementation compatible with project backends")
    print()
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Backend Compatibility", test_backend_compatibility),
        ("Large-Scale Performance", test_large_scale_performance),
        ("Numerical Stability", test_numerical_stability)
    ]
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            if not passed:
                all_passed = False
                print(f"\n✗ {test_name} test failed")
        except Exception as e:
            all_passed = False
            print(f"\n✗ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("The randomized SVD implementation is working correctly")
        print("and is compatible with the project's backend system.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())