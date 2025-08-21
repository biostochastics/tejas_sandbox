#!/usr/bin/env python3
"""
Comprehensive test to verify randomized SVD against sklearn.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from randomized_svd import RandomizedSVD

def test_accuracy():
    """Test accuracy compared to sklearn's TruncatedSVD."""
    print("="*60)
    print("RANDOMIZED SVD vs SKLEARN TRUNCATEDSVD COMPARISON")
    print("="*60)
    
    test_cases = [
        (100, 50, 10),    # Small matrix
        (500, 200, 30),   # Medium matrix
        (1000, 500, 50),  # Larger matrix
    ]
    
    for n_samples, n_features, n_components in test_cases:
        print(f"\n[{n_samples}x{n_features}] Components: {n_components}")
        
        # Generate test data
        np.random.seed(42)
        U_true, _ = np.linalg.qr(np.random.randn(n_samples, n_components))
        V_true, _ = np.linalg.qr(np.random.randn(n_features, n_components))
        S_true = np.exp(-np.arange(n_components))
        X = U_true @ np.diag(S_true) @ V_true.T
        X += 0.01 * np.random.randn(n_samples, n_features)
        
        # sklearn TruncatedSVD
        sklearn_svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
        X_transformed_sklearn = sklearn_svd.fit_transform(X)
        S_sklearn = sklearn_svd.singular_values_
        Vt_sklearn = sklearn_svd.components_
        # Extract U from sklearn (it returns U*S)
        U_sklearn = X_transformed_sklearn / S_sklearn[np.newaxis, :]
        
        # Our RandomizedSVD
        custom_svd = RandomizedSVD(n_components=n_components, n_iter=5, 
                                  n_oversamples=10, random_state=42)
        U_custom, S_custom, Vt_custom = custom_svd.fit_transform(X)
        
        # Compare singular values
        s_error = np.linalg.norm(S_custom - S_sklearn) / np.linalg.norm(S_sklearn)
        
        # Compare reconstruction
        X_recon_sklearn = (U_sklearn * S_sklearn) @ Vt_sklearn
        X_recon_custom = U_custom @ np.diag(S_custom) @ Vt_custom
        
        recon_error_sklearn = np.linalg.norm(X - X_recon_sklearn, 'fro') / np.linalg.norm(X, 'fro')
        recon_error_custom = np.linalg.norm(X - X_recon_custom, 'fro') / np.linalg.norm(X, 'fro')
        
        # Compare subspaces using principal angles
        # The columns should span similar subspaces even if signs differ
        M = np.abs(U_custom.T @ U_sklearn)
        principal_angles = np.arccos(np.clip(np.linalg.svd(M)[1], -1, 1))
        max_angle = np.max(principal_angles) * 180 / np.pi
        
        print(f"  Singular values rel. error: {s_error:.3e}")
        print(f"  Reconstruction error sklearn: {recon_error_sklearn:.3e}")
        print(f"  Reconstruction error custom:  {recon_error_custom:.3e}")
        print(f"  Max principal angle: {max_angle:.1f}°")
        
        # Test passes if errors are small
        if s_error < 0.01 and recon_error_custom < recon_error_sklearn * 1.2 and max_angle < 10:
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED")

if __name__ == "__main__":
    test_accuracy()
