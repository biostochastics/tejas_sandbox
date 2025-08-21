#!/usr/bin/env python3
import numpy as np
from sklearn.decomposition import TruncatedSVD
from randomized_svd import RandomizedSVD

# Test with larger matrix
np.random.seed(42)
n_samples, n_features = 500, 200
n_components = 30

# Create test matrix
X = np.random.randn(n_samples, n_features)

# sklearn
sklearn_svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=42)
X_transformed_sklearn = sklearn_svd.fit_transform(X)
S_sklearn = sklearn_svd.singular_values_
Vt_sklearn = sklearn_svd.components_
U_sklearn = X_transformed_sklearn / S_sklearn[np.newaxis, :]

# Our implementation
custom_svd = RandomizedSVD(n_components=n_components, n_iter=5, n_oversamples=10, random_state=42)
U_custom, S_custom, Vt_custom = custom_svd.fit_transform(X)

# Directly compare reconstruction without caring about signs
X_recon_sklearn = X_transformed_sklearn @ Vt_sklearn
X_recon_custom = (U_custom * S_custom) @ Vt_custom

# Check reconstruction errors
error_sklearn = np.linalg.norm(X - X_recon_sklearn, 'fro')
error_custom = np.linalg.norm(X - X_recon_custom, 'fro')

print(f"Reconstruction error sklearn: {error_sklearn:.3f}")
print(f"Reconstruction error custom:  {error_custom:.3f}")
print(f"Relative difference: {abs(error_custom - error_sklearn)/error_sklearn:.3%}")

# Check if singular values match
s_diff = np.linalg.norm(S_custom - S_sklearn) / np.linalg.norm(S_sklearn)
print(f"\nSingular value relative error: {s_diff:.3e}")

# Check orthogonality 
print(f"\nOrthogonality check:")
print(f"U^T @ U (should be I): max off-diagonal = {np.max(np.abs(U_custom.T @ U_custom - np.eye(n_components))):.3e}")
print(f"V^T @ V (should be I): max off-diagonal = {np.max(np.abs(Vt_custom @ Vt_custom.T - np.eye(n_components))):.3e}")
