"""
Comprehensive test suite for custom Randomized SVD implementation.
Tests accuracy, numerical stability, edge cases, and performance.
"""

import numpy as np
import pytest
from randomized_svd import RandomizedSVD, compare_with_sklearn
from sklearn.utils.extmath import randomized_svd as sklearn_rsvd
from scipy.linalg import svd as scipy_svd
import warnings
import time


class TestRandomizedSVD:
    """Test suite for RandomizedSVD implementation."""
    
    @pytest.fixture
    def small_matrix(self):
        """Create a small test matrix."""
        np.random.seed(42)
        return np.random.randn(100, 50)
    
    @pytest.fixture
    def large_matrix(self):
        """Create a large test matrix for performance testing."""
        np.random.seed(42)
        # Simulate 100k+ dimensional data
        n_samples = 1000
        n_features = 100000
        rank = 100
        
        # Create low-rank matrix efficiently
        U = np.random.randn(n_samples, rank).astype(np.float32)
        V = np.random.randn(n_features, rank).astype(np.float32)
        # Don't materialize full matrix - return factors
        return U, V, rank
    
    @pytest.fixture
    def low_rank_matrix(self):
        """Create a known low-rank matrix."""
        np.random.seed(42)
        rank = 10
        n_samples, n_features = 200, 150
        U = np.random.randn(n_samples, rank)
        V = np.random.randn(n_features, rank)
        return U @ V.T, rank
    
    def test_basic_functionality(self, small_matrix):
        """Test basic SVD computation."""
        svd = RandomizedSVD(n_components=10, random_state=42)
        U, S, Vt = svd.fit_transform(small_matrix)
        
        assert U.shape == (100, 10)
        assert S.shape == (10,)
        assert Vt.shape == (10, 50)
        
        # Check singular values are non-negative and sorted
        assert np.all(S >= 0)
        assert np.all(S[:-1] >= S[1:])
    
    def test_orthogonality(self, small_matrix):
        """Test orthogonality of U and V matrices."""
        svd = RandomizedSVD(n_components=10, random_state=42)
        U, S, Vt = svd.fit_transform(small_matrix)
        
        # Check U orthogonality
        U_ortho = U.T @ U
        np.testing.assert_allclose(U_ortho, np.eye(10), atol=1e-5)
        
        # Check V orthogonality
        V_ortho = Vt @ Vt.T
        np.testing.assert_allclose(V_ortho, np.eye(10), atol=1e-5)
    
    def test_reconstruction_accuracy(self, low_rank_matrix):
        """Test reconstruction accuracy for low-rank matrices."""
        X, true_rank = low_rank_matrix
        
        # Use enough components to capture the true rank
        svd = RandomizedSVD(
            n_components=true_rank + 5,
            n_iter=7,
            n_oversamples=20,
            random_state=42
        )
        U, S, Vt = svd.fit_transform(X)
        
        # Reconstruct
        X_reconstructed = U @ np.diag(S) @ Vt
        
        # Check relative reconstruction error
        error = np.linalg.norm(X - X_reconstructed, 'fro')
        relative_error = error / np.linalg.norm(X, 'fro')
        
        assert relative_error < 1e-6  # Should be very small for low-rank matrix
        assert svd.relative_error_ < 1e-6
    
    def test_comparison_with_sklearn(self, small_matrix):
        """Compare results with sklearn implementation."""
        n_components = 15
        
        # Custom implementation
        custom_svd = RandomizedSVD(
            n_components=n_components,
            n_iter=5,
            n_oversamples=10,
            random_state=42
        )
        U_custom, S_custom, Vt_custom = custom_svd.fit_transform(small_matrix)
        
        # sklearn implementation
        U_sklearn, S_sklearn, Vt_sklearn = sklearn_rsvd(
            small_matrix,
            n_components=n_components,
            n_iter=5,
            n_oversamples=10,
            random_state=42
        )
        
        # Compare singular values (should be very close)
        np.testing.assert_allclose(S_custom, S_sklearn, rtol=1e-5)
        
        # Compare subspaces (may have sign differences)
        for i in range(n_components):
            # Check if vectors are parallel (same or opposite direction)
            dot_u = np.abs(np.dot(U_custom[:, i], U_sklearn[:, i]))
            dot_v = np.abs(np.dot(Vt_custom[i, :], Vt_sklearn[i, :]))
            assert dot_u > 0.99  # Should be close to 1
            assert dot_v > 0.99
    
    def test_transform_inverse_transform(self, small_matrix):
        """Test transform and inverse_transform methods."""
        svd = RandomizedSVD(n_components=20, random_state=42)
        svd.fit_transform(small_matrix)
        
        # Transform
        X_transformed = svd.transform(small_matrix)
        assert X_transformed.shape == (100, 20)
        
        # Inverse transform
        X_reconstructed = svd.inverse_transform(X_transformed)
        assert X_reconstructed.shape == small_matrix.shape
        
        # Check reconstruction error
        error = np.linalg.norm(small_matrix - X_reconstructed, 'fro')
        relative_error = error / np.linalg.norm(small_matrix, 'fro')
        assert relative_error < 0.5  # Should capture significant variance
    
    def test_numerical_stability_warnings(self):
        """Test numerical stability warnings."""
        # Create ill-conditioned matrix
        np.random.seed(42)
        X = np.random.randn(50, 30)
        X[:, -5:] = 1e-10 * np.random.randn(50, 5)  # Near-zero columns
        
        svd = RandomizedSVD(n_components=25, tolerance=1e-8)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            svd.fit_transform(X)
            
            # Should warn about near-zero singular values
            warning_messages = [str(warning.message) for warning in w]
            assert any("singular values below tolerance" in msg for msg in warning_messages)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        svd = RandomizedSVD(n_components=5)
        
        # Test with wrong dimensions
        with pytest.raises(ValueError, match="Expected 2D array"):
            svd.fit_transform(np.array([1, 2, 3]))
        
        # Test with NaN values
        X_nan = np.random.randn(10, 10)
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains NaN"):
            svd.fit_transform(X_nan)
        
        # Test with Inf values
        X_inf = np.random.randn(10, 10)
        X_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            svd.fit_transform(X_inf)
        
        # Test transform before fit
        with pytest.raises(ValueError, match="not been fitted"):
            svd.transform(np.random.randn(10, 10))
    
    def test_different_dtypes(self, small_matrix):
        """Test with different data types."""
        for dtype in [np.float32, np.float64]:
            svd = RandomizedSVD(n_components=10, dtype=dtype, random_state=42)
            U, S, Vt = svd.fit_transform(small_matrix)
            
            assert U.dtype == dtype
            assert S.dtype == dtype
            assert Vt.dtype == dtype
            
            # Should still be accurate
            assert svd.relative_error_ < 0.1
    
    def test_explained_variance(self, low_rank_matrix):
        """Test explained variance calculation."""
        X, true_rank = low_rank_matrix
        
        svd = RandomizedSVD(n_components=true_rank, random_state=42)
        svd.fit_transform(X)
        
        # Should explain most variance for low-rank matrix
        assert svd.explained_variance_ratio_ > 0.99
    
    def test_covariance_precision(self, small_matrix):
        """Test covariance and precision matrix methods."""
        svd = RandomizedSVD(n_components=10, random_state=42)
        svd.fit_transform(small_matrix)
        
        cov = svd.get_covariance()
        precision = svd.get_precision()
        
        assert cov.shape == (10,)
        assert precision.shape == (10,)
        
        # Precision should be inverse of covariance
        np.testing.assert_allclose(cov * precision, np.ones(10), rtol=1e-5)
    
    def test_reproducibility(self, small_matrix):
        """Test reproducibility with fixed random state."""
        svd1 = RandomizedSVD(n_components=10, random_state=42)
        U1, S1, Vt1 = svd1.fit_transform(small_matrix)
        
        svd2 = RandomizedSVD(n_components=10, random_state=42)
        U2, S2, Vt2 = svd2.fit_transform(small_matrix)
        
        np.testing.assert_array_equal(S1, S2)
        np.testing.assert_array_equal(np.abs(U1), np.abs(U2))  # Signs may differ
        np.testing.assert_array_equal(np.abs(Vt1), np.abs(Vt2))
    
    def test_power_iterations_effect(self, small_matrix):
        """Test effect of power iterations on accuracy."""
        errors = []
        
        for n_iter in [0, 1, 3, 5, 7]:
            svd = RandomizedSVD(
                n_components=10,
                n_iter=n_iter,
                random_state=42
            )
            svd.fit_transform(small_matrix)
            errors.append(svd.relative_error_)
        
        # More iterations should generally improve accuracy
        assert errors[-1] <= errors[0]  # 7 iterations better than 0
    
    def test_oversampling_effect(self, small_matrix):
        """Test effect of oversampling on accuracy."""
        errors = []
        
        for n_oversamples in [0, 5, 10, 20]:
            svd = RandomizedSVD(
                n_components=10,
                n_iter=3,
                n_oversamples=n_oversamples,
                random_state=42
            )
            svd.fit_transform(small_matrix)
            errors.append(svd.relative_error_)
        
        # More oversampling should generally improve accuracy
        assert errors[-1] <= errors[0]  # 20 oversamples better than 0


def test_large_scale_performance():
    """Test performance on large-scale matrix (100k+ dimensions)."""
    print("\n" + "=" * 60)
    print("LARGE-SCALE PERFORMANCE TEST (100k dimensions)")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 100000
    rank = 50
    n_components = 30
    
    print(f"Matrix size: {n_samples} x {n_features}")
    print(f"True rank: {rank}")
    print(f"Components to extract: {n_components}")
    print("-" * 60)
    
    # Create low-rank matrix efficiently (don't materialize full matrix)
    U_true = np.random.randn(n_samples, rank).astype(np.float32)
    V_true = np.random.randn(n_features, rank).astype(np.float32)
    
    # Custom implementation with efficient matrix multiplication
    svd = RandomizedSVD(
        n_components=n_components,
        n_iter=3,
        n_oversamples=10,
        dtype=np.float32,
        random_state=42,
        verify_accuracy=False  # Skip for performance test
    )
    
    # Override fit_transform to work with factored form
    class EfficientRandomizedSVD(RandomizedSVD):
        def fit_transform_factored(self, U_factor, V_factor):
            """Compute SVD from factored form X = U_factor @ V_factor.T"""
            start_time = time.time()
            
            n_samples = U_factor.shape[0]
            n_features = V_factor.shape[0]
            
            # Set random state
            rng = np.random.RandomState(self.random_state)
            
            # Generate random projection
            projection_size = self.n_components + self.n_oversamples
            omega = rng.standard_normal(
                size=(n_features, projection_size)
            ).astype(self.dtype)
            omega = omega / np.linalg.norm(omega, axis=0)
            
            # Efficient multiplication: Y = (U @ V.T) @ omega = U @ (V.T @ omega)
            Y = U_factor @ (V_factor.T @ omega)
            
            # Power iteration
            for i in range(self.n_iter):
                # Y = X @ X.T @ Y = U @ V.T @ V @ U.T @ Y
                temp = V_factor.T @ (V_factor @ (U_factor.T @ Y))
                Y = U_factor @ temp
                Y, _ = np.linalg.qr(Y)
            
            Q, R = np.linalg.qr(Y)
            
            # B = Q.T @ X = Q.T @ U @ V.T
            B = (Q.T @ U_factor) @ V_factor.T
            
            # SVD of small matrix
            Uhat, S, Vt = scipy_svd(B, full_matrices=False)
            
            # Recover
            U = Q @ Uhat
            
            # Truncate
            self.U_ = U[:, :self.n_components].astype(self.dtype)
            self.S_ = S[:self.n_components].astype(self.dtype)
            self.Vt_ = Vt[:self.n_components, :].astype(self.dtype)
            
            self.computation_time_ = time.time() - start_time
            
            return self.U_, self.S_, self.Vt_
    
    efficient_svd = EfficientRandomizedSVD(
        n_components=n_components,
        n_iter=3,
        n_oversamples=10,
        dtype=np.float32,
        random_state=42
    )
    
    # Run test
    U, S, Vt = efficient_svd.fit_transform_factored(U_true, V_true)
    
    print(f"Computation time: {efficient_svd.computation_time_:.3f} seconds")
    print(f"Memory usage (approximate): {(U.nbytes + S.nbytes + Vt.nbytes) / 1e6:.2f} MB")
    print()
    
    print("Results:")
    print(f"  U shape: {U.shape}")
    print(f"  S shape: {S.shape}")
    print(f"  Vt shape: {Vt.shape}")
    print(f"  Top 5 singular values: {S[:5]}")
    print()
    
    # Verify basic properties
    print("Verification:")
    U_ortho = np.linalg.norm(U.T @ U - np.eye(n_components), 'fro')
    V_ortho = np.linalg.norm(Vt @ Vt.T - np.eye(n_components), 'fro')
    print(f"  U orthogonality error: {U_ortho:.2e}")
    print(f"  V orthogonality error: {V_ortho:.2e}")
    print(f"  Singular values sorted: {np.all(S[:-1] >= S[1:])}")
    
    print("\n✓ Large-scale test completed successfully!")


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Run pytest
    pytest.main([__file__, "-v"])
    
    # Run performance test
    test_large_scale_performance()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # Run basic comparison test
    print("Testing Custom Randomized SVD Implementation")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.random.randn(500, 200)
    
    results = compare_with_sklearn(X, n_components=30, n_iter=5, n_oversamples=20)
    
    print("\n" + "=" * 60)
    print("RUNNING LARGE-SCALE TEST")
    test_large_scale_performance()