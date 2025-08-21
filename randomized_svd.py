"""
Custom Randomized SVD Implementation for Large-Scale Matrices (>100k dimensions)
Based on Halko, Martinsson, and Tropp (2011) algorithm
Optimized for accuracy, numerical stability, and verification
Compatible with Tejas project backend system (numpy, numba, torch, auto)
"""

import numpy as np
from typing import Tuple, Optional, Union, Literal
import warnings
from scipy.linalg import qr, svd as scipy_svd
import time
import logging

logger = logging.getLogger(__name__)

# Check for optional backends
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
    
    @jit(nopython=True, parallel=True)
    def _power_iteration_numba(X_data, X_indices, X_indptr, Y, n_features):
        """Numba-accelerated power iteration for sparse matrices."""
        n_samples, n_components = Y.shape
        result = np.zeros_like(Y)
        
        for i in prange(n_samples):
            for k in range(n_components):
                val = 0.0
                for idx in range(X_indptr[i], X_indptr[i+1]):
                    j = X_indices[idx]
                    val += X_data[idx] * Y[j, k]
                result[i, k] = val
        return result
    
    logger.info("Numba backend available for accelerated SVD operations")
except ImportError:
    HAS_NUMBA = False
    _power_iteration_numba = None
    logger.info("Numba not available for SVD, using numpy backend")

try:
    import torch
    HAS_TORCH = True
    logger.info("PyTorch backend available for GPU-accelerated SVD")
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available for SVD")

# CuPy is not in requirements.txt, so disable by default
HAS_CUPY = False
try:
    # Only enable if explicitly available (user can install separately)
    import cupy as cp
    if cp.cuda.is_available():
        HAS_CUPY = True
        logger.info("CuPy backend available for GPU-accelerated SVD")
except ImportError:
    pass


class RandomizedSVD:
    """
    Randomized SVD implementation with accuracy verification and numerical stability.
    
    Optimized for matrices with dimensions > 100k where only top-k components needed.
    Includes multiple verification methods and numerical stability checks.
    Supports multiple backends: numpy, numba, torch (GPU), cupy (GPU), auto.
    """
    
    def __init__(
        self,
        n_components: int = 100,
        n_oversamples: int = 20,
        n_iter: int = 5,
        random_state: Optional[int] = None,
        dtype: np.dtype = np.float32,
        verify_accuracy: bool = True,
        tolerance: float = 1e-6,
        backend: Literal['numpy', 'numba', 'torch', 'cupy', 'auto'] = 'auto',
        device: str = 'cpu'  # For torch backend: 'cpu', 'cuda', 'mps'
    ):
        """
        Initialize Randomized SVD.
        
        Parameters:
        -----------
        n_components : int
            Number of singular values/vectors to compute (k)
        n_oversamples : int
            Additional random samples for improved accuracy (typically 10-20)
        n_iter : int
            Number of power iterations for improved accuracy (typically 3-7)
        random_state : int, optional
            Random seed for reproducibility
        dtype : np.dtype
            Data type for computation (float32 for memory, float64 for precision)
        verify_accuracy : bool
            Whether to compute and verify reconstruction error
        tolerance : float
            Numerical tolerance for stability checks
        backend : str
            Computation backend: 'numpy', 'numba', 'torch', 'cupy', or 'auto'
        device : str
            Device for torch backend: 'cpu', 'cuda', or 'mps'
        """
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state
        self.dtype = dtype
        self.verify_accuracy = verify_accuracy
        self.tolerance = tolerance
        self.backend = backend
        self.device = device
        
        # Select backend
        self._select_backend()
        
        # Results storage
        self.U_ = None
        self.S_ = None
        self.Vt_ = None
        self.reconstruction_error_ = None
        self.relative_error_ = None
        self.condition_number_ = None
        
        # Performance metrics
        self.computation_time_ = None
        self.memory_peak_ = None
    
    def _select_backend(self):
        """Select and validate the computation backend."""
        if self.backend == 'auto':
            # Auto-select based on availability and hardware
            # Prioritize backends that are in requirements.txt
            if HAS_TORCH:
                # Check for GPU availability
                if torch.cuda.is_available():
                    self.backend = 'torch'
                    self.device = 'cuda'
                    logger.info("Auto-selected PyTorch backend with CUDA")
                else:
                    # MPS has incomplete support for linalg.qr, so use CPU for now
                    # if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    #     self.backend = 'torch'
                    #     self.device = 'mps'
                    #     logger.info("Auto-selected PyTorch backend with MPS (Apple Silicon)")
                    # Use torch on CPU if no GPU (still can be faster for large matrices)
                    self.backend = 'torch'
                    self.device = 'cpu'
                    logger.info("Auto-selected PyTorch backend on CPU")
            elif HAS_NUMBA:
                self.backend = 'numba'
                logger.info("Auto-selected Numba backend (CPU acceleration)")
            elif HAS_CUPY:  # Only if user explicitly installed it
                self.backend = 'cupy'
                logger.info("Auto-selected CuPy backend (GPU acceleration)")
            else:
                self.backend = 'numpy'
                logger.info("Auto-selected NumPy backend")
        
        # Validate selected backend
        if self.backend == 'numba' and not HAS_NUMBA:
            warnings.warn("Numba not available, falling back to numpy", RuntimeWarning)
            self.backend = 'numpy'
        elif self.backend == 'torch' and not HAS_TORCH:
            warnings.warn("PyTorch not available, falling back to numpy", RuntimeWarning)
            self.backend = 'numpy'
        elif self.backend == 'cupy' and not HAS_CUPY:
            warnings.warn("CuPy not available, falling back to numpy", RuntimeWarning)
            self.backend = 'numpy'
        
        logger.info(f"Using backend: {self.backend}" + (f" on {self.device}" if self.backend == 'torch' else ""))
        
    def fit_transform(
        self,
        X: np.ndarray,
        return_components: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute randomized SVD of matrix X.
        
        Parameters:
        -----------
        X : np.ndarray
            Input matrix of shape (n_samples, n_features)
        return_components : bool
            If True, return (U, S, Vt). If False, return only U @ diag(S)
            
        Returns:
        --------
        U, S, Vt or transformed data depending on return_components
        """
        start_time = time.time()
        
        # Input validation and conversion
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        
        # Check if randomized SVD is appropriate
        if self.n_components >= min(n_samples, n_features) * 0.8:
            warnings.warn(
                f"n_components ({self.n_components}) is close to matrix rank. "
                "Consider using standard SVD for better accuracy.",
                UserWarning
            )
        
        # Route to backend-specific implementation
        if self.backend == 'torch':
            U, S, Vt = self._fit_transform_torch(X)
        elif self.backend == 'cupy':
            U, S, Vt = self._fit_transform_cupy(X)
        elif self.backend == 'numba':
            U, S, Vt = self._fit_transform_numba(X)
        else:
            U, S, Vt = self._fit_transform_numpy(X)
        
        # Truncate to requested components
        U = U[:, :self.n_components]
        S = S[:self.n_components]
        Vt = Vt[:self.n_components, :]
        
        # Ensure correct signs (largest element in each column should be positive)
        self._ensure_deterministic_signs(U, Vt)
        
        # Store results
        self.U_ = U.astype(self.dtype)
        self.S_ = S.astype(self.dtype)
        self.Vt_ = Vt.astype(self.dtype)
        
        # Compute verification metrics
        if self.verify_accuracy:
            self._compute_accuracy_metrics(X)
        
        # Check numerical stability
        self._check_numerical_stability()
        
        self.computation_time_ = time.time() - start_time
        
        if return_components:
            return self.U_, self.S_, self.Vt_
        else:
            return self.U_ * self.S_
    
    def _fit_transform_numpy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """NumPy implementation of randomized SVD."""
        n_samples, n_features = X.shape
        
        # Set random state
        rng = np.random.RandomState(self.random_state)
        
        # Generate random projection matrix
        projection_size = self.n_components + self.n_oversamples
        omega = rng.standard_normal(size=(n_features, projection_size)).astype(self.dtype)
        # Don't normalize - it introduces bias in the random projection
        
        # Form Y = X @ Omega with power iteration
        Y = X @ omega
        
        # Power iteration for improved accuracy (Halko et al. 2011)
        for i in range(self.n_iter):
            # Orthogonalize Y first
            Y, _ = qr(Y, mode='economic')
            # Apply A^T
            Z = X.T @ Y
            # Orthogonalize Z
            Z, _ = qr(Z, mode='economic')
            # Apply A
            Y = X @ Z
        
        # QR decomposition
        Q, R = qr(Y, mode='economic')
        
        # Form B = Q.T @ X
        B = Q.T @ X
        
        # SVD of small matrix
        Uhat, S, Vt = scipy_svd(B, full_matrices=False)
        
        # Recover left singular vectors
        U = Q @ Uhat
        
        return U, S, Vt
    
    def _fit_transform_numba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-accelerated implementation of randomized SVD."""
        if not HAS_NUMBA:
            return self._fit_transform_numpy(X)
        
        # For dense matrices, Numba helps mainly with large matrix multiplications
        # Fall back to numpy for now as the main bottleneck is the SVD itself
        return self._fit_transform_numpy(X)
    
    def _fit_transform_torch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PyTorch implementation for GPU acceleration."""
        if not HAS_TORCH:
            return self._fit_transform_numpy(X)
        
        # Convert to torch tensor
        X_torch = torch.from_numpy(X).to(self.device).float()
        n_samples, n_features = X_torch.shape
        
        # Set random state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # Generate random projection
        projection_size = self.n_components + self.n_oversamples
        omega = torch.randn(n_features, projection_size, device=self.device)
        # Don't normalize - it introduces bias in the random projection
        
        # Power iteration
        Y = X_torch @ omega
        
        # Power iteration for improved accuracy (Halko et al. 2011)
        for i in range(self.n_iter):
            # Orthogonalize Y first
            Y, _ = torch.linalg.qr(Y, mode='reduced')
            # Apply A^T
            Z = X_torch.T @ Y
            # Orthogonalize Z
            Z, _ = torch.linalg.qr(Z, mode='reduced')
            # Apply A
            Y = X_torch @ Z
        
        # QR decomposition
        Q, R = torch.linalg.qr(Y, mode='reduced')
        
        # Form B = Q.T @ X
        B = Q.T @ X_torch
        
        # SVD of small matrix
        Uhat, S, Vt = torch.linalg.svd(B, full_matrices=False)
        
        # Recover left singular vectors
        U = Q @ Uhat
        
        # Convert back to numpy
        return U.cpu().numpy(), S.cpu().numpy(), Vt.cpu().numpy()
    
    def _fit_transform_cupy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CuPy implementation for GPU acceleration."""
        if not HAS_CUPY:
            return self._fit_transform_numpy(X)
        
        # Convert to CuPy array
        X_gpu = cp.asarray(X)
        n_samples, n_features = X_gpu.shape
        
        # Set random state
        if self.random_state is not None:
            cp.random.seed(self.random_state)
        
        # Generate random projection
        projection_size = self.n_components + self.n_oversamples
        omega = cp.random.standard_normal(size=(n_features, projection_size)).astype(cp.float32)
        # Don't normalize - it introduces bias in the random projection
        
        # Power iteration
        Y = X_gpu @ omega
        
        # Power iteration for improved accuracy (Halko et al. 2011)
        for i in range(self.n_iter):
            # Orthogonalize Y first
            Y, _ = cp.linalg.qr(Y, mode='reduced')
            # Apply A^T
            Z = X_gpu.T @ Y
            # Orthogonalize Z
            Z, _ = cp.linalg.qr(Z, mode='reduced')
            # Apply A
            Y = X_gpu @ Z
        
        # QR decomposition
        Q, R = cp.linalg.qr(Y, mode='reduced')
        
        # Form B = Q.T @ X
        B = Q.T @ X_gpu
        
        # SVD of small matrix
        Uhat, S, Vt = cp.linalg.svd(B, full_matrices=False)
        
        # Recover left singular vectors
        U = Q @ Uhat
        
        # Convert back to numpy
        return cp.asnumpy(U), cp.asnumpy(S), cp.asnumpy(Vt)
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and prepare input matrix."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array")
        
        # Check for NaN or Inf
        if not np.isfinite(X).all():
            raise ValueError("Input contains NaN or Inf values")
        
        # Convert to specified dtype if needed
        if X.dtype != self.dtype:
            X = X.astype(self.dtype, copy=False)
        
        # Ensure C-contiguous for performance
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        
        return X
    
    def _ensure_deterministic_signs(self, U: np.ndarray, Vt: np.ndarray):
        """Ensure deterministic signs for reproducibility."""
        # Make the largest element in each column positive
        for j in range(U.shape[1]):
            max_idx = np.argmax(np.abs(U[:, j]))
            if U[max_idx, j] < 0:
                U[:, j] *= -1
                Vt[j, :] *= -1
    
    def _compute_accuracy_metrics(self, X: np.ndarray):
        """Compute reconstruction error and other accuracy metrics."""
        # Reconstruction error: ||X - U @ S @ Vt||_F
        X_reconstructed = self.U_ @ np.diag(self.S_) @ self.Vt_
        
        # Frobenius norm of error
        error_matrix = X - X_reconstructed
        self.reconstruction_error_ = np.linalg.norm(error_matrix, 'fro')
        
        # Relative error
        X_norm = np.linalg.norm(X, 'fro')
        self.relative_error_ = self.reconstruction_error_ / X_norm if X_norm > 0 else np.inf
        
        # Explained variance ratio
        total_variance = np.sum(X ** 2)
        explained_variance = np.sum(self.S_ ** 2)
        self.explained_variance_ratio_ = explained_variance / total_variance
        
        # Condition number (indicates numerical stability)
        if len(self.S_) > 0:
            self.condition_number_ = self.S_[0] / self.S_[-1] if self.S_[-1] > 0 else np.inf
    
    def _check_numerical_stability(self):
        """Check for numerical stability issues."""
        # Check singular values
        if np.any(self.S_ < 0):
            warnings.warn("Negative singular values detected - numerical instability", RuntimeWarning)
        
        # Check for near-zero singular values
        near_zero = np.sum(self.S_ < self.tolerance)
        if near_zero > 0:
            warnings.warn(f"{near_zero} singular values below tolerance {self.tolerance}", RuntimeWarning)
        
        # Check orthogonality of U
        if self.U_ is not None:
            ortho_error = np.linalg.norm(self.U_.T @ self.U_ - np.eye(self.n_components), 'fro')
            if ortho_error > 1e-3:
                warnings.warn(f"U orthogonality error: {ortho_error:.2e}", RuntimeWarning)
        
        # Check orthogonality of V
        if self.Vt_ is not None:
            V = self.Vt_.T
            ortho_error = np.linalg.norm(V.T @ V - np.eye(self.n_components), 'fro')
            if ortho_error > 1e-3:
                warnings.warn(f"V orthogonality error: {ortho_error:.2e}", RuntimeWarning)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using computed SVD.
        
        Parameters:
        -----------
        X : np.ndarray
            Data to transform
            
        Returns:
        --------
        X_transformed : np.ndarray
            Transformed data in reduced space
        """
        if self.Vt_ is None:
            raise ValueError("Model has not been fitted yet")
        
        X = self._validate_input(X)
        return X @ self.Vt_.T
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : np.ndarray
            Transformed data
            
        Returns:
        --------
        X : np.ndarray
            Data in original space
        """
        if self.Vt_ is None:
            raise ValueError("Model has not been fitted yet")
        
        return X_transformed @ self.Vt_
    
    def get_covariance(self) -> np.ndarray:
        """Get covariance matrix in transformed space."""
        if self.S_ is None:
            raise ValueError("Model has not been fitted yet")
        
        n_samples = self.U_.shape[0]
        return (self.S_ ** 2) / (n_samples - 1)
    
    def get_precision(self) -> np.ndarray:
        """Get precision matrix (inverse covariance) in transformed space."""
        cov = self.get_covariance()
        return 1.0 / cov
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute average log-likelihood of samples.
        
        Parameters:
        -----------
        X : np.ndarray
            Test samples
            
        Returns:
        --------
        score : float
            Average log-likelihood
        """
        X_transformed = self.transform(X)
        precision = self.get_precision()
        
        # Mahalanobis distance
        log_likelihood = -0.5 * np.sum(X_transformed ** 2 * precision)
        log_likelihood -= 0.5 * np.sum(np.log(2 * np.pi / precision))
        
        return log_likelihood / X.shape[0]
    
    def print_diagnostics(self):
        """Print comprehensive diagnostics of the SVD computation."""
        print("=" * 60)
        print("RANDOMIZED SVD DIAGNOSTICS")
        print("=" * 60)
        
        if self.S_ is not None:
            print(f"Components computed: {len(self.S_)}")
            print(f"Computation time: {self.computation_time_:.3f} seconds")
            print(f"Data type: {self.dtype}")
            print()
            
            print("SINGULAR VALUES:")
            print(f"  Max: {self.S_[0]:.6e}")
            print(f"  Min: {self.S_[-1]:.6e}")
            print(f"  Condition number: {self.condition_number_:.2e}")
            print()
            
            if self.verify_accuracy:
                print("ACCURACY METRICS:")
                print(f"  Reconstruction error (Frobenius): {self.reconstruction_error_:.6e}")
                print(f"  Relative error: {self.relative_error_:.6e}")
                print(f"  Explained variance ratio: {self.explained_variance_ratio_:.4%}")
                print()
            
            print("NUMERICAL STABILITY:")
            # Check orthogonality
            if self.U_ is not None:
                U_ortho = np.linalg.norm(self.U_.T @ self.U_ - np.eye(self.n_components), 'fro')
                print(f"  U orthogonality error: {U_ortho:.2e}")
            
            if self.Vt_ is not None:
                V = self.Vt_.T
                V_ortho = np.linalg.norm(V.T @ V - np.eye(self.n_components), 'fro')
                print(f"  V orthogonality error: {V_ortho:.2e}")
            
            # Singular value decay
            if len(self.S_) > 1:
                decay_rate = self.S_[1:] / self.S_[:-1]
                print(f"  Average singular value decay rate: {np.mean(decay_rate):.4f}")
        else:
            print("No SVD computed yet")
        
        print("=" * 60)


def compare_with_sklearn(
    X: np.ndarray,
    n_components: int = 50,
    n_iter: int = 5,
    n_oversamples: int = 20
) -> dict:
    """
    Compare custom implementation with sklearn's randomized SVD.
    
    Parameters:
    -----------
    X : np.ndarray
        Test matrix
    n_components : int
        Number of components
    n_iter : int
        Number of power iterations
    n_oversamples : int
        Number of oversamples
        
    Returns:
    --------
    results : dict
        Comparison results
    """
    from sklearn.utils.extmath import randomized_svd as sklearn_rsvd
    
    print("Comparing Custom vs sklearn Randomized SVD...")
    print(f"Matrix shape: {X.shape}")
    print(f"Components: {n_components}, Iterations: {n_iter}, Oversamples: {n_oversamples}")
    print("-" * 60)
    
    # Custom implementation
    custom_svd = RandomizedSVD(
        n_components=n_components,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        random_state=42,
        verify_accuracy=True
    )
    
    start = time.time()
    U_custom, S_custom, Vt_custom = custom_svd.fit_transform(X)
    custom_time = time.time() - start
    
    # sklearn implementation
    start = time.time()
    U_sklearn, S_sklearn, Vt_sklearn = sklearn_rsvd(
        X,
        n_components=n_components,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        random_state=42
    )
    sklearn_time = time.time() - start
    
    # Compare singular values
    s_diff = np.linalg.norm(S_custom - S_sklearn) / np.linalg.norm(S_sklearn)
    
    # Compare subspaces (using principal angles)
    U_overlap = np.linalg.norm(U_custom.T @ U_sklearn - np.eye(n_components), 'fro')
    V_overlap = np.linalg.norm(Vt_custom @ Vt_sklearn.T - np.eye(n_components), 'fro')
    
    # Reconstruction errors
    X_custom = U_custom @ np.diag(S_custom) @ Vt_custom
    X_sklearn = U_sklearn @ np.diag(S_sklearn) @ Vt_sklearn
    
    custom_error = np.linalg.norm(X - X_custom, 'fro')
    sklearn_error = np.linalg.norm(X - X_sklearn, 'fro')
    
    results = {
        'custom_time': custom_time,
        'sklearn_time': sklearn_time,
        'singular_value_diff': s_diff,
        'U_subspace_diff': U_overlap,
        'V_subspace_diff': V_overlap,
        'custom_reconstruction_error': custom_error,
        'sklearn_reconstruction_error': sklearn_error,
        'custom_relative_error': custom_svd.relative_error_,
        'speedup': sklearn_time / custom_time
    }
    
    print(f"TIMING:")
    print(f"  Custom: {custom_time:.3f}s")
    print(f"  sklearn: {sklearn_time:.3f}s")
    print(f"  Speedup: {results['speedup']:.2f}x")
    print()
    
    print(f"ACCURACY:")
    print(f"  Singular value difference: {s_diff:.2e}")
    print(f"  U subspace difference: {U_overlap:.2e}")
    print(f"  V subspace difference: {V_overlap:.2e}")
    print()
    
    print(f"RECONSTRUCTION:")
    print(f"  Custom error: {custom_error:.2e}")
    print(f"  sklearn error: {sklearn_error:.2e}")
    print(f"  Custom relative error: {custom_svd.relative_error_:.2e}")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Custom Randomized SVD Implementation")
    print("=" * 60)
    
    # Create test matrix
    np.random.seed(42)
    n_samples, n_features = 1000, 500
    rank = 50
    
    # Low-rank matrix with noise
    U_true = np.random.randn(n_samples, rank)
    V_true = np.random.randn(n_features, rank)
    X = U_true @ V_true.T + 0.1 * np.random.randn(n_samples, n_features)
    
    # Test custom implementation
    svd = RandomizedSVD(n_components=30, n_iter=5, n_oversamples=20, random_state=42)
    U, S, Vt = svd.fit_transform(X)
    
    # Print diagnostics
    svd.print_diagnostics()
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH SKLEARN")
    print("=" * 60)
    compare_with_sklearn(X, n_components=30)