"""
Randomized SVD Implementation
==============================
Based on Halko, Martinsson, and Tropp (2011) algorithm.
Optimized for large-scale matrices with multiple backend support.
"""

import numpy as np
from typing import Tuple, Optional, Union, Literal
import warnings
from scipy.linalg import qr, svd as scipy_svd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from scipy.sparse.linalg import svds
import time
import logging

logger = logging.getLogger(__name__)

# Check for optional backends
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import numba
    from numba import jit, prange, njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    jit = None
    njit = None
    prange = None


class RandomizedSVD:
    """
    Randomized SVD implementation based on Halko, Martinsson, and Tropp (2011).

    Two-stage algorithm for computing approximate low-rank SVD:
    1. Range finding: Approximate the column space of X
    2. SVD computation: Compute SVD on reduced matrix

    Parameters
    ----------
    n_components : int
        Number of singular values and vectors to compute
    n_oversamples : int, default=10
        Number of additional random samples (improves accuracy)
    n_iter : int or 'auto', default='auto'
        Number of power iterations. 'auto' selects based on n_components
    center : bool, default=True
        Whether to center the data before SVD
    random_state : int, optional
        Random seed for reproducibility
    dtype : np.dtype, default=np.float32
        Data type for computations
    backend : str, default='numpy'
        Computation backend: 'numpy', 'torch', 'numba', 'auto'
    device : str, default='cpu'
        Device for torch backend: 'cpu', 'cuda', 'mps'
    verify_accuracy : bool, default=False
        Whether to compute reconstruction error metrics
    """

    def __init__(
        self,
        n_components: int,
        n_oversamples: int = 10,
        n_iter: Union[int, str] = "auto",
        center: bool = True,
        random_state: Optional[int] = None,
        dtype: np.dtype = np.float32,
        backend: Literal["numpy", "torch", "numba", "auto"] = "numpy",
        device: str = "cpu",
        verify_accuracy: bool = False,
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.center = center
        self.random_state = random_state
        self.dtype = dtype
        self.backend = backend
        self.device = device
        self.verify_accuracy = verify_accuracy

        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Select backend
        self._select_backend()

        # Results storage
        self.U_ = None
        self.S_ = None
        self.Vt_ = None
        self.mean_ = None
        self.components_ = None  # sklearn compatibility
        self.singular_values_ = None
        self.explained_variance_ratio_ = None
        self.is_fitted = False

        # Performance metrics
        self.computation_time_ = None
        self.reconstruction_error_ = None
        self.relative_error_ = None

    def _select_backend(self):
        """Select and validate computation backend."""
        if self.backend == "auto":
            if HAS_TORCH and torch.cuda.is_available():
                self.backend = "torch"
                self.device = "cuda"
                logger.info("Auto-selected PyTorch backend with CUDA")
            elif HAS_NUMBA:
                self.backend = "numba"
                logger.info("Auto-selected Numba backend")
            else:
                self.backend = "numpy"
                logger.info("Auto-selected NumPy backend")

        # Validate selected backend
        if self.backend == "torch" and not HAS_TORCH:
            warnings.warn(
                "PyTorch not available, falling back to numpy", RuntimeWarning
            )
            self.backend = "numpy"
        elif self.backend == "numba" and not HAS_NUMBA:
            warnings.warn("Numba not available, falling back to numpy", RuntimeWarning)
            self.backend = "numpy"

    def _auto_select_n_iter(self, n_samples: int, n_features: int):
        """Auto-select number of iterations based on problem size."""
        if self.n_iter == "auto":
            # Based on scikit-learn heuristics
            if self.n_components < 0.1 * min(n_samples, n_features):
                self.n_iter = 7
            else:
                self.n_iter = 4
        elif not isinstance(self.n_iter, int):
            self.n_iter = 4  # Default fallback

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : object
            Returns the instance itself
        """
        self.fit_transform(X, return_components=False)
        return self

    def fit_transform(
        self,
        X: Union[np.ndarray, csr_matrix, csc_matrix],
        return_components: bool = False,
        y=None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute randomized SVD of matrix X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input matrix
        return_components : bool, default=False
            If True, return (U, S, Vt). If False, return transformed data.

        Returns
        -------
        U, S, Vt or X_transformed depending on return_components
        """
        start_time = time.time()

        # Check if sparse
        if issparse(X):
            return self._fit_transform_sparse(X, return_components)

        # Input validation
        X = self._validate_input(X)
        n_samples, n_features = X.shape

        # Auto-select n_iter if needed
        self._auto_select_n_iter(n_samples, n_features)

        # Validate n_components
        max_components = min(n_samples, n_features)
        if self.n_components > max_components:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed "
                f"min(n_samples, n_features) = {max_components}"
            )

        # Center data if requested
        if self.center:
            self.mean_ = X.mean(axis=0)
            X_centered = X - self.mean_
        else:
            self.mean_ = None
            X_centered = X

        # Perform randomized SVD based on backend
        if self.backend == "torch":
            U, S, Vt = self._randomized_svd_torch(X_centered)
        elif self.backend == "numba":
            U, S, Vt = self._randomized_svd_numba(X_centered)
        else:
            U, S, Vt = self._randomized_svd_numpy(X_centered)

        # Store results
        self.U_ = U[:, : self.n_components]
        self.S_ = S[: self.n_components]
        self.Vt_ = Vt[: self.n_components, :]

        # sklearn compatibility
        self.components_ = self.Vt_
        self.singular_values_ = self.S_
        self.is_fitted = True

        # Compute accuracy metrics if requested
        if self.verify_accuracy:
            self._compute_accuracy_metrics(X_centered)

        self.computation_time_ = time.time() - start_time

        if return_components:
            return self.U_, self.S_, self.Vt_
        else:
            # Return transformed data
            return self.U_ * self.S_[np.newaxis, :]

    def _randomized_svd_numpy(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        NumPy implementation of randomized SVD.

        Implements Algorithm 4.3 from Halko et al. 2011
        """
        n_samples, n_features = X.shape

        # Stage A: Find orthonormal matrix Q whose range approximates X
        # Step 1: Generate random Gaussian matrix
        rng = np.random.RandomState(self.random_state)
        projection_size = self.n_components + self.n_oversamples
        omega = rng.randn(n_features, projection_size).astype(self.dtype)

        # Step 2: Form Y = X @ Omega
        Y = X @ omega

        # Step 3: Orthogonalize Y to get Q
        Q, _ = qr(Y, mode="economic")

        # Step 4: Power iterations for improved accuracy
        for _ in range(self.n_iter):
            # Apply (X.T @ X)^q to improve subspace
            Z = X.T @ Q
            Q, _ = qr(Z, mode="economic")
            Z = X @ Q
            Q, _ = qr(Z, mode="economic")

        # Stage B: Compute SVD on reduced matrix
        # Step 1: Form B = Q.T @ X
        B = Q.T @ X

        # Step 2: Compute SVD of small matrix B
        Uhat, S, Vt = scipy_svd(B, full_matrices=False)

        # Step 3: Recover left singular vectors
        U = Q @ Uhat

        return U, S, Vt

    def _randomized_svd_torch(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PyTorch implementation for GPU acceleration."""
        if not HAS_TORCH:
            return self._randomized_svd_numpy(X)

        # Convert to torch tensor
        X_torch = torch.from_numpy(X).to(self.device).float()
        n_samples, n_features = X_torch.shape

        # Stage A: Range finding
        projection_size = self.n_components + self.n_oversamples

        # Random Gaussian matrix
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        omega = torch.randn(
            n_features, projection_size, device=self.device, dtype=torch.float32
        )

        # Form Y = X @ Omega
        Y = X_torch @ omega

        # QR decomposition
        Q, _ = torch.linalg.qr(Y, mode="reduced")

        # Power iterations
        for _ in range(self.n_iter):
            Z = X_torch.T @ Q
            Q, _ = torch.linalg.qr(Z, mode="reduced")
            Z = X_torch @ Q
            Q, _ = torch.linalg.qr(Z, mode="reduced")

        # Stage B: SVD on reduced matrix
        B = Q.T @ X_torch
        Uhat, S, Vt = torch.linalg.svd(B, full_matrices=False)

        # Recover U
        U = Q @ Uhat

        # Convert back to numpy
        return U.cpu().numpy(), S.cpu().numpy(), Vt.cpu().numpy()

    def _randomized_svd_numba(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-accelerated implementation."""
        if not HAS_NUMBA:
            return self._randomized_svd_numpy(X)

        # Use numpy for main operations but accelerate matrix multiplications
        n_samples, n_features = X.shape

        # Create numba-accelerated matmul if not exists
        if not hasattr(self, "_matmul_numba"):

            @jit(nopython=True, parallel=True, fastmath=True)
            def matmul_numba(A, B):
                m, k = A.shape
                k2, n = B.shape
                C = np.zeros((m, n), dtype=A.dtype)
                for i in prange(m):
                    for j in range(n):
                        sum_val = 0.0
                        for l in range(k):
                            sum_val += A[i, l] * B[l, j]
                        C[i, j] = sum_val
                return C

            self._matmul_numba = matmul_numba

        # Stage A: Range finding (use numba for matrix multiplications)
        rng = np.random.RandomState(self.random_state)
        projection_size = self.n_components + self.n_oversamples
        omega = rng.randn(n_features, projection_size).astype(self.dtype)

        # Use numba for Y = X @ omega
        Y = self._matmul_numba(X, omega)
        Q, _ = qr(Y, mode="economic")

        # Power iterations with numba acceleration
        for _ in range(self.n_iter):
            # Z = X.T @ Q using numba
            Z = self._matmul_numba(X.T, Q)
            Q, _ = qr(Z, mode="economic")
            # Z = X @ Q using numba
            Z = self._matmul_numba(X, Q)
            Q, _ = qr(Z, mode="economic")

        # Stage B: SVD on reduced matrix
        B = self._matmul_numba(Q.T, X)
        Uhat, S, Vt = scipy_svd(B, full_matrices=False)
        U = self._matmul_numba(Q, Uhat)

        return U, S, Vt

    def _fit_transform_sparse(
        self, X: Union[csr_matrix, csc_matrix], return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute randomized SVD for sparse matrices using true randomized algorithm.
        Implements Halko et al. 2011 with sparse-aware operations.
        """
        n_samples, n_features = X.shape

        # Auto-select n_iter if needed
        self._auto_select_n_iter(n_samples, n_features)

        # Compute mean if centering (but don't actually center sparse matrix)
        if self.center:
            self.mean_ = np.asarray(X.mean(axis=0)).ravel()
        else:
            self.mean_ = None

        # Use sparse-aware randomized SVD if small enough
        # For very large sparse matrices, fall back to iterative methods
        if self.n_components < min(n_samples, n_features) * 0.1:
            # Use true randomized algorithm for sparse matrices
            U, S, Vt = self._sparse_randomized_svd(X)
        else:
            # Fall back to scipy's iterative solver for large component requests
            max_components = min(n_samples, n_features) - 1
            if self.n_components > max_components:
                raise ValueError(
                    f"For sparse matrices, n_components ({self.n_components}) must be < "
                    f"min(n_samples, n_features) = {max_components}"
                )

            k = min(self.n_components + self.n_oversamples, max_components)
            U, S, Vt = svds(X, k=k, which="LM", random_state=self.random_state)

            # svds returns singular values in ascending order, reverse them
            idx = np.argsort(S)[::-1]
            U = U[:, idx]
            S = S[idx]
            Vt = Vt[idx, :]

        # Truncate to requested components
        self.U_ = U[:, : self.n_components]
        self.S_ = S[: self.n_components]
        self.Vt_ = Vt[: self.n_components, :]

        # sklearn compatibility
        self.components_ = self.Vt_
        self.singular_values_ = self.S_
        self.is_fitted = True

        if return_components:
            return self.U_, self.S_, self.Vt_
        else:
            return self.U_ * self.S_[np.newaxis, :]

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
        if not X.flags["C_CONTIGUOUS"]:
            X = np.ascontiguousarray(X)

        return X

    def _sparse_randomized_svd(
        self, X: Union[csr_matrix, csc_matrix], use_countsketch: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimized sparse-aware randomized SVD implementation.
        Uses CountSketch or sparse Gaussian projections for efficiency.
        """
        n_samples, n_features = X.shape
        k = self.n_components
        p = self.n_oversamples
        q = self.n_iter if isinstance(self.n_iter, int) else 4

        # Random state
        rng = np.random.RandomState(self.random_state)

        # Stage A: Randomized Range Finder
        l = k + p  # Total projection size

        # Choose projection method based on sparsity
        sparsity = 1.0 - (X.nnz / (n_samples * n_features))

        if use_countsketch and sparsity > 0.95:
            # Use CountSketch for very sparse matrices
            Y = self._countsketch_projection(X, l, rng)
        elif sparsity > 0.9:
            # Use sparse Gaussian projection
            Y = self._sparse_gaussian_projection(X, l, rng)
        else:
            # Standard Gaussian projection for moderately sparse
            Omega = rng.randn(n_features, l).astype(self.dtype)
            Y = X.dot(Omega)  # Use .dot() for better sparse performance

        # Power iterations with proper orthogonalization
        for iter_idx in range(q):
            # Orthogonalize Y
            Q, _ = np.linalg.qr(Y, mode="reduced")
            # Project to row space and back
            Z = X.T.dot(Q)  # (n_features, l)
            Q, _ = np.linalg.qr(Z, mode="reduced")
            Y = X.dot(Q)  # (n_samples, l)

        # Final orthogonalization
        Q, _ = np.linalg.qr(Y, mode="reduced")

        # Q is already computed above

        # Stage B: Compute SVD on reduced matrix
        # B = Q^T @ X (dense-sparse multiplication)
        B = Q.T @ X

        # Convert B to dense for SVD (it's small: l x n_features)
        B_dense = B.toarray() if issparse(B) else B

        # Compute SVD of small matrix
        Uhat, S, Vt = scipy_svd(B_dense, full_matrices=False)

        # Recover left singular vectors
        U = Q @ Uhat

        # Truncate to k components
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]

        return U, S, Vt

    def _countsketch_projection(
        self, X: Union[csr_matrix, csc_matrix], l: int, rng
    ) -> np.ndarray:
        """Dispatch to JIT-compiled version if available."""
        if HAS_NUMBA and issparse(X):
            return self._countsketch_projection_numba(X, l, rng)
        else:
            return self._countsketch_projection_python(X, l, rng)

    def _countsketch_projection_python(
        self, X: Union[csr_matrix, csc_matrix], l: int, rng
    ) -> np.ndarray:
        """
        CountSketch projection - extremely efficient for very sparse matrices.
        Maps features to random buckets with random signs.
        Correctly implements Y = X @ S where S is the implicit sketch matrix.
        """
        n_samples, n_features = X.shape

        # Create hash functions
        # h: maps each feature to a bucket in [0, l-1]
        # s: assigns random sign to each feature
        h = rng.randint(0, l, size=n_features)
        s = rng.choice([-1, 1], size=n_features).astype(self.dtype)

        # Initialize output
        Y = np.zeros((n_samples, l), dtype=self.dtype)

        # Convert to CSC for efficient column access
        if issparse(X):
            X_csc = X.tocsc()
            # Iterate over features (columns)
            for j in range(n_features):
                col_j = X_csc.getcol(j)  # Get j-th column as sparse vector
                target_bucket = h[j]
                sign = s[j]
                # Add scaled column to target bucket
                Y[:, target_bucket] += sign * col_j.toarray().ravel()
        else:
            # Dense version
            for j in range(n_features):
                target_bucket = h[j]
                sign = s[j]
                Y[:, target_bucket] += sign * X[:, j]

        return Y

    def _countsketch_projection_numba(
        self, X: Union[csr_matrix, csc_matrix], l: int, rng
    ) -> np.ndarray:
        """Numba JIT-compiled CountSketch for maximum performance."""
        n_samples, n_features = X.shape

        # Create hash functions
        h = rng.randint(0, l, size=n_features).astype(np.int32)
        s = rng.choice([-1.0, 1.0], size=n_features).astype(self.dtype)

        # Convert to CSR for efficient row iteration
        X_csr = X.tocsr()

        # Call JIT-compiled kernel
        Y = _countsketch_kernel_csr(
            X_csr.data, X_csr.indices, X_csr.indptr, h, s, n_samples, l, self.dtype
        )

        return Y

    def _sparse_gaussian_projection(
        self, X: Union[csr_matrix, csc_matrix], l: int, rng
    ) -> np.ndarray:
        """Dispatch to JIT-compiled version if available."""
        if HAS_NUMBA and issparse(X):
            return self._sparse_gaussian_projection_numba(X, l, rng)
        else:
            return self._sparse_gaussian_projection_python(X, l, rng)

    def _sparse_gaussian_projection_python(
        self, X: Union[csr_matrix, csc_matrix], l: int, rng
    ) -> np.ndarray:
        """
        Sparse Gaussian projection following Achlioptas 2003.
        Each element of projection matrix is:
        - 0 with probability 1 - 1/s
        - ±√s with probability 1/(2s)
        """
        n_samples, n_features = X.shape

        # Sparsity parameter (s=3 gives 1/3 non-zero density)
        s = 3
        nnz_prob = 1.0 / s
        scale = np.sqrt(s)

        # Create sparse projection matrix column by column
        # This is more memory efficient than creating full matrix
        Y = np.zeros((n_samples, l), dtype=self.dtype)

        # Convert to CSC for efficient column operations if sparse
        if issparse(X):
            X_csc = X.tocsc()

        for j in range(l):
            # Generate sparse column of projection matrix
            mask = rng.random(n_features) < nnz_prob
            nnz_indices = np.where(mask)[0]
            nnz_values = rng.choice([-scale, scale], size=len(nnz_indices))

            # Compute Y[:, j] = X @ omega_j
            if issparse(X):
                for idx, val in zip(nnz_indices, nnz_values):
                    # X_csc.getcol(idx) returns column as sparse vector
                    col = X_csc.getcol(idx)
                    Y[:, j] += val * col.toarray().ravel()
            else:
                # Dense case: direct indexing works
                if len(nnz_indices) > 0:
                    Y[:, j] = X[:, nnz_indices] @ nnz_values

        return Y

    def _sparse_gaussian_projection_numba(
        self, X: Union[csr_matrix, csc_matrix], l: int, rng
    ) -> np.ndarray:
        """Numba JIT-compiled sparse Gaussian projection."""
        n_samples, n_features = X.shape

        # Parameters
        s = 3
        nnz_prob = 1.0 / s
        scale = np.sqrt(s)

        # Convert to CSR for row-wise operations
        X_csr = X.tocsr()

        # Pre-generate all random projections
        omega_nnz_masks = rng.random((l, n_features)) < nnz_prob
        omega_signs = rng.choice([-scale, scale], size=(l, n_features)).astype(
            self.dtype
        )

        # Call JIT-compiled kernel
        Y = _sparse_gaussian_kernel_csr(
            X_csr.data,
            X_csr.indices,
            X_csr.indptr,
            omega_nnz_masks,
            omega_signs,
            n_samples,
            n_features,
            l,
        )

        return Y

    def _estimate_leverage_scores_unused(
        self, X: Union[csr_matrix, csc_matrix], Y: np.ndarray
    ) -> np.ndarray:
        """
        Estimate leverage scores for adaptive sampling.
        Leverage scores indicate importance of each row/column.
        """
        # Approximate leverage scores using current sketch
        # l_i ≈ ||e_i^T * X * (X^T X)^{-1/2}||^2

        # Use Y as approximation to range of X
        Q, _ = np.linalg.qr(Y, mode="reduced")

        # Compute row norms of X*Q (approximate leverage scores)
        if issparse(X):
            XQ = X.dot(Q)
        else:
            XQ = X @ Q

        leverage_scores = np.sum(XQ**2, axis=1)

        # Normalize
        leverage_scores = leverage_scores / np.sum(leverage_scores)

        return leverage_scores

    def _adaptive_orthogonalize_unused(
        self, Y: np.ndarray, leverage_scores: np.ndarray
    ) -> np.ndarray:
        """
        Adaptive orthogonalization based on leverage scores.
        Focuses computational effort on important directions.
        """
        # Sort columns by importance (estimated from leverage scores)
        col_importance = Y.T @ leverage_scores
        important_cols = np.argsort(-np.abs(col_importance))

        # Reorder columns by importance
        Y_ordered = Y[:, important_cols]

        # Use Cholesky QR on reordered matrix
        Q_ordered = self._cholesky_qr(Y_ordered)

        # Restore original order
        Q = np.zeros_like(Q_ordered)
        Q[:, important_cols] = Q_ordered

        return Q

    def adaptive_regularization_with_jitter(
        self, G: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numerically stable regularization with exponential backoff.
        Implements jitter backoff strategy from FIX_ROADMAP_FINAL.md.

        Parameters
        ----------
        G : ndarray
            Gram matrix to regularize

        Returns
        -------
        G_reg : ndarray
            Regularized Gram matrix
        L : ndarray
            Cholesky factor of G_reg
        """
        dtype = G.dtype
        eps = np.finfo(dtype).eps
        scale = np.linalg.norm(G, "fro") / G.shape[0]

        # Start with conservative jitter
        jitter = max(eps * 1000, scale * eps * 100)
        max_attempts = 10

        for attempt in range(max_attempts):
            try:
                G_reg = G + jitter * np.eye(G.shape[0], dtype=dtype)
                L = np.linalg.cholesky(G_reg)

                # Log successful regularization
                if attempt > 0:
                    logger.debug(
                        f"Cholesky succeeded after {attempt} attempts with jitter={jitter:.2e}"
                    )

                return G_reg, L
            except np.linalg.LinAlgError:
                jitter *= 10  # Exponential backoff
                logger.debug(
                    f"Cholesky failed (attempt {attempt + 1}), increasing jitter to {jitter:.2e}"
                )

        # Last resort: upcast to float64
        if dtype == np.float32:
            logger.warning("Float32 Cholesky failed, upcasting to float64")
            G64 = G.astype(np.float64)
            G64_reg = G64 + jitter * np.eye(G.shape[0])
            L64 = np.linalg.cholesky(G64_reg)
            return G64_reg.astype(dtype), L64.astype(dtype)

        raise np.linalg.LinAlgError("Cholesky failed after jitter escalation")

    def _cholesky_qr(self, Y: np.ndarray) -> np.ndarray:
        """
        Cholesky QR decomposition - faster than Gram-Schmidt for sparse operations.
        Uses adaptive regularization with jitter backoff for numerical stability.
        """
        # Compute Gram matrix G = Y^T Y (small: k x k)
        G = Y.T @ Y

        try:
            # Use adaptive regularization with jitter backoff
            G_reg, R = self.adaptive_regularization_with_jitter(G)

            # Q = Y @ R^{-1}
            Q = Y @ np.linalg.inv(R.T)

            # One refinement step for numerical stability
            G2 = Q.T @ Q - np.eye(Q.shape[1])
            if np.linalg.norm(G2, ord="fro") > 1e-6:
                # Re-orthogonalize once
                W = Q @ (G2 / 2)
                Q = Q - W

            return Q
        except np.linalg.LinAlgError as e:
            # Fall back to Gram-Schmidt if Cholesky fails completely
            logger.warning(f"Cholesky QR failed even with adaptive regularization: {e}")
            return self._orthogonalize_sparse(Y)

    def _orthogonalize_sparse(self, Y: np.ndarray) -> np.ndarray:
        """
        Orthogonalize columns of Y using block Gram-Schmidt with re-orthogonalization.
        Optimized for better cache performance.
        """
        n, k = Y.shape
        block_size = min(8, k)  # Process in blocks for better cache usage
        Q = np.zeros_like(Y)

        for i in range(0, k, block_size):
            # Process block of columns
            end = min(i + block_size, k)
            block = Y[:, i:end].copy()

            # First pass: orthogonalize against previous blocks
            if i > 0:
                coeffs = Q[:, :i].T @ block
                block -= Q[:, :i] @ coeffs

            # QR decomposition within block
            block_q, _ = np.linalg.qr(block, mode="reduced")

            # Second pass: re-orthogonalize for numerical stability
            if i > 0:
                coeffs2 = Q[:, :i].T @ block_q
                block_q -= Q[:, :i] @ coeffs2
                # Re-normalize
                for j in range(block_q.shape[1]):
                    norm = np.linalg.norm(block_q[:, j])
                    if norm > 1e-10:
                        block_q[:, j] /= norm

            Q[:, i:end] = block_q

        return Q


# Numba JIT-compiled kernels
if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _countsketch_kernel_csr(data, indices, indptr, h, s, n_samples, l, dtype):
        """Numba kernel for CountSketch with CSR matrix."""
        Y = np.zeros((n_samples, l), dtype=dtype)

        # Parallel loop over rows
        for i in prange(n_samples):
            row_start = indptr[i]
            row_end = indptr[i + 1]

            for idx in range(row_start, row_end):
                j = indices[idx]  # Column index
                val = data[idx]  # Value
                bucket = h[j]
                sign = s[j]
                Y[i, bucket] += sign * val

        return Y

    @njit(parallel=True, cache=True)
    def _sparse_gaussian_kernel_csr(
        data, indices, indptr, omega_masks, omega_vals, n_samples, n_features, l
    ):
        """Numba kernel for sparse Gaussian projection with CSR matrix."""
        Y = np.zeros((n_samples, l), dtype=data.dtype)

        # Loop over projection dimensions
        for proj_idx in prange(l):
            mask = omega_masks[proj_idx]
            vals = omega_vals[proj_idx]

            # For each row in X
            for i in range(n_samples):
                row_start = indptr[i]
                row_end = indptr[i + 1]

                # Accumulate dot product
                dot_product = 0.0
                for idx in range(row_start, row_end):
                    j = indices[idx]  # Column index
                    if mask[j]:  # If this feature is selected
                        dot_product += data[idx] * vals[j]

                Y[i, proj_idx] = dot_product

        return Y

    @njit(cache=True)
    def _cholesky_qr_kernel(Y):
        """Numba kernel for Cholesky QR decomposition."""
        # Compute Gram matrix G = Y^T Y
        G = Y.T @ Y
        n = G.shape[0]

        # Add regularization
        for i in range(n):
            G[i, i] += 1e-10

        # Cholesky decomposition
        L = np.linalg.cholesky(G)

        # Q = Y @ inv(L.T)
        # Solve L.T @ Q.T = Y.T for Q.T
        Q_T = np.linalg.solve(L.T, Y.T)
        Q = Q_T.T

        return Q
else:
    # Dummy functions if Numba not available
    def _countsketch_kernel_csr(*args):
        raise ImportError("Numba not available")

    def _sparse_gaussian_kernel_csr(*args):
        raise ImportError("Numba not available")

    def _cholesky_qr_kernel(*args):
        raise ImportError("Numba not available")


class RSVDAutoTuner:
    """
    Automatic parameter tuning for RandomizedSVD.
    Finds optimal oversamples and iterations for specific data.
    """

    def __init__(self, backend="auto", device="cpu", random_state=42):
        """
        Initialize auto-tuner.

        Parameters
        ----------
        backend : str
            Backend to use for testing
        device : str
            Device for torch backend
        random_state : int
            Random seed
        """
        self.backend = backend
        self.device = device
        self.random_state = random_state
        self.results = []

    def evaluate_config(
        self, X, n_components, n_oversamples, n_iter, X_baseline=None, S_baseline=None
    ):
        """
        Evaluate a single RSVD configuration.

        Parameters
        ----------
        X : array-like
            Input data matrix
        n_components : int
            Number of components to compute
        n_oversamples : int
            Oversampling parameter
        n_iter : int
            Number of iterations
        X_baseline : array-like, optional
            Baseline SVD projection for quality comparison
        S_baseline : array-like, optional
            Baseline singular values

        Returns
        -------
        metrics : dict
            Performance and quality metrics
        """
        import time

        # Create RSVD instance
        rsvd = RandomizedSVD(
            n_components=n_components,
            n_oversamples=n_oversamples,
            n_iter=n_iter,
            backend=self.backend,
            device=self.device,
            random_state=self.random_state,
            verify_accuracy=False,
        )

        # Time the computation
        start_time = time.time()
        U, S, Vt = rsvd.fit_transform(X, return_components=True)
        compute_time = time.time() - start_time

        metrics = {
            "n_oversamples": n_oversamples,
            "n_iter": n_iter,
            "time": compute_time,
            "backend": self.backend,
        }

        # Quality metrics if baseline provided
        if S_baseline is not None:
            # Singular value accuracy
            n_compare = min(len(S), len(S_baseline))
            s_ratio = S[:n_compare] / (S_baseline[:n_compare] + 1e-10)
            metrics["singular_value_accuracy"] = float(np.mean(s_ratio))
            metrics["singular_value_std"] = float(np.std(s_ratio))

            # Effective rank
            s_normalized = S / (S[0] + 1e-10)
            metrics["effective_rank"] = int(np.sum(s_normalized > 0.01))

        if X_baseline is not None:
            # Neighbor preservation
            X_reduced = U * S

            # Sample queries for speed
            n_samples = min(50, X.shape[0])
            query_indices = np.random.RandomState(self.random_state).choice(
                X.shape[0], n_samples, replace=False
            )

            preservation_scores = []
            for q_idx in query_indices:
                # Original space neighbors
                sims_orig = X_baseline @ X_baseline[q_idx]
                top_k = min(20, X.shape[0] - 1)
                top_k_orig = np.argsort(sims_orig)[-top_k:]

                # RSVD space neighbors
                sims_rsvd = X_reduced @ X_reduced[q_idx]
                top_k_rsvd = np.argsort(sims_rsvd)[-top_k:]

                # Overlap
                preserved = len(set(top_k_orig) & set(top_k_rsvd)) / top_k
                preservation_scores.append(preserved)

            metrics["neighbor_preservation"] = float(np.mean(preservation_scores))

            # Overall quality score
            metrics["quality_score"] = (
                0.5 * metrics["neighbor_preservation"]
                + 0.3 * metrics["singular_value_accuracy"]
                + 0.2 * (metrics["effective_rank"] / n_components)
            )

        return metrics

    def auto_tune(
        self, X, n_components=None, target_quality=0.9, max_time=None, verbose=True
    ):
        """
        Automatically find optimal RSVD parameters.

        Parameters
        ----------
        X : array-like
            Input data matrix
        n_components : int, optional
            Number of components (default: min(100, min(X.shape) // 2))
        target_quality : float
            Target quality score (0-1)
        max_time : float, optional
            Maximum time per configuration
        verbose : bool
            Print progress

        Returns
        -------
        best_params : dict
            Optimal parameters found
        """
        if n_components is None:
            n_components = min(100, min(X.shape) // 2)

        if verbose:
            print("=" * 60)
            print("RSVD AUTO-TUNING")
            print("=" * 60)
            print(f"Data shape: {X.shape}")
            print(f"Components: {n_components}")
            print(f"Target quality: {target_quality:.1%}")
            print(f"Backend: {self.backend}")

        # Compute baseline with high-quality settings
        if verbose:
            print("\nComputing baseline...")

        baseline_rsvd = RandomizedSVD(
            n_components=n_components,
            n_oversamples=max(20, n_components // 2),
            n_iter=7,
            backend=self.backend,
            device=self.device,
            random_state=self.random_state,
        )
        U_base, S_base, Vt_base = baseline_rsvd.fit_transform(X, return_components=True)
        X_baseline = U_base * S_base

        # Test configurations
        configs = [
            (5, 2),  # Minimal
            (10, 4),  # Standard (scikit-learn default)
            (10, 7),  # More iterations
            (20, 4),  # More oversamples
            (20, 7),  # Balanced
            (30, 7),  # High quality (current recommendation)
        ]

        if verbose:
            print("\nTesting configurations...")

        best_score = 0
        best_config = None

        for n_over, n_iter in configs:
            if verbose:
                print(f"  Testing oversamples={n_over}, iter={n_iter}...", end=" ")

            metrics = self.evaluate_config(
                X, n_components, n_over, n_iter, X_baseline, S_base
            )

            self.results.append(metrics)

            if "quality_score" in metrics:
                score = metrics["quality_score"]
                if verbose:
                    print(f"quality={score:.3f}, time={metrics['time']:.3f}s")

                if score > best_score:
                    best_score = score
                    best_config = (n_over, n_iter)

                # Early stopping if target met
                if score >= target_quality:
                    if verbose:
                        print("  → Target quality achieved!")
                    break

        # Return best configuration
        if best_config:
            best_params = {
                "n_oversamples": best_config[0],
                "n_iter": best_config[1],
                "quality_score": best_score,
                "recommendation": f"Use n_oversamples={best_config[0]}, n_iter={best_config[1]} for {best_score:.1%} quality",
            }
        else:
            # Fallback to reasonable defaults
            best_params = {
                "n_oversamples": 10,
                "n_iter": 4,
                "quality_score": None,
                "recommendation": "Using scikit-learn defaults (n_oversamples=10, n_iter=4)",
            }

        if verbose:
            print("\n" + "=" * 60)
            print("OPTIMAL PARAMETERS FOUND:")
            print("=" * 60)
            print(best_params["recommendation"])

        return best_params


# Add auto_tune method to RandomizedSVD class
def auto_tune(self, X, target_quality=0.9, verbose=True):
    """
    Auto-tune parameters for this RSVD instance.

    Parameters
    ----------
    X : array-like
        Sample of your data
    target_quality : float
        Target quality (0-1)
    verbose : bool
        Print progress

    Returns
    -------
    self : RandomizedSVD
        Self with updated parameters
    """
    tuner = RSVDAutoTuner(
        backend=self.backend, device=self.device, random_state=self.random_state
    )

    best_params = tuner.auto_tune(
        X,
        n_components=self.n_components,
        target_quality=target_quality,
        verbose=verbose,
    )

    # Update self with best parameters
    self.n_oversamples = best_params["n_oversamples"]
    self.n_iter = best_params["n_iter"]

    if verbose:
        print("\nRandomizedSVD parameters updated:")
        print(f"  n_oversamples: {self.n_oversamples}")
        print(f"  n_iter: {self.n_iter}")

    return self


# Add method to class
RandomizedSVD.auto_tune = auto_tune


# Testing function
def compare_with_sklearn(X, n_components=50, n_iter=4, n_oversamples=10):
    """
    Compare with sklearn's randomized SVD.

    Parameters
    ----------
    X : ndarray
        Test matrix
    n_components : int
        Number of components
    n_iter : int
        Number of iterations
    n_oversamples : int
        Number of oversamples

    Returns
    -------
    results : dict
        Comparison results
    """
    from sklearn.utils.extmath import randomized_svd as sklearn_rsvd
    import time

    print("Comparing Custom vs sklearn Randomized SVD...")
    print(f"Matrix shape: {X.shape}")
    print(
        f"Components: {n_components}, Iterations: {n_iter}, Oversamples: {n_oversamples}"
    )
    print("-" * 60)

    # Custom implementation
    custom_svd = RandomizedSVD(
        n_components=n_components,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        random_state=42,
        verify_accuracy=True,
        center=False,  # sklearn doesn't center by default
    )

    start = time.time()
    U_custom, S_custom, Vt_custom = custom_svd.fit_transform(X, return_components=True)
    custom_time = time.time() - start

    # sklearn implementation
    start = time.time()
    U_sklearn, S_sklearn, Vt_sklearn = sklearn_rsvd(
        X,
        n_components=n_components,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        random_state=42,
    )
    sklearn_time = time.time() - start

    # Compare singular values
    s_diff = np.linalg.norm(S_custom - S_sklearn) / np.linalg.norm(S_sklearn)

    results = {
        "custom_time": custom_time,
        "sklearn_time": sklearn_time,
        "singular_value_diff": s_diff,
        "speedup": sklearn_time / custom_time,
    }

    print("TIMING:")
    print(f"  Custom: {custom_time:.3f}s")
    print(f"  sklearn: {sklearn_time:.3f}s")
    print(f"  Speedup: {results['speedup']:.2f}x")
    print()

    print("ACCURACY:")
    print(f"  Singular value difference: {s_diff:.2e}")

    return results
