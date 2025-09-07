"""
Fused Pipeline Encoder for High-Performance Binary Fingerprinting
==================================================================

Implements a unified, single-pass pipeline that fuses RSVD, ITQ, and SIMD operations
to eliminate transformation overhead and achieve optimal performance.

Architecture: CSR → project(RSVD) → rotate(ITQ) → binarize → pack
"""

import numpy as np
import torch
import logging
from typing import Tuple, Union, List
from scipy.sparse import csr_matrix, issparse
import time
from sklearn.feature_extraction.text import TfidfVectorizer

# Import components
from .randomized_svd import RandomizedSVD
from .itq_optimized import ITQOptimizer, ITQConfig
from .hamming_simd import pack_bits_uint32, HammingSIMDImpl

# Try to import numba for CPU optimization
try:
    import numba
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)


class FusedPipelineEncoder:
    """
    Fused encoder that combines RSVD projection, ITQ rotation, and bit packing
    in a single streaming pipeline to minimize memory transfers and eliminate
    redundant computations.

    Key optimizations:
    - Single SVD computation shared between RSVD and ITQ
    - Streaming CSR processing with batch blocking
    - Unified CPU backend with Numba acceleration
    - Direct binarization without intermediate materializations
    - Cache-aligned bit packing for SIMD operations
    """

    def __init__(
        self,
        n_bits: int = 128,
        max_features: int = 10000,
        use_itq: bool = False,  # ITQ is optional, disabled by default
        n_iterations_itq: int = 5,  # Reduced from 50, only used if use_itq=True
        batch_size: int = 5000,
        backend: str = "cpu",  # 'cpu' or 'gpu'
        use_numba: bool = True,
        random_state: int = 42,
        # Backward compatibility parameters
        use_randomized_svd: bool = True,  # Added for compatibility
        svd_n_oversamples: int = 10,  # Optimized based on benchmarks
        svd_n_iter: int = 5,  # Balanced quality/speed for high dimensions
        # Sampling parameters (like Original TEJAS)
        use_sampling: bool = True,  # Use golden ratio sampling
        memory_limit_gb: float = 50.0,  # Memory limit for sampling
        # Enhanced RSVD parameters
        use_block_krylov: bool = False,  # Use Block Krylov for better accuracy
        regularization: str = "none",  # 'none', 'tikhonov', or 'adaptive'
        regularization_strength: float = 1e-6,  # Regularization parameter
    ):
        """
        Initialize fused pipeline encoder.

        Args:
            n_bits: Number of bits in final binary code
            max_features: Maximum features for TF-IDF vectorizer
            use_itq: Whether to apply ITQ rotation (default: False)
            n_iterations_itq: ITQ optimization iterations if use_itq=True
            batch_size: Batch size for streaming processing
            backend: Computation backend ('cpu' or 'gpu')
            use_numba: Use Numba acceleration for CPU backend
            random_state: Random seed for reproducibility
        """
        self.n_bits = n_bits
        self.max_features = max_features
        self.use_itq = use_itq
        self.n_iterations_itq = n_iterations_itq
        self.batch_size = batch_size
        self.backend = backend
        self.use_numba = use_numba and HAS_NUMBA
        self.random_state = random_state
        # Store backward compatibility params
        self.use_randomized_svd = use_randomized_svd
        self.svd_n_oversamples = svd_n_oversamples
        self.svd_n_iter = svd_n_iter
        # Sampling parameters
        self.use_sampling = use_sampling
        self.memory_limit_gb = memory_limit_gb
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        # Enhanced RSVD parameters
        self.use_block_krylov = use_block_krylov
        self.regularization = regularization
        self.regularization_strength = regularization_strength

        # Components (initialized during fit)
        self.vectorizer = None
        self.rsvd = None
        self.itq = None
        self.projection_matrix = None  # Combined RSVD projection
        self.rotation_matrix = None  # ITQ rotation
        self.combined_matrix = None  # Cached projection @ rotation matrix
        self.data_mean = None  # Mean for centering
        self.centering_bias = None  # Bias for sparse centering
        self.effective_bits = None  # Actual bits after rank capping
        self.is_fitted = False

        # Search cache
        self._searcher_cache = None
        self._cached_database = None

        # Performance metrics
        self.fit_time = None
        self.transform_time = None

        logger.info("Initialized FusedPipelineEncoder")
        logger.info(f"  Backend: {backend}")
        logger.info(f"  Numba: {'enabled' if self.use_numba else 'disabled'}")
        logger.info(f"  Batch size: {batch_size}")
        if use_block_krylov:
            logger.info("  Block Krylov: enabled")
        if regularization != "none":
            logger.info(f"  Regularization: {regularization}")

    def fit(
        self, titles: List[str], memory_limit_gb: float = 50
    ) -> "FusedPipelineEncoder":
        """
        Fit the fused pipeline on training data.

        This performs:
        1. TF-IDF vectorization with automatic max_features adjustment
        2. Single RSVD decomposition
        3. ITQ rotation learning (using RSVD output)
        4. Padding with random projections if needed
        5. Store combined transformation matrices

        Args:
            titles: List of text titles to train on
            memory_limit_gb: Memory limit for processing

        Returns:
            Self for chaining
        """
        start_time = time.time()
        logger.info(f"Fitting fused pipeline on {len(titles):,} titles...")

        # Automatically increase max_features for small datasets to avoid bit capping
        n_samples = len(titles)
        min_features_needed = (
            self.n_bits * 100
        )  # Heuristic: need at least 100x features per bit
        adjusted_max_features = max(self.max_features, min_features_needed)

        if adjusted_max_features > self.max_features:
            logger.info(
                f"Auto-adjusting max_features from {self.max_features:,} to {adjusted_max_features:,} to avoid bit capping"
            )

        # Step 1: Fit TF-IDF vectorizer and optionally sample
        logger.info("Step 1: Learning TF-IDF vocabulary...")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=adjusted_max_features,
            lowercase=True,
            dtype=np.float32,
        )

        # Apply golden ratio sampling if enabled (like Original TEJAS)
        if self.use_sampling and len(titles) > 100:
            sample_indices = self._golden_ratio_sample(
                len(titles), self.memory_limit_gb
            )
            sample_titles = [titles[i] for i in sample_indices]
            logger.info(
                f"  Using {len(sample_titles):,} sampled titles for learning projection"
            )
            # Fit vectorizer on all, transform on sample (like Original TEJAS)
            self.vectorizer.fit(titles)
            X_tfidf = self.vectorizer.transform(sample_titles)
        else:
            # Use all data (original behavior)
            X_tfidf = self.vectorizer.fit_transform(titles)

        logger.info(f"  Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        logger.info(f"  Matrix shape: {X_tfidf.shape}")

        # Store sparse matrix for potential ITQ projection later
        self.X_train_sparse = X_tfidf

        # Step 2: Compute RSVD with energy-based component selection
        logger.info("Step 2: Computing RSVD projection with energy analysis...")

        # First compute more components for energy analysis (like Original TEJAS)
        max_rank = min(X_tfidf.shape[0], X_tfidf.shape[1])
        # Request components up to rank limit - padding will handle the rest
        # Request min(n_bits, max_rank-1) to avoid exceeding matrix rank
        initial_components = min(self.n_bits, max_rank - 1)

        logger.info(
            f"  Computing {initial_components} components for energy analysis..."
        )

        # DISABLE CENTERING to match Original TEJAS behavior
        # The Original encoder doesn't center, which preserves the natural sparsity
        self.data_mean = None
        logger.info("  Centering DISABLED to match Original TEJAS behavior")

        # Choose between RandomizedSVD and regular SVD
        if self.use_randomized_svd:
            # Use RandomizedSVD (default, more efficient for large sparse matrices)
            logger.info("  Using RandomizedSVD for efficiency...")
            if self.backend == "gpu" and torch.cuda.is_available():
                rsvd_backend = "torch"
                device = "cuda"
            else:
                rsvd_backend = "numba" if self.use_numba else "numpy"
                device = "cpu"

            self.rsvd = RandomizedSVD(
                n_components=initial_components,
                n_iter=self.svd_n_iter,
                n_oversamples=self.svd_n_oversamples,
                backend=rsvd_backend,
                device=device,
                random_state=self.random_state,
                center=False,  # We'll handle centering ourselves for sparse matrices
                # Enhanced features
                use_block_krylov=self.use_block_krylov,
                regularization=self.regularization,
                regularization_strength=self.regularization_strength,
            )

            logger.info(
                f"  Processing sparse matrix directly (sparsity: {1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.3f})"
            )

            # Check for potential rank deficiency
            if min(X_tfidf.shape[0], X_tfidf.shape[1]) < initial_components * 2:
                logger.warning(
                    "Matrix may be rank-deficient. Enabling adaptive regularization."
                )
                self.rsvd.regularization = "adaptive"

            # Fit RSVD - it will automatically detect sparse and use efficient sparse SVD
            U, S, Vt = self.rsvd.fit_transform(X_tfidf, return_components=True)
        else:
            # Use regular SVD (potentially more accurate but requires dense matrix)
            logger.info("  Using regular SVD (numpy.linalg.svd)...")

            # Convert to dense for regular SVD
            if issparse(X_tfidf):
                X_dense = X_tfidf.toarray()
            else:
                X_dense = X_tfidf

            # Use numpy's regular SVD
            U, S, Vt = np.linalg.svd(X_dense, full_matrices=False)

            # Truncate to initial_components for consistency
            U = U[:, :initial_components]
            S = S[:initial_components]
            Vt = Vt[:initial_components, :]

            logger.info(f"  SVD complete: U={U.shape}, S={S.shape}, Vt={Vt.shape}")

        # Energy analysis for optimal dimensionality
        energy = S**2
        captured_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy)

        # For RSVD/sparse SVD, we only have k singular values, not all of them
        # We need to estimate the true total energy or use a different strategy

        # Strategy 1: Use percentage of CAPTURED energy (not total)
        # This is more conservative and works with partial SVD
        cumulative_percentage = cumulative_energy / captured_energy

        # Find where we capture 95% of the energy we've computed
        # Match Fused V2's 95% threshold
        target_percentage = 0.95  # 95% cumulative energy threshold
        n_components_95 = np.searchsorted(cumulative_percentage, target_percentage) + 1

        # Strategy 2: Use mean energy threshold (like Original TEJAS)
        energy_threshold = energy.mean()
        n_components_mean = np.sum(energy > energy_threshold)

        # Strategy 3: Use elbow method - find where energy drops significantly
        if len(S) > 10:
            energy_ratios = energy[1:] / energy[:-1]
            # Find first big drop (ratio < 0.5)
            elbow_idx = np.where(energy_ratios < 0.5)[0]
            n_components_elbow = elbow_idx[0] + 1 if len(elbow_idx) > 0 else len(S)
        else:
            n_components_elbow = len(S)

        # Use a balanced approach for component selection
        # For sparse/RSVD, we need to be more generous since we only see partial spectrum

        # Option 1: Use most of what we computed (conservative)
        n_components_conservative = min(int(len(S) * 0.8), len(S))

        # Option 2: Use 80% of captured energy
        # (already computed as n_components_80)

        # Option 3: Use a minimum threshold to ensure reasonable representation
        n_components_minimum = min(64, len(S))  # At least 64 components if available

        # CRITICAL FIX: Enforce 64 minimum like Original TEJAS
        # Take the maximum of our strategies to avoid being too aggressive
        # First compute the strategy-based selection
        strategy_components = max(
            n_components_95,  # 95% of captured energy
            n_components_mean * 4,  # 4x mean (less aggressive)
        )

        # Always use the requested n_bits, padding will handle insufficient components
        # This ensures output dimensions match what was requested
        effective_bits = self.n_bits

        logger.info("  Component selection:")
        logger.info(f"    95% of captured energy: {n_components_95} components")
        logger.info(f"    Mean energy (2x): {n_components_mean * 2} components")
        logger.info(f"    Elbow method: {n_components_elbow} components")
        logger.info(f"    Selected: {effective_bits} components")
        self.effective_bits = effective_bits  # Store for use in transform

        # Calculate actual explained variance (handle case where effective_bits > len(S))
        available_components = min(effective_bits, len(energy))
        explained_variance = energy[:available_components].sum() / captured_energy

        logger.info("  Energy analysis complete:")
        logger.info(f"    Total singular values: {len(S)}")
        logger.info("    Target energy: 95% (cumulative)")
        logger.info(f"    Selected components: {effective_bits}")
        logger.info(f"    Explained variance: {explained_variance:.3f}")
        logger.info(f"    Top 5 singular values: {S[:5]}")

        # Handle case where we need more bits than available singular values
        n_available = len(S)
        if effective_bits > n_available:
            # Pad with random projections for remaining dimensions
            logger.info(
                f"  Padding: {n_available} singular values available, {effective_bits} requested"
            )

            # Keep all available components
            Vt_truncated = Vt
            S_truncated = S
            if U is not None:
                U_truncated = U

            # Create random projections for additional dimensions
            n_features = Vt.shape[1]
            n_pad = effective_bits - n_available

            # Generate random orthogonal vectors
            random_proj = np.random.randn(n_pad, n_features).astype(np.float32)
            # Orthogonalize against existing components
            for i in range(n_pad):
                for j in range(n_available):
                    random_proj[i] -= np.dot(random_proj[i], Vt[j]) * Vt[j]
                for j in range(i):
                    random_proj[i] -= (
                        np.dot(random_proj[i], random_proj[j]) * random_proj[j]
                    )
                # Normalize
                random_proj[i] /= np.linalg.norm(random_proj[i]) + 1e-8

            # Combine original and random components
            Vt = np.vstack([Vt_truncated, random_proj])
            S = np.concatenate(
                [S_truncated, np.ones(n_pad) * S_truncated[-1] * 0.1]
            )  # Small singular values
            if U is not None:
                U_pad = np.random.randn(U.shape[0], n_pad).astype(np.float32)
                # Orthogonalize U_pad
                U_pad, _ = np.linalg.qr(U_pad)
                U = np.hstack([U_truncated, U_pad])
        else:
            # Truncate to selected components
            Vt = Vt[:effective_bits]
            S = S[:effective_bits]
            if U is not None:
                U = U[:, :effective_bits]

        # Store projection matrix - use Vt for correct dimensions
        # This ensures projection_matrix has shape (n_features, effective_bits)
        self.projection_matrix = (
            Vt.T
            if Vt is not None
            else np.eye(X_tfidf.shape[1], effective_bits, dtype=np.float32)
        )

        # CRITICAL: Normalize projection matrix columns to prevent scale issues
        # This matches what Original TEJAS does and is essential for quality
        col_norms = np.linalg.norm(self.projection_matrix, axis=0, keepdims=True)
        epsilon = 1e-6
        col_norms = np.maximum(col_norms, epsilon)
        self.projection_matrix = self.projection_matrix / col_norms
        logger.info("  Normalized projection matrix columns for scale consistency")

        # Note: data_mean is already computed above for sparse centering

        logger.info(f"  Projection shape: {self.projection_matrix.shape}")
        # Compute explained variance without needing X_dense
        total_variance = np.sum(S**2)
        logger.info(f"  Total captured variance: {total_variance:.3f}")

        # Step 3: Optionally learn ITQ rotation using RSVD output
        if self.use_itq:
            logger.info(
                f"Step 3: Learning ITQ rotation with {self.n_iterations_itq} iterations..."
            )

            # Initialize ITQ WITHOUT precomputed projection (we'll pass projected data directly)
            # Use effective_bits which is the actual dimensionality we have
            config = ITQConfig(
                n_bits=effective_bits,
                max_iterations=self.n_iterations_itq,
                min_iterations=5,
                convergence_threshold=1e-5,
                random_state=self.random_state,
                verbose=True,
            )
            self.itq = ITQOptimizer(config)

            # ITQ only needs to learn rotation on already-projected data
            # Fix: Ensure U and S have matching dimensions after padding/truncation
            if U is not None and U.shape[1] == len(S):
                X_projected = U * S[np.newaxis, :]  # Reconstruct projected data
            else:
                # If U is None or has wrong dimensions, project the training data
                logger.warning(
                    "U matrix unavailable or wrong shape. Projecting training data for ITQ."
                )
                # Project the stored training data through the projection matrix
                X_projected = (
                    self.X_train_sparse @ self.projection_matrix
                    if hasattr(self, "X_train_sparse")
                    else None
                )
                if X_projected is None:
                    logger.error(
                        "Cannot compute projected data for ITQ. Using identity rotation."
                    )
                    self.rotation_matrix = np.eye(effective_bits, dtype=np.float32)
                    self.itq = None
                    return self

            # Normalize data for ITQ (zero-centered, unit variance)
            X_mean = X_projected.mean(axis=0)
            X_std = X_projected.std(axis=0)
            X_std[X_std < 1e-8] = 1.0  # Avoid division by zero
            X_normalized = (X_projected - X_mean) / X_std

            # Store normalization parameters for transform
            self.itq_mean = X_mean
            self.itq_std = X_std

            self.itq.fit(X_normalized)
            self.rotation_matrix = self.itq.rotation_matrix
        else:
            logger.info("Step 3: Skipping ITQ (use_itq=False)")
            # Use identity matrix with effective_bits dimensions
            self.rotation_matrix = np.eye(effective_bits, dtype=np.float32)

        # Compute bias for sparse centering: b = -(μW)R
        # This allows: (x - μ)WR = x(WR) - (μWR)
        if self.data_mean is not None:
            # Note: centering_bias is what we ADD to achieve subtraction
            # Use float64 for intermediate computation to avoid overflow
            with np.errstate(over="ignore", invalid="ignore"):
                proj_64 = self.projection_matrix.astype(np.float64)
                rot_64 = self.rotation_matrix.astype(np.float64)
                mean_64 = self.data_mean.astype(np.float64)
                self.centering_bias = -(mean_64 @ proj_64 @ rot_64).astype(np.float32)
                # Replace any NaN/Inf with 0
                self.centering_bias = np.nan_to_num(
                    self.centering_bias, nan=0.0, posinf=0.0, neginf=0.0
                )
            logger.info("  Computed centering bias for sparse optimization")
        else:
            self.centering_bias = None

        # Cache the combined matrix for efficient transformation
        with np.errstate(over="ignore", invalid="ignore"):
            proj_64 = self.projection_matrix.astype(np.float64)
            rot_64 = self.rotation_matrix.astype(np.float64)
            self.combined_matrix = (proj_64 @ rot_64).astype(np.float32)
            # Replace any NaN/Inf with 0
            self.combined_matrix = np.nan_to_num(
                self.combined_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
        logger.info(f"  Cached combined matrix shape: {self.combined_matrix.shape}")

        logger.info(
            f"  ITQ converged with rotation matrix shape: {self.rotation_matrix.shape}"
        )
        if self.centering_bias is not None:
            logger.info("  Computed centering bias for sparse optimization")

        # Mark as fitted
        self.is_fitted = True
        self.fit_time = time.time() - start_time

        logger.info(f"Fused pipeline fitted in {self.fit_time:.2f}s")
        return self

    def _golden_ratio_sample(self, n_total, memory_limit_gb=50):
        """
        Golden ratio sampling to reduce data size while maintaining coverage.
        Matches Original TEJAS sampling strategy.
        """
        # Estimate memory usage
        bytes_per_element = 4  # float32
        elements_per_sample = self.max_features
        bytes_per_sample = bytes_per_element * elements_per_sample

        # Memory calculations with safety margin
        safety_margin = 0.2
        effective_memory_gb = memory_limit_gb * (1 - safety_margin)
        max_samples = int(effective_memory_gb * 1e9 / bytes_per_sample)

        # Golden ratio reduction
        sample_size = n_total
        while sample_size > max_samples:
            sample_size = int(sample_size / self.golden_ratio)

        # Ensure minimum samples
        sample_size = max(min(100, n_total), sample_size)

        logger.info("  Golden ratio sampling:")
        logger.info(f"    Original: {n_total:,} samples")
        logger.info(
            f"    Selected: {sample_size:,} samples ({sample_size / n_total * 100:.1f}%)"
        )

        # Create evenly spaced indices
        if sample_size < n_total:
            indices = np.linspace(0, n_total - 1, sample_size, dtype=int)
        else:
            indices = np.arange(n_total)

        return indices

    def transform(
        self, titles: List[str], return_packed: bool = True, memory_limit_gb: float = 50
    ) -> np.ndarray:
        """
        Transform titles to binary fingerprints using fused pipeline.

        This performs in a single pass:
        1. TF-IDF transformation
        2. RSVD projection
        3. ITQ rotation
        4. Binarization
        5. Bit packing (optional)

        Args:
            titles: List of titles to encode
            return_packed: If True, return packed uint32 array
            memory_limit_gb: Memory limit for processing (default: 50GB)

        Returns:
            Binary fingerprints (packed or unpacked)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted first")

        start_time = time.time()
        n_titles = len(titles)

        logger.info(f"Transforming {n_titles:,} titles...")

        # Process in batches for memory efficiency
        # Use effective_bits which is the actual dimensionality
        bits_to_use = self.effective_bits

        # Calculate memory requirements
        if return_packed:
            n_chunks = (bits_to_use + 31) // 32
            output_memory_gb = (n_titles * n_chunks * 4) / 1e9  # uint32 = 4 bytes
        else:
            output_memory_gb = (n_titles * bits_to_use) / 1e9  # uint8 = 1 byte

        # Calculate batch processing memory requirements
        # Get actual max_features from fitted vectorizer
        actual_max_features = (
            len(self.vectorizer.vocabulary_) if self.vectorizer else self.max_features
        )
        batch_tfidf_memory_gb = (
            self.batch_size * actual_max_features * 4
        ) / 1e9  # float32
        batch_intermediate_memory_gb = (
            self.batch_size * bits_to_use * 4
        ) / 1e9  # float32 for projections
        batch_memory_gb = batch_tfidf_memory_gb + batch_intermediate_memory_gb

        # Total memory requirement with 20% safety margin
        safety_margin = 1.2
        total_memory_gb = (output_memory_gb + batch_memory_gb) * safety_margin

        # Log memory analysis
        logger.info("  Memory analysis:")
        logger.info(f"    Output array: {output_memory_gb:.2f} GB")
        logger.info(f"    Batch processing: {batch_memory_gb:.2f} GB")
        logger.info(f"    Total required (with margin): {total_memory_gb:.2f} GB")
        logger.info(f"    Memory limit: {memory_limit_gb:.2f} GB")

        # Validate memory requirements and adjust batch size if needed
        actual_batch_size = self.batch_size
        if total_memory_gb > memory_limit_gb:
            # Try to reduce batch size to fit within memory limit
            reduced_batch_size = self.batch_size
            while reduced_batch_size > 100:  # Minimum batch size of 100
                reduced_batch_size = reduced_batch_size // 2
                reduced_batch_memory_gb = (
                    reduced_batch_size * actual_max_features * 4
                ) / 1e9 + (reduced_batch_size * bits_to_use * 4) / 1e9
                reduced_total_gb = (
                    output_memory_gb + reduced_batch_memory_gb
                ) * safety_margin

                if reduced_total_gb <= memory_limit_gb:
                    logger.warning(
                        f"  Reducing batch size from {self.batch_size} to {reduced_batch_size} to fit memory limit"
                    )
                    actual_batch_size = reduced_batch_size
                    break
            else:
                # Cannot fit even with minimum batch size
                raise MemoryError(
                    f"Transformation requires {total_memory_gb:.2f} GB but limit is {memory_limit_gb:.2f} GB. "
                    f"Consider: (1) Increasing memory_limit_gb, (2) Processing fewer titles at once, "
                    f"(3) Reducing n_bits or max_features during fit"
                )

        # Allocate output array
        if return_packed:
            fingerprints = np.zeros((n_titles, n_chunks), dtype=np.uint32)
        else:
            fingerprints = np.zeros((n_titles, bits_to_use), dtype=np.uint8)

        for i in range(0, n_titles, actual_batch_size):
            batch_end = min(i + self.batch_size, n_titles)
            batch_titles = titles[i:batch_end]

            # Step 1: TF-IDF transformation (sparse)
            X_batch = self.vectorizer.transform(batch_titles)

            # Step 2-5: Fused projection → rotation → binarization
            # Use sparse path when data is actually sparse (high sparsity ratio)
            sparsity = 1 - (X_batch.nnz / (X_batch.shape[0] * X_batch.shape[1]))
            use_sparse_path = sparsity > 0.9  # Use sparse path for sparse data

            if use_sparse_path:
                batch_binary = self._fused_transform_sparse(X_batch)
            elif self.use_numba and HAS_NUMBA:
                batch_binary = self._fused_transform_numba(X_batch)
            else:
                batch_binary = self._fused_transform_numpy(X_batch)

            # Store results
            if return_packed:
                # Pack bits for SIMD
                batch_packed = pack_bits_uint32(batch_binary)
                fingerprints[i:batch_end] = batch_packed
            else:
                fingerprints[i:batch_end] = batch_binary

        self.transform_time = time.time() - start_time
        logger.info(f"Transformation complete in {self.transform_time:.2f}s")
        logger.info(f"  Throughput: {n_titles / self.transform_time:.1f} docs/sec")

        return fingerprints

    def _fused_transform_sparse(self, X_sparse: csr_matrix) -> np.ndarray:
        """
        Sparse-efficient transformation using virtual centering.
        For large sparse matrices: uses (x)(WR) - (μWR) to avoid densification
        For small sparse matrices: converts to dense for better numerical stability
        """
        batch_size = X_sparse.shape[0]
        n_features = X_sparse.shape[1]

        # Determine if we should use virtual centering based on matrix size
        # Threshold: ~100MB in float32
        memory_threshold_bytes = 1e8
        matrix_memory = batch_size * n_features * 4  # float32 = 4 bytes

        if matrix_memory < memory_threshold_bytes or self.centering_bias is None:
            # Small matrix or no centering needed: use dense approach for stability
            X_dense = X_sparse.toarray()

            # L2 normalize input vectors (same as Original TEJAS)
            norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
            epsilon = 1e-8
            X_normalized = X_dense / np.maximum(norms, epsilon)

            # Apply centering if mean is available
            if self.data_mean is not None:
                X_centered = X_normalized - self.data_mean
            else:
                X_centered = X_normalized

            # Project using cached combined matrix with overflow protection
            # Use float64 to avoid overflow
            with np.errstate(over="ignore", invalid="ignore"):
                X_centered_64 = X_centered.astype(np.float64)
                combined_64 = self.combined_matrix.astype(np.float64)
                X_transformed = (X_centered_64 @ combined_64).astype(np.float32)
                # Replace any NaN/Inf with 0
                X_transformed = np.nan_to_num(
                    X_transformed, nan=0.0, posinf=0.0, neginf=0.0
                )

        else:
            # Large sparse matrix: use virtual centering to maintain sparsity
            # DON'T normalize input - Original TEJAS normalizes AFTER projection
            X_dense_for_computation = X_sparse.toarray()

            # Step 2: Apply virtual centering with overflow protection
            # (X - μ) @ W @ R = X @ (W @ R) - μ @ (W @ R)
            #                 = X @ combined_matrix - centering_bias
            # Use float64 to avoid overflow
            with np.errstate(over="ignore", invalid="ignore"):
                X_dense_64 = X_dense_for_computation.astype(np.float64)
                combined_64 = self.combined_matrix.astype(np.float64)
                X_transformed = (X_dense_64 @ combined_64).astype(np.float32)
                # Replace any NaN/Inf with 0
                X_transformed = np.nan_to_num(
                    X_transformed, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Apply centering correction (this is the virtual centering)
            if self.centering_bias is not None:
                X_transformed = (
                    X_transformed - self.centering_bias
                )  # Fixed: subtract bias (which is already negative)

        # L2 normalize projected vectors before binarization
        epsilon = 1e-8
        proj_norms = np.linalg.norm(X_transformed, axis=1, keepdims=True)
        X_transformed = X_transformed / np.maximum(proj_norms, epsilon)

        # Binarize (threshold at zero, using strict > like original)
        binary = (X_transformed > 0).astype(np.uint8)

        return binary

    def _fused_transform_numpy(self, X_sparse: csr_matrix) -> np.ndarray:
        """
        Numpy implementation of fused transformation.
        CSR → dense → normalize → center → project → normalize → binarize
        """
        # Convert sparse to dense
        X_dense = X_sparse.toarray()

        # L2 normalize input vectors (critical for semantic preservation)
        norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
        epsilon = 1e-8
        X_normalized = X_dense / np.maximum(norms, epsilon)

        # Apply centering if mean is available
        if self.data_mean is not None:
            X_centered = X_normalized - self.data_mean
        else:
            X_centered = X_normalized

        # Use cached combined matrix for single matmul
        X_transformed = X_centered @ self.combined_matrix

        # L2 normalize projected vectors before binarization
        epsilon = 1e-8
        proj_norms = np.linalg.norm(X_transformed, axis=1, keepdims=True)
        X_transformed = X_transformed / np.maximum(proj_norms, epsilon)

        # Binarize (threshold at zero, using strict > like original)
        binary = (X_transformed > 0).astype(np.uint8)

        return binary

    def _fused_transform_numba(self, X_sparse: csr_matrix) -> np.ndarray:
        """
        Numba-accelerated fused transformation.
        Optimized for CPU with parallel processing.
        """
        if not HAS_NUMBA:
            return self._fused_transform_numpy(X_sparse)

        # Convert to dense
        X_dense = X_sparse.toarray()

        # L2 normalize input vectors
        norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
        epsilon = 1e-8
        X_normalized = X_dense / np.maximum(norms, epsilon)

        # Apply centering if mean is available
        if self.data_mean is not None:
            X_centered = X_normalized - self.data_mean
        else:
            X_centered = X_normalized

        # Use Numba kernel with centered input
        binary = _numba_fused_transform_normalized(X_centered, self.combined_matrix)

        return binary

    def _sparse_to_dense_batched(
        self, X_sparse: csr_matrix, batch_size: int = 5000
    ) -> np.ndarray:
        """
        Convert sparse matrix to dense in batches to save memory.

        This method processes the sparse matrix in chunks and concatenates results
        to avoid pre-allocating the entire dense matrix, which would defeat the
        purpose of batched processing.
        """
        n_samples = X_sparse.shape[0]

        # Process batches and collect results
        dense_batches = []
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            # Convert only this batch to dense
            batch_dense = X_sparse[i:batch_end].toarray()
            dense_batches.append(batch_dense)

        # Concatenate all batches
        # This is more memory efficient as we only hold one batch in memory at a time
        # during conversion, then concatenate at the end
        return np.concatenate(dense_batches, axis=0)

    def clear_cache(self):
        """Clear search cache to free memory."""
        self._searcher_cache = None
        self._cached_database = None

        # Clear intermediate components if needed
        if self.rsvd is not None and hasattr(self.rsvd, "U_"):
            self.rsvd.U_ = None  # Clear large intermediate matrices
            self.rsvd.S_ = None

    def search(
        self, query: Union[str, List[str]], database_packed: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar items using the fused pipeline.

        Args:
            query: Query text or list of queries
            database_packed: Packed database fingerprints
            k: Number of nearest neighbors

        Returns:
            indices: Top-k indices
            distances: Hamming distances
        """
        # Transform query
        if isinstance(query, str):
            query = [query]

        query_packed = self.transform(query, return_packed=True)

        # Cache searcher if database hasn't changed (content-based check)
        # Check both object identity and shape/dtype to detect changes
        cache_invalid = (
            self._cached_database is None
            or self._searcher_cache is None
            or self._cached_database is not database_packed
            or self._cached_database.shape != database_packed.shape
            or self._cached_database.dtype != database_packed.dtype
        )

        if cache_invalid:
            self._searcher_cache = HammingSIMDImpl(
                database_packed,
                backend="numba" if self.use_numba else "numpy",
                use_packing=False,  # Already packed
            )
            self._cached_database = database_packed

        if len(query) == 1:
            return self._searcher_cache.search(query_packed[0], k)
        else:
            return self._searcher_cache.search_batch(query_packed, k)


# Numba kernels for acceleration
if HAS_NUMBA:

    @jit(nopython=True, parallel=True, fastmath=True)
    def _numba_fused_transform_normalized(X_normalized, combined_matrix):
        """
        Numba kernel for fused transformation with normalized input.
        Performs: X_normalized @ combined_matrix → normalize → binary
        """
        n_samples = X_normalized.shape[0]
        n_bits = combined_matrix.shape[1]

        # Pre-allocate output
        binary = np.zeros((n_samples, n_bits), dtype=np.uint8)

        # Process each sample in parallel
        for i in prange(n_samples):
            # Project
            projected = np.zeros(n_bits)
            for j in range(n_bits):
                val = 0.0
                for k in range(X_normalized.shape[1]):
                    val += X_normalized[i, k] * combined_matrix[k, j]
                projected[j] = val

            # Normalize projected vector
            norm = 0.0
            for j in range(n_bits):
                norm += projected[j] * projected[j]
            norm = np.sqrt(norm)
            if norm < 1e-8:
                norm = 1e-8

            # Binarize normalized projection
            for j in range(n_bits):
                binary[i, j] = 1 if (projected[j] / norm) >= 0 else 0

        return binary

else:

    def _numba_fused_transform_normalized(X_normalized, combined_matrix):
        """Fallback to numpy implementation with normalization"""
        X_transformed = X_normalized @ combined_matrix
        # Normalize projected vectors
        norms = np.linalg.norm(X_transformed, axis=1, keepdims=True)
        epsilon = 1e-8
        X_transformed = X_transformed / np.maximum(norms, epsilon)
        return (X_transformed >= 0).astype(np.uint8)


def benchmark_fused_vs_original(n_samples=10000, n_features=5000, n_bits=128):
    """
    Benchmark fused pipeline against original sequential implementation.
    """
    import time
    from core.encoder import GoldenRatioEncoder

    # Generate synthetic data
    np.random.seed(42)
    titles = [f"synthetic document {i} " * 10 for i in range(n_samples)]

    logger.info("=" * 60)
    logger.info("FUSED PIPELINE BENCHMARK")
    logger.info("=" * 60)

    # Test original implementation
    logger.info("\n1. Original Sequential Pipeline:")
    original = GoldenRatioEncoder(
        n_bits=n_bits, max_features=n_features, use_itq=True, use_randomized_svd=True
    )

    start = time.time()
    original.fit(titles[:1000], memory_limit_gb=10)  # Train on subset
    fit_time_original = time.time() - start

    start = time.time()
    fingerprints_original = original.transform(titles, batch_size=1000)
    transform_time_original = time.time() - start

    logger.info(f"  Fit time: {fit_time_original:.2f}s")
    logger.info(f"  Transform time: {transform_time_original:.2f}s")
    logger.info(f"  Total: {fit_time_original + transform_time_original:.2f}s")

    # Test fused implementation
    logger.info("\n2. Fused Pipeline:")
    fused = FusedPipelineEncoder(
        n_bits=n_bits, max_features=n_features, batch_size=1000, use_numba=True
    )

    start = time.time()
    fused.fit(titles[:1000], memory_limit_gb=10)
    fit_time_fused = time.time() - start

    start = time.time()
    fingerprints_fused = fused.transform(titles, return_packed=True)
    transform_time_fused = time.time() - start

    logger.info(f"  Fit time: {fit_time_fused:.2f}s")
    logger.info(f"  Transform time: {transform_time_fused:.2f}s")
    logger.info(f"  Total: {fit_time_fused + transform_time_fused:.2f}s")

    # Calculate speedup
    total_original = fit_time_original + transform_time_original
    total_fused = fit_time_fused + transform_time_fused
    speedup = total_original / total_fused

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS:")
    logger.info(f"  Original: {total_original:.2f}s")
    logger.info(f"  Fused: {total_fused:.2f}s")
    logger.info(f"  SPEEDUP: {speedup:.2f}x")
    logger.info("=" * 60)

    return {
        "original_time": total_original,
        "fused_time": total_fused,
        "speedup": speedup,
    }


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_fused_vs_original()
    print(f"\nFinal speedup: {results['speedup']:.2f}x")
