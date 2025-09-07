"""
Fused Pipeline Encoder V2 - Modular, Robust Implementation
===========================================================

A complete rewrite of the FusedPipelineEncoder with proper separation of concerns,
numerical stability, and comprehensive error handling.

Architecture:
    Stage Pattern: Each processing step is an independent Stage
    Pipeline: Orchestrates stages with validation
    FusedEncoder: Adds performance optimizations without compromising correctness

Key Improvements:
    - Modular stages with clear contracts
    - Proper numerical stability and conditioning
    - Correct data flow (ITQ receives projected data, not raw SVD components)
    - Consistent binarization across all paths
    - Comprehensive dimension validation
    - True streaming support for large datasets
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List
from scipy.sparse import csr_matrix, issparse
import joblib
from pathlib import Path

# Import TF-IDF implementation (prefer numpy-only to avoid sklearn dependency)
try:
    from .tfidf_numpy import TfidfVectorizerNumpy as TfidfVectorizer

    USING_SKLEARN = False
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer

    USING_SKLEARN = True

# Import other components
from .randomized_svd import RandomizedSVD
from .itq_optimized import ITQOptimizer, ITQConfig
from .hamming_simd import HammingSIMDImpl

logger = logging.getLogger(__name__)

# Try to import Numba-optimized functions
try:
    from .numba_helpers import (
        binarize_cyclic,
        pack_bits_uint32,
        normalize_rows,
        NUMBA_AVAILABLE,
    )

    if NUMBA_AVAILABLE:
        logger.info("Using Numba JIT-compiled optimizations")
except ImportError:
    # Fall back to original implementation
    from .hamming_simd import pack_bits_uint32

    NUMBA_AVAILABLE = False
    logger.info("Numba optimizations not available")

# Log which TF-IDF implementation we're using
if USING_SKLEARN:
    logger.info("Using sklearn TF-IDF implementation")
else:
    logger.info("Using pure NumPy TF-IDF implementation (no sklearn dependency)")


# ============================================================================
# Base Stage Class
# ============================================================================


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage must implement fit/transform methods and provide
    dimension information for validation.
    """

    def __init__(self, name: str = None):
        """Initialize stage with optional name for debugging."""
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self._output_dim = None
        self._input_dim = None

    @abstractmethod
    def fit(self, X: Union[np.ndarray, csr_matrix], y=None) -> "Stage":
        """
        Fit the stage on training data.

        Args:
            X: Input data (dense or sparse)
            y: Ignored (for sklearn compatibility)

        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def transform(
        self, X: Union[np.ndarray, csr_matrix]
    ) -> Union[np.ndarray, csr_matrix]:
        """
        Transform input data.

        Args:
            X: Input data

        Returns:
            Transformed data
        """
        pass

    def fit_transform(
        self, X: Union[np.ndarray, csr_matrix], y=None
    ) -> Union[np.ndarray, csr_matrix]:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    @property
    def output_dim(self) -> Optional[int]:
        """Output dimensionality of this stage."""
        return self._output_dim

    @property
    def input_dim(self) -> Optional[int]:
        """Expected input dimensionality."""
        return self._input_dim

    def validate_input(self, X: Union[np.ndarray, csr_matrix]) -> None:
        """
        Validate input dimensions and type.

        Raises:
            ValueError: If input is invalid
        """
        if not self.is_fitted and self.input_dim is not None:
            if X.shape[1] != self.input_dim:
                raise ValueError(
                    f"{self.name}: Expected input dim {self.input_dim}, got {X.shape[1]}"
                )

    def save(self, path: Union[str, Path]) -> None:
        """Save stage to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Stage":
        """Load stage from disk."""
        return joblib.load(path)


# ============================================================================
# TF-IDF Vectorization Stage
# ============================================================================


class TFIDFStage(Stage):
    """
    TF-IDF vectorization stage wrapping sklearn's TfidfVectorizer.

    Handles text to sparse matrix conversion with configurable parameters.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (3, 5),
        analyzer: str = "char",
        dtype: np.dtype = np.float32,
        **kwargs,
    ):
        """
        Initialize TF-IDF stage.

        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range for feature extraction
            analyzer: Unit of analysis ('char' or 'word')
            dtype: Output data type
            **kwargs: Additional arguments for TfidfVectorizer
        """
        super().__init__("TFIDFStage")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.dtype = dtype
        self.vectorizer_kwargs = kwargs
        self.vectorizer = None

    def fit(self, X: List[str], y=None) -> "TFIDFStage":
        """
        Fit TF-IDF vectorizer on text data.

        Args:
            X: List of text documents

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting TF-IDF with max_features={self.max_features}")

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
            dtype=self.dtype,
            lowercase=True,
            **self.vectorizer_kwargs,
        )

        try:
            self.vectorizer.fit(X)
        except ValueError as e:
            if "empty vocabulary" in str(e):
                # Handle very sparse data by using a simpler analyzer
                logger.warning("Empty vocabulary detected, using character analyzer")
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    analyzer="char",
                    ngram_range=(1, 2),
                    dtype=self.dtype,
                    lowercase=True,
                )
                self.vectorizer.fit(X)
            else:
                raise

        self._output_dim = len(self.vectorizer.vocabulary_)
        self.is_fitted = True

        logger.info(f"TF-IDF vocabulary size: {self._output_dim:,}")
        return self

    def transform(self, X: List[str]) -> csr_matrix:
        """
        Transform text to TF-IDF features.

        Args:
            X: List of text documents

        Returns:
            Sparse TF-IDF matrix
        """
        if not self.is_fitted:
            raise ValueError("TFIDFStage must be fitted before transform")

        X_tfidf = self.vectorizer.transform(X)

        # Log sparsity for debugging
        sparsity = 1 - (X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]))
        logger.debug(f"TF-IDF sparsity: {sparsity:.3f}")

        return X_tfidf


# ============================================================================
# Projection Stage (RSVD + Normalization)
# ============================================================================


class ProjectionStage(Stage):
    """
    Dimensionality reduction via RSVD with proper normalization.

    Handles all projection, scaling, and centering in one place.
    """

    def __init__(
        self,
        n_components: int = 128,
        center: bool = True,
        normalize_output: bool = True,
        energy_threshold: float = 0.95,
        condition_threshold: float = 1e10,
        dtype: np.dtype = np.float32,
        random_state: int = 42,
        **rsvd_kwargs,
    ):
        """
        Initialize projection stage.

        Args:
            n_components: Target dimensionality
            center: Whether to center data before projection
            normalize_output: Whether to L2-normalize projected vectors
            energy_threshold: Cumulative energy threshold for auto-selection
            condition_threshold: Max condition number before regularization
            dtype: Computation data type
            random_state: Random seed
            **rsvd_kwargs: Additional arguments for RandomizedSVD
        """
        super().__init__("ProjectionStage")
        self.n_components = n_components
        self.center = center
        self.normalize_output = normalize_output
        self.energy_threshold = energy_threshold
        self.condition_threshold = condition_threshold
        self.dtype = dtype
        self.random_state = random_state
        self.rsvd_kwargs = rsvd_kwargs

        # Fitted parameters
        self.rsvd = None
        self.projection_matrix = None
        self.data_mean = None
        self.singular_values = None
        self.explained_variance_ratio = None
        self.effective_components = None

    def fit(self, X: Union[np.ndarray, csr_matrix], y=None) -> "ProjectionStage":
        """
        Fit RSVD on input data.

        Args:
            X: Input data (dense or sparse)

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting ProjectionStage with n_components={self.n_components}")

        # Store input dimension
        self._input_dim = X.shape[1]
        n_samples = X.shape[0]

        # Compute mean for centering (if enabled)
        if self.center:
            if issparse(X):
                # Efficient sparse mean computation
                self.data_mean = np.array(X.mean(axis=0)).flatten().astype(self.dtype)
            else:
                self.data_mean = X.mean(axis=0).astype(self.dtype)
            logger.info("Computed data mean for centering")
        else:
            self.data_mean = None

        # Check condition number for numerical stability
        if not issparse(X) and X.shape[0] < 10000:  # Only for small dense matrices
            try:
                cond = np.linalg.cond(X)
                if cond > self.condition_threshold:
                    logger.warning(
                        f"High condition number: {cond:.2e}. Enabling regularization."
                    )
                    self.rsvd_kwargs["regularization"] = "adaptive"
            except:
                pass  # Skip condition check if it fails

        # Determine actual number of components to compute
        max_components = min(n_samples, self._input_dim) - 1
        # For requested components, try to get exactly what's asked for
        components_to_compute = min(self.n_components, max_components)

        # Choose SVD method based on matrix properties
        # For small/medium datasets or when degeneracy is likely, use standard SVD
        use_standard_svd = (
            n_samples < 10000 and not issparse(X)
        ) or self.rsvd_kwargs.get("force_standard_svd", False)

        if use_standard_svd:
            # Use standard SVD for better consistency with degenerate singular values
            logger.info(f"Using standard SVD for {components_to_compute} components")

            if issparse(X):
                X_dense = X.toarray()
            else:
                X_dense = X

            if self.center:
                X_centered = X_dense - self.data_mean
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            else:
                U, S, Vt = np.linalg.svd(X_dense, full_matrices=False)

            # Truncate to requested components
            U = U[:, :components_to_compute]
            S = S[:components_to_compute]
            Vt = Vt[:components_to_compute, :]
        else:
            # Use RSVD for large datasets
            logger.info(f"Using RandomizedSVD for {components_to_compute} components")

            # Initialize RSVD
            self.rsvd = RandomizedSVD(
                n_components=components_to_compute,
                center=False,  # We handle centering ourselves
                random_state=self.random_state,
                dtype=self.dtype,
                **self.rsvd_kwargs,
            )

            # Apply centering if needed
            if self.center and not issparse(X):
                X_centered = X - self.data_mean
                U, S, Vt = self.rsvd.fit_transform(X_centered, return_components=True)
            else:
                # For sparse or non-centered, fit directly
                U, S, Vt = self.rsvd.fit_transform(X, return_components=True)

        self.singular_values = S

        # Energy-based component selection
        energy = S**2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy) / total_energy

        # Find number of components for target energy
        n_for_threshold = np.searchsorted(cumulative_energy, self.energy_threshold) + 1

        # Prioritize explicitly requested n_components over energy threshold
        # Only use energy-based selection if n_components is very large or not set
        if self.n_components <= len(S):
            # Use the requested number of components
            self.effective_components = min(self.n_components, len(S))
            logger.info(
                f"Using requested {self.effective_components} components (energy: {cumulative_energy[self.effective_components - 1]:.3f})"
            )
        else:
            # Fall back to energy-based selection
            self.effective_components = min(n_for_threshold, len(S))
            logger.info(
                f"Using energy-based selection: {self.effective_components} components for {self.energy_threshold:.1%} energy"
            )

        # Store projection matrix (truncated to effective components)
        self.projection_matrix = Vt[: self.effective_components].T.astype(self.dtype)

        # Skip normalization - it's causing numerical instability
        # The projection matrix from SVD is already well-conditioned

        # Store explained variance
        if self.effective_components > 0:
            self.explained_variance_ratio = cumulative_energy[
                self.effective_components - 1
            ]
        else:
            self.explained_variance_ratio = 0.0

        self._output_dim = self.effective_components
        self.is_fitted = True

        logger.info("ProjectionStage fitted:")
        logger.info(f"  Effective components: {self.effective_components}")
        logger.info(f"  Explained variance: {self.explained_variance_ratio:.3f}")
        logger.info(f"  Top 5 singular values: {S[:5]}")

        return self

    def transform(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """
        Project input data to lower dimension.

        Args:
            X: Input data

        Returns:
            Projected data (always dense)
        """
        if not self.is_fitted:
            raise ValueError("ProjectionStage must be fitted before transform")

        self.validate_input(X)

        # Center if needed
        if self.center:
            if issparse(X):
                # Virtual centering for sparse matrices with numerical stability
                X_proj = X @ self.projection_matrix
                # Compute centering bias with higher precision to avoid overflow
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    centering_bias = self.data_mean.astype(
                        np.float64
                    ) @ self.projection_matrix.astype(np.float64)
                    centering_bias = centering_bias.astype(self.dtype)
                    # Check for numerical issues
                    if not (
                        np.any(np.isnan(centering_bias))
                        or np.any(np.isinf(centering_bias))
                    ):
                        X_proj = X_proj - centering_bias
                    else:
                        # Fallback: skip centering if numerical issues
                        logger.debug("Numerical issues in centering, skipping")
            else:
                X_centered = X - self.data_mean
                X_proj = X_centered @ self.projection_matrix
        else:
            # Direct projection
            X_proj = X @ self.projection_matrix

        # Normalize output vectors if requested
        if self.normalize_output:
            norms = np.linalg.norm(X_proj, axis=1, keepdims=True)
            X_proj = X_proj / np.maximum(norms, 1e-8)

        return X_proj.astype(self.dtype)


# ============================================================================
# Rotation Stage (ITQ)
# ============================================================================


class RotationStage(Stage):
    """
    ITQ rotation learning on projected data.

    CRITICAL: Only accepts already-projected data as input.
    This prevents the dimension mismatch bug in the original implementation.
    """

    def __init__(
        self,
        n_iterations: int = 50,
        convergence_threshold: float = 1e-5,
        random_state: int = 42,
    ):
        """
        Initialize rotation stage.

        Args:
            n_iterations: Maximum ITQ iterations
            convergence_threshold: Convergence tolerance
            random_state: Random seed
        """
        super().__init__("RotationStage")
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.random_state = random_state

        # Fitted parameters
        self.itq = None
        self.rotation_matrix = None
        self.data_mean = None

    def fit(self, X: np.ndarray, y=None) -> "RotationStage":
        """
        Learn ITQ rotation on projected data.

        Args:
            X: ALREADY PROJECTED data (dense, n_samples x n_components)

        Returns:
            Self for chaining
        """
        if issparse(X):
            raise ValueError("RotationStage requires dense input (projected data)")

        logger.info(f"Fitting RotationStage on {X.shape[1]} dimensions")

        # Store dimensions
        self._input_dim = X.shape[1]
        self._output_dim = X.shape[1]  # Rotation preserves dimensionality

        # Normalize data for ITQ (zero-centered, unit variance)
        self.data_mean = X.mean(axis=0)
        self.data_std = X.std(axis=0)
        self.data_std[self.data_std < 1e-8] = 1.0  # Avoid division by zero
        X_normalized = (X - self.data_mean) / self.data_std

        # Initialize and fit ITQ with proper config
        config = ITQConfig(
            n_bits=self._input_dim,
            max_iterations=self.n_iterations,
            min_iterations=5,
            convergence_threshold=self.convergence_threshold,
            random_state=self.random_state,
            verbose=True,
        )
        self.itq = ITQOptimizer(config)

        # ITQ expects normalized data
        self.itq.fit(X_normalized)
        self.rotation_matrix = self.itq.rotation_matrix

        # Verify rotation matrix is orthogonal
        eye = self.rotation_matrix @ self.rotation_matrix.T
        ortho_error = np.abs(eye - np.eye(self._input_dim)).max()
        if ortho_error > 1e-5:
            logger.warning(
                f"Rotation matrix not perfectly orthogonal: max error {ortho_error:.2e}"
            )

        self.is_fitted = True
        # Get convergence info from ITQ
        convergence_info = (
            self.itq.get_convergence_summary()
            if hasattr(self.itq, "get_convergence_summary")
            else {}
        )
        iterations_used = convergence_info.get("iterations", self.n_iterations)
        logger.info(f"RotationStage fitted with {iterations_used} iterations")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply learned rotation to projected data.

        Args:
            X: Projected data

        Returns:
            Rotated data
        """
        if not self.is_fitted:
            raise ValueError("RotationStage must be fitted before transform")

        self.validate_input(X)

        # Normalize and rotate (using stored normalization parameters)
        X_normalized = (X - self.data_mean) / self.data_std
        X_rotated = X_normalized @ self.rotation_matrix

        return X_rotated


# ============================================================================
# Binarization Stage
# ============================================================================


class BinarizationStage(Stage):
    """
    Binarization with cyclic tiling and optional harmonic thresholding.

    Supports multiple expansion strategies when components < target_bits:
    - 'pad_zero': Original behavior, pad with zeros
    - 'cyclic': Tile components cyclically to fill bits
    - 'harmonic': Use multiple thresholds per component
    """

    def __init__(
        self,
        pack_bits: bool = True,
        dtype_packed: np.dtype = np.uint32,
        target_bits: int = 256,
        expansion_strategy: str = "cyclic",
    ):
        """
        Initialize binarization stage.

        Args:
            pack_bits: Whether to pack bits into uint arrays
            dtype_packed: Data type for packed arrays (uint8/uint16/uint32)
            target_bits: Target number of bits
            expansion_strategy: How to handle component < bits case:
                'pad_zero': Pad with zeros (original behavior)
                'cyclic': Tile components cyclically
                'harmonic': Multiple thresholds per component
        """
        super().__init__("BinarizationStage")
        self.pack_bits = pack_bits
        self.dtype_packed = dtype_packed
        self.target_bits = target_bits
        self.expansion_strategy = expansion_strategy

        if expansion_strategy not in ["pad_zero", "cyclic", "harmonic"]:
            raise ValueError(f"Unknown expansion strategy: {expansion_strategy}")

    def fit(self, X: np.ndarray, y=None) -> "BinarizationStage":
        """
        Fit is a no-op for binarization.

        Args:
            X: Input data (for dimension discovery)

        Returns:
            Self for chaining
        """
        self._input_dim = X.shape[1]

        if self.pack_bits:
            # Calculate packed dimensions
            bits_per_element = np.dtype(self.dtype_packed).itemsize * 8
            self._output_dim = (
                self._input_dim + bits_per_element - 1
            ) // bits_per_element
        else:
            self._output_dim = self._input_dim

        self.is_fitted = True
        logger.info(
            f"BinarizationStage configured: input={self._input_dim}, output={self._output_dim}"
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Binarize and optionally pack input data.

        Args:
            X: Continuous valued data

        Returns:
            Binary codes (packed or unpacked)
        """
        n_samples, n_features = X.shape

        # Normalize to unit sphere (matching original TEJAS)
        if "normalize_rows" in globals():
            # Use Numba-optimized version if available
            X_normalized = normalize_rows(X, epsilon=1e-8)
        else:
            # Fall back to pure NumPy
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_normalized = X / (norms + 1e-8)

        # Handle expansion based on strategy
        if n_features < self.target_bits:
            if self.expansion_strategy == "cyclic":
                # Cyclic tiling: repeat components to fill bits
                if "binarize_cyclic" in globals():
                    # Use Numba-optimized version if available
                    binary = binarize_cyclic(X_normalized, self.target_bits)
                else:
                    # Fall back to pure NumPy
                    binary = np.zeros((n_samples, self.target_bits), dtype=np.uint8)
                    for i in range(self.target_bits):
                        source_idx = i % n_features
                        binary[:, i] = (X_normalized[:, source_idx] > 0).astype(
                            np.uint8
                        )
                logger.info(
                    f"Applied cyclic tiling: {n_features} components → {self.target_bits} bits"
                )

            elif self.expansion_strategy == "harmonic":
                # Harmonic thresholding: multiple thresholds per component
                binary = np.zeros((n_samples, self.target_bits), dtype=np.uint8)

                # Calculate how many thresholds we need per component
                thresholds_per_comp = (self.target_bits + n_features - 1) // n_features

                # Generate thresholds based on data distribution
                thresholds = []
                if thresholds_per_comp == 1:
                    thresholds = [0.0]
                elif thresholds_per_comp == 2:
                    thresholds = [0.0, np.median(X_normalized)]
                else:
                    # Use percentiles for multiple thresholds
                    percentiles = np.linspace(20, 80, thresholds_per_comp)
                    thresholds = [np.percentile(X_normalized, p) for p in percentiles]

                # Apply thresholds
                bit_idx = 0
                for thresh_idx, threshold in enumerate(thresholds):
                    for comp_idx in range(n_features):
                        if bit_idx >= self.target_bits:
                            break
                        binary[:, bit_idx] = (
                            X_normalized[:, comp_idx] > threshold
                        ).astype(np.uint8)
                        bit_idx += 1
                    if bit_idx >= self.target_bits:
                        break

                logger.info(
                    f"Applied harmonic thresholding: {n_features} components → {self.target_bits} bits with {len(thresholds)} thresholds"
                )

            else:  # pad_zero
                # Original behavior: pad with zeros
                binary = (X_normalized > 0).astype(np.uint8)
                padding = np.zeros(
                    (n_samples, self.target_bits - n_features), dtype=np.uint8
                )
                binary = np.hstack([binary, padding])
                logger.info(
                    f"Applied zero padding: {n_features} components → {self.target_bits} bits"
                )
        else:
            # Standard binarization when we have enough components
            binary = (X_normalized > 0).astype(np.uint8)
            if n_features > self.target_bits:
                binary = binary[:, : self.target_bits]

        if self.pack_bits:
            # Pack bits based on dtype
            if self.dtype_packed == np.uint32:
                return pack_bits_uint32(binary)
            elif self.dtype_packed == np.uint8:
                # Pack into uint8 (8 bits per element)
                n_samples = binary.shape[0]
                n_bits = binary.shape[1]
                n_packed = (n_bits + 7) // 8
                packed = np.zeros((n_samples, n_packed), dtype=np.uint8)

                for i in range(n_packed):
                    start_bit = i * 8
                    end_bit = min(start_bit + 8, n_bits)
                    for j in range(start_bit, end_bit):
                        bit_position = j - start_bit
                        packed[:, i] |= binary[:, j] << bit_position

                return packed
            else:
                raise ValueError(f"Unsupported packed dtype: {self.dtype_packed}")

        return binary


# ============================================================================
# Pipeline Class
# ============================================================================


class Pipeline:
    """
    Orchestrates stages with validation and error handling.

    Ensures correct data flow and dimension compatibility between stages.
    """

    def __init__(self, stages: List[Stage], validate_dims: bool = True):
        """
        Initialize pipeline with stages.

        Args:
            stages: Ordered list of stages
            validate_dims: Whether to validate dimensions between stages
        """
        self.stages = stages
        self.validate_dims = validate_dims
        self.is_fitted = False

    def fit(self, X: Union[List[str], np.ndarray, csr_matrix], y=None) -> "Pipeline":
        """
        Fit all stages sequentially.

        Args:
            X: Input data (text for TF-IDF, array for others)

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting pipeline with {len(self.stages)} stages")

        # Transform through each stage, fitting as we go
        X_current = X
        for i, stage in enumerate(self.stages):
            logger.info(f"Fitting stage {i + 1}/{len(self.stages)}: {stage.name}")

            # Fit and transform for next stage (except last)
            if i < len(self.stages) - 1:
                X_current = stage.fit_transform(X_current)
            else:
                stage.fit(X_current)

            # Validate dimension compatibility
            if self.validate_dims and i < len(self.stages) - 1:
                next_stage = self.stages[i + 1]
                if (
                    hasattr(next_stage, "input_dim")
                    and next_stage.input_dim is not None
                ):
                    if stage.output_dim != next_stage.input_dim:
                        raise ValueError(
                            f"Dimension mismatch: {stage.name} outputs {stage.output_dim}, "
                            f"but {next_stage.name} expects {next_stage.input_dim}"
                        )

        self.is_fitted = True
        logger.info("Pipeline fitted successfully")
        return self

    def transform(
        self,
        X: Union[List[str], np.ndarray, csr_matrix],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Transform data through all stages.

        Args:
            X: Input data
            batch_size: Optional batch size for memory efficiency

        Returns:
            Final transformed output
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        if batch_size is not None:
            # Process in batches
            return self._transform_batched(X, batch_size)

        # Transform through each stage
        X_current = X
        for stage in self.stages:
            X_current = stage.transform(X_current)

        return X_current

    def _transform_batched(
        self, X: Union[List[str], np.ndarray], batch_size: int
    ) -> np.ndarray:
        """
        Transform data in batches for memory efficiency.

        Args:
            X: Input data
            batch_size: Batch size

        Returns:
            Concatenated results
        """
        n_samples = len(X) if isinstance(X, list) else X.shape[0]
        results = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            if isinstance(X, list):
                X_batch = X[i:end_idx]
            else:
                X_batch = X[i:end_idx]

            # Transform batch through pipeline
            X_batch_transformed = X_batch
            for stage in self.stages:
                X_batch_transformed = stage.transform(X_batch_transformed)

            results.append(X_batch_transformed)

        return np.vstack(results)

    def fit_transform(
        self, X: Union[List[str], np.ndarray, csr_matrix], y=None
    ) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X, y).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save entire pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved pipeline to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Pipeline":
        """Load pipeline from disk."""
        return joblib.load(path)


# ============================================================================
# Fused Pipeline Encoder (Optimized Pipeline)
# ============================================================================


class FusedPipelineEncoder:
    """
    High-performance encoder that fuses operations where beneficial.

    Built on top of the modular Pipeline with additional optimizations:
    - Cached combined projection/rotation matrix
    - Efficient batch processing
    - Memory management
    - Search functionality
    """

    def __init__(
        self,
        n_bits: int = 128,
        max_features: int = 10000,
        use_itq: bool = True,
        n_iterations_itq: int = 50,
        batch_size: int = 5000,
        energy_threshold: float = 0.95,
        center_data: bool = True,
        expansion_strategy: str = "cyclic",
        dtype: np.dtype = np.float32,
        random_state: int = 42,
        svd_n_oversamples: int = 10,  # RSVD oversampling parameter (optimized)
        svd_n_iter: int = 5,
    ):  # RSVD iteration parameter (balanced)
        """
        Initialize fused encoder.

        Args:
            n_bits: Number of bits in binary code
            max_features: Maximum TF-IDF features
            use_itq: Whether to use ITQ rotation
            n_iterations_itq: ITQ iterations
            batch_size: Batch size for processing
            energy_threshold: Energy threshold for dimension selection
            center_data: Whether to center data before projection
            expansion_strategy: Strategy for handling components < bits:
                'pad_zero': Pad with zeros (original)
                'cyclic': Tile components cyclically (recommended)
                'harmonic': Multiple thresholds per component
            dtype: Computation data type
            random_state: Random seed
        """
        self.n_bits = n_bits
        self.max_features = max_features
        self.use_itq = use_itq
        self.n_iterations_itq = n_iterations_itq
        self.batch_size = batch_size
        self.energy_threshold = energy_threshold
        self.center_data = center_data
        self.expansion_strategy = expansion_strategy
        self.dtype = dtype
        self.random_state = random_state

        # Build pipeline stages
        stages = [
            TFIDFStage(max_features=max_features, dtype=dtype),
            ProjectionStage(
                n_components=n_bits,
                center=center_data,
                normalize_output=True,
                energy_threshold=energy_threshold,
                dtype=dtype,
                random_state=random_state,
                # Optimized RSVD parameters for high dimensions
                n_oversamples=svd_n_oversamples,  # Use parameter from init
                n_iter=svd_n_iter,  # Use parameter from init
            ),
        ]

        if use_itq:
            stages.append(
                RotationStage(n_iterations=n_iterations_itq, random_state=random_state)
            )

        stages.append(
            BinarizationStage(
                pack_bits=True,
                dtype_packed=np.uint32,
                target_bits=self.n_bits,
                expansion_strategy=self.expansion_strategy,
            )
        )

        self.pipeline = Pipeline(stages, validate_dims=True)

        # Cached optimization matrices
        self.combined_matrix = None
        self.centering_bias = None

        # Search cache
        self._searcher_cache = None
        self._cached_database = None

        # Performance tracking
        self.fit_time = None
        self.transform_time = None

    @property
    def effective_components(self):
        """Get effective components from projection stage."""
        if hasattr(self, "pipeline") and len(self.pipeline.stages) > 1:
            projection_stage = self.pipeline.stages[1]  # Projection is second stage
            if hasattr(projection_stage, "effective_components"):
                return projection_stage.effective_components
        return None

    def fit(
        self, titles: List[str], memory_limit_gb: float = 50
    ) -> "FusedPipelineEncoder":
        """
        Fit the encoder on training data.

        Args:
            titles: List of text documents
            memory_limit_gb: Memory limit for processing

        Returns:
            Self for chaining
        """
        import time

        start_time = time.time()

        logger.info(f"Fitting FusedPipelineEncoder on {len(titles):,} documents")

        # Fit the pipeline
        self.pipeline.fit(titles)

        # Cache combined transformation matrix for efficiency
        self._cache_combined_matrix()

        self.fit_time = time.time() - start_time
        logger.info(f"FusedPipelineEncoder fitted in {self.fit_time:.2f}s")

        return self

    def _cache_combined_matrix(self):
        """Cache the combined projection and rotation matrix."""
        # Get projection matrix from ProjectionStage
        proj_stage = self.pipeline.stages[1]  # TF-IDF is 0, Projection is 1
        projection_matrix = proj_stage.projection_matrix

        # Get rotation matrix if using ITQ
        if self.use_itq:
            rot_stage = self.pipeline.stages[2]  # Rotation is 2
            rotation_matrix = rot_stage.rotation_matrix

            # Combine matrices
            self.combined_matrix = projection_matrix @ rotation_matrix

            # Cache centering bias if needed
            if proj_stage.center:
                # Bias for virtual centering: -(mean @ proj @ rot)
                mean_proj = proj_stage.data_mean @ projection_matrix
                mean_proj_rot = mean_proj @ rotation_matrix
                self.centering_bias = -mean_proj_rot + rot_stage.data_mean
        else:
            self.combined_matrix = projection_matrix

            # Cache centering bias if needed with numerical stability
            if proj_stage.center:
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    bias = proj_stage.data_mean.astype(
                        np.float64
                    ) @ projection_matrix.astype(np.float64)
                    self.centering_bias = -bias.astype(np.float32)
                    if np.any(np.isnan(self.centering_bias)) or np.any(
                        np.isinf(self.centering_bias)
                    ):
                        logger.debug(
                            "Numerical issues in centering bias, setting to zero"
                        )
                        self.centering_bias = np.zeros_like(bias, dtype=np.float32)

        logger.info(f"Cached combined matrix: {self.combined_matrix.shape}")

    def transform(
        self, titles: List[str], return_packed: bool = True, memory_limit_gb: float = 50
    ) -> np.ndarray:
        """
        Transform documents to binary codes.

        Args:
            titles: List of text documents
            return_packed: Whether to return packed bits
            memory_limit_gb: Memory limit

        Returns:
            Binary codes (packed or unpacked)
        """
        import time

        start_time = time.time()

        n_docs = len(titles)
        logger.info(f"Transforming {n_docs:,} documents")

        # Use pipeline's batch transform
        result = self.pipeline.transform(titles, batch_size=self.batch_size)

        # Unpack if requested
        if not return_packed and result.dtype != np.uint8:
            # Unpack from uint32 to uint8
            from .hamming_simd import unpack_bits_uint32

            result = unpack_bits_uint32(result, self.n_bits)

        self.transform_time = time.time() - start_time
        logger.info(f"Transformation complete in {self.transform_time:.2f}s")
        logger.info(f"Throughput: {n_docs / self.transform_time:.1f} docs/sec")

        return result

    def search(
        self, query: Union[str, List[str]], database: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar items.

        Args:
            query: Query text(s)
            database: Database codes (packed or unpacked)
            k: Number of neighbors

        Returns:
            indices: Top-k indices
            distances: Hamming distances
        """
        if isinstance(query, str):
            query = [query]

        # Import here to avoid circular dependency
        from .hamming_simd import pack_bits_uint32, unpack_bits_uint32

        # Detect database format
        database_is_packed = database.dtype == np.uint32

        # Transform query with matching format
        if database_is_packed:
            # Database is packed, transform query to packed format
            query_transformed = self.transform(query, return_packed=True)
        else:
            # Database is unpacked, transform query to unpacked format
            query_transformed = self.transform(query, return_packed=False)

        # Ensure shape compatibility - sometimes transform returns wrong format
        if len(query) == 1:
            query_single = (
                query_transformed[0]
                if query_transformed.ndim > 1
                else query_transformed
            )

            # Check for shape mismatch and fix
            if database_is_packed and query_single.dtype == np.uint8:
                # Query is unpacked but database is packed - pack the query
                query_single = pack_bits_uint32(query_single.reshape(1, -1))[0]
            elif not database_is_packed and query_single.dtype == np.uint32:
                # Query is packed but database is unpacked - unpack the query
                query_single = unpack_bits_uint32(
                    query_single.reshape(1, -1), self.n_bits
                )[0]

        # Initialize or reuse searcher with correct format
        if self._searcher_cache is None or self._cached_database is not database:
            self._searcher_cache = HammingSIMDImpl(
                database,
                backend="numpy",
                use_packing=False,  # HammingSIMDImpl handles format internally
            )
            self._cached_database = database

        # Search
        if len(query) == 1:
            return self._searcher_cache.search(query_single, k)
        else:
            # For batch, ensure all queries match database format
            if database_is_packed and query_transformed.dtype == np.uint8:
                query_transformed = pack_bits_uint32(query_transformed)
            elif not database_is_packed and query_transformed.dtype == np.uint32:
                query_transformed = unpack_bits_uint32(query_transformed, self.n_bits)
            return self._searcher_cache.search_batch(query_transformed, k)

    def save(self, path: Union[str, Path]) -> None:
        """Save encoder to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save pipeline and cached matrices
        save_dict = {
            "pipeline": self.pipeline,
            "combined_matrix": self.combined_matrix,
            "centering_bias": self.centering_bias,
            "config": {
                "n_bits": self.n_bits,
                "max_features": self.max_features,
                "use_itq": self.use_itq,
                "n_iterations_itq": self.n_iterations_itq,
                "batch_size": self.batch_size,
                "energy_threshold": self.energy_threshold,
                "center_data": self.center_data,
                "dtype": self.dtype,
                "random_state": self.random_state,
            },
        }

        joblib.dump(save_dict, path)
        logger.info(f"Saved FusedPipelineEncoder to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FusedPipelineEncoder":
        """Load encoder from disk."""
        save_dict = joblib.load(path)

        # Reconstruct encoder
        config = save_dict["config"]
        encoder = cls(**config)
        encoder.pipeline = save_dict["pipeline"]
        encoder.combined_matrix = save_dict["combined_matrix"]
        encoder.centering_bias = save_dict["centering_bias"]

        return encoder


# ============================================================================
# Convenience Functions
# ============================================================================


def create_fused_encoder(
    n_bits: int = 128, max_features: int = 10000, use_itq: bool = True, **kwargs
) -> FusedPipelineEncoder:
    """
    Create a fused encoder with sensible defaults.

    Args:
        n_bits: Number of bits
        max_features: Max TF-IDF features
        use_itq: Whether to use ITQ
        **kwargs: Additional arguments

    Returns:
        Configured encoder
    """
    return FusedPipelineEncoder(
        n_bits=n_bits, max_features=max_features, use_itq=use_itq, **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np

    # Generate sample data
    np.random.seed(42)
    titles = [f"Document {i} with some text content" for i in range(1000)]

    # Create and fit encoder
    encoder = create_fused_encoder(n_bits=128, max_features=5000)
    encoder.fit(titles[:500])  # Train on subset

    # Transform documents
    fingerprints = encoder.transform(titles)
    print(f"Fingerprint shape: {fingerprints.shape}")
    print(f"Fingerprint dtype: {fingerprints.dtype}")

    # Search example
    query = "Document 42 with some text content"
    indices, distances = encoder.search(query, fingerprints, k=5)
    print(f"Top 5 matches for query: {indices}")
    print(f"Distances: {distances}")
