"""
Streamlined Fused Pipeline Encoder - Fast and Simple
=====================================================

A cleaned-up fused pipeline that removes unnecessary optimizations:
- NO ITQ (adds 0.4s overhead for minimal gain)
- NO bit packing (adds complexity, slows down search)
- NO complex backend management (overhead exceeds benefits)
- KEEPS: Randomized SVD for memory efficiency when needed
- KEEPS: Direct numpy operations for speed
- KEEPS: Simple, fast transformations

This aims to match or exceed baseline performance while maintaining
the architectural benefits of a fused pipeline.
"""

import numpy as np
import logging
import time
from typing import List, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class StreamlinedFusedEncoder:
    """
    Streamlined fused encoder that removes unnecessary overhead.

    Key design decisions:
    - No ITQ: Direct binarization is sufficient
    - No bit packing: Keep uint8 for fast operations
    - No backend management: NumPy only for consistency
    - Simple golden ratio sampling like baseline
    - Randomized SVD only when actually beneficial (>10k features)
    """

    def __init__(
        self,
        n_bits: int = 128,
        max_features: int = 10000,
        use_randomized_svd: bool = True,
        svd_n_iter: int = 2,  # Reduced iterations for speed
        memory_limit_gb: float = 2.0,  # Same as baseline
        random_state: int = 42,
    ):
        """
        Initialize streamlined encoder.

        Args:
            n_bits: Number of bits in binary code
            max_features: Maximum TF-IDF features
            use_randomized_svd: Use RSVD for large matrices (auto-disabled for small)
            svd_n_iter: Power iterations for RSVD (2 is usually sufficient)
            memory_limit_gb: Memory limit for golden ratio sampling
            random_state: Random seed
        """
        self.n_bits = n_bits
        self.max_features = max_features
        self.use_randomized_svd = use_randomized_svd
        self.svd_n_iter = svd_n_iter
        self.memory_limit_gb = memory_limit_gb
        self.random_state = random_state
        self.golden_ratio = (1 + np.sqrt(5)) / 2

        # Components
        self.vectorizer = None
        self.projection_matrix = None
        self.mean_ = None
        self.sample_indices = None
        self.is_fitted = False

        # Performance tracking
        self.fit_time = None
        self.transform_time = None

        logger.info("Initialized StreamlinedFusedEncoder")
        logger.info(f"  n_bits: {n_bits}")
        logger.info(f"  max_features: {max_features}")
        logger.info(f"  memory_limit: {memory_limit_gb}GB")

    def _golden_ratio_sample(self, n_total: int) -> np.ndarray:
        """
        Golden ratio sampling matching baseline implementation.

        Args:
            n_total: Total number of samples

        Returns:
            Sample indices
        """
        # Calculate memory-constrained sample size
        bytes_per_element = 4  # float32
        elements_per_sample = self.max_features
        bytes_per_sample = bytes_per_element * elements_per_sample
        max_samples = int(self.memory_limit_gb * 1e9 / bytes_per_sample)

        # Apply golden ratio reduction
        sample_size = n_total
        reduction_level = 0

        while sample_size > max_samples:
            sample_size = int(sample_size / self.golden_ratio)
            reduction_level += 1

        logger.info(f"Golden ratio sampling: {n_total:,} -> {sample_size:,} samples")

        # Use uniform sampling for better coverage (improvement over baseline)
        if sample_size < n_total:
            # Uniform sampling instead of logarithmic for better representation
            indices = np.linspace(0, n_total - 1, sample_size, dtype=int)
            indices = np.unique(indices)
        else:
            indices = np.arange(n_total)

        return indices

    def _compute_svd(
        self, X: np.ndarray, n_components: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SVD using either randomized or standard method.

        Args:
            X: Input matrix (n_samples, n_features)
            n_components: Number of components

        Returns:
            U, S: Left singular vectors and singular values
        """
        n_samples, n_features = X.shape

        # Use randomized SVD only when beneficial
        use_rsvd = (
            self.use_randomized_svd
            and n_features > 5000  # Only for high-dimensional data
            and n_components
            < min(n_samples, n_features) // 4  # And requesting few components
        )

        if use_rsvd:
            logger.info(f"Using Randomized SVD for {n_features} features")
            try:
                from .randomized_svd import RandomizedSVD

                rsvd = RandomizedSVD(
                    n_components=n_components,
                    n_oversamples=10,  # Minimal oversampling
                    n_iter=self.svd_n_iter,  # Fast iterations
                    random_state=self.random_state,
                    dtype=np.float32,
                    backend="numpy",  # Simple numpy backend
                )

                # Fit and get components
                rsvd.fit(X)
                U = rsvd.U_
                S = rsvd.S_
                Vt = rsvd.Vt_

                # Build projection matrix
                self.projection_matrix = Vt.T

            except Exception as e:
                logger.warning(f"RSVD failed, falling back to standard SVD: {e}")
                use_rsvd = False

        if not use_rsvd:
            logger.info(f"Using standard SVD for {n_features} features")
            # Standard SVD with economical mode
            U, S, Vt = np.linalg.svd(X, full_matrices=False)

            # Keep only n_components
            U = U[:, :n_components]
            S = S[:n_components]
            Vt = Vt[:n_components, :]

            # Build projection matrix
            self.projection_matrix = Vt.T

        return U, S

    def fit(self, texts: List[str]) -> "StreamlinedFusedEncoder":
        """
        Fit the encoder using streamlined pipeline.

        Args:
            texts: List of text documents

        Returns:
            Self for chaining
        """
        start_time = time.time()
        n_texts = len(texts)
        logger.info(f"Fitting streamlined encoder on {n_texts:,} texts...")

        # Step 1: TF-IDF vectorization
        logger.info("Step 1: TF-IDF vectorization...")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=self.max_features,
            lowercase=True,
            dtype=np.float32,
        )

        # Fit on all texts (learns vocabulary)
        self.vectorizer.fit(texts)
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"  Vocabulary size: {vocab_size:,}")

        # Step 2: Golden ratio sampling if needed
        if n_texts > 10000:  # Only sample for large datasets
            logger.info("Step 2: Golden ratio sampling...")
            self.sample_indices = self._golden_ratio_sample(n_texts)
            sample_texts = [texts[i] for i in self.sample_indices]
        else:
            self.sample_indices = np.arange(n_texts)
            sample_texts = texts

        # Step 3: Transform samples
        logger.info(f"Step 3: Transforming {len(sample_texts):,} samples...")
        X_tfidf = self.vectorizer.transform(sample_texts)

        # Convert to dense for SVD (unavoidable)
        if hasattr(X_tfidf, "toarray"):
            X_dense = X_tfidf.toarray()
        else:
            X_dense = X_tfidf

        # Step 4: Center the data
        logger.info("Step 4: Centering data...")
        self.mean_ = X_dense.mean(axis=0, keepdims=True)
        X_centered = X_dense - self.mean_

        # Step 5: Compute SVD projection
        logger.info("Step 5: Computing SVD projection...")
        n_components = min(self.n_bits, X_centered.shape[0] - 1, X_centered.shape[1])

        if n_components < self.n_bits:
            logger.warning(f"Capping bits to {n_components} due to data constraints")
            self.n_bits = n_components

        U, S = self._compute_svd(X_centered, n_components)

        # Log singular value decay for diagnostics
        if len(S) > 1:
            energy_ratio = S[0] / S[-1]
            logger.info(f"  Singular value ratio: {energy_ratio:.2f}")

        self.is_fitted = True
        self.fit_time = time.time() - start_time
        logger.info(f"Fitting complete in {self.fit_time:.2f}s")

        return self

    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Transform texts to binary codes using streamlined pipeline.
        NO ITQ, NO bit packing - just fast, simple binarization.

        Args:
            texts: Text or list of texts

        Returns:
            Binary codes as uint8 array (n_texts, n_bits)
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        start_time = time.time()

        # Step 1: TF-IDF transform
        X_tfidf = self.vectorizer.transform(texts)

        # Step 2: Convert to dense
        if hasattr(X_tfidf, "toarray"):
            X_dense = X_tfidf.toarray()
        else:
            X_dense = X_tfidf

        # Step 3: Center using training mean
        X_centered = X_dense - self.mean_

        # Step 4: Project using SVD components
        X_projected = X_centered @ self.projection_matrix

        # Step 5: Simple binarization (threshold at 0)
        # This is fast and effective - no ITQ needed
        X_binary = (X_projected > 0).astype(np.uint8)

        # Ensure correct number of bits
        if X_binary.shape[1] < self.n_bits:
            # Pad with zeros if needed
            padding = np.zeros(
                (X_binary.shape[0], self.n_bits - X_binary.shape[1]), dtype=np.uint8
            )
            X_binary = np.hstack([X_binary, padding])

        self.transform_time = time.time() - start_time

        if single_input:
            return X_binary[0]
        return X_binary

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one call.

        Args:
            texts: List of texts

        Returns:
            Binary codes
        """
        return self.fit(texts).transform(texts)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Alias for transform for compatibility.
        """
        return self.transform(texts)

    def save(self, path: str):
        """
        Save encoder to disk.
        """
        import pickle

        state = {
            "n_bits": self.n_bits,
            "max_features": self.max_features,
            "vectorizer": self.vectorizer,
            "projection_matrix": self.projection_matrix,
            "mean_": self.mean_,
            "sample_indices": self.sample_indices,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Saved encoder to {path}")

    def load(self, path: str):
        """
        Load encoder from disk.
        """
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.n_bits = state["n_bits"]
        self.max_features = state["max_features"]
        self.vectorizer = state["vectorizer"]
        self.projection_matrix = state["projection_matrix"]
        self.mean_ = state["mean_"]
        self.sample_indices = state["sample_indices"]
        self.is_fitted = state["is_fitted"]

        logger.info(f"Loaded encoder from {path}")

    def __repr__(self):
        return (
            f"StreamlinedFusedEncoder("
            f"n_bits={self.n_bits}, "
            f"max_features={self.max_features}, "
            f"fitted={self.is_fitted})"
        )
