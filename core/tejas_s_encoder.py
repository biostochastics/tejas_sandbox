"""
Tejas-S (Streamlined) Encoder
==============================
Streamlined implementation with caching and always using RSVD for efficiency.
Based on StreamlinedFusedEncoder but enhanced with query caching.
"""

import numpy as np
import logging
import time
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

logger = logging.getLogger(__name__)


class TejasSEncoder:
    """
    Tejas-S (Streamlined) encoder with mandatory RSVD and query caching.

    Key features:
    - Always uses Randomized SVD for consistency
    - Query caching for repeated searches
    - No ITQ (direct binarization)
    - No bit packing (uint8 for speed)
    - Simple and fast
    """

    def __init__(
        self,
        n_bits: int = 256,
        max_features: int = 10000,
        svd_n_iter: int = 3,  # Balanced for speed/accuracy
        memory_limit_gb: float = 4.0,
        cache_size: int = 1000,
        random_state: int = 42,
    ):
        """
        Initialize Tejas-S encoder.

        Args:
            n_bits: Number of bits in binary code
            max_features: Maximum TF-IDF features
            svd_n_iter: Power iterations for RSVD (always used)
            memory_limit_gb: Memory limit for golden ratio sampling
            cache_size: Size of query cache
            random_state: Random seed
        """
        self.n_bits = n_bits
        self.max_features = max_features
        self.svd_n_iter = svd_n_iter
        self.memory_limit_gb = memory_limit_gb
        self.cache_size = cache_size
        self.random_state = random_state
        self.golden_ratio = (1 + np.sqrt(5)) / 2

        # Components
        self.vectorizer = None
        self.projection_matrix = None
        self.mean_ = None
        self.sample_indices = None
        self.is_fitted = False

        # Query cache
        self._query_cache = {}
        self._cache_order = []  # For LRU

        # Performance tracking
        self.fit_time = None
        self.n_components = n_bits  # For compatibility

        logger.info("Initialized Tejas-S Encoder")
        logger.info(f"  n_bits: {n_bits}")
        logger.info(f"  max_features: {max_features}")
        logger.info(f"  RSVD iterations: {svd_n_iter}")
        logger.info(f"  cache_size: {cache_size}")

    def _golden_ratio_sample(self, n_total: int) -> np.ndarray:
        """Golden ratio sampling for memory efficiency."""
        bytes_per_element = 4  # float32
        elements_per_sample = self.max_features
        bytes_per_sample = bytes_per_element * elements_per_sample
        max_samples = int(self.memory_limit_gb * 1e9 / bytes_per_sample)

        sample_size = n_total
        while sample_size > max_samples:
            sample_size = int(sample_size / self.golden_ratio)

        if sample_size < n_total:
            # Use uniform sampling for better coverage
            indices = np.linspace(0, n_total - 1, sample_size, dtype=int)
            logger.info(f"  Sampled {sample_size:,}/{n_total:,} documents")
        else:
            indices = np.arange(n_total)

        return indices

    def _apply_randomized_svd(self, X: np.ndarray) -> np.ndarray:
        """Always use Randomized SVD for dimensionality reduction."""
        from core.randomized_svd import RandomizedSVD

        logger.info(f"  Applying Randomized SVD to {X.shape}")

        svd = RandomizedSVD(
            n_components=self.n_bits,
            n_iter=self.svd_n_iter,
            random_state=self.random_state,
        )

        # Fit and get projection
        svd.fit(X)

        # Store components as projection matrix
        # V.T gives us the projection from feature space to component space
        self.projection_matrix = svd.components_.T.astype(np.float32)

        # Store singular values for diagnostics
        if hasattr(svd, "singular_values_"):
            self.singular_values_ = svd.singular_values_
            # Calculate explained variance
            total_variance = np.sum(svd.singular_values_**2)
            if total_variance > 0:
                self.explained_variance_ratio_ = (
                    svd.singular_values_**2
                ) / total_variance

        return self.projection_matrix

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for a query."""
        return hashlib.md5(text.encode()).hexdigest()

    def _update_cache(self, key: str, value: np.ndarray):
        """Update cache with LRU eviction."""
        if key in self._query_cache:
            # Move to end (most recent)
            self._cache_order.remove(key)
            self._cache_order.append(key)
        else:
            # Add new entry
            if len(self._query_cache) >= self.cache_size:
                # Evict oldest
                oldest = self._cache_order.pop(0)
                del self._query_cache[oldest]

            self._query_cache[key] = value.copy()
            self._cache_order.append(key)

    def fit(self, texts: List[str]) -> "TejasSEncoder":
        """
        Fit the encoder on training texts.

        Args:
            texts: List of text documents

        Returns:
            Self for chaining
        """
        start_time = time.time()
        logger.info(f"Fitting Tejas-S on {len(texts):,} texts")

        # Step 1: TF-IDF Vectorization
        logger.info("  Step 1: TF-IDF vectorization")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            dtype=np.float32,
        )
        X_tfidf = self.vectorizer.fit_transform(texts)

        # Step 2: Golden ratio sampling
        logger.info("  Step 2: Golden ratio sampling")
        n_samples = X_tfidf.shape[0]
        self.sample_indices = self._golden_ratio_sample(n_samples)

        # Convert to dense for sampled data
        X_sampled = X_tfidf[self.sample_indices].toarray()

        # Step 3: Compute mean for centering
        logger.info("  Step 3: Computing mean")
        self.mean_ = np.mean(X_sampled, axis=0, keepdims=True)
        X_centered = X_sampled - self.mean_

        # Step 4: Always apply Randomized SVD
        logger.info("  Step 4: Randomized SVD")
        self._apply_randomized_svd(X_centered)

        self.is_fitted = True
        self.fit_time = time.time() - start_time

        logger.info(f"Fitting complete in {self.fit_time:.2f}s")
        logger.info(f"  Projection shape: {self.projection_matrix.shape}")

        # Clear cache after fitting
        self._query_cache.clear()
        self._cache_order.clear()

        return self

    def transform(
        self, texts: Union[str, List[str]], use_cache: bool = True
    ) -> np.ndarray:
        """
        Transform texts to binary fingerprints.

        Args:
            texts: Single text or list of texts
            use_cache: Whether to use query cache for single texts

        Returns:
            Binary fingerprints as uint8 array
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        # Handle single text
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

            # Check cache for single queries
            if use_cache and len(texts) == 1:
                cache_key = self._get_cache_key(texts[0])
                if cache_key in self._query_cache:
                    logger.debug("Cache hit for query")
                    return self._query_cache[cache_key]

        # Vectorize
        X_tfidf = self.vectorizer.transform(texts)
        X_dense = X_tfidf.toarray()

        # Center using training mean
        X_centered = X_dense - self.mean_

        # Project to binary space
        X_projected = X_centered @ self.projection_matrix

        # Binarize (simple zero threshold)
        fingerprints = (X_projected > 0).astype(np.uint8)

        # Update cache for single queries
        if single_text and use_cache:
            self._update_cache(cache_key, fingerprints)

        return fingerprints[0] if single_text else fingerprints

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts, use_cache=False)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Alias for transform (compatibility)."""
        return self.transform(texts)

    # For compatibility with unified benchmark
    def __str__(self):
        return f"Tejas-S(n_bits={self.n_bits}, cache={self.cache_size})"
