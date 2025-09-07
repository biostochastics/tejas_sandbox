"""
Optimized Fused Encoder WITHOUT Bit Packing
============================================

This version removes all bit packing operations for pure fingerprint comparison.
Uses the same optimizations as optimized_fused but returns raw binary arrays.
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Union, Optional, Dict, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OptimizedFusedEncoderNoPack:
    """
    Optimized fused encoder without bit packing.
    Returns raw binary fingerprints for maximum speed in pure comparisons.
    """

    def __init__(
        self,
        n_bits: int = 256,
        n_components: Optional[int] = None,
        max_features: int = 10000,
        tokenizer: str = "char",
        ngram_range: tuple = (3, 5),
        energy_threshold: float = 0.95,
        device: str = "cpu",
        batch_size: int = 512,
        use_cache: bool = True,
        cache_size: int = 10000,
    ):
        self.n_bits = n_bits
        self.n_components = n_components
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.ngram_range = ngram_range
        self.energy_threshold = energy_threshold
        self.device = device
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.cache_size = cache_size

        # Components to be learned
        self.vectorizer = None
        self.projection_matrix = None
        self.singular_values = None
        self.fitted = False

        # Cache for repeated encodings
        self.cache = {} if use_cache else None

        logger.info("Initialized OptimizedFusedEncoderNoPack")
        logger.info(f"  n_bits: {n_bits}, tokenizer: {tokenizer}")
        logger.info(f"  energy_threshold: {energy_threshold}")
        logger.info("  NO BIT PACKING - returns raw binary arrays")

    def fit(self, texts: List[str], y=None):
        """Fit the encoder on training texts."""
        logger.info(f"Fitting encoder on {len(texts)} texts...")

        # Vectorize texts
        logger.info("Vectorizing texts...")
        self.vectorizer = TfidfVectorizer(
            analyzer=self.tokenizer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=True,
            dtype=np.float32,
        )

        X = self.vectorizer.fit_transform(texts)
        X_dense = X.toarray()

        # Convert to torch for SVD
        X_tensor = torch.from_numpy(X_dense).float()
        if self.device != "cpu" and torch.cuda.is_available():
            X_tensor = X_tensor.to(self.device)

        # Compute SVD
        logger.info("Computing SVD...")
        U, S, Vh = torch.linalg.svd(X_tensor, full_matrices=False)

        # Determine number of components based on energy threshold
        if self.n_components is None:
            energy = S**2
            total_energy = energy.sum()
            cumulative_energy = torch.cumsum(energy, dim=0) / total_energy

            # Find components needed for energy threshold
            n_for_threshold = (
                torch.searchsorted(cumulative_energy, self.energy_threshold).item() + 1
            )
            self.n_components = min(n_for_threshold, self.n_bits, len(S))

            logger.info(
                f"Selected {self.n_components} components for {self.energy_threshold:.1%} energy"
            )

        # Store projection matrix
        self.projection_matrix = Vh[: self.n_components].T.cpu().numpy()
        self.singular_values = S[: self.n_components].cpu().numpy()

        self.fitted = True
        logger.info("Fitting complete")

        return self

    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Transform texts into binary fingerprints WITHOUT bit packing.

        Returns:
            Binary numpy array of shape (n_texts, n_bits) with values 0 or 1
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        all_fingerprints = []

        # Process in batches
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Encoding",
            disable=len(texts) < 100,
        ):
            batch = texts[i : i + self.batch_size]

            # Check cache
            if self.use_cache:
                fingerprints = []
                uncached_texts = []
                uncached_indices = []

                for j, text in enumerate(batch):
                    if text in self.cache:
                        fingerprints.append(self.cache[text])
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(j)

                if uncached_texts:
                    # Process uncached texts
                    new_fps = self._transform_batch(uncached_texts)

                    # Update cache
                    for text, fp in zip(uncached_texts, new_fps):
                        if len(self.cache) < self.cache_size:
                            self.cache[text] = fp

                    # Merge cached and new fingerprints
                    result = np.zeros((len(batch), self.n_bits), dtype=np.uint8)
                    cached_idx = 0
                    uncached_idx = 0

                    for j in range(len(batch)):
                        if j in uncached_indices:
                            result[j] = new_fps[uncached_idx]
                            uncached_idx += 1
                        else:
                            result[j] = fingerprints[cached_idx]
                            cached_idx += 1

                    all_fingerprints.append(result)
                else:
                    all_fingerprints.append(np.array(fingerprints))
            else:
                # No caching
                fingerprints = self._transform_batch(batch)
                all_fingerprints.append(fingerprints)

        # Combine all batches
        result = np.vstack(all_fingerprints) if all_fingerprints else np.array([])

        if single_input and len(result) > 0:
            result = result[0]

        return result

    def _transform_batch(self, texts: List[str]) -> np.ndarray:
        """Transform a batch of texts without caching."""
        # Vectorize
        X = self.vectorizer.transform(texts)
        X_dense = X.toarray()

        # Project
        X_projected = X_dense @ self.projection_matrix

        # Apply weights
        if self.singular_values is not None:
            X_projected = X_projected * self.singular_values

        # Pad or truncate to n_bits
        if X_projected.shape[1] < self.n_bits:
            padding = np.zeros(
                (X_projected.shape[0], self.n_bits - X_projected.shape[1])
            )
            X_projected = np.hstack([X_projected, padding])
        elif X_projected.shape[1] > self.n_bits:
            X_projected = X_projected[:, : self.n_bits]

        # Convert to binary (NO PACKING)
        binary = (X_projected > 0).astype(np.uint8)

        return binary

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Alias for transform."""
        return self.transform(texts)

    def get_params(self) -> Dict[str, Any]:
        """Get encoder parameters."""
        return {
            "n_bits": self.n_bits,
            "n_components": self.n_components,
            "max_features": self.max_features,
            "tokenizer": self.tokenizer,
            "ngram_range": self.ngram_range,
            "energy_threshold": self.energy_threshold,
            "use_cache": self.use_cache,
            "cache_size": self.cache_size,
            "fitted": self.fitted,
        }

    def clear_cache(self):
        """Clear the encoding cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
