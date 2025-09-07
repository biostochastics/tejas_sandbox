"""
Randomized SVD Encoder
======================
Drop-in replacement for original Tejas encoder using Randomized SVD.
Matches original's variance-based component selection logic.
"""

import time
import logging
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from core.randomized_svd import RandomizedSVD

logger = logging.getLogger(__name__)


class RandomizedSVDEncoder:
    """
    Binary encoder using Randomized SVD for scalability.
    Matches original Tejas energy-based component selection.
    """

    def __init__(self, n_bits=256, max_features=10000, device="cpu"):
        self.n_bits = n_bits
        self.max_features = max_features
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.device = device

        # Components to be learned
        self.vectorizer = None
        self.projection = None
        self.singular_values = None
        self.n_components = None
        self.sample_indices = None
        self.training_stats = {}

        logger.info("Initialized RandomizedSVDEncoder")
        logger.info(f"  n_bits: {n_bits}")
        logger.info(f"  max_features: {max_features}")

    def _golden_ratio_sample(self, n_total, target_memory_gb=50):
        """Sample using golden ratio until it fits in memory."""
        bytes_per_element = 4
        elements_per_sample = self.max_features
        bytes_per_sample = bytes_per_element * elements_per_sample

        max_samples = int(target_memory_gb * 1e9 / bytes_per_sample)

        sample_size = n_total
        reduction_level = 0

        while sample_size > max_samples:
            sample_size = int(sample_size / self.golden_ratio)
            reduction_level += 1

        logger.info("Golden ratio sampling:")
        logger.info(f"  Original: {n_total:,} samples")
        logger.info(f"  Reduced: {sample_size:,} samples")
        logger.info(f"  Reduction levels: {reduction_level}")

        if sample_size < n_total:
            indices = np.unique(
                np.logspace(0, np.log10(n_total - 1), sample_size).astype(int)
            )
        else:
            indices = np.arange(n_total)

        return indices

    def fit(self, titles, memory_limit_gb=50):
        """
        Fit encoder using Randomized SVD with variance-based selection.
        Matches original Tejas component selection logic.
        """
        start_time = time.time()
        logger.info(f"Training RandomizedSVD encoder on {len(titles):,} titles...")

        # Step 1: Fit vectorizer
        logger.info("Step 1: Learning vocabulary...")
        t0 = time.time()

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=self.max_features,
            lowercase=True,
            dtype=np.float32,
        )
        self.vectorizer.fit(titles)

        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"  Vocabulary size: {vocab_size:,}")
        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Step 2: Golden ratio sampling
        logger.info("Step 2: Golden ratio sampling...")
        t0 = time.time()

        self.sample_indices = self._golden_ratio_sample(len(titles), memory_limit_gb)
        sample_titles = [titles[i] for i in self.sample_indices]
        logger.info(f"  Sampled {len(sample_titles):,} titles")
        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Step 3: Transform and prepare matrix
        logger.info("Step 3: Transforming sampled titles...")
        t0 = time.time()

        X_sample = self.vectorizer.transform(sample_titles)
        X_dense = X_sample.toarray()
        logger.info(f"  Matrix shape: {X_dense.shape}")
        logger.info(f"  Matrix memory: {X_dense.nbytes / 1e9:.2f} GB")
        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Step 4: Randomized SVD with adaptive component selection
        logger.info("Step 4: Computing Randomized SVD...")
        t0 = time.time()

        # First, estimate how many components we might need
        # Start with min(n_bits * 2, features, samples) to ensure we get enough
        max_possible = min(X_dense.shape)
        initial_components = min(self.n_bits * 2, max_possible)

        # Use randomized SVD
        rsvd = RandomizedSVD(
            n_components=initial_components, n_oversamples=10, n_iter=2, random_state=42
        )

        # Fit and get decomposition
        rsvd.fit(X_dense)

        # Get the components (stored after fit)
        U = rsvd.U_
        S = rsvd.S_  # Note: capital S_
        Vh = rsvd.components_  # This is Vt_

        logger.info(f"  Computed {len(S)} singular values")
        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Step 5: Energy-based component selection (matching original Tejas)
        logger.info("Step 5: Selecting components based on energy threshold...")
        t0 = time.time()

        # Energy analysis - EXACTLY matching original Tejas logic
        energy = S**2
        total_energy = energy.sum()
        energy_threshold = energy.mean()  # Use mean energy as threshold

        # Find components above mean energy
        n_components = np.sum(energy > energy_threshold)

        # Constrain to reasonable range (matching original)
        n_components = np.clip(n_components, 64, min(self.n_bits, len(S)))

        # Calculate explained variance
        explained_variance = energy[:n_components].sum() / total_energy

        logger.info(f"  Total singular values: {len(S)}")
        logger.info(f"  Energy threshold: {energy_threshold:.2f}")
        logger.info(f"  Selected components: {n_components}")
        logger.info(f"  Explained variance: {explained_variance:.3f}")
        logger.info(f"  Top 5 singular values: {S[:5]}")
        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Step 6: Store projection matrix with cyclical padding
        # Ensure we always output n_bits dimensions
        if n_components < self.n_bits:
            # Use cyclical padding to maintain structure
            logger.info(
                f"  Applying cyclical padding from {n_components} to {self.n_bits} components"
            )

            # Calculate how many full cycles and remainder
            n_cycles = self.n_bits // n_components
            remainder = self.n_bits % n_components

            # Build padded projection matrix cyclically
            Vh_padded = np.zeros((self.n_bits, Vh.shape[1]), dtype=np.float32)
            S_padded = np.zeros(self.n_bits, dtype=np.float32)

            # Fill with cycles
            for i in range(n_cycles):
                start_idx = i * n_components
                end_idx = min((i + 1) * n_components, self.n_bits)
                copy_size = end_idx - start_idx

                # Apply decay factor for each cycle to maintain energy hierarchy
                decay = 1.0 / (i + 1)  # Each cycle has less influence

                Vh_padded[start_idx:end_idx] = Vh[:copy_size] * decay
                S_padded[start_idx:end_idx] = S[:copy_size] * decay

            # Fill remainder if needed
            if remainder > 0:
                start_idx = n_cycles * n_components
                decay = 1.0 / (n_cycles + 1)
                Vh_padded[start_idx:] = Vh[:remainder] * decay
                S_padded[start_idx:] = S[:remainder] * decay

            self.projection = Vh_padded.T  # Shape: (features, n_bits)
            self.singular_values = S_padded
        else:
            # We have more components than needed, just truncate
            self.projection = Vh[: self.n_bits].T  # Shape: (features, n_bits)
            self.singular_values = S[: self.n_bits]
            n_components = self.n_bits

        self.n_components = (
            n_components  # Actual meaningful components (before padding)
        )

        # Store training statistics
        self.training_stats = {
            "n_titles": len(titles),
            "n_samples": len(sample_titles),
            "sampling_ratio": len(sample_titles) / len(titles),
            "vocab_size": vocab_size,
            "n_components": n_components,
            "explained_variance": float(explained_variance),
            "energy_threshold": float(energy_threshold),
            "total_energy": float(total_energy),
            "train_time": time.time() - start_time,
        }

        logger.info(f"Training complete in {time.time() - start_time:.2f}s")
        return self

    def train(self, titles, memory_limit_gb=50, batch_size=10000):
        """Compatibility method that calls fit."""
        return self.fit(titles, memory_limit_gb)

    def encode(self, titles, batch_size=10000):
        """
        Encode titles into binary fingerprints.
        """
        if self.projection is None:
            raise ValueError("Encoder must be fitted before encoding")

        n_titles = len(titles)
        n_batches = (n_titles + batch_size - 1) // batch_size

        fingerprints = []

        for i in tqdm(range(n_batches), desc="Encoding titles"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_titles)
            batch_titles = titles[start_idx:end_idx]

            # Transform to TF-IDF
            X_batch = self.vectorizer.transform(batch_titles)
            X_dense = X_batch.toarray()

            # Project using learned components
            projected = X_dense @ self.projection

            # Binarize (sign function)
            binary = (projected > 0).astype(np.uint8)

            fingerprints.append(binary)

        # Concatenate all batches
        all_fingerprints = np.vstack(fingerprints)

        # Pack bits if n_bits is divisible by 8
        if self.n_bits % 8 == 0:
            return self._pack_bits(all_fingerprints)
        else:
            return all_fingerprints

    def transform(self, titles, batch_size=10000):
        """Alias for encode for compatibility."""
        return self.encode(titles, batch_size)

    def _pack_bits(self, binary_matrix):
        """Pack binary matrix into uint8 array."""
        n_samples, n_bits = binary_matrix.shape
        n_bytes = n_bits // 8

        packed = np.zeros((n_samples, n_bytes), dtype=np.uint8)

        for i in range(n_bytes):
            byte_bits = binary_matrix[:, i * 8 : (i + 1) * 8]
            for j in range(8):
                packed[:, i] |= byte_bits[:, j] << j

        return packed

    def compute_similarity(self, fingerprints1, fingerprints2):
        """
        Compute Hamming similarity between fingerprints.
        """
        if fingerprints1.dtype == np.uint8 and len(fingerprints1.shape) == 2:
            # Packed format
            xor = np.bitwise_xor(fingerprints1[:, None, :], fingerprints2[None, :, :])
            distances = np.sum(np.unpackbits(xor, axis=-1), axis=-1)
        else:
            # Unpacked format
            distances = np.sum(
                fingerprints1[:, None, :] != fingerprints2[None, :, :], axis=-1
            )

        # Convert distance to similarity
        max_distance = self.n_bits
        similarities = 1.0 - (distances / max_distance)

        return similarities
