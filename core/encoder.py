"""
Binary Semantic Encoder with Golden Ratio Sampling
=================================================

Transforms TF-IDF vectors into binary fingerprints using SVD and phase collapse.
Implements golden ratio sampling for optimal pattern capture.
"""

import time
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
from tqdm import tqdm
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class GoldenRatioEncoder:
    """
    Encodes text into binary fingerprints using quantum-inspired phase collapse.
    Based on quantum consciousness principles for optimal pattern capture.
    """

    def __init__(self, n_bits=128, max_features=10000, device="cpu"):
        self.n_bits = n_bits
        self.max_features = max_features
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.device = device

        # Components to be learned
        self.vectorizer = None
        self.projection = None
        self.singular_values = None
        self.sample_indices = None
        self.training_stats = {}

        logger.info("Initialized GoldenRatioEncoder")
        logger.info(f"  n_bits: {n_bits}")
        logger.info(f"  max_features: {max_features}")
        logger.info(f"  golden_ratio: {self.golden_ratio:.6f}")

    def _should_use_randomized_svd(self, n_samples):
        """
        Determine if we should use randomized SVD based on dataset size.

        Args:
            n_samples: Number of samples

        Returns:
            bool: True if randomized SVD should be used
        """
        # Never use randomized SVD - always use golden ratio sampling
        # The original TEJAS approach is to subsample then do full SVD
        return False

    def _golden_ratio_sample(self, n_total, target_memory_gb=2):
        """
        Sample using golden ratio until it fits in memory.

        Args:
            n_total: Total number of items
            target_memory_gb: Target memory usage (default 2GB for reasonable SVD)

        Returns:
            sample_indices: Indices to sample
        """
        # Calculate how many samples we can fit
        bytes_per_element = 4  # float32
        elements_per_sample = self.max_features
        bytes_per_sample = bytes_per_element * elements_per_sample

        max_samples = int(target_memory_gb * 1e9 / bytes_per_sample)

        # Apply golden ratio reduction until it fits
        sample_size = n_total
        reduction_level = 0

        while sample_size > max_samples:
            sample_size = int(sample_size / self.golden_ratio)
            reduction_level += 1

        logger.info("Golden ratio sampling:")
        logger.info(f"  Original: {n_total:,} samples")
        logger.info(f"  Reduced: {sample_size:,} samples")
        logger.info(f"  Reduction levels: {reduction_level}")
        logger.info(f"  Coverage: {sample_size / n_total * 100:.1f}%")

        # Create indices with logarithmic distribution
        if sample_size < n_total:
            indices = np.unique(
                np.logspace(0, np.log10(n_total - 1), sample_size).astype(int)
            )
        else:
            indices = np.arange(n_total)

        logger.info(f"  Selected {len(indices):,} unique indices")
        return indices

    def train(self, titles, memory_limit_gb=2, batch_size=10000):
        """
        Train encoder using golden ratio sampling.
        This is the method called by the training script.

        Args:
            titles: List of all titles
            memory_limit_gb: Memory limit for computation (default 2GB)
            batch_size: Not used in fit, but kept for compatibility
        """
        self.fit(titles, memory_limit_gb)

    def fit(self, titles, memory_limit_gb=2):
        """
        Fit encoder using golden ratio sampling.

        Args:
            titles: List of all titles
            memory_limit_gb: Memory limit for computation (default 2GB)
        """
        start_time = time.time()
        logger.info(f"Training encoder on {len(titles):,} titles...")

        # Step 1: Fit vectorizer on ALL titles (learns vocabulary)
        logger.info("Step 1: Learning vocabulary from all titles...")
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
        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Step 3: Transform sample and compute SVD
        logger.info(f"Step 3: Transforming {len(sample_titles):,} sampled titles...")
        t0 = time.time()

        X_sample = self.vectorizer.transform(sample_titles)
        X_dense = X_sample.toarray()
        logger.info(f"  Matrix shape: {X_dense.shape}")
        logger.info(f"  Matrix memory: {X_dense.nbytes / 1e9:.2f} GB")

        # Convert to PyTorch for SVD
        X_tensor = torch.from_numpy(X_dense).float()
        if self.device != "cpu" and torch.cuda.is_available():
            X_tensor = X_tensor.to(self.device)

        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Step 4: SVD with energy analysis
        logger.info("Step 4: Computing SVD with energy analysis...")
        t0 = time.time()

        # Check if we should use randomized SVD for large datasets
        use_rsvd = self._should_use_randomized_svd(len(sample_titles))

        if use_rsvd:
            logger.info(f"  Using Randomized SVD for {len(sample_titles):,} samples")
            # Import RandomizedSVD
            try:
                from core.randomized_svd import RandomizedSVD
            except ImportError:
                logger.warning("RandomizedSVD not available, falling back to full SVD")
                use_rsvd = False

        if use_rsvd:
            # Use RandomizedSVD for large datasets
            # Determine n_components adaptively based on data size
            n_components_rsvd = min(self.n_bits, min(X_dense.shape) // 4, 512)

            rsvd = RandomizedSVD(
                n_components=n_components_rsvd,
                n_oversamples=20,  # Good balance of speed/accuracy
                n_iter=5,  # Sufficient for most cases
                random_state=42,
                dtype=np.float32,
                backend="numpy",  # Use numpy for stability
                device=self.device,
            )

            # Compute randomized SVD
            U_np, S_np, Vt_np = rsvd.fit_transform(X_dense, return_components=True)

            # Convert back to PyTorch tensors
            U = torch.from_numpy(U_np).float()
            S = torch.from_numpy(S_np).float()
            Vh = torch.from_numpy(Vt_np).float()

            if self.device != "cpu" and torch.cuda.is_available():
                U = U.to(self.device)
                S = S.to(self.device)
                Vh = Vh.to(self.device)
        else:
            # Use full SVD for smaller datasets
            # Add small regularization for numerical stability
            reg_strength = 1e-6
            X_regularized = (
                X_tensor
                + reg_strength
                * torch.eye(
                    X_tensor.shape[0],
                    X_tensor.shape[1],
                    device=X_tensor.device,
                    dtype=X_tensor.dtype,
                )[: X_tensor.shape[0], : X_tensor.shape[1]]
            )

            U, S, Vh = torch.linalg.svd(X_regularized, full_matrices=False)

        # Filter out very small singular values for stability
        min_singular_value = 1e-10
        valid_mask = S > min_singular_value
        S = S[valid_mask]
        Vh = Vh[valid_mask]

        # Energy analysis
        energy = S**2
        total_energy = energy.sum()
        energy_threshold = energy.mean()

        # Find components above mean energy
        n_components = torch.sum(energy > energy_threshold).item()

        # Constrain to reasonable range
        n_components = np.clip(n_components, 64, min(self.n_bits, len(S)))

        # Calculate explained variance
        explained_variance = energy[:n_components].sum() / total_energy

        logger.info(f"  Total singular values: {len(S)}")
        logger.info(f"  Energy threshold: {energy_threshold:.2f}")
        logger.info(f"  Selected components: {n_components}")
        logger.info(f"  Explained variance: {explained_variance:.3f}")
        logger.info(f"  Top 5 singular values: {S[:5].cpu().numpy()}")
        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Step 5: Store projection matrix
        self.projection = Vh[:n_components].T.cpu().numpy()
        self.singular_values = S[:n_components].cpu().numpy()
        self.n_components = n_components

        # Step 6: Validate coherence
        logger.info("Step 5: Validating projection coherence...")
        t0 = time.time()

        coherence = self._validate_coherence()
        logger.info(f"  Projection coherence: {coherence:.4f}")
        logger.info(f"  Time: {time.time() - t0:.2f}s")

        # Store training statistics
        self.training_stats = {
            "n_titles": len(titles),
            "n_samples": len(sample_titles),
            "sample_ratio": len(sample_titles) / len(titles),
            "n_features": vocab_size,
            "n_components": n_components,
            "explained_variance": float(explained_variance),
            "coherence": float(coherence),
            "training_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Training complete in {self.training_stats['training_time']:.2f}s")

    def encode(self, titles, batch_size=10000, show_progress=True):
        """
        Transform titles to binary fingerprints.
        This method is called by the training script.

        Args:
            titles: Titles to encode
            batch_size: Processing batch size
            show_progress: Show progress bar

        Returns:
            Binary fingerprints tensor (n_titles, n_bits)
        """
        return self.transform(titles, batch_size, show_progress)

    def transform(self, titles, batch_size=10000, show_progress=False):
        """
        Transform titles to binary fingerprints.

        Args:
            titles: Titles to encode
            batch_size: Processing batch size
            show_progress: Show progress bar

        Returns:
            Binary fingerprints as torch tensor (n_titles, n_bits)
        """
        if self.vectorizer is None:
            raise ValueError("Encoder must be fitted first")

        n_titles = len(titles)
        fingerprints = np.zeros((n_titles, self.n_bits), dtype=np.uint8)

        # Process in batches
        iterator = range(0, n_titles, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding titles")

        for i in iterator:
            batch_end = min(i + batch_size, n_titles)
            batch = titles[i:batch_end]

            # Transform to TF-IDF
            X_batch = self.vectorizer.transform(batch)
            # Handle both sparse and dense matrices
            if hasattr(X_batch, "toarray"):
                X_dense = X_batch.toarray()
            else:
                X_dense = X_batch  # Already dense

            # Project using learned components
            X_projected = X_dense @ self.projection

            # Check for NaN/inf and handle gracefully
            if np.any(np.isnan(X_projected)) or np.any(np.isinf(X_projected)):
                # Fallback to random projection if numerical issues
                logger.warning(
                    "Numerical instability detected in projection, using fallback"
                )
                X_projected = (
                    np.random.randn(X_projected.shape[0], X_projected.shape[1]) * 0.1
                )

            # Normalize to unit sphere
            norms = np.linalg.norm(X_projected, axis=1, keepdims=True)
            X_normalized = X_projected / (norms + 1e-8)

            # Extract binary phases
            binary = (X_normalized > 0).astype(np.uint8)

            # Store (handling case where n_components < n_bits)
            actual_bits = min(binary.shape[1], self.n_bits)
            fingerprints[i:batch_end, :actual_bits] = binary[:, :actual_bits]

        # Convert to PyTorch tensor for compatibility
        return torch.from_numpy(fingerprints)

    def encode_single(self, title):
        """Encode a single title."""
        return self.encode([title], show_progress=False)[0]

    def _validate_coherence(self):
        """Measure coherence of projection using quantum principle."""
        # Create random test vectors
        test_vectors = np.random.randn(100, self.projection.shape[0])

        # Project with numerical stability check
        try:
            projected = test_vectors @ self.projection
            # Check for NaN/inf
            if np.any(np.isnan(projected)) or np.any(np.isinf(projected)):
                logger.warning("Numerical instability in coherence validation")
                return 0.0  # Return low coherence for unstable projection
        except Exception as e:
            logger.warning(f"Error in coherence validation: {e}")
            return 0.0

        # Convert to complex for phase analysis
        projected_complex = projected.astype(np.complex64)

        # Measure phase coherence
        phases = np.angle(np.sum(projected_complex, axis=1))
        phase_factors = np.exp(1j * phases)
        coherence = np.abs(np.mean(phase_factors))

        return coherence

    def save(self, save_dir):
        """Save encoder to disk."""
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving encoder to {save_path}")

            # Save vectorizer vocabulary and IDF as numpy arrays
            if self.vectorizer is None:
                raise ValueError("Cannot save encoder: vectorizer is None")

            vocab_items = sorted(
                self.vectorizer.vocabulary_.items(), key=lambda x: x[1]
            )
            vocab_array = np.array([item[0] for item in vocab_items], dtype=object)

            vocab_path = save_path / "vocabulary.npy"
            logger.info(f"Saving vocabulary to {vocab_path}")
            np.save(vocab_path, vocab_array)

            idf_path = save_path / "idf_weights.npy"
            logger.info(f"Saving IDF weights to {idf_path}")
            np.save(idf_path, self.vectorizer.idf_)

            # Save projection and parameters
            if self.projection is None:
                raise ValueError("Cannot save encoder: projection matrix is None")

            projection_path = save_path / "projection.npy"
            logger.info(f"Saving projection matrix to {projection_path}")
            np.save(projection_path, self.projection)

            if self.singular_values is None:
                raise ValueError("Cannot save encoder: singular values are None")

            singular_path = save_path / "singular_values.npy"
            logger.info(f"Saving singular values to {singular_path}")
            np.save(singular_path, self.singular_values)

            # Save configuration
            config = {
                "n_bits": int(self.n_bits),
                "n_components": int(self.n_components),
                "max_features": int(self.max_features),
                "golden_ratio": float(self.golden_ratio),
                "sample_indices": self.sample_indices.tolist()
                if self.sample_indices is not None
                else None,
                "training_stats": {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in self.training_stats.items()
                },
            }

            config_path = save_path / "config.json"
            logger.info(f"Saving config to {config_path}")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # Verify all files were created
            expected_files = [
                "vocabulary.npy",
                "idf_weights.npy",
                "projection.npy",
                "singular_values.npy",
                "config.json",
            ]

            for file in expected_files:
                file_path = save_path / file
                if not file_path.exists():
                    raise FileNotFoundError(
                        f"Failed to save {file} - file does not exist after save"
                    )
                logger.info(f"  Verified: {file} ({file_path.stat().st_size} bytes)")

            logger.info(f"Encoder saved successfully to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save encoder: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise

    def load(self, save_dir):
        """Load encoder from disk."""
        save_path = Path(save_dir)

        # Load configuration
        with open(save_path / "config.json", "r") as f:
            config = json.load(f)

        self.n_bits = config["n_bits"]
        self.n_components = config["n_components"]
        self.max_features = config["max_features"]
        self.golden_ratio = config["golden_ratio"]
        self.training_stats = config.get("training_stats", {})

        # Load projection and singular values
        self.projection = np.load(save_path / "projection.npy")
        self.singular_values = np.load(save_path / "singular_values.npy")

        # Recreate vectorizer
        vocab_array = np.load(save_path / "vocabulary.npy", allow_pickle=True)
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=self.max_features,
            lowercase=True,
            dtype=np.float32,
        )

        # Restore vocabulary
        self.vectorizer.vocabulary_ = {
            word: idx for idx, word in enumerate(vocab_array)
        }
        self.vectorizer.idf_ = np.load(save_path / "idf_weights.npy")

        logger.info(f"Encoder loaded from {save_path}")
