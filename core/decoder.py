"""
Binary Fingerprint Decoder
=========================

Reconstructs semantic meaning from binary fingerprints.
Provides interpretation and analysis of binary patterns.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SemanticDecoder:
    """
    Decoder for reconstructing semantic information from binary fingerprints.

    Capabilities:
    - Pattern explanation and interpretation
    - Semantic interpolation between fingerprints
    - Channel analysis and statistics
    - Similarity explanation
    """

    def __init__(
        self,
        projection_matrix: Optional[np.ndarray] = None,
        vocabulary: Optional[Dict[str, int]] = None,
        singular_values: Optional[np.ndarray] = None,
        n_bits: int = 128,
        n_components: Optional[int] = None,
    ):
        """
        Initialize the decoder.

        Args:
            projection_matrix: Projection matrix from encoder (numpy array)
            vocabulary: N-gram vocabulary mapping
            singular_values: Singular values from SVD (numpy array)
            n_bits: Number of bits in fingerprints
            n_components: Number of components used in encoding
        """
        self.projection_matrix = projection_matrix
        self.vocabulary = vocabulary
        self.singular_values = singular_values
        self.n_bits = n_bits
        self.n_components = n_components if n_components else n_bits

        # Reverse vocabulary for decoding
        if vocabulary:
            self.reverse_vocabulary = {v: k for k, v in vocabulary.items()}
        else:
            self.reverse_vocabulary = None

        logger.info("Initialized SemanticDecoder")
        logger.info(f"  Vocabulary size: {len(vocabulary) if vocabulary else 0}")
        logger.info(f"  Binary dimensions: {n_bits}")
        logger.info(f"  Components: {self.n_components}")

    def decode_patterns(
        self, fingerprint: Union[np.ndarray, torch.Tensor], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extract the most likely n-gram patterns from a fingerprint.

        This is an approximation - true inverse is not possible due to:
        1. Binary quantization loses information
        2. Dimensionality reduction loses information

        Args:
            fingerprint: Binary fingerprint
            top_k: Number of top patterns to return

        Returns:
            List of (n-gram, score) tuples
        """
        if self.projection_matrix is None or self.vocabulary is None:
            raise ValueError("Decoder requires projection matrix and vocabulary")

        # Convert to numpy if torch
        if isinstance(fingerprint, torch.Tensor):
            fingerprint = fingerprint.cpu().numpy()

        # Convert binary to continuous (-1, 1)
        continuous = fingerprint.astype(np.float32) * 2 - 1

        # Use only the components that were used in encoding
        if len(continuous) > self.n_components:
            continuous = continuous[: self.n_components]

        # Approximate inverse projection
        # Note: This is not a true inverse, just an approximation
        try:
            # Use pseudo-inverse of projection matrix
            projection_pinv = np.linalg.pinv(self.projection_matrix.T)
            reconstructed = continuous @ projection_pinv

            # Get top features by magnitude
            feature_scores = np.abs(reconstructed)
            top_indices = np.argsort(feature_scores)[-top_k:][::-1]

            # Get n-grams
            patterns = []
            for idx in top_indices:
                if idx < len(self.reverse_vocabulary):
                    ngram = self.reverse_vocabulary.get(idx, f"<unknown-{idx}>")
                    score = feature_scores[idx]
                    patterns.append((ngram, float(score)))

            return patterns

        except Exception as e:
            logger.warning(f"Pattern decoding failed: {e}")
            return [("<decoding-failed>", 0.0)]

    def explain_similarity(
        self, fp1: Union[np.ndarray, torch.Tensor], fp2: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Union[float, int]]:
        """
        Explain why two fingerprints are similar.

        Args:
            fp1: First fingerprint
            fp2: Second fingerprint

        Returns:
            Explanation of shared patterns
        """
        # Convert to torch for efficient operations
        if isinstance(fp1, np.ndarray):
            fp1 = torch.from_numpy(fp1)
        if isinstance(fp2, np.ndarray):
            fp2 = torch.from_numpy(fp2)

        # Ensure same device
        if fp1.device != fp2.device:
            fp2 = fp2.to(fp1.device)

        # Find shared patterns using torch operations
        shared_active = (fp1 == 1) & (fp2 == 1)
        shared_inactive = (fp1 == 0) & (fp2 == 0)
        xor_result = fp1 ^ fp2

        # Calculate statistics
        explanation = {
            "shared_active_channels": int(shared_active.sum().item()),
            "shared_inactive_channels": int(shared_inactive.sum().item()),
            "total_shared": int((fp1 == fp2).sum().item()),
            "similarity": float((fp1 == fp2).sum().item() / len(fp1)),
            "hamming_distance": int(xor_result.sum().item()),
        }

        return explanation

    def interpolate(
        self,
        fp1: Union[np.ndarray, torch.Tensor],
        fp2: Union[np.ndarray, torch.Tensor],
        steps: int = 5,
    ) -> List[torch.Tensor]:
        """
        Create interpolated fingerprints between two endpoints.

        Args:
            fp1: Start fingerprint
            fp2: End fingerprint
            steps: Number of interpolation steps

        Returns:
            List of interpolated fingerprints (as torch tensors)
        """
        # Convert to torch
        if isinstance(fp1, np.ndarray):
            fp1 = torch.from_numpy(fp1)
        if isinstance(fp2, np.ndarray):
            fp2 = torch.from_numpy(fp2)

        # Find differing positions
        diff_mask = fp1 != fp2
        diff_positions = torch.where(diff_mask)[0]
        n_diffs = len(diff_positions)

        # Create interpolated fingerprints
        interpolated = []

        for i in range(steps + 2):  # Include endpoints
            # Calculate how many bits to flip
            flip_ratio = i / (steps + 1)
            n_flips = int(n_diffs * flip_ratio)

            # Create interpolated fingerprint
            fp_interp = fp1.clone()

            # Flip the first n_flips differing positions
            if n_flips > 0:
                positions_to_flip = diff_positions[:n_flips]
                fp_interp[positions_to_flip] = fp2[positions_to_flip]

            interpolated.append(fp_interp)

        return interpolated

    def analyze_channels(
        self, fingerprints: Union[np.ndarray, torch.Tensor]
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze the role of each binary channel.

        Args:
            fingerprints: Multiple fingerprints (n_samples, n_bits)

        Returns:
            Channel analysis
        """
        # Convert to torch for efficient computation
        if isinstance(fingerprints, np.ndarray):
            fingerprints = torch.from_numpy(fingerprints)

        n_samples, n_bits = fingerprints.shape

        channel_analysis = {}

        # Compute all statistics at once using torch
        activations = fingerprints.float()
        channel_means = activations.mean(dim=0)
        channel_vars = activations.var(dim=0)

        for channel in range(n_bits):
            mean_val = channel_means[channel].item()
            var_val = channel_vars[channel].item()

            channel_analysis[channel] = {
                "activation_rate": mean_val,
                "variance": var_val,
                "entropy": self._calculate_entropy(mean_val),
                "is_balanced": bool(0.4 <= mean_val <= 0.6),
            }

        return channel_analysis

    def _calculate_entropy(self, p1: float) -> float:
        """Calculate Shannon entropy for binary channel."""
        p0 = 1 - p1

        if p1 == 0 or p1 == 1:
            return 0.0

        return -p1 * np.log2(p1) - p0 * np.log2(p0)

    def find_pattern_fingerprints(
        self,
        pattern: str,
        fingerprints: torch.Tensor,
        titles: List[str],
        threshold: float = 0.8,
    ) -> List[Tuple[int, str, float]]:
        """
        Find fingerprints that likely contain a specific pattern.

        Args:
            pattern: Pattern to search for
            fingerprints: All fingerprints
            titles: Corresponding titles
            threshold: Similarity threshold

        Returns:
            List of (index, title, similarity) for likely matches
        """
        # This would require encoding the pattern first
        # For now, return titles that actually contain the pattern
        matches = []
        pattern_lower = pattern.lower()

        for idx, title in enumerate(titles):
            if pattern_lower in title.lower():
                matches.append((idx, title, 1.0))

        return matches

    def save(self, save_dir: Union[str, Path]):
        """Save decoder state."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save arrays
        if self.projection_matrix is not None:
            np.save(save_path / "decoder_projection.npy", self.projection_matrix)

        if self.singular_values is not None:
            np.save(save_path / "decoder_singular_values.npy", self.singular_values)

        # Save vocabulary
        if self.vocabulary is not None:
            vocab_items = sorted(self.vocabulary.items(), key=lambda x: x[1])
            vocab_array = np.array([item[0] for item in vocab_items], dtype=object)
            np.save(save_path / "decoder_vocabulary.npy", vocab_array)

        # Save config
        config = {
            "n_bits": int(self.n_bits),  # Ensure Python int
            "n_components": int(self.n_components),  # Ensure Python int
            "has_projection": self.projection_matrix is not None,
            "has_vocabulary": self.vocabulary is not None,
            "has_singular_values": self.singular_values is not None,
        }

        with open(save_path / "decoder_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Decoder saved to {save_path}")

    def load(self, save_dir: Union[str, Path]):
        """Load decoder state."""
        save_path = Path(save_dir)

        # Load config
        with open(save_path / "decoder_config.json", "r") as f:
            config = json.load(f)

        self.n_bits = config["n_bits"]
        self.n_components = config["n_components"]

        # Load arrays if they exist
        if config["has_projection"]:
            self.projection_matrix = np.load(save_path / "decoder_projection.npy")

        if config["has_singular_values"]:
            self.singular_values = np.load(save_path / "decoder_singular_values.npy")

        if config["has_vocabulary"]:
            vocab_array = np.load(
                save_path / "decoder_vocabulary.npy", allow_pickle=True
            )
            self.vocabulary = {word: idx for idx, word in enumerate(vocab_array)}
            self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

        logger.info(f"Decoder loaded from {save_path}")

    @classmethod
    def from_encoder(cls, encoder_dir: Union[str, Path]) -> "SemanticDecoder":
        """
        Create decoder from a trained encoder.

        Args:
            encoder_dir: Directory containing saved encoder

        Returns:
            Configured decoder
        """
        encoder_path = Path(encoder_dir)

        # Load encoder config
        with open(encoder_path / "config.json", "r") as f:
            encoder_config = json.load(f)

        # Load encoder components
        projection = np.load(encoder_path / "projection.npy")
        singular_values = np.load(encoder_path / "singular_values.npy")
        vocab_array = np.load(encoder_path / "vocabulary.npy", allow_pickle=True)

        # Create vocabulary dict
        vocabulary = {word: idx for idx, word in enumerate(vocab_array)}

        # Create decoder
        decoder = cls(
            projection_matrix=projection,
            vocabulary=vocabulary,
            singular_values=singular_values,
            n_bits=encoder_config["n_bits"],
            n_components=encoder_config["n_components"],
        )

        logger.info(f"Created decoder from encoder at {encoder_path}")
        return decoder


def demonstrate_decoder():
    """
    Demonstrate decoder capabilities.
    """
    # Create sample fingerprints as torch tensors
    n_samples = 100
    n_bits = 128
    fingerprints = torch.randint(0, 2, (n_samples, n_bits), dtype=torch.uint8)

    # Create decoder
    decoder = SemanticDecoder(n_bits=n_bits)

    print("\nSemantic Decoder Demo:")
    print("=" * 50)

    # Explain similarity
    fp1 = fingerprints[0]
    fp2 = fingerprints[1]

    explanation = decoder.explain_similarity(fp1, fp2)
    print("\nSimilarity explanation between fingerprints 0 and 1:")
    for key, value in explanation.items():
        print(f"  {key}: {value}")

    # Interpolation
    interpolated = decoder.interpolate(fp1, fp2, steps=3)
    print(f"\nInterpolation path ({len(interpolated)} steps):")
    for i, fp in enumerate(interpolated):
        dist_to_start = (fp != fp1).sum().item()
        dist_to_end = (fp != fp2).sum().item()
        print(f"  Step {i}: distance to start={dist_to_start}, to end={dist_to_end}")

    # Channel analysis
    channel_stats = decoder.analyze_channels(fingerprints)

    balanced_channels = sum(1 for ch in channel_stats.values() if ch["is_balanced"])
    print("\nChannel analysis:")
    print(f"  Total channels: {n_bits}")
    print(f"  Balanced channels: {balanced_channels}")
    print(
        f"  Average entropy: {np.mean([ch['entropy'] for ch in channel_stats.values()]):.3f}"
    )


if __name__ == "__main__":
    demonstrate_decoder()
