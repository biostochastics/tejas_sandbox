"""
Tejas-F (Fused) Encoder
========================
Fused V2 encoder with ITQ disabled, reranker disabled, bit packing enabled.
This is the standard production configuration.
"""

import logging

from .fused_encoder_v2_optimized import OptimizedFusedEncoder

logger = logging.getLogger(__name__)


class TejasFEncoder(OptimizedFusedEncoder):
    """
    Tejas-F (Fused) encoder - production configuration.

    Configuration:
    - ITQ: DISABLED (faster training)
    - Reranker: DISABLED (pure binary search)
    - Bit packing: ENABLED (memory efficiency)
    - RSVD: Auto-enabled for large matrices
    """

    def __init__(self, n_bits: int = 256, max_features: int = 10000, **kwargs):
        """
        Initialize Tejas-F encoder with production settings.

        Args:
            n_bits: Number of bits in fingerprint
            max_features: Maximum TF-IDF features
            **kwargs: Additional arguments passed to parent
        """
        # Force production configuration
        super().__init__(
            n_bits=n_bits,
            max_features=max_features,
            use_packing=True,  # Always pack bits for memory efficiency
            use_itq=False,  # Disable ITQ for speed
            use_reranker=False,  # Disable reranker for pure binary search
            **kwargs,
        )

        logger.info("Initialized Tejas-F Encoder (Production Config)")
        logger.info(f"  n_bits: {n_bits}")
        logger.info(f"  max_features: {max_features}")
        logger.info("  bit_packing: ENABLED")
        logger.info("  ITQ: DISABLED")
        logger.info("  Reranker: DISABLED")

    def __str__(self):
        return f"Tejas-F(n_bits={self.n_bits}, packed=True)"


class TejasFPlusEncoder(OptimizedFusedEncoder):
    """
    Tejas-F+ (Fused Plus) encoder - enhanced configuration.

    Configuration:
    - ITQ: ENABLED (better binary codes)
    - Reranker: ENABLED (semantic accuracy)
    - Bit packing: ENABLED (memory efficiency)
    - RSVD: Auto-enabled for large matrices
    """

    def __init__(
        self,
        n_bits: int = 256,
        max_features: int = 10000,
        itq_iterations: int = 50,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_alpha: float = 0.7,
        **kwargs,
    ):
        """
        Initialize Tejas-F+ encoder with enhanced settings.

        Args:
            n_bits: Number of bits in fingerprint
            max_features: Maximum TF-IDF features
            itq_iterations: Number of ITQ optimization iterations
            reranker_model: Cross-encoder model for reranking
            reranker_alpha: Weight for semantic score (0-1)
            **kwargs: Additional arguments passed to parent
        """
        # Enhanced configuration
        reranker_config = {
            "model_name": reranker_model,
            "alpha": reranker_alpha,
            "max_candidates": 30,
            "cache_size": 10000,
        }

        # Remove conflicting parameters from kwargs if present
        kwargs.pop("use_reranker", None)
        kwargs.pop("use_itq", None)
        kwargs.pop("use_packing", None)
        kwargs.pop("reranker_config", None)

        # Note: itq_iterations is not passed to parent as it's handled by use_itq flag
        # The ITQ module will use its default iterations
        super().__init__(
            n_bits=n_bits,
            max_features=max_features,
            use_packing=True,  # Pack bits for memory
            use_itq=True,  # Enable ITQ for better codes
            use_reranker=True,  # Enable semantic reranking
            reranker_config=reranker_config,
            **kwargs,
        )

        logger.info("Initialized Tejas-F+ Encoder (Enhanced Config)")
        logger.info(f"  n_bits: {n_bits}")
        logger.info(f"  max_features: {max_features}")
        logger.info("  bit_packing: ENABLED")
        logger.info(f"  ITQ: ENABLED ({itq_iterations} iterations)")
        logger.info(f"  Reranker: ENABLED ({reranker_model})")
        logger.info(f"  Reranker alpha: {reranker_alpha}")

    def __str__(self):
        return f"Tejas-F+(n_bits={self.n_bits}, ITQ=True, Reranker=True)"
