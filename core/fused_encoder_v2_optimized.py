"""
Optimized Fused Encoder V2 with performance fixes.

Key optimizations:
1. Single-query fast path with cached vectorizer state
2. Optional unpacked format for better performance without Numba
3. Compatible output format with original encoder
"""

import numpy as np
import torch
from typing import Union, List, Optional, Dict
import logging

from .fused_encoder_v2 import FusedPipelineEncoder
from .reranker import TEJASReranker, RerankerConfig

logger = logging.getLogger(__name__)


class OptimizedFusedEncoder(FusedPipelineEncoder):
    """
    Optimized version of FusedPipelineEncoder with:
    - Fast single-query encoding path
    - Optional unpacked format for better search performance
    - Torch tensor compatibility
    """

    def __init__(
        self,
        *args,
        use_packing: bool = None,
        use_reranker: bool = False,
        reranker_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize optimized encoder.

        Args:
            use_packing: Whether to pack bits. None = auto-detect based on Numba availability
            use_reranker: Whether to enable semantic reranking
            reranker_config: Configuration for reranker
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)

        # Auto-detect packing based on Numba availability
        if use_packing is None:
            try:
                import numba

                self.use_packing = True
                logger.info("Numba available - using packed format for SIMD")
            except ImportError:
                self.use_packing = False
                logger.info(
                    "Numba not available - using unpacked format for better performance"
                )
        else:
            self.use_packing = use_packing

        # Cache for single-query optimization
        self._query_cache = {}
        self._cache_size = 100  # LRU cache size

        # Initialize reranker if requested
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            try:
                reranker_cfg = RerankerConfig(**(reranker_config or {}))
                self.reranker = TEJASReranker(reranker_cfg)
                logger.info(f"Initialized reranker: {reranker_cfg.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.use_reranker = False

        # Storage for document texts (for reranking)
        self.document_texts = None

    def transform_single(
        self, query: str, return_torch: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Optimized single-query transformation with caching.

        Args:
            query: Single query text
            return_torch: Whether to return torch.Tensor for compatibility

        Returns:
            Binary fingerprint (unpacked by default for single queries)
        """
        # Check cache first
        if query in self._query_cache:
            result = self._query_cache[query].copy()
            if return_torch:
                return torch.from_numpy(result)
            return result

        # Get TF-IDF stage
        tfidf_stage = self.pipeline.stages[0]

        # Vectorize single query efficiently
        query_tfidf = tfidf_stage.vectorizer.transform([query])

        # Apply rest of pipeline stages
        X_current = query_tfidf
        for stage in self.pipeline.stages[1:]:
            # Control packing based on self.use_packing
            if hasattr(stage, "pack_bits"):
                original_pack = stage.pack_bits
                stage.pack_bits = (
                    self.use_packing
                )  # Respect the encoder's packing setting
                X_current = stage.transform(X_current)
                stage.pack_bits = original_pack
            else:
                X_current = stage.transform(X_current)

        # Get single result
        result = X_current[0] if len(X_current.shape) > 1 else X_current

        # Cache result (manage cache size)
        if len(self._query_cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO, could be improved to LRU)
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[query] = result.copy()

        if return_torch:
            return torch.from_numpy(result)
        return result

    def transform(
        self,
        titles: Union[str, List[str]],
        return_packed: Optional[bool] = None,
        return_torch: bool = False,
        memory_limit_gb: float = 50,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Enhanced transform with single-query optimization.

        Args:
            titles: Text or list of texts
            return_packed: Whether to pack bits (None = use default)
            return_torch: Whether to return torch.Tensor
            memory_limit_gb: Memory limit

        Returns:
            Binary codes (format depends on options)
        """
        # Determine packing preference
        if return_packed is None:
            return_packed = self.use_packing

        # Handle single query specially - but respect packing preference
        if isinstance(titles, str):
            titles = [titles]
            is_single = True
        elif len(titles) == 1:
            is_single = True
        else:
            is_single = False

        # Use parent implementation which respects return_packed
        result = super().transform(
            titles, return_packed=return_packed, memory_limit_gb=memory_limit_gb
        )

        if return_torch:
            # Unpack if needed for torch compatibility
            if return_packed and result.dtype != np.uint8:
                from .hamming_simd import unpack_bits_uint32

                result = unpack_bits_uint32(result, self.n_bits)
            return torch.from_numpy(result)

        return result

    def encode(
        self, titles: Union[str, List[str]], **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compatibility method matching original encoder interface.

        Returns numpy array by default for better compatibility with benchmarks.
        Use return_torch=True in kwargs to get torch.Tensor.
        """
        return_torch = kwargs.pop("return_torch", False)
        return self.transform(titles, return_torch=return_torch, **kwargs)

    def encode_single(self, title: str) -> torch.Tensor:
        """
        Encode single title - compatible with original encoder.
        """
        return self.transform_single(title, return_torch=True)

    def search_optimized(
        self,
        query: Union[str, List[str]],
        database: np.ndarray,
        k: int = 10,
        use_fast_path: bool = True,
    ) -> tuple:
        """
        Optimized search with format awareness.

        Args:
            query: Query text(s)
            database: Database fingerprints
            k: Number of neighbors
            use_fast_path: Use optimized path for unpacked data

        Returns:
            indices, distances
        """
        # Determine if database is packed based on dtype
        database_is_packed = database.dtype == np.uint32

        # Encode query to match database format
        if isinstance(query, str):
            query_fp = self.transform_single(query, return_torch=False)
            # Pack if needed to match database
            if database_is_packed and query_fp.dtype == np.uint8:
                from .hamming_simd import pack_bits_uint32

                query_fp = pack_bits_uint32(query_fp.reshape(1, -1))[0]
        else:
            # Transform with same packing as database
            query_fp = self.transform(
                query, return_packed=database_is_packed, return_torch=False
            )

        # Check database format
        if database.dtype == np.uint8:
            # Unpacked data - use scipy cdist
            from scipy.spatial.distance import cdist

            if len(query_fp.shape) == 1:
                query_fp = query_fp.reshape(1, -1)

            distances = cdist(query_fp, database, metric="hamming")
            indices = np.argsort(distances, axis=1)[:, :k]
            distances = np.sort(distances, axis=1)[:, :k]

            if indices.shape[0] == 1:
                return indices[0], distances[0]
            return indices, distances
        else:
            # Packed data (uint32) - use SIMD optimized search
            from .hamming_simd import pack_bits_uint32

            # Import best available backend
            try:
                from .hamming_simd import (
                    hamming_distance_numba_fixed as hamming_search_func,
                )

                logger.debug("Using Numba-optimized Hamming distance")
            except ImportError:
                from .hamming_simd import (
                    hamming_distance_numpy_packed as hamming_search_func,
                )

                logger.debug("Using NumPy Hamming distance (Numba not available)")

            # Pack query fingerprint if needed
            if query_fp.dtype == np.uint8:
                query_fp_packed = pack_bits_uint32(
                    query_fp.reshape(1, -1) if len(query_fp.shape) == 1 else query_fp
                )
            else:
                query_fp_packed = query_fp

            # Compute distances using SIMD - need to handle batch properly
            if len(query_fp_packed.shape) == 1:
                query_fp_packed = query_fp_packed.reshape(1, -1)

            # Compute distances for all queries against database
            n_queries = query_fp_packed.shape[0]

            # For single query, use the optimized function directly
            if n_queries == 1:
                indices, dists = hamming_search_func(
                    database,
                    query_fp_packed[0],
                    k,  # Pass k, not n_bits!
                )
                return indices, dists / self.n_bits  # Normalize distances

            # For batch queries, compute all distances
            n_docs = database.shape[0]
            all_indices = []
            all_distances = []

            for i in range(n_queries):
                idx, dist = hamming_search_func(database, query_fp_packed[i], k)
                all_indices.append(idx)
                all_distances.append(dist / self.n_bits)  # Normalize

            return np.array(all_indices), np.array(all_distances)

    def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")

    def set_document_texts(self, texts: List[str]):
        """
        Store document texts for reranking.

        Args:
            texts: List of document texts corresponding to database entries
        """
        self.document_texts = texts
        logger.info(f"Stored {len(texts)} document texts for reranking")

    def search_with_reranking(
        self,
        query: Union[str, List[str]],
        database: np.ndarray,
        k: int = 10,
        rerank_top_k: int = 30,
        titles: Optional[List[str]] = None,
    ) -> Dict:
        """
        Search with optional semantic reranking.

        Args:
            query: Query text(s)
            database: Database fingerprints
            k: Final number of results
            rerank_top_k: Number of candidates to rerank
            titles: Optional document titles for results

        Returns:
            Dictionary with indices, distances, scores, and reranked results
        """
        # Get initial results using Hamming distance
        indices, distances = self.search_optimized(query, database, k=rerank_top_k)

        results = {
            "indices": indices[:k],
            "distances": distances[:k],
            "hamming_scores": 1.0
            - (distances[:k] / self.n_bits),  # Convert to similarity
        }

        # Apply reranking if available
        if self.use_reranker and self.reranker is not None:
            if isinstance(query, list):
                query = query[0]  # Handle single query for now

            # Prepare candidates
            if titles is None and self.document_texts is not None:
                titles = [self.document_texts[i] for i in indices[:rerank_top_k]]
            elif titles is not None:
                titles = [titles[i] for i in indices[:rerank_top_k]]

            if titles:
                # Create candidate tuples for reranker
                candidates = [
                    (
                        titles[i],
                        float(1.0 - distances[i] / self.n_bits),
                        int(distances[i]),
                    )
                    for i in range(min(rerank_top_k, len(indices)))
                ]

                # Rerank
                reranked = self.reranker.rerank(query, candidates, top_k=k)

                # Update results with reranked order
                reranked_indices = []
                reranked_scores = []
                for title, score, _ in reranked:
                    # Find original index
                    for i, idx in enumerate(indices[:rerank_top_k]):
                        if titles[i] == title:
                            reranked_indices.append(idx)
                            reranked_scores.append(score)
                            break

                results["reranked_indices"] = np.array(reranked_indices)
                results["reranked_scores"] = np.array(reranked_scores)
                results["reranker_stats"] = self.reranker.get_stats()

        return results


def create_optimized_encoder(
    n_bits: int = 128,
    max_features: int = 10000,
    use_itq: bool = True,
    use_packing: bool = None,
    **kwargs,
) -> OptimizedFusedEncoder:
    """
    Create optimized fused encoder with sensible defaults.

    Args:
        n_bits: Number of bits
        max_features: Max TF-IDF features
        use_itq: Whether to use ITQ
        use_packing: Whether to pack bits (None = auto-detect)
        **kwargs: Additional arguments

    Returns:
        Optimized encoder instance
    """
    return OptimizedFusedEncoder(
        n_bits=n_bits,
        max_features=max_features,
        use_itq=use_itq,
        use_packing=use_packing,
        **kwargs,
    )
