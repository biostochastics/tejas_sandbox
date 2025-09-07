#!/usr/bin/env python3
"""
TEJAS Semantic Reranker
=======================

Lightweight cross-encoder reranking for improved search quality.
Maintains TEJAS's speed advantages while adding semantic understanding.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder

    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    logger.warning("sentence-transformers not available, reranking disabled")


@dataclass
class RerankerConfig:
    """Configuration for TEJAS reranker."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512
    alpha: float = 0.7  # Weight for cross-encoder vs Hamming similarity
    cache_size: int = 10000  # LRU cache for frequent queries
    use_cache: bool = True
    fallback_on_error: bool = True
    max_candidates: int = 30  # Maximum candidates to rerank (for speed)


class TEJASReranker:
    """
    Semantic reranker for TEJAS search results.

    Uses a lightweight cross-encoder to rerank initial Hamming distance results,
    providing much better semantic understanding while maintaining speed.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize the reranker.

        Args:
            config: Reranker configuration
        """
        self.config = config or RerankerConfig()
        self.model = None
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if HAS_CROSS_ENCODER:
            try:
                self.model = CrossEncoder(
                    self.config.model_name,
                    device=self.config.device,
                    max_length=self.config.max_length,
                )
                logger.info(f"Loaded reranker model: {self.config.model_name}")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, int]],
        top_k: int = 10,
        texts: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, int]]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Search query text
            candidates: List of (title, similarity, distance) tuples from Hamming search
            top_k: Number of results to return
            texts: Optional full texts for reranking (if different from titles)

        Returns:
            Reranked list of (title, score, original_distance) tuples
        """
        if not candidates:
            return []

        # Limit candidates for speed
        if len(candidates) > self.config.max_candidates:
            candidates = candidates[: self.config.max_candidates]
            if texts:
                texts = texts[: self.config.max_candidates]

        # If no model available, return original results
        if self.model is None:
            if self.config.fallback_on_error:
                return candidates[:top_k]
            else:
                raise RuntimeError("Reranker model not available")

        # Check cache for this query
        cache_key = self._get_cache_key(query, candidates)
        if self.config.use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key][:top_k]

        self.cache_misses += 1

        try:
            # Prepare pairs for cross-encoder
            if texts:
                pairs = [[query, text] for text in texts]
            else:
                pairs = [[query, cand[0]] for cand in candidates]

            # Get cross-encoder scores
            start_time = time.time()
            ce_scores = self.model.predict(pairs, batch_size=self.config.batch_size)
            inference_time = time.time() - start_time

            # Normalize scores to [0, 1]
            ce_scores = self._normalize_scores(ce_scores)

            # Get original Hamming similarities
            hamming_sims = np.array([cand[1] for cand in candidates])

            # Combine scores (weighted fusion)
            combined_scores = (
                self.config.alpha * ce_scores + (1 - self.config.alpha) * hamming_sims
            )

            # Sort by combined score
            ranked_indices = np.argsort(combined_scores)[::-1]

            # Create reranked results
            reranked = [
                (
                    candidates[i][0],  # title
                    float(combined_scores[i]),  # combined score
                    candidates[i][2],  # original distance
                )
                for i in ranked_indices
            ]

            # Update cache
            if self.config.use_cache:
                self._update_cache(cache_key, reranked)

            # Log performance metrics
            if len(candidates) > 0:
                logger.debug(
                    f"Reranked {len(candidates)} results in {inference_time:.3f}s "
                    f"({len(candidates) / inference_time:.1f} docs/sec)"
                )

            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            if self.config.fallback_on_error:
                return candidates[:top_k]
            else:
                raise

    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Tuple[str, float, int]]],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float, int]]]:
        """
        Rerank multiple queries in batch for efficiency.

        Args:
            queries: List of query texts
            candidates_list: List of candidate lists
            top_k: Number of results per query

        Returns:
            List of reranked results for each query
        """
        if not HAS_CROSS_ENCODER or self.model is None:
            return [cands[:top_k] for cands in candidates_list]

        # Prepare all pairs
        all_pairs = []
        query_boundaries = [0]

        for query, candidates in zip(queries, candidates_list):
            pairs = [[query, cand[0]] for cand in candidates]
            all_pairs.extend(pairs)
            query_boundaries.append(query_boundaries[-1] + len(pairs))

        # Get all scores at once
        if all_pairs:
            all_scores = self.model.predict(
                all_pairs, batch_size=self.config.batch_size
            )
            all_scores = self._normalize_scores(all_scores)
        else:
            all_scores = []

        # Process each query's results
        reranked_results = []
        for i, (query, candidates) in enumerate(zip(queries, candidates_list)):
            start_idx = query_boundaries[i]
            end_idx = query_boundaries[i + 1]

            if start_idx < end_idx:
                ce_scores = all_scores[start_idx:end_idx]
                hamming_sims = np.array([cand[1] for cand in candidates])

                combined_scores = (
                    self.config.alpha * ce_scores
                    + (1 - self.config.alpha) * hamming_sims
                )

                ranked_indices = np.argsort(combined_scores)[::-1][:top_k]

                reranked = [
                    (candidates[j][0], float(combined_scores[j]), candidates[j][2])
                    for j in ranked_indices
                ]
                reranked_results.append(reranked)
            else:
                reranked_results.append(candidates[:top_k])

        return reranked_results

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score = scores.min()
        max_score = scores.max()

        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return np.ones_like(scores) * 0.5

    def _get_cache_key(self, query: str, candidates: List) -> str:
        """Generate cache key for query-candidates pair."""
        # Use query and top candidate titles as key
        candidate_str = "|".join([c[0][:20] for c in candidates[:5]])
        return f"{query}:{candidate_str}"

    def _update_cache(self, key: str, results: List):
        """Update LRU cache with new results."""
        # Simple LRU: remove oldest if cache is full
        if len(self.cache) >= self.config.cache_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = results

    def get_stats(self) -> Dict:
        """Get reranker statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "model": self.config.model_name,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "alpha": self.config.alpha,
        }


class DenseEmbeddingReranker:
    """
    Alternative reranker using pre-computed dense embeddings.
    Faster but requires more storage.
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize dense embedding reranker.

        Args:
            embedding_dim: Dimension of dense embeddings
        """
        self.embedding_dim = embedding_dim
        self.doc_embeddings = {}  # doc_id -> embedding
        self.query_encoder = None

        try:
            from sentence_transformers import SentenceTransformer

            self.query_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded query encoder for dense reranking")
        except ImportError:
            logger.warning("SentenceTransformer not available for dense reranking")

    def index_documents(
        self,
        doc_ids: List[str],
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        """
        Pre-compute and store document embeddings.

        Args:
            doc_ids: Document identifiers
            texts: Document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        """
        if self.query_encoder is None:
            raise RuntimeError("Query encoder not available")

        embeddings = self.query_encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=False,
        )

        for doc_id, embedding in zip(doc_ids, embeddings):
            self.doc_embeddings[doc_id] = embedding

        logger.info(
            f"Indexed {len(doc_ids)} documents with {self.embedding_dim}D embeddings"
        )

    def rerank(
        self, query: str, candidate_ids: List[str], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rerank using cosine similarity of dense embeddings.

        Args:
            query: Query text
            candidate_ids: Candidate document IDs
            top_k: Number of results

        Returns:
            List of (doc_id, score) tuples
        """
        if self.query_encoder is None:
            raise RuntimeError("Query encoder not available")

        # Encode query
        query_embedding = self.query_encoder.encode(query, convert_to_tensor=False)

        # Compute similarities
        scores = []
        for doc_id in candidate_ids:
            if doc_id in self.doc_embeddings:
                doc_embedding = self.doc_embeddings[doc_id]
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                scores.append((doc_id, float(similarity)))
            else:
                scores.append((doc_id, 0.0))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


def create_reranker(
    method: str = "cross-encoder", **kwargs
) -> Union[TEJASReranker, DenseEmbeddingReranker]:
    """
    Factory function to create appropriate reranker.

    Args:
        method: 'cross-encoder' or 'dense-embedding'
        **kwargs: Additional configuration

    Returns:
        Reranker instance
    """
    if method == "cross-encoder":
        config = RerankerConfig(**kwargs)
        return TEJASReranker(config)
    elif method == "dense-embedding":
        return DenseEmbeddingReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker method: {method}")
