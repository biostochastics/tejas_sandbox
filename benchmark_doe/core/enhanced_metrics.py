#!/usr/bin/env python3
"""
Enhanced Metrics Module for IR Evaluation
Supports both binary hash codes and dense vectors with appropriate distance metrics
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricResults:
    """Container for comprehensive IR metrics."""
    # Precision at various cutoffs
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0
    precision_at_50: float = 0.0
    precision_at_100: float = 0.0
    
    # Recall at various cutoffs
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    recall_at_50: float = 0.0
    recall_at_100: float = 0.0
    recall_at_500: float = 0.0
    recall_at_1000: float = 0.0
    
    # Ranking metrics
    ndcg_at_1: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    
    map_at_10: float = 0.0
    map_at_100: float = 0.0
    
    mrr: float = 0.0  # Mean Reciprocal Rank (no cutoff)
    mrr_at_10: float = 0.0
    
    # Success rates (at least 1 relevant doc retrieved)
    success_at_1: float = 0.0
    success_at_5: float = 0.0
    success_at_10: float = 0.0
    
    # Efficiency metrics
    queries_per_second: float = 0.0
    encoding_speed_docs_per_sec: float = 0.0
    index_size_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Dataset statistics
    num_queries_evaluated: int = 0
    num_queries_with_relevance: int = 0
    avg_relevant_per_query: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}


class SimilarityScorer:
    """Base class for similarity scoring."""
    
    def compute_similarities(self, query_embeddings: np.ndarray, 
                           doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity matrix between queries and documents."""
        raise NotImplementedError
    
    def get_top_k(self, similarities: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k document indices and scores for each query."""
        if len(similarities.shape) == 1:
            similarities = similarities.reshape(1, -1)
        
        # Get top-k indices
        if k >= similarities.shape[1]:
            # Return all documents sorted
            indices = np.argsort(-similarities, axis=1)
            scores = np.take_along_axis(similarities, indices, axis=1)
        else:
            # Efficient partial sort for top-k
            indices = np.argpartition(-similarities, k-1, axis=1)[:, :k]
            scores = np.take_along_axis(similarities, indices, axis=1)
            # Sort the top-k
            sorted_idx = np.argsort(-scores, axis=1)
            indices = np.take_along_axis(indices, sorted_idx, axis=1)
            scores = np.take_along_axis(scores, sorted_idx, axis=1)
        
        return indices, scores


class HammingScorer(SimilarityScorer):
    """Scorer for binary hash codes using Hamming distance."""
    
    def __init__(self, n_bits: int):
        self.n_bits = n_bits
    
    def compute_similarities(self, query_embeddings: np.ndarray, 
                           doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute Hamming similarity (inverse of Hamming distance).
        Higher similarity = fewer differing bits
        """
        # Convert to numpy if needed (handles torch tensors)
        if hasattr(query_embeddings, 'cpu'):
            query_embeddings = query_embeddings.cpu().numpy()
        elif hasattr(query_embeddings, 'numpy'):
            query_embeddings = query_embeddings.numpy()
        if hasattr(doc_embeddings, 'cpu'):
            doc_embeddings = doc_embeddings.cpu().numpy()
        elif hasattr(doc_embeddings, 'numpy'):
            doc_embeddings = doc_embeddings.numpy()
            
        # Ensure numpy arrays
        query_embeddings = np.asarray(query_embeddings)
        doc_embeddings = np.asarray(doc_embeddings)
        
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # For binary codes, compute XOR and count differing bits
        # Then convert to similarity (1 - normalized_hamming_distance)
        similarities = []
        for q in query_embeddings:
            # XOR to find differing bits
            diff = np.bitwise_xor(doc_embeddings, q)
            # Count differing bits (Hamming distance)
            if diff.dtype == np.uint8:
                # For byte arrays, count set bits using numpy
                hamming_dist = np.unpackbits(diff, axis=1).sum(axis=1)
            else:
                # For bit arrays
                hamming_dist = np.sum(diff != 0, axis=1)
            # Convert to similarity (higher = more similar)
            similarity = 1.0 - (hamming_dist / self.n_bits)
            similarities.append(similarity)
        
        return np.array(similarities)


class CosineScorer(SimilarityScorer):
    """Scorer for dense vectors using cosine similarity."""
    
    def compute_similarities(self, query_embeddings: np.ndarray, 
                           doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between queries and documents."""
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Normalize vectors
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-10)
        doc_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarity
        similarities = np.dot(query_norm, doc_norm.T)
        
        return similarities


class DotProductScorer(SimilarityScorer):
    """Scorer using raw dot product (for compatibility)."""
    
    def compute_similarities(self, query_embeddings: np.ndarray, 
                           doc_embeddings: np.ndarray) -> np.ndarray:
        """Compute dot product between queries and documents."""
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        return np.dot(query_embeddings, doc_embeddings.T)


class EnhancedMetricsCalculator:
    """Enhanced metrics calculator with support for multiple distance metrics."""
    
    def __init__(self, encoding_type: str = "binary", n_bits: int = 256):
        """
        Initialize metrics calculator.
        
        Args:
            encoding_type: Type of encoding ("binary", "dense", "dot_product")
            n_bits: Number of bits for binary encodings
        """
        self.encoding_type = encoding_type
        self.n_bits = n_bits
        
        # Select appropriate scorer
        if encoding_type == "binary":
            self.scorer = HammingScorer(n_bits)
        elif encoding_type == "dense":
            self.scorer = CosineScorer()
        else:  # dot_product or fallback
            self.scorer = DotProductScorer()
        
        logger.info(f"Initialized {encoding_type} metrics calculator with {self.scorer.__class__.__name__}")
    
    def calculate_all_metrics(self, 
                            query_embeddings: np.ndarray,
                            doc_embeddings: np.ndarray,
                            relevance_data: Dict[int, List[int]],
                            query_times: Optional[List[float]] = None,
                            encoding_time: Optional[float] = None,
                            memory_usage: Optional[float] = None) -> MetricResults:
        """
        Calculate comprehensive IR metrics.
        
        Args:
            query_embeddings: Query embeddings/codes
            doc_embeddings: Document embeddings/codes
            relevance_data: Dict mapping query_idx to list of relevant doc indices
            query_times: List of query latencies in ms
            encoding_time: Total encoding time in seconds
            memory_usage: Peak memory usage in MB
        
        Returns:
            MetricResults object with all metrics
        """
        results = MetricResults()
        
        if len(query_embeddings) == 0 or len(doc_embeddings) == 0:
            return results
        
        # NOTE: Similarities are now computed per-query to save memory
        # similarities = self.scorer.compute_similarities(query_embeddings, doc_embeddings)
        
        # Cutoff values to evaluate
        k_values = [1, 5, 10, 20, 50, 100, 500, 1000]
        
        # Store per-query metrics
        precision_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        ndcg_scores = {k: [] for k in [1, 5, 10, 20]}
        map_scores = {k: [] for k in [10, 100]}
        mrr_scores = []
        mrr_at_10_scores = []  # Separate list for MRR@10
        success_scores = {k: [] for k in [1, 5, 10]}
        
        # Track statistics
        queries_with_relevance = 0
        total_relevant = []
        
        # Process each query
        for q_idx in range(len(query_embeddings)):
            # Get relevant documents for this query
            relevant_docs = set(relevance_data.get(q_idx, []))
            
            if not relevant_docs:
                continue  # Skip queries without relevance judgments
            
            queries_with_relevance += 1
            total_relevant.append(len(relevant_docs))
            
            # Compute similarities for this query (memory efficient)
            query_emb = query_embeddings[q_idx:q_idx+1]
            q_similarities = self.scorer.compute_similarities(query_emb, doc_embeddings)
            
            # Get ranking for this query
            ranked_docs, scores = self.scorer.get_top_k(q_similarities, 
                                                       min(1000, len(doc_embeddings)))
            ranked_docs = ranked_docs[0]  # Remove batch dimension
            
            # Calculate metrics at different cutoffs
            for k in k_values:
                if k > len(doc_embeddings):
                    continue
                
                top_k = ranked_docs[:k]
                relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
                
                # Precision@k
                if k in precision_scores:
                    precision_scores[k].append(relevant_in_top_k / k)
                
                # Recall@k
                if k in recall_scores:
                    recall_scores[k].append(relevant_in_top_k / len(relevant_docs))
                
                # Success@k (at least 1 relevant doc)
                if k in success_scores:
                    success_scores[k].append(1.0 if relevant_in_top_k > 0 else 0.0)
            
            # NDCG at various cutoffs
            for k in [1, 5, 10, 20]:
                if k > len(doc_embeddings):
                    continue
                ndcg = self._calculate_ndcg_single(ranked_docs[:k], relevant_docs, k)
                ndcg_scores[k].append(ndcg)
            
            # MAP at cutoffs
            for k in [10, 100]:
                if k > len(doc_embeddings):
                    continue
                ap = self._calculate_ap_single(ranked_docs[:k], relevant_docs)
                map_scores[k].append(ap)
            
            # MRR (Mean Reciprocal Rank)
            mrr, rank = self._calculate_mrr_single(ranked_docs, relevant_docs)
            mrr_scores.append(mrr)
            
            # Correctly calculate MRR@10
            if rank is not None and rank <= 10:
                mrr_at_10_scores.append(1.0 / rank)
            else:
                mrr_at_10_scores.append(0.0)
        
        # Aggregate metrics
        results.num_queries_evaluated = len(query_embeddings)
        results.num_queries_with_relevance = queries_with_relevance
        
        if queries_with_relevance > 0:
            results.avg_relevant_per_query = np.mean(total_relevant)
            
            # Precision metrics
            results.precision_at_1 = np.mean(precision_scores[1]) if precision_scores[1] else 0.0
            results.precision_at_5 = np.mean(precision_scores[5]) if precision_scores[5] else 0.0
            results.precision_at_10 = np.mean(precision_scores[10]) if precision_scores[10] else 0.0
            results.precision_at_20 = np.mean(precision_scores[20]) if precision_scores[20] else 0.0
            results.precision_at_50 = np.mean(precision_scores[50]) if precision_scores[50] else 0.0
            results.precision_at_100 = np.mean(precision_scores[100]) if precision_scores[100] else 0.0
            
            # Recall metrics
            results.recall_at_10 = np.mean(recall_scores[10]) if recall_scores[10] else 0.0
            results.recall_at_20 = np.mean(recall_scores[20]) if recall_scores[20] else 0.0
            results.recall_at_50 = np.mean(recall_scores[50]) if recall_scores[50] else 0.0
            results.recall_at_100 = np.mean(recall_scores[100]) if recall_scores[100] else 0.0
            results.recall_at_500 = np.mean(recall_scores[500]) if recall_scores[500] else 0.0
            results.recall_at_1000 = np.mean(recall_scores[1000]) if recall_scores[1000] else 0.0
            
            # NDCG metrics
            results.ndcg_at_1 = np.mean(ndcg_scores[1]) if ndcg_scores[1] else 0.0
            results.ndcg_at_5 = np.mean(ndcg_scores[5]) if ndcg_scores[5] else 0.0
            results.ndcg_at_10 = np.mean(ndcg_scores[10]) if ndcg_scores[10] else 0.0
            results.ndcg_at_20 = np.mean(ndcg_scores[20]) if ndcg_scores[20] else 0.0
            
            # MAP metrics
            results.map_at_10 = np.mean(map_scores[10]) if map_scores[10] else 0.0
            results.map_at_100 = np.mean(map_scores[100]) if map_scores[100] else 0.0
            
            # MRR metrics
            results.mrr = np.mean(mrr_scores) if mrr_scores else 0.0
            results.mrr_at_10 = np.mean(mrr_at_10_scores) if mrr_at_10_scores else 0.0
            
            # Success rates
            results.success_at_1 = np.mean(success_scores[1]) if success_scores[1] else 0.0
            results.success_at_5 = np.mean(success_scores[5]) if success_scores[5] else 0.0
            results.success_at_10 = np.mean(success_scores[10]) if success_scores[10] else 0.0
        
        # Efficiency metrics
        if query_times:
            results.queries_per_second = 1000.0 / np.mean(query_times) if np.mean(query_times) > 0 else 0.0
        
        if encoding_time and len(doc_embeddings) > 0:
            results.encoding_speed_docs_per_sec = len(doc_embeddings) / encoding_time
        
        if memory_usage:
            results.peak_memory_mb = memory_usage
        
        # Index size estimation
        if hasattr(doc_embeddings, 'nbytes'):
            results.index_size_mb = doc_embeddings.nbytes / (1024 * 1024)
        
        return results
    
    def _calculate_ndcg_single(self, ranked_docs: np.ndarray, 
                              relevant_docs: set, k: int) -> float:
        """Calculate NDCG@k for a single query."""
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, doc_id in enumerate(ranked_docs[:k]):
            rel = 1.0 if doc_id in relevant_docs else 0.0
            dcg += rel / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate ideal DCG
        num_relevant = min(len(relevant_docs), k)
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_ap_single(self, ranked_docs: np.ndarray, 
                            relevant_docs: set) -> float:
        """Calculate Average Precision for a single query."""
        precisions = []
        num_relevant = 0
        
        for i, doc_id in enumerate(ranked_docs):
            if doc_id in relevant_docs:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_mrr_single(self, ranked_docs: np.ndarray, 
                             relevant_docs: set) -> Tuple[float, Optional[int]]:
        """Calculate Reciprocal Rank and rank position for a single query."""
        for i, doc_id in enumerate(ranked_docs):
            if doc_id in relevant_docs:
                rank = i + 1
                return 1.0 / rank, rank
        return 0.0, None


def compute_confidence_intervals(values: List[float], 
                                confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute median and confidence intervals.
    
    Returns:
        (median, lower_bound, upper_bound)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    median = np.median(values)
    
    # Bootstrap confidence intervals
    n_bootstrap = 10000
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_medians.append(np.median(sample))
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
    upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))
    
    return median, lower, upper


def statistical_significance_test(values1: List[float], 
                                 values2: List[float],
                                 test_type: str = "wilcoxon") -> Tuple[float, bool]:
    """
    Test statistical significance between two sets of values.
    
    Args:
        values1: First set of values
        values2: Second set of values
        test_type: "wilcoxon" for paired test, "mannwhitney" for unpaired
    
    Returns:
        (p_value, is_significant at 0.05 level)
    """
    if len(values1) == 0 or len(values2) == 0:
        return 1.0, False
    
    if test_type == "wilcoxon" and len(values1) == len(values2):
        # Paired test
        statistic, p_value = stats.wilcoxon(values1, values2)
    else:
        # Unpaired test
        statistic, p_value = stats.mannwhitneyu(values1, values2)
    
    return p_value, p_value < 0.05


# Dataset-specific baseline scores for normalization
DATASET_BASELINES = {
    "msmarco": {
        "ndcg_at_10": 0.32,  # Typical neural IR baseline
        "map_at_10": 0.28,
        "mrr": 0.30,
        "recall_at_100": 0.85
    },
    "beir-scifact": {
        "ndcg_at_10": 0.65,
        "map_at_10": 0.60,
        "mrr": 0.63,
        "recall_at_100": 0.92
    },
    "wikipedia": {
        "ndcg_at_10": 0.45,  # Estimated baseline
        "map_at_10": 0.40,
        "mrr": 0.42,
        "recall_at_100": 0.75
    }
}


def normalize_to_baseline(metric_value: float, baseline_value: float) -> float:
    """Normalize metric to baseline (0-1 scale where 1 = baseline performance)."""
    if baseline_value > 0:
        return metric_value / baseline_value
    return metric_value