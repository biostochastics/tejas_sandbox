"""
Binary Fingerprint Operations and Search
=======================================

High-performance binary operations for semantic fingerprints.
Implements XOR-based Hamming distance for hardware speed search. 
"""

import torch
import time
import logging
from typing import List, Tuple
logger = logging.getLogger(__name__)


class BinaryFingerprintSearch:
    """
    Ultra-fast search using binary fingerprints and XOR operations.
    Achieves near-theoretical speed limits for pattern matching.
    """
    
    def __init__(self, fingerprints: torch.Tensor, titles: List[str], device: str = 'auto'):
        """
        Initialize search engine.
        
        Args:
            fingerprints: Binary fingerprint tensor (n_items, n_bits)
            titles: List of titles corresponding to fingerprints
            device: Device for computation ('cpu', 'cuda', or 'auto')
        """
        self.fingerprints = fingerprints
        self.titles = titles
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move to device
        self.fingerprints = self.fingerprints.to(self.device)
        
        logger.info(f"Loaded {len(self.titles):,} fingerprints")
        logger.info(f"Device: {self.device}")
        logger.info("Ready for search!")
    
    def search(self, query_fingerprint: torch.Tensor, k: int = 10, show_pattern_analysis: bool = True) -> List[Tuple[str, float, int]]:
        """
        Search for similar titles using XOR-based Hamming distance.
        
        Args:
            query_fingerprint: Query fingerprint tensor
            k: Number of results to return
            show_pattern_analysis: Show pattern family analysis
            
        Returns:
            List of (title, similarity, distance) tuples
        """
        start_time = time.time()
        
        # Move query to device
        query_fingerprint = query_fingerprint.to(self.device)
        
        # Compute Hamming distances using XOR
        xor_result = self.fingerprints ^ query_fingerprint.unsqueeze(0)
        
        # Count differing bits (Hamming distance)
        hamming_distances = xor_result.sum(dim=1)
        
        # Get top-k nearest
        distances, indices = torch.topk(hamming_distances, k, largest=False)
        
        search_time = time.time() - start_time
        
        # Convert to similarities
        n_bits = self.fingerprints.shape[1]
        similarities = 1.0 - (distances.float() / n_bits)
        
        # Prepare results
        results = []
        for idx, sim, dist in zip(indices.cpu(), similarities.cpu(), distances.cpu()):
            results.append((
                self.titles[idx],
                float(sim),
                int(dist)
            ))
        
        # Log performance
        comparisons_per_sec = len(self.titles) / search_time
        logger.info(f"Search time: {search_time*1000:.2f} ms")
        logger.info(f"Comparisons/sec: {comparisons_per_sec:,.0f}")
        
        # Pattern analysis
        if show_pattern_analysis:
            self._analyze_patterns(results)
        
        return results
    
    def search_pattern(self, pattern: str, encoder, max_results: int = 100) -> List[Tuple[str, float, int]]:
        """
        Search for titles containing a specific pattern.
        Demonstrates zero false positives for pattern matching.
        
        Args:
            pattern: Pattern to search for (e.g., "List of", "University of")
            encoder: Encoder to create query fingerprint
            max_results: Maximum results to return
            
        Returns:
            Matching titles with similarities
        """
        logger.info(f"Pattern search for: '{pattern}'")
        
        # Encode the pattern
        pattern_fingerprint = encoder.encode_single(pattern)
        
        # Search with larger k to find true matches
        results = self.search(pattern_fingerprint, k=min(1000, len(self.titles)), show_pattern_analysis=False)
        
        # Filter to only those that ACTUALLY contain the pattern
        pattern_matches = []
        false_positives = []
        
        for title, sim, dist in results:
            if pattern.lower() in title.lower():
                pattern_matches.append((title, sim, dist))
            else:
                false_positives.append((title, sim, dist))
            
            if len(pattern_matches) >= max_results:
                break
        
        # Report findings
        logger.info(f"Pattern Match Analysis:")
        logger.info(f"  Checked: {len(results)} similar fingerprints")
        logger.info(f"  True matches: {len(pattern_matches)}")
        logger.info(f"  False positives: {len(false_positives)}")
        if len(pattern_matches) + len(false_positives) > 0:
            logger.info(f"  Precision: {len(pattern_matches)/(len(pattern_matches)+len(false_positives))*100:.1f}%")
        
        return pattern_matches[:max_results]
    
    def _analyze_patterns(self, results: List[Tuple[str, float, int]]):
        """Analyze pattern families in search results."""
        # Common patterns to check
        patterns = {
            'List of': 0,
            'University': 0,
            'County': 0,
            'Battle of': 0,
            '(disambiguation)': 0,
            '(film)': 0,
            '(album)': 0,
            'History of': 0
        }
        
        # Count patterns in results
        for title, _, _ in results:
            for pattern in patterns:
                if pattern in title:
                    patterns[pattern] += 1
        
        # Show if any patterns dominate
        if any(count > len(results) * 0.3 for count in patterns.values()):
            logger.info("Pattern Family Analysis:")
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    logger.info(f"  {pattern}: {count}/{len(results)} ({count/len(results)*100:.0f}%)")
    
    def benchmark(self, n_queries: int = 100):
        """
        Benchmark search performance.
        
        Args:
            n_queries: Number of random queries to test
        """
        logger.info(f"Benchmarking with {n_queries} random queries...")
        
        # Select random fingerprints as queries
        query_indices = torch.randperm(len(self.titles))[:n_queries]
        
        # Time searches
        search_times = []
        
        for idx in query_indices:
            query = self.fingerprints[idx]
            start = time.time()
            _ = self.search(query, k=10, show_pattern_analysis=False)
            search_times.append(time.time() - start)
        
        # Calculate statistics
        search_times = torch.tensor(search_times) * 1000  # Convert to ms
        
        logger.info(f"Benchmark Results:")
        logger.info(f"  Average search time: {search_times.mean():.2f} ms")
        logger.info(f"  Median search time: {search_times.median():.2f} ms")
        logger.info(f"  Min search time: {search_times.min():.2f} ms")
        logger.info(f"  Max search time: {search_times.max():.2f} ms")
        logger.info(f"  Comparisons/sec: {len(self.titles)/search_times.mean()*1000:,.0f}")


def demonstrate_fingerprint_search():
    """
    Demonstrate fingerprint search capabilities.
    """
    # Create sample data
    n_items = 10000
    n_bits = 128
    
    # Generate random fingerprints and titles
    fingerprints = torch.randint(0, 2, (n_items, n_bits), dtype=torch.uint8)
    titles = [f"Sample Title {i}" for i in range(n_items)]
    
    # Create search engine
    search_engine = BinaryFingerprintSearch(fingerprints, titles)
    
    print("\nBinary Fingerprint Search Demo:")
    print("=" * 50)
    print(f"Database: {n_items:,} items, {n_bits} bits each")
    
    # Perform search
    query = fingerprints[0]
    results = search_engine.search(query, k=5)
    
    print(f"\nSearch results:")
    for i, (title, sim, dist) in enumerate(results):
        print(f"  {i+1}. {title}: similarity={sim:.3f}, distance={dist}")
    
    # Benchmark
    search_engine.benchmark(n_queries=10)


if __name__ == "__main__":
    demonstrate_fingerprint_search()