"""
BERT Encoder Wrapper for DOE Benchmarking
Provides BERT baselines for comparison with TEJAS encoders
"""

import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for sentence-transformers availability
try:
    from sentence_transformers import SentenceTransformer
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logger.warning("sentence-transformers not installed. BERT baselines unavailable.")
    logger.warning("Install with: pip install sentence-transformers transformers torch")


@dataclass
class BERTConfig:
    """Configuration for BERT encoder."""
    model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cpu'  # 'cpu', 'cuda', 'mps'
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress: bool = False
    max_seq_length: Optional[int] = None
    use_fp16: bool = False  # Half precision for faster inference
    

class BERTEncoder:
    """
    BERT encoder wrapper for DOE benchmarking.
    Provides consistent interface with TEJAS encoders.
    """
    
    def __init__(self, config: Optional[BERTConfig] = None):
        """Initialize BERT encoder with configuration."""
        if not BERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers transformers torch"
            )
        
        self.config = config or BERTConfig()
        self.model = None
        self.embedding_dim = None
        self.encoding_times = []
        
    def fit(self, documents: List[str]) -> 'BERTEncoder':
        """
        Initialize BERT model. No training needed as it's pre-trained.
        
        Args:
            documents: List of documents (unused, kept for interface compatibility)
            
        Returns:
            Self for chaining
        """
        logger.info(f"Initializing BERT model: {self.config.model_name}")
        
        # Initialize model
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        
        # Set max sequence length if specified
        if self.config.max_seq_length:
            self.model.max_seq_length = self.config.max_seq_length
            
        # Enable fp16 if requested and available
        if self.config.use_fp16 and self.config.device != 'cpu':
            self.model.half()
        
        # Get embedding dimension
        dummy_embedding = self.model.encode(['test'], convert_to_numpy=True)
        self.embedding_dim = dummy_embedding.shape[1]
        
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Max sequence length: {self.model.max_seq_length}")
        
        return self
    
    def encode(self, texts: List[str], return_binary: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            return_binary: If True, binarize embeddings (for comparison with TEJAS)
            
        Returns:
            Embeddings array (n_texts, embedding_dim) or binary codes
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call fit() first.")
        
        start_time = time.time()
        
        # Encode texts
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=self.config.show_progress,
            convert_to_numpy=True
        )
        
        encoding_time = time.time() - start_time
        self.encoding_times.append(encoding_time)
        
        # Optionally binarize for fair comparison with TEJAS
        if return_binary:
            embeddings = self._binarize(embeddings)
        
        return embeddings
    
    def _binarize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Binarize continuous embeddings for comparison with TEJAS.
        Uses median threshold per dimension.
        
        Args:
            embeddings: Continuous embeddings
            
        Returns:
            Binary embeddings
        """
        # Use median as threshold for each dimension
        thresholds = np.median(embeddings, axis=0, keepdims=True)
        binary = (embeddings > thresholds).astype(np.uint8)
        return binary
    
    def search(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of results to return
            
        Returns:
            List of (doc_idx, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call fit() first.")
        
        # Encode query
        query_embedding = self.encode([query])[0]
        
        # Encode documents (could be cached in production)
        doc_embeddings = self.encode(documents)
        
        # Compute cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results with scores
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        if self.model is None:
            return {'model_memory_mb': 0, 'total_memory_mb': 0}
        
        # Estimate model memory
        model_params = sum(p.numel() for p in self.model.parameters())
        bytes_per_param = 4 if not self.config.use_fp16 else 2
        model_memory_mb = (model_params * bytes_per_param) / (1024 * 1024)
        
        # Get GPU memory if using CUDA
        gpu_memory_mb = 0
        if self.config.device == 'cuda' and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return {
            'model_memory_mb': model_memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'total_memory_mb': model_memory_mb + gpu_memory_mb
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.encoding_times:
            return {}
        
        return {
            'avg_encoding_time': np.mean(self.encoding_times),
            'total_encoding_time': sum(self.encoding_times),
            'num_encodings': len(self.encoding_times),
            'embedding_dim': self.embedding_dim,
            'model_name': self.config.model_name,
            'device': self.config.device,
            'memory_usage': self.get_memory_usage()
        }


class BERTBenchmarkAdapter:
    """
    Adapter to make BERT encoder compatible with DOE benchmark runner.
    Provides the same interface as TEJAS encoders.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 device: str = 'cpu', **kwargs):
        """Initialize adapter with BERT configuration."""
        self.config = BERTConfig(
            model_name=model_name,
            device=device,
            **kwargs
        )
        self.encoder = BERTEncoder(self.config)
        self.fitted = False
        
    def fit(self, documents: List[str]) -> 'BERTBenchmarkAdapter':
        """Fit the encoder (initialize BERT model)."""
        self.encoder.fit(documents)
        self.fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to embeddings (compatible with TEJAS interface)."""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        return self.encoder.encode(texts, return_binary=False)
    
    def encode_binary(self, texts: List[str]) -> np.ndarray:
        """Encode texts to binary codes (for comparison with TEJAS)."""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")
        return self.encoder.encode(texts, return_binary=True)
    
    def get_index_size(self) -> int:
        """Get index size in bytes (memory usage)."""
        memory_stats = self.encoder.get_memory_usage()
        return int(memory_stats['total_memory_mb'] * 1024 * 1024)
    
    def search(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar documents."""
        return self.encoder.search(query, documents, top_k)
    
    def get_params(self) -> Dict[str, Any]:
        """Get encoder parameters."""
        return {
            'model_name': self.config.model_name,
            'embedding_dim': self.encoder.embedding_dim,
            'device': self.config.device,
            'normalize': self.config.normalize_embeddings,
            'batch_size': self.config.batch_size
        }


# Factory function for benchmark runner
def create_bert_encoder(pipeline_type: str, config: Dict[str, Any]) -> BERTBenchmarkAdapter:
    """
    Factory function to create BERT encoder for benchmarking.
    
    Args:
        pipeline_type: Type of BERT model ('bert_mini', 'bert_base', etc.)
        config: Configuration dictionary
        
    Returns:
        BERT encoder adapter
    """
    # Model mapping
    model_map = {
        'bert_mini': 'all-MiniLM-L6-v2',  # 384 dims, fast
        'bert_base': 'all-mpnet-base-v2',  # 768 dims, accurate
        'bert_distil': 'all-distilroberta-v1',  # 768 dims, balanced
        'bert_e5': 'intfloat/e5-small-v2',  # 384 dims, state-of-art
    }
    
    model_name = config.get('model_name', model_map.get(pipeline_type, 'all-MiniLM-L6-v2'))
    device = config.get('device', 'cpu')
    
    # Check for GPU availability
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    return BERTBenchmarkAdapter(
        model_name=model_name,
        device=device,
        batch_size=config.get('batch_size', 32),
        normalize_embeddings=config.get('normalize_embeddings', True),
        use_fp16=config.get('use_fp16', False)
    )