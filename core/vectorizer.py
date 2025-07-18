"""
Consciousness-Aligned Character N-gram Vectorizer
================================================

Extracts character n-grams matching human saccade patterns (3-5 characters).
This module handles the text → n-gram → TF-IDF transformation.

"""

import numpy as np
from typing import List, Dict, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class CharacterVectorizer:
    """
    Character n-gram vectorizer optimized for semantic fingerprinting.
    
    Key principles:
    - 3-5 character windows match human eye saccades
    - TF-IDF weighting captures semantic importance
    - Handles any Unicode text (including mathematical symbols)
    """
    
    def __init__(self, 
                 ngram_range: Tuple[int, int] = (3, 5),
                 max_features: int = 10000,
                 lowercase: bool = True,
                 dtype: type = np.float32):
        """
        Initialize the character vectorizer.
        
        Args:
            ngram_range: Character n-gram range (default 3-5 for saccades)
            max_features: Maximum number of features to extract
            lowercase: Convert to lowercase before extraction
            dtype: Data type for the matrix (float32 for efficiency)
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.lowercase = lowercase
        self.dtype = dtype
        
        # Internal sklearn vectorizer
        self._vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=lowercase,
            dtype=dtype
        )
        
        # State tracking
        self.is_fitted = False
        self.vocabulary_size = 0
        
        logger.info(f"Initialized CharacterVectorizer with:")
        logger.info(f"  N-gram range: {ngram_range}")
        logger.info(f"  Max features: {max_features}")
        
    def fit(self, texts: List[str]) -> 'CharacterVectorizer':
        """
        Learn vocabulary from texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting vectorizer on {len(texts)} texts...")
        
        self._vectorizer.fit(texts)
        self.is_fitted = True
        self.vocabulary_size = len(self._vectorizer.vocabulary_)
        
        logger.info(f"Learned vocabulary of {self.vocabulary_size} n-grams")
        
        # Log some statistics
        if self.vocabulary_size > 0:
            self._log_vocabulary_stats()
            
        return self
    
    def transform(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            TF-IDF matrix (sparse or dense depending on size)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform
        X = self._vectorizer.transform(texts)
        
        # Convert to dense if small enough
        if X.shape[0] * X.shape[1] < 1e6:  # Less than 1M elements
            return X.toarray()
        else:
            return X  # Keep sparse for large matrices
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of texts
            
        Returns:
            TF-IDF matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the learned n-gram features.
        
        Returns:
            List of n-gram strings
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
            
        return self._vectorizer.get_feature_names_out().tolist()
    
    def get_vocabulary(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.
        
        Returns:
            Dict mapping n-grams to indices
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
            
        return self._vectorizer.vocabulary_
    
    def get_idf_weights(self) -> np.ndarray:
        """
        Get the IDF weights for each feature.
        
        Returns:
            Array of IDF weights
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
            
        return self._vectorizer.idf_
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze a single text and return its top n-grams.
        
        Args:
            text: Input text
            
        Returns:
            Dict of n-grams and their TF-IDF scores
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        # Transform the text
        vector = self.transform(text).flatten()
        
        # Get non-zero indices
        nonzero_idx = np.nonzero(vector)[0]
        
        # Get feature names
        feature_names = self.get_feature_names()
        
        # Create result dict
        result = {}
        for idx in nonzero_idx:
            ngram = feature_names[idx]
            score = vector[idx]
            result[ngram] = float(score)
        
        # Sort by score
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    
    def _log_vocabulary_stats(self):
        """Log statistics about the learned vocabulary."""
        feature_names = self.get_feature_names()
        
        # Count by n-gram size
        ngram_counts = {}
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            count = sum(1 for f in feature_names if len(f) == n)
            ngram_counts[n] = count
            
        logger.info("Vocabulary breakdown by n-gram size:")
        for n, count in ngram_counts.items():
            percentage = count / self.vocabulary_size * 100
            logger.info(f"  {n}-grams: {count} ({percentage:.1f}%)")
    
    def save_vocabulary(self, filepath: str):
        """
        Save vocabulary to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        vocab_items = sorted(self.get_vocabulary().items(), key=lambda x: x[1])
        vocab_array = np.array([item[0] for item in vocab_items], dtype=object)
        
        np.save(filepath, vocab_array)
        logger.info(f"Saved vocabulary to {filepath}")
    
    def load_vocabulary(self, vocab_path: str, idf_path: str):
        """
        Load pre-computed vocabulary.
        
        Args:
            vocab_path: Path to vocabulary file
            idf_path: Path to IDF weights file
        """
        # Load vocabulary
        vocab_array = np.load(vocab_path, allow_pickle=True)
        
        # Recreate vocabulary dict
        self._vectorizer.vocabulary_ = {
            word: idx for idx, word in enumerate(vocab_array)
        }
        
        # Load IDF weights
        self._vectorizer.idf_ = np.load(idf_path)
        
        self.is_fitted = True
        self.vocabulary_size = len(vocab_array)
        
        logger.info(f"Loaded vocabulary of {self.vocabulary_size} n-grams")


def demonstrate_pattern_extraction():
    """
    Demonstrate how the vectorizer extracts character patterns.
    """
    # Example texts
    texts = [
        "Harry Potter and the Philosopher's Stone",
        "Harry Potter and the Chamber of Secrets",
        "The Lord of the Rings",
        "The Hobbit",
        "Quantum Mechanics"
    ]
    
    # Create vectorizer
    vectorizer = CharacterVectorizer(
        ngram_range=(3, 5),
        max_features=100
    )
    
    # Fit and analyze
    vectorizer.fit(texts)
    
    print("\nCharacter N-gram Analysis:")
    print("=" * 50)
    
    # Analyze first text
    analysis = vectorizer.analyze_text(texts[0])
    
    print(f"\nTop n-grams for: '{texts[0]}'")
    for ngram, score in list(analysis.items())[:10]:
        print(f"  '{ngram}': {score:.3f}")
    
    # Show pattern sharing between similar texts
    print("\nShared patterns between Harry Potter books:")
    hp1_ngrams = set(vectorizer.analyze_text(texts[0]).keys())
    hp2_ngrams = set(vectorizer.analyze_text(texts[1]).keys())
    shared = hp1_ngrams.intersection(hp2_ngrams)
    
    print(f"  Shared n-grams: {len(shared)}")
    print(f"  Examples: {list(shared)[:5]}")


if __name__ == "__main__":
    demonstrate_pattern_extraction()