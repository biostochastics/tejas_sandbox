"""
Modern Tokenizer Implementations for tejas
==========================================

Provides multiple tokenization strategies:
1. HuggingFace-based character n-grams (drop-in replacement)
2. Byte-level BPE tokenizer
3. SupraTok-style cross-boundary tokenizer
4. Hybrid hierarchical tokenizer

All implementations maintain TF-IDF weighting and sparse matrix output.
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack
from typing import List, Dict, Tuple, Union, Any
from collections import Counter
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from tokenizers import Tokenizer, pre_tokenizers, models, trainers, normalizers
    from tokenizers.pre_tokenizers import PreTokenizer

    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    # Define dummy types to avoid NameError
    Tokenizer = Any
    PreTokenizer = Any
    logger.warning(
        "HuggingFace tokenizers not available. Install with: pip install tokenizers"
    )


class TokenizerType(Enum):
    """Available tokenizer types."""

    CHAR_NGRAM = "char_ngram"
    BYTE_BPE = "byte_bpe"
    SUPRATOK = "supratok"
    HYBRID = "hybrid"


@dataclass
class TokenizerConfig:
    """Configuration for tokenizers."""

    tokenizer_type: TokenizerType = TokenizerType.CHAR_NGRAM
    ngram_range: Tuple[int, int] = (3, 5)
    max_features: int = 10000
    vocab_size: int = 10000
    min_frequency: int = 2
    lowercase: bool = True
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False
    dtype: type = np.float32

    # Byte-BPE specific
    byte_fallback: bool = True

    # SupraTok specific
    cross_boundary_threshold: float = 0.7
    max_phrase_length: int = 4

    # Hybrid specific
    level_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)


class CharNgramTokenizer:
    """
    HuggingFace-accelerated character n-gram tokenizer.
    Drop-in replacement for sklearn TfidfVectorizer with character analyzer.
    """

    def __init__(self, config: TokenizerConfig):
        """Initialize the character n-gram tokenizer."""
        self.config = config
        self.vocabulary_ = {}
        self.idf_ = None
        self.n_features_ = 0
        self.is_fitted = False

        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "HuggingFace tokenizers required. Install with: pip install tokenizers"
            )

    def _create_char_ngram_tokenizer(self) -> Tokenizer:
        """Create a custom tokenizer for character n-grams."""
        # Use a dummy model, we'll override with custom pre-tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Add normalizers
        normalizer_list = []
        if self.config.lowercase:
            normalizer_list.append(normalizers.Lowercase())
        normalizer_list.append(normalizers.NFKC())  # Unicode normalization

        if normalizer_list:
            tokenizer.normalizer = normalizers.Sequence(normalizer_list)

        # Custom pre-tokenizer for sliding window n-grams
        class SlidingCharNgram:
            def __init__(self, min_n: int, max_n: int):
                self.min_n = min_n
                self.max_n = max_n

            def pre_tokenize(self, pretok):
                text = pretok.normalized
                tokens = []

                for n in range(self.min_n, self.max_n + 1):
                    for i in range(len(text) - n + 1):
                        ngram = text[i : i + n]
                        tokens.append((ngram, (i, i + n)))

                return tokens

        # Note: In production, we'd implement this as a proper PreTokenizer
        # For now, we'll use the tokenizer's splitting capabilities
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        return tokenizer

    def _extract_char_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from text."""
        if self.config.lowercase:
            text = text.lower()

        ngrams = []
        min_n, max_n = self.config.ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i : i + n]
                ngrams.append(ngram)

        return ngrams

    def fit(self, texts: List[str]) -> "CharNgramTokenizer":
        """
        Learn vocabulary and IDF from texts.
        """
        n_docs = len(texts)

        # Build vocabulary using character n-grams
        doc_freq = Counter()
        all_ngrams = []

        for text in texts:
            ngrams = self._extract_char_ngrams(text)
            unique_ngrams = set(ngrams)

            for ngram in unique_ngrams:
                doc_freq[ngram] += 1
            all_ngrams.extend(ngrams)

        # Get term frequencies for ranking
        term_freq = Counter(all_ngrams)

        # Filter by minimum frequency
        filtered_terms = [
            term
            for term, count in term_freq.items()
            if count >= self.config.min_frequency
        ]

        # Sort by frequency and limit to max_features
        filtered_terms.sort(key=lambda x: term_freq[x], reverse=True)
        if len(filtered_terms) > self.config.max_features:
            filtered_terms = filtered_terms[: self.config.max_features]

        # Create vocabulary mapping
        self.vocabulary_ = {term: idx for idx, term in enumerate(filtered_terms)}
        self.n_features_ = len(self.vocabulary_)

        # Calculate IDF
        if self.config.use_idf:
            self.idf_ = np.zeros(self.n_features_, dtype=self.config.dtype)
            for term, idx in self.vocabulary_.items():
                df = doc_freq.get(term, 0)
                if self.config.smooth_idf:
                    idf = np.log((n_docs + 1) / (df + 1)) + 1
                else:
                    idf = np.log(n_docs / max(df, 1)) + 1
                self.idf_[idx] = idf

        self.is_fitted = True
        logger.info(f"CharNgramTokenizer fitted with {self.n_features_} features")

        return self

    def transform(self, texts: Union[str, List[str]]) -> csr_matrix:
        """
        Transform texts to TF-IDF sparse matrix.
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        if isinstance(texts, str):
            texts = [texts]

        n_docs = len(texts)

        # Build sparse matrix
        row_indices = []
        col_indices = []
        values = []

        for doc_idx, text in enumerate(texts):
            ngrams = self._extract_char_ngrams(text)

            # Count term frequencies
            tf = Counter(ngrams)

            # Calculate TF-IDF for each term
            doc_values = {}
            for term, count in tf.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]

                    # Calculate TF
                    if self.config.sublinear_tf:
                        tf_value = 1 + np.log(count)
                    else:
                        tf_value = count

                    # Apply IDF
                    if self.config.use_idf:
                        doc_values[term_idx] = tf_value * self.idf_[term_idx]
                    else:
                        doc_values[term_idx] = tf_value

            # L2 normalization
            if doc_values:
                norm = np.sqrt(sum(v**2 for v in doc_values.values()))
                if norm > 0:
                    for term_idx in doc_values:
                        doc_values[term_idx] /= norm

                # Add to sparse matrix lists
                for term_idx, value in doc_values.items():
                    row_indices.append(doc_idx)
                    col_indices.append(term_idx)
                    values.append(value)

        # Create sparse matrix
        if values:
            X = csr_matrix(
                (values, (row_indices, col_indices)),
                shape=(n_docs, self.n_features_),
                dtype=self.config.dtype,
            )
        else:
            X = csr_matrix((n_docs, self.n_features_), dtype=self.config.dtype)

        return X

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)


class ByteBPETokenizer:
    """
    Byte-level BPE tokenizer similar to GPT-2.
    Handles any text without OOV issues.
    """

    def __init__(self, config: TokenizerConfig):
        """Initialize the byte-level BPE tokenizer."""
        self.config = config
        self.tokenizer = None
        self.vocabulary_ = {}
        self.idf_ = None
        self.n_features_ = 0
        self.is_fitted = False

        if not TOKENIZERS_AVAILABLE:
            raise ImportError(
                "HuggingFace tokenizers required. Install with: pip install tokenizers"
            )

    def fit(self, texts: List[str]) -> "ByteBPETokenizer":
        """
        Train BPE model and learn IDF weights.
        """
        # Create BPE tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Use byte-level pre-tokenizer (like GPT-2)
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Add normalizers
        normalizer_list = []
        if self.config.lowercase:
            normalizer_list.append(normalizers.Lowercase())
        normalizer_list.append(normalizers.NFKC())

        if normalizer_list:
            self.tokenizer.normalizer = normalizers.Sequence(normalizer_list)

        # Create trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        )

        # Train the tokenizer
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        # Build vocabulary mapping
        vocab = self.tokenizer.get_vocab()
        self.vocabulary_ = {token: idx for token, idx in vocab.items()}
        self.n_features_ = len(self.vocabulary_)

        # Calculate IDF weights
        if self.config.use_idf:
            n_docs = len(texts)
            doc_freq = Counter()

            for text in texts:
                tokens = set(self.tokenizer.encode(text).tokens)
                for token in tokens:
                    doc_freq[token] += 1

            self.idf_ = np.zeros(self.n_features_, dtype=self.config.dtype)
            for token, idx in self.vocabulary_.items():
                df = doc_freq.get(token, 0)
                if self.config.smooth_idf:
                    idf = np.log((n_docs + 1) / (df + 1)) + 1
                else:
                    idf = np.log(n_docs / max(df, 1)) + 1
                self.idf_[idx] = idf

        self.is_fitted = True
        logger.info(f"ByteBPETokenizer fitted with {self.n_features_} features")

        return self

    def transform(self, texts: Union[str, List[str]]) -> csr_matrix:
        """
        Transform texts to TF-IDF sparse matrix.
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        if isinstance(texts, str):
            texts = [texts]

        n_docs = len(texts)

        # Build sparse matrix
        row_indices = []
        col_indices = []
        values = []

        for doc_idx, text in enumerate(texts):
            # Tokenize with BPE
            encoding = self.tokenizer.encode(text)
            tokens = encoding.tokens

            # Count term frequencies
            tf = Counter(tokens)

            # Calculate TF-IDF
            doc_values = {}
            for token, count in tf.items():
                if token in self.vocabulary_:
                    token_idx = self.vocabulary_[token]

                    # Calculate TF
                    if self.config.sublinear_tf:
                        tf_value = 1 + np.log(count)
                    else:
                        tf_value = count

                    # Apply IDF
                    if self.config.use_idf and self.idf_ is not None:
                        doc_values[token_idx] = tf_value * self.idf_[token_idx]
                    else:
                        doc_values[token_idx] = tf_value

            # L2 normalization
            if doc_values:
                norm = np.sqrt(sum(v**2 for v in doc_values.values()))
                if norm > 0:
                    for token_idx in doc_values:
                        doc_values[token_idx] /= norm

                # Add to sparse matrix
                for token_idx, value in doc_values.items():
                    row_indices.append(doc_idx)
                    col_indices.append(token_idx)
                    values.append(value)

        # Create sparse matrix
        if values:
            X = csr_matrix(
                (values, (row_indices, col_indices)),
                shape=(n_docs, self.n_features_),
                dtype=self.config.dtype,
            )
        else:
            X = csr_matrix((n_docs, self.n_features_), dtype=self.config.dtype)

        return X

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)


class SupraTokTokenizer:
    """
    SupraTok-style tokenizer that captures cross-boundary patterns.
    Identifies multi-word semantic units and common phrases.
    """

    def __init__(self, config: TokenizerConfig):
        """Initialize the SupraTok tokenizer."""
        self.config = config
        self.vocabulary_ = {}
        self.phrase_patterns_ = {}
        self.idf_ = None
        self.n_features_ = 0
        self.is_fitted = False

    def _extract_phrases(self, texts: List[str]) -> Dict[str, float]:
        """
        Extract high-quality phrases using PMI (Pointwise Mutual Information).
        """
        # Count unigrams and bigrams/trigrams
        unigram_counts = Counter()
        phrase_counts = Counter()
        total_words = 0

        for text in texts:
            if self.config.lowercase:
                text = text.lower()

            # Simple word tokenization
            words = re.findall(r"\b\w+\b", text)
            total_words += len(words)

            # Count unigrams
            for word in words:
                unigram_counts[word] += 1

            # Count phrases (2-4 words)
            for n in range(2, min(self.config.max_phrase_length + 1, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i : i + n])
                    phrase_counts[phrase] += 1

        # Calculate PMI scores for phrases
        phrase_scores = {}
        for phrase, count in phrase_counts.items():
            if count < self.config.min_frequency:
                continue

            words = phrase.split()

            # Calculate expected frequency if words were independent
            expected = 1.0
            for word in words:
                expected *= unigram_counts.get(word, 0) / total_words
            expected *= total_words * (len(words) - 1)

            # PMI score
            if expected > 0:
                pmi = np.log(count / expected)
                # Normalize by phrase length to favor longer coherent phrases
                normalized_pmi = pmi / np.sqrt(len(words))

                if normalized_pmi > self.config.cross_boundary_threshold:
                    phrase_scores[phrase] = normalized_pmi

        return phrase_scores

    def _tokenize_with_phrases(self, text: str) -> List[str]:
        """
        Tokenize text recognizing learned phrases.
        """
        if self.config.lowercase:
            text = text.lower()

        words = re.findall(r"\b\w+\b", text)
        tokens = []
        i = 0

        while i < len(words):
            # Try to match longest phrase first
            matched = False
            for n in range(min(self.config.max_phrase_length, len(words) - i), 1, -1):
                phrase = " ".join(words[i : i + n])
                if phrase in self.phrase_patterns_:
                    tokens.append(
                        phrase.replace(" ", "_")
                    )  # Use underscore for multi-word
                    i += n
                    matched = True
                    break

            if not matched:
                # Fall back to single word
                tokens.append(words[i])
                i += 1

        # Also add character n-grams for words not in phrases
        char_tokens = []
        for token in tokens:
            if "_" not in token:  # Single word
                min_n, max_n = self.config.ngram_range
                for n in range(min_n, min(max_n + 1, len(token) + 1)):
                    for j in range(len(token) - n + 1):
                        char_tokens.append(token[j : j + n])

        return tokens + char_tokens

    def fit(self, texts: List[str]) -> "SupraTokTokenizer":
        """
        Learn phrases and vocabulary from texts.
        """
        n_docs = len(texts)

        # Extract high-quality phrases
        self.phrase_patterns_ = self._extract_phrases(texts)
        logger.info(f"Extracted {len(self.phrase_patterns_)} cross-boundary phrases")

        # Build vocabulary
        doc_freq = Counter()
        all_tokens = []

        for text in texts:
            tokens = self._tokenize_with_phrases(text)
            unique_tokens = set(tokens)

            for token in unique_tokens:
                doc_freq[token] += 1
            all_tokens.extend(tokens)

        # Get term frequencies
        term_freq = Counter(all_tokens)

        # Filter and sort
        filtered_terms = [
            term
            for term, count in term_freq.items()
            if count >= self.config.min_frequency
        ]
        filtered_terms.sort(key=lambda x: term_freq[x], reverse=True)

        if len(filtered_terms) > self.config.max_features:
            filtered_terms = filtered_terms[: self.config.max_features]

        # Create vocabulary
        self.vocabulary_ = {term: idx for idx, term in enumerate(filtered_terms)}
        self.n_features_ = len(self.vocabulary_)

        # Calculate IDF
        if self.config.use_idf:
            self.idf_ = np.zeros(self.n_features_, dtype=self.config.dtype)
            for term, idx in self.vocabulary_.items():
                df = doc_freq.get(term, 0)
                if self.config.smooth_idf:
                    idf = np.log((n_docs + 1) / (df + 1)) + 1
                else:
                    idf = np.log(n_docs / max(df, 1)) + 1
                self.idf_[idx] = idf

        self.is_fitted = True

        # Log statistics
        multi_word_count = sum(1 for term in self.vocabulary_ if "_" in term)
        logger.info(f"SupraTokTokenizer fitted with {self.n_features_} features")
        logger.info(
            f"  Multi-word units: {multi_word_count} ({multi_word_count / self.n_features_ * 100:.1f}%)"
        )

        return self

    def transform(self, texts: Union[str, List[str]]) -> csr_matrix:
        """
        Transform texts to TF-IDF sparse matrix.
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        if isinstance(texts, str):
            texts = [texts]

        n_docs = len(texts)

        # Build sparse matrix
        row_indices = []
        col_indices = []
        values = []

        for doc_idx, text in enumerate(texts):
            tokens = self._tokenize_with_phrases(text)

            # Count term frequencies
            tf = Counter(tokens)

            # Calculate TF-IDF
            doc_values = {}
            for token, count in tf.items():
                if token in self.vocabulary_:
                    token_idx = self.vocabulary_[token]

                    # Calculate TF
                    if self.config.sublinear_tf:
                        tf_value = 1 + np.log(count)
                    else:
                        tf_value = count

                    # Apply IDF
                    if self.config.use_idf:
                        doc_values[token_idx] = tf_value * self.idf_[token_idx]
                    else:
                        doc_values[token_idx] = tf_value

            # L2 normalization
            if doc_values:
                norm = np.sqrt(sum(v**2 for v in doc_values.values()))
                if norm > 0:
                    for token_idx in doc_values:
                        doc_values[token_idx] /= norm

                # Add to sparse matrix
                for token_idx, value in doc_values.items():
                    row_indices.append(doc_idx)
                    col_indices.append(token_idx)
                    values.append(value)

        # Create sparse matrix
        if values:
            X = csr_matrix(
                (values, (row_indices, col_indices)),
                shape=(n_docs, self.n_features_),
                dtype=self.config.dtype,
            )
        else:
            X = csr_matrix((n_docs, self.n_features_), dtype=self.config.dtype)

        return X

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)


class HybridTokenizer:
    """
    Hierarchical tokenizer combining multiple levels:
    - Level 1: Character n-grams (base)
    - Level 2: Byte-level BPE (morphology)
    - Level 3: SupraTok phrases (semantics)
    """

    def __init__(self, config: TokenizerConfig):
        """Initialize the hybrid tokenizer."""
        self.config = config

        # Create sub-tokenizers
        self.char_tokenizer = CharNgramTokenizer(config)
        self.bpe_tokenizer = ByteBPETokenizer(config)
        self.phrase_tokenizer = SupraTokTokenizer(config)

        self.is_fitted = False
        self.n_features_ = 0

    def fit(self, texts: List[str]) -> "HybridTokenizer":
        """
        Fit all sub-tokenizers.
        """
        logger.info("Fitting hybrid tokenizer...")

        # Fit each level
        self.char_tokenizer.fit(texts)
        self.bpe_tokenizer.fit(texts)
        self.phrase_tokenizer.fit(texts)

        # Calculate total features
        self.n_features_ = (
            self.char_tokenizer.n_features_
            + self.bpe_tokenizer.n_features_
            + self.phrase_tokenizer.n_features_
        )

        self.is_fitted = True
        logger.info(f"Hybrid tokenizer fitted with {self.n_features_} total features")

        return self

    def transform(self, texts: Union[str, List[str]]) -> csr_matrix:
        """
        Transform texts using all levels and combine.
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transform")

        if isinstance(texts, str):
            texts = [texts]

        # Get representations from each level
        X_char = self.char_tokenizer.transform(texts)
        X_bpe = self.bpe_tokenizer.transform(texts)
        X_phrase = self.phrase_tokenizer.transform(texts)

        # Apply level weights
        weights = self.config.level_weights
        X_char = X_char * weights[0]
        X_bpe = X_bpe * weights[1]
        X_phrase = X_phrase * weights[2]

        # Concatenate horizontally
        X_combined = hstack([X_char, X_bpe, X_phrase], format="csr")

        # Re-normalize the combined vector
        for i in range(X_combined.shape[0]):
            row = X_combined.getrow(i)
            norm = np.sqrt(row.multiply(row).sum())
            if norm > 0:
                X_combined[i] = row / norm

        return X_combined

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)


def create_tokenizer(
    tokenizer_type: TokenizerType, **kwargs
) -> Union[CharNgramTokenizer, ByteBPETokenizer, SupraTokTokenizer, HybridTokenizer]:
    """
    Factory function to create tokenizers.

    Args:
        tokenizer_type: Type of tokenizer to create
        **kwargs: Configuration parameters

    Returns:
        Configured tokenizer instance
    """
    config = TokenizerConfig(tokenizer_type=tokenizer_type, **kwargs)

    if tokenizer_type == TokenizerType.CHAR_NGRAM:
        return CharNgramTokenizer(config)
    elif tokenizer_type == TokenizerType.BYTE_BPE:
        return ByteBPETokenizer(config)
    elif tokenizer_type == TokenizerType.SUPRATOK:
        return SupraTokTokenizer(config)
    elif tokenizer_type == TokenizerType.HYBRID:
        return HybridTokenizer(config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


if __name__ == "__main__":
    # Quick demonstration
    texts = [
        "Harry Potter and the Philosopher's Stone",
        "Harry Potter and the Chamber of Secrets",
        "The Lord of the Rings is a classic",
        "Machine learning is transforming the world",
        "Natural language processing with transformers",
    ]

    print("Testing modern tokenizers...")
    print("=" * 50)

    # Test each tokenizer type
    for tok_type in [
        TokenizerType.CHAR_NGRAM,
        TokenizerType.BYTE_BPE,
        TokenizerType.SUPRATOK,
        TokenizerType.HYBRID,
    ]:
        print(f"\n{tok_type.value} tokenizer:")
        try:
            tokenizer = create_tokenizer(tok_type, max_features=100)
            X = tokenizer.fit_transform(texts)
            print(f"  Shape: {X.shape}")
            print(f"  Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")
        except Exception as e:
            print(f"  Error: {e}")
