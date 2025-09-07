"""
Pure NumPy TF-IDF implementation to eliminate sklearn dependency
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple
from collections import Counter
import re


class TfidfVectorizerNumpy:
    """
    Pure NumPy implementation of TF-IDF vectorizer with sparse matrix output.
    Replaces sklearn.feature_extraction.text.TfidfVectorizer
    """

    def __init__(
        self,
        max_features: int = 10000,
        analyzer: str = "word",
        ngram_range: Tuple[int, int] = (1, 1),
        lowercase: bool = True,
        min_df: int = 1,
        max_df: float = 1.0,
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        dtype: type = np.float32,
        **kwargs,
    ):
        """
        Initialize TF-IDF vectorizer.

        Args:
            max_features: Maximum number of features (vocabulary size)
            analyzer: 'word' or 'char' for tokenization
            ngram_range: (min_n, max_n) for n-gram extraction
            lowercase: Convert text to lowercase
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as fraction)
            use_idf: Use inverse document frequency weighting
            smooth_idf: Add 1 to document frequencies for smoothing
            sublinear_tf: Use logarithmic term frequency
            dtype: Data type for output matrix
            **kwargs: Additional parameters (ignored for compatibility)
        """
        self.max_features = max_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.dtype = dtype

        self.vocabulary_ = {}
        self.idf_ = None
        self.n_features_ = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words or characters."""
        if self.lowercase:
            text = text.lower()

        if self.analyzer == "word":
            # Simple word tokenization
            tokens = re.findall(r"\b\w+\b", text)
        elif self.analyzer == "char":
            # Character tokenization
            tokens = list(text)
        else:
            raise ValueError(f"Unknown analyzer: {self.analyzer}")

        return tokens

    def _get_ngrams(self, tokens: List[str]) -> List[str]:
        """Extract n-grams from tokens."""
        min_n, max_n = self.ngram_range
        ngrams = []

        for n in range(min_n, max_n + 1):
            if self.analyzer == "word":
                # Word n-grams
                for i in range(len(tokens) - n + 1):
                    ngram = " ".join(tokens[i : i + n])
                    ngrams.append(ngram)
            else:
                # Character n-grams
                for i in range(len(tokens) - n + 1):
                    ngram = "".join(tokens[i : i + n])
                    ngrams.append(ngram)

        return ngrams if ngrams else tokens

    def fit(self, texts: List[str]) -> "TfidfVectorizerNumpy":
        """
        Learn vocabulary and IDF from texts.

        Args:
            texts: List of text documents

        Returns:
            Self for chaining
        """
        n_docs = len(texts)

        # Build vocabulary
        doc_freq = Counter()
        all_tokens = []

        for text in texts:
            tokens = self._tokenize(text)
            ngrams = self._get_ngrams(tokens)
            unique_ngrams = set(ngrams)

            for ngram in unique_ngrams:
                doc_freq[ngram] += 1
            all_tokens.extend(ngrams)

        # Filter by document frequency
        min_df_count = (
            self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        )
        max_df_count = (
            int(self.max_df * n_docs) if isinstance(self.max_df, float) else self.max_df
        )

        # Get term frequencies for ranking
        term_freq = Counter(all_tokens)

        # Filter and sort by frequency
        filtered_terms = [
            term for term, df in doc_freq.items() if min_df_count <= df <= max_df_count
        ]
        filtered_terms.sort(key=lambda x: term_freq[x], reverse=True)

        # Limit to max_features
        if self.max_features and len(filtered_terms) > self.max_features:
            filtered_terms = filtered_terms[: self.max_features]

        # Create vocabulary mapping
        self.vocabulary_ = {term: idx for idx, term in enumerate(filtered_terms)}
        self.n_features_ = len(self.vocabulary_)

        # Calculate IDF
        if self.use_idf:
            self.idf_ = np.zeros(self.n_features_)
            for term, idx in self.vocabulary_.items():
                df = doc_freq[term]
                if self.smooth_idf:
                    idf = np.log((n_docs + 1) / (df + 1)) + 1
                else:
                    idf = np.log(n_docs / df) + 1
                self.idf_[idx] = idf

        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to TF-IDF sparse matrix.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF sparse matrix (CSR format)
        """
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")

        n_docs = len(texts)

        # Build sparse matrix using COO format (then convert to CSR)
        # Lists to store non-zero entries
        row_indices = []
        col_indices = []
        values = []

        for doc_idx, text in enumerate(texts):
            tokens = self._tokenize(text)
            ngrams = self._get_ngrams(tokens)

            # Count term frequencies
            tf = Counter(ngrams)

            # Temporary storage for this document's values
            doc_values = {}

            # Calculate TF-IDF for each term
            for term, count in tf.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]

                    # Calculate TF
                    if self.sublinear_tf:
                        tf_value = 1 + np.log(count)
                    else:
                        tf_value = count

                    # Apply IDF
                    if self.use_idf:
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
                dtype=self.dtype,
            )
        else:
            # Empty matrix
            X = csr_matrix((n_docs, self.n_features_), dtype=self.dtype)

        return X

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """
        Fit vocabulary and transform texts in one step.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF sparse matrix (CSR format)
        """
        self.fit(texts)
        return self.transform(texts)


class TfidfVectorizerNumbaBoosted:
    """
    NumPy TF-IDF with optional Numba acceleration for performance-critical parts.
    Falls back to pure NumPy if Numba is not available.
    """

    def __init__(self, **kwargs):
        """Initialize with same parameters as TfidfVectorizerNumpy."""
        self.base_vectorizer = TfidfVectorizerNumpy(**kwargs)
        self.use_numba = False

        # Try to import and use numba if available
        try:
            import numba

            self.use_numba = True
            self._setup_numba_functions()
        except ImportError:
            pass

    def _setup_numba_functions(self):
        """Setup Numba-accelerated functions."""
        import numba

        @numba.jit(nopython=True)
        def compute_tfidf_values_numba(counts, idf_values):
            """Numba-accelerated TF-IDF value computation."""
            n_terms = len(counts)
            values = np.zeros(n_terms, dtype=np.float32)

            for i in range(n_terms):
                if counts[i] > 0:
                    values[i] = counts[i] * idf_values[i]

            # L2 normalization
            norm = 0.0
            for i in range(n_terms):
                norm += values[i] * values[i]
            norm = np.sqrt(norm)

            if norm > 0:
                for i in range(n_terms):
                    values[i] /= norm

            return values

        self._compute_tfidf_values_numba = compute_tfidf_values_numba

    def fit(self, texts: List[str]) -> "TfidfVectorizerNumbaBoosted":
        """Fit vocabulary and IDF."""
        self.base_vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """Transform texts to TF-IDF sparse matrix."""
        # For now, just use base implementation
        # Numba optimization for sparse matrices is complex
        return self.base_vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    @property
    def vocabulary_(self):
        return self.base_vectorizer.vocabulary_

    @property
    def idf_(self):
        return self.base_vectorizer.idf_
