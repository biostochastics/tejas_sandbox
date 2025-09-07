#!/usr/bin/env python3
"""
Dataset Loading for DOE Benchmarks

This module provides proper dataset loading functionality to replace
the hardcoded fake data that was making all benchmarks invalid.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Handles loading of real datasets for benchmarking.
    
    Supports multiple data formats and sources including:
    - Wikipedia titles
    - MS MARCO passages
    - BEIR datasets (SciFact, NFCorpus, etc.)
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Base directory containing datasets
        """
        if data_dir is None:
            # Default to project's data directory
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
            
        logger.info(f"Initialized DatasetLoader with data_dir: {self.data_dir}")
    
    def load_wikipedia(
        self, 
        size: str = "10k",
        return_format: str = "list"
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Load Wikipedia dataset.
        
        Args:
            size: Dataset size ("10k", "50k", "125k", or "full")
            return_format: Format for documents ("list" or "dict")
            
        Returns:
            Tuple of (documents, queries, relevance)
        """
        wiki_dir = self.data_dir / "wikipedia"
        
        # Map size to file
        size_map = {
            "10k": "wikipedia_10000.txt",
            "50k": "wikipedia_50000.txt", 
            "125k": "wikipedia_125000.txt",
            "250k": "wikipedia_250000.txt",
            "full": "wikipedia_en_20231101_titles.txt"
        }
        
        if size not in size_map:
            raise ValueError(f"Invalid size: {size}. Must be one of {list(size_map.keys())}")
            
        file_path = wiki_dir / size_map[size]
        
        if not file_path.exists():
            # Try pickle format
            pickle_path = self.data_dir / f"wikipedia_sample_{size.replace('k', 'k.pkl')}"
            if pickle_path.exists():
                logger.info(f"Loading Wikipedia from pickle: {pickle_path}")
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        documents = data.get('titles', data.get('documents', []))
                    else:
                        documents = data
            else:
                raise FileNotFoundError(f"Wikipedia dataset not found: {file_path}")
        else:
            logger.info(f"Loading Wikipedia titles from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = [line.strip() for line in f if line.strip()]
        
        # For Wikipedia, we create synthetic queries from titles
        # In production, you'd have real query-document pairs
        n_queries = min(1000, len(documents) // 10)
        queries = []
        relevance = {}
        
        for i in range(n_queries):
            # Create query from document title (simplified)
            doc_idx = i * 10
            if doc_idx < len(documents):
                doc = documents[doc_idx]
                # Simple query generation - take first few words
                query = " ".join(doc.split()[:3])
                queries.append(query)
                # Mark surrounding documents as relevant
                relevance[i] = list(range(max(0, doc_idx-5), 
                                         min(len(documents), doc_idx+5)))
        
        logger.info(f"Loaded {len(documents)} documents and {len(queries)} queries")
        return documents, queries, relevance
    
    def load_msmarco(
        self,
        subset: str = "dev",
        max_docs: int = None
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Load MS MARCO dataset.
        
        Args:
            subset: Which subset to load ("dev" or "train")
            max_docs: Maximum number of documents to load
            
        Returns:
            Tuple of (documents, queries, relevance)
        """
        msmarco_dir = self.data_dir / "msmarco"
        
        # Load documents
        collection_path = msmarco_dir / "collection.tsv"
        if not collection_path.exists():
            raise FileNotFoundError(f"MS MARCO collection not found: {collection_path}")
            
        logger.info(f"Loading MS MARCO collection from: {collection_path}")
        documents = []
        doc_id_map = {}
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_docs and idx >= max_docs:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc_id = parts[0]
                    doc_text = parts[1]
                    documents.append(doc_text)
                    doc_id_map[doc_id] = idx
        
        # Load queries
        queries_path = msmarco_dir / f"queries.{subset}.small.tsv"
        if not queries_path.exists():
            logger.warning(f"Queries file not found: {queries_path}")
            # Create synthetic queries
            n_queries = min(100, len(documents) // 10)
            queries = [f"query {i}" for i in range(n_queries)]
            query_id_map = {str(i): i for i in range(n_queries)}
        else:
            logger.info(f"Loading queries from: {queries_path}")
            queries = []
            query_id_map = {}
            
            with open(queries_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        query_id = parts[0]
                        query_text = parts[1]
                        queries.append(query_text)
                        query_id_map[query_id] = idx
        
        # Load relevance judgments
        qrels_path = msmarco_dir / f"qrels.{subset}.small.tsv"
        relevance = {}
        
        if qrels_path.exists():
            logger.info(f"Loading relevance from: {qrels_path}")
            with open(qrels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        query_id = parts[0]
                        doc_id = parts[2]
                        
                        if query_id in query_id_map and doc_id in doc_id_map:
                            q_idx = query_id_map[query_id]
                            d_idx = doc_id_map[doc_id]
                            
                            if q_idx not in relevance:
                                relevance[q_idx] = []
                            relevance[q_idx].append(d_idx)
        else:
            logger.warning("No relevance file found, creating synthetic relevance")
            # Create synthetic relevance
            for i in range(len(queries)):
                relevance[i] = list(range(i*10, min((i+1)*10, len(documents))))
        
        logger.info(f"Loaded {len(documents)} documents, {len(queries)} queries, "
                   f"{len(relevance)} relevance judgments")
        return documents, queries, relevance
    
    def load_beir(
        self,
        dataset_name: str = "scifact"
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Load BEIR dataset.
        
        Args:
            dataset_name: Name of BEIR dataset (e.g., "scifact", "nfcorpus")
            
        Returns:
            Tuple of (documents, queries, relevance)
        """
        beir_dir = self.data_dir / "beir" / dataset_name
        
        if not beir_dir.exists():
            # Try datasets directory
            beir_dir = Path(__file__).parent.parent.parent / "datasets" / dataset_name
            
        if not beir_dir.exists():
            raise FileNotFoundError(f"BEIR dataset not found: {dataset_name}")
            
        # Load corpus
        corpus_path = beir_dir / "corpus.jsonl"
        documents = []
        doc_id_map = {}
        
        logger.info(f"Loading BEIR corpus from: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                doc_id = data.get('_id', str(idx))
                # Combine title and text
                title = data.get('title', '')
                text = data.get('text', '')
                doc_text = f"{title} {text}".strip()
                documents.append(doc_text)
                doc_id_map[doc_id] = idx
        
        # Load queries
        queries_path = beir_dir / "queries.jsonl"
        queries = []
        query_id_map = {}
        
        logger.info(f"Loading queries from: {queries_path}")
        with open(queries_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                query_id = data.get('_id', str(idx))
                query_text = data.get('text', '')
                queries.append(query_text)
                query_id_map[query_id] = idx
        
        # Load relevance
        relevance = {}
        qrels_dir = beir_dir / "qrels"
        
        # Try test set first, then train
        for subset in ["test.tsv", "train.tsv"]:
            qrels_path = qrels_dir / subset
            if qrels_path.exists():
                logger.info(f"Loading relevance from: {qrels_path}")
                with open(qrels_path, 'r', encoding='utf-8') as f:
                    # Skip header
                    next(f)
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            query_id = parts[0]
                            doc_id = parts[1]
                            score = int(parts[2])
                            
                            if score > 0 and query_id in query_id_map and doc_id in doc_id_map:
                                q_idx = query_id_map[query_id]
                                d_idx = doc_id_map[doc_id]
                                
                                if q_idx not in relevance:
                                    relevance[q_idx] = []
                                relevance[q_idx].append(d_idx)
                break
        
        logger.info(f"Loaded {len(documents)} documents, {len(queries)} queries, "
                   f"{len(relevance)} relevance judgments")
        return documents, queries, relevance
    
    def load_dataset(
        self,
        dataset_type: str = "wikipedia",
        **kwargs
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Load dataset based on type.
        
        Args:
            dataset_type: Type of dataset ("wikipedia", "msmarco", "beir")
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Tuple of (documents, queries, relevance)
        """
        loaders = {
            "wikipedia": self.load_wikipedia,
            "msmarco": self.load_msmarco,
            "beir": self.load_beir
        }
        
        if dataset_type not in loaders:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Must be one of {list(loaders.keys())}")
        
        return loaders[dataset_type](**kwargs)
    
    def get_sample(
        self,
        documents: List[str],
        queries: List[str],
        relevance: Dict,
        n_docs: int = 1000,
        n_queries: int = 100
    ) -> Tuple[List[str], List[str], Dict]:
        """
        Get a sample of the dataset for quick testing.
        
        Args:
            documents: Full document list
            queries: Full query list
            relevance: Full relevance dictionary
            n_docs: Number of documents to sample
            n_queries: Number of queries to sample
            
        Returns:
            Sampled tuple of (documents, queries, relevance)
        """
        # Sample documents
        sampled_docs = documents[:n_docs]
        
        # Sample queries and adjust relevance
        sampled_queries = queries[:n_queries]
        sampled_relevance = {}
        
        for q_idx in range(min(n_queries, len(queries))):
            if q_idx in relevance:
                # Filter relevant docs to only those in sample
                relevant_docs = [d for d in relevance[q_idx] if d < n_docs]
                if relevant_docs:
                    sampled_relevance[q_idx] = relevant_docs
        
        logger.info(f"Sampled {len(sampled_docs)} documents, "
                   f"{len(sampled_queries)} queries")
        
        return sampled_docs, sampled_queries, sampled_relevance


# Convenience function
def load_benchmark_dataset(
    dataset_type: str = "wikipedia",
    size: str = "10k",
    sample: bool = False,
    n_docs: int = 1000,
    n_queries: int = 100,
    **kwargs
) -> Tuple[List[str], List[str], Dict]:
    """
    Load dataset for benchmarking.
    
    Args:
        dataset_type: Type of dataset
        size: Size variant (for wikipedia)
        sample: Whether to return a small sample
        n_docs: Number of documents to sample (if sample=True)
        n_queries: Number of queries to sample (if sample=True)
        **kwargs: Additional loader arguments
        
    Returns:
        Tuple of (documents, queries, relevance)
    """
    loader = DatasetLoader()
    
    # Separate sampling params from loader params
    loader_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['n_docs', 'n_queries']}
    
    if dataset_type == "wikipedia":
        docs, queries, relevance = loader.load_wikipedia(size=size)
    elif dataset_type == "msmarco":
        # MS MARCO specific params
        subset = loader_kwargs.pop('subset', 'dev')
        max_docs = loader_kwargs.pop('max_docs', None)
        docs, queries, relevance = loader.load_msmarco(subset=subset, max_docs=max_docs)
    elif dataset_type == "beir":
        # BEIR specific params
        dataset_name = loader_kwargs.pop('dataset_name', 'scifact')
        docs, queries, relevance = loader.load_beir(dataset_name=dataset_name)
    else:
        docs, queries, relevance = loader.load_dataset(dataset_type, **loader_kwargs)
    
    if sample:
        docs, queries, relevance = loader.get_sample(
            docs, queries, relevance,
            n_docs=n_docs,
            n_queries=n_queries
        )
    
    return docs, queries, relevance