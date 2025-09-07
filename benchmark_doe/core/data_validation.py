#!/usr/bin/env python3
"""
Data Validation and Schema Support for DOE Benchmarks

Ensures data integrity and proper formatting for all dataset types.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates dataset formats and ensures data integrity.
    
    Provides schema validation for:
    - Document collections
    - Query sets
    - Relevance judgments
    - Experiment configurations
    """
    
    @staticmethod
    def validate_documents(documents: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate document collection.
        
        Args:
            documents: List of document strings
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check type
        if not isinstance(documents, list):
            errors.append(f"Documents must be a list, got {type(documents)}")
            return False, errors
            
        # Check not empty
        if len(documents) == 0:
            errors.append("Document collection is empty")
            return False, errors
            
        # Check each document
        for i, doc in enumerate(documents[:100]):  # Check first 100
            if not isinstance(doc, str):
                errors.append(f"Document {i} is not a string: {type(doc)}")
            elif len(doc.strip()) == 0:
                errors.append(f"Document {i} is empty")
            elif len(doc) > 1000000:  # 1MB limit per doc
                errors.append(f"Document {i} is too large: {len(doc)} chars")
                
        # Check for duplicates
        if len(documents) != len(set(documents)):
            n_duplicates = len(documents) - len(set(documents))
            errors.append(f"Found {n_duplicates} duplicate documents")
            
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_queries(queries: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate query collection.
        
        Args:
            queries: List of query strings
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check type
        if not isinstance(queries, list):
            errors.append(f"Queries must be a list, got {type(queries)}")
            return False, errors
            
        # Check not empty
        if len(queries) == 0:
            errors.append("Query collection is empty")
            return False, errors
            
        # Check each query
        for i, query in enumerate(queries[:100]):  # Check first 100
            if not isinstance(query, str):
                errors.append(f"Query {i} is not a string: {type(query)}")
            elif len(query.strip()) == 0:
                errors.append(f"Query {i} is empty")
            elif len(query) > 10000:  # 10KB limit per query
                errors.append(f"Query {i} is too large: {len(query)} chars")
                
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_relevance(
        relevance: Dict[int, List[int]], 
        n_queries: int,
        n_docs: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate relevance judgments.
        
        Args:
            relevance: Dictionary mapping query indices to relevant doc indices
            n_queries: Total number of queries
            n_docs: Total number of documents
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check type
        if not isinstance(relevance, dict):
            errors.append(f"Relevance must be a dict, got {type(relevance)}")
            return False, errors
            
        # Check query indices
        for q_idx in relevance.keys():
            if not isinstance(q_idx, int):
                errors.append(f"Query index must be int, got {type(q_idx)}")
            elif q_idx < 0 or q_idx >= n_queries:
                errors.append(f"Query index {q_idx} out of range [0, {n_queries})")
                
        # Check document indices
        for q_idx, doc_indices in relevance.items():
            if not isinstance(doc_indices, list):
                errors.append(f"Relevance for query {q_idx} must be list")
                continue
                
            for d_idx in doc_indices:
                if not isinstance(d_idx, int):
                    errors.append(f"Doc index must be int, got {type(d_idx)}")
                elif d_idx < 0 or d_idx >= n_docs:
                    errors.append(f"Doc index {d_idx} out of range [0, {n_docs})")
                    
        # Check for reasonable relevance coverage
        if len(relevance) < min(10, n_queries // 10):
            errors.append(f"Too few queries have relevance: {len(relevance)}/{n_queries}")
            
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_dataset(
        documents: List[str],
        queries: List[str],
        relevance: Dict[int, List[int]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate complete dataset.
        
        Args:
            documents: Document collection
            queries: Query collection
            relevance: Relevance judgments
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Validate documents
        valid, errors = DataValidator.validate_documents(documents)
        if not valid:
            report["valid"] = False
            report["errors"].extend([f"[Documents] {e}" for e in errors])
            
        # Validate queries
        valid, errors = DataValidator.validate_queries(queries)
        if not valid:
            report["valid"] = False
            report["errors"].extend([f"[Queries] {e}" for e in errors])
            
        # Validate relevance
        valid, errors = DataValidator.validate_relevance(
            relevance, len(queries), len(documents)
        )
        if not valid:
            report["valid"] = False
            report["errors"].extend([f"[Relevance] {e}" for e in errors])
            
        # Collect statistics
        report["stats"] = {
            "n_documents": len(documents),
            "n_queries": len(queries),
            "n_relevance_judgments": sum(len(v) for v in relevance.values()),
            "queries_with_relevance": len(relevance),
            "avg_relevant_per_query": np.mean([len(v) for v in relevance.values()]) if relevance else 0,
            "avg_doc_length": np.mean([len(d) for d in documents[:1000]]) if documents else 0,
            "avg_query_length": np.mean([len(q) for q in queries[:1000]]) if queries else 0,
        }
        
        # Add warnings for potential issues
        if report["stats"]["avg_relevant_per_query"] < 1:
            report["warnings"].append("Very few relevant documents per query")
            
        if report["stats"]["avg_doc_length"] < 10:
            report["warnings"].append("Documents seem very short")
            
        if report["stats"]["n_queries"] < 10:
            report["warnings"].append("Very few queries for meaningful evaluation")
            
        return report["valid"], report


class ConfigValidator:
    """
    Validates experiment configurations.
    """
    
    REQUIRED_FIELDS = {
        "experiment_id": str,
        "pipeline_architecture": str,
        "n_bits": int,
        "batch_size": int
    }
    
    VALID_PIPELINES = [
        "original_tejas", "fused_char", "fused_byte", 
        "modular", "fused", "fused_v2"
    ]
    
    VALID_N_BITS = [64, 128, 256, 512, 1024]
    VALID_BATCH_SIZES = [100, 500, 1000, 5000, 10000]
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate experiment configuration.
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field, expected_type in cls.REQUIRED_FIELDS.items():
            if field not in config:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(config[field], expected_type):
                errors.append(f"Field {field} must be {expected_type.__name__}, "
                            f"got {type(config[field]).__name__}")
                
        # Validate pipeline
        if "pipeline_architecture" in config:
            if config["pipeline_architecture"] not in cls.VALID_PIPELINES:
                errors.append(f"Invalid pipeline: {config['pipeline_architecture']}. "
                            f"Must be one of {cls.VALID_PIPELINES}")
                
        # Validate n_bits
        if "n_bits" in config:
            if config["n_bits"] not in cls.VALID_N_BITS:
                errors.append(f"Invalid n_bits: {config['n_bits']}. "
                            f"Must be one of {cls.VALID_N_BITS}")
                            
        # Validate batch_size
        if "batch_size" in config:
            if config["batch_size"] not in cls.VALID_BATCH_SIZES:
                errors.append(f"Invalid batch_size: {config['batch_size']}. "
                            f"Must be one of {cls.VALID_BATCH_SIZES}")
                
        return len(errors) == 0, errors
    
    @classmethod
    def validate_batch(cls, configs: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a batch of experiment configurations.
        
        Args:
            configs: List of experiment configurations
            
        Returns:
            Tuple of (all_valid, validation_report)
        """
        report = {
            "valid": True,
            "total_configs": len(configs),
            "valid_configs": 0,
            "invalid_configs": 0,
            "errors": {}
        }
        
        # Check each config
        for i, config in enumerate(configs):
            valid, errors = cls.validate_config(config)
            if valid:
                report["valid_configs"] += 1
            else:
                report["invalid_configs"] += 1
                report["valid"] = False
                exp_id = config.get("experiment_id", f"config_{i}")
                report["errors"][exp_id] = errors
                
        return report["valid"], report


def validate_benchmark_data(
    dataset_path: Path,
    config_path: Optional[Path] = None
) -> bool:
    """
    Validate benchmark dataset and configurations.
    
    Args:
        dataset_path: Path to dataset
        config_path: Optional path to configuration file
        
    Returns:
        True if all validations pass
    """
    all_valid = True
    
    # Load and validate dataset
    try:
        from .dataset_loader import load_benchmark_dataset
        
        documents, queries, relevance = load_benchmark_dataset(
            dataset_type="wikipedia",
            size="10k",
            sample=False
        )
        
        valid, report = DataValidator.validate_dataset(documents, queries, relevance)
        
        if not valid:
            logger.error("Dataset validation failed:")
            for error in report["errors"]:
                logger.error(f"  - {error}")
            all_valid = False
        else:
            logger.info("Dataset validation passed")
            logger.info(f"  Documents: {report['stats']['n_documents']}")
            logger.info(f"  Queries: {report['stats']['n_queries']}")
            logger.info(f"  Relevance judgments: {report['stats']['n_relevance_judgments']}")
            
        # Show warnings
        for warning in report.get("warnings", []):
            logger.warning(f"  ⚠️  {warning}")
            
    except Exception as e:
        logger.error(f"Failed to validate dataset: {e}")
        all_valid = False
        
    # Validate configurations if provided
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                configs = json.load(f)
                
            valid, report = ConfigValidator.validate_batch(configs)
            
            if not valid:
                logger.error("Configuration validation failed:")
                for exp_id, errors in report["errors"].items():
                    logger.error(f"  {exp_id}:")
                    for error in errors:
                        logger.error(f"    - {error}")
                all_valid = False
            else:
                logger.info("Configuration validation passed")
                logger.info(f"  Valid configs: {report['valid_configs']}/{report['total_configs']}")
                
        except Exception as e:
            logger.error(f"Failed to validate configurations: {e}")
            all_valid = False
            
    return all_valid