#!/usr/bin/env python3
"""
Unified TEJAS Benchmark Suite V3 - Multi-Dataset Edition
=========================================================
Comprehensive multi-encoder, multi-dataset benchmarking with automatic table generation.
Tests all encoder implementations across BEIR, MS MARCO, and Wikipedia datasets.
"""

import sys
import time
import json
import gc
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Type, Protocol
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import importlib
import inspect
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))


# ============================================================================
# Data Classes for Type Safety
# ============================================================================

class EncoderStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    NOT_FOUND = "not_found"


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    dataset_name: str
    encoder_name: str
    run_number: int
    status: EncoderStatus
    train_time: Optional[float] = None
    encode_speed: Optional[float] = None
    search_speed: Optional[float] = None
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    recall_at_1: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    doc_count: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['status'] = self.status.value
        return d


# ============================================================================
# Dataset Loaders
# ============================================================================

class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load_documents(self, path: str, limit: Optional[int] = None) -> List[str]:
        """Load documents from dataset."""
        pass
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return dataset name."""
        pass


class WikipediaLoader(BaseDatasetLoader):
    """Loader for Wikipedia dataset (text format)."""
    
    def load_documents(self, path: str, limit: Optional[int] = None) -> List[str]:
        """Load Wikipedia titles from text file."""
        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = line.strip()
                if doc:
                    documents.append(doc)
                    if limit and len(documents) >= limit:
                        break
        
        print(f"  📚 Loaded {len(documents):,} Wikipedia documents")
        return documents
    
    def get_dataset_name(self) -> str:
        return "Wikipedia"


class MSMARCOLoader(BaseDatasetLoader):
    """Loader for MS MARCO dataset (TSV format)."""
    
    def load_documents(self, path: str, limit: Optional[int] = None) -> List[str]:
        """Load MS MARCO passages from TSV file."""
        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # Format: doc_id \t passage_text
                    documents.append(parts[1])
                    if limit and len(documents) >= limit:
                        break
        
        print(f"  📚 Loaded {len(documents):,} MS MARCO documents")
        return documents
    
    def get_dataset_name(self) -> str:
        return "MS-MARCO"


class BEIRLoader(BaseDatasetLoader):
    """Loader for BEIR datasets (JSONL format)."""
    
    def load_documents(self, path: str, limit: Optional[int] = None) -> List[str]:
        """Load BEIR corpus from JSONL file."""
        documents = []
        
        # BEIR corpus format: {"_id": "...", "title": "...", "text": "..."}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc_json = json.loads(line.strip())
                    # Combine title and text
                    title = doc_json.get('title', '').strip()
                    text = doc_json.get('text', '').strip()
                    
                    if title and text:
                        document = f"{title}. {text}"
                    elif title:
                        document = title
                    elif text:
                        document = text
                    else:
                        continue
                    
                    documents.append(document)
                    if limit and len(documents) >= limit:
                        break
                except json.JSONDecodeError:
                    continue
        
        print(f"  📚 Loaded {len(documents):,} BEIR documents")
        return documents
    
    def get_dataset_name(self) -> str:
        return "BEIR-SciFact"


# ============================================================================
# Encoder Registry with Dynamic Discovery
# ============================================================================

class EncoderRegistry:
    """Dynamic encoder discovery and registration."""
    
    # Define all known encoders with their import paths
    ENCODER_DEFINITIONS = [
        # Original implementations
        {
            "name": "Original Tejas",
            "module": "core.encoder",
            "class": "GoldenRatioEncoder",
            "params": {}
        },
        {
            "name": "Original Tejas (Legacy)",
            "module": "original_tejas.core.encoder",
            "class": "GoldenRatioEncoder",
            "params": {}
        },
        # Randomized SVD variants
        {
            "name": "Tejas + RSVD",
            "module": "core.randomized_svd_encoder",
            "class": "RandomizedSVDEncoder",
            "params": {"svd_n_iter": 3}
        },
        # Streamlined variants
        {
            "name": "Tejas-S (Streamlined)",
            "module": "core.tejas_s_encoder",
            "class": "TejasSEncoder",
            "params": {"memory_limit_gb": 16.0, "cache_size": 1000}
        },
        {
            "name": "Streamlined Fused",
            "module": "core.streamlined_fused_encoder",
            "class": "StreamlinedFusedEncoder",
            "params": {}
        },
        # Fused variants
        {
            "name": "Tejas-F (Fused)",
            "module": "core.tejas_f_encoder",
            "class": "TejasFEncoder",
            "params": {}
        },
        {
            "name": "Tejas-F+ (Enhanced)",
            "module": "core.tejas_f_encoder",
            "class": "TejasFPlusEncoder",
            "params": {"itq_iterations": 50}
        }
    ]
    
    def __init__(self):
        self.encoders = {}
        self._discover_encoders()
    
    def _discover_encoders(self):
        """Dynamically discover and load available encoders."""
        print("\n🔍 Discovering Encoders...")
        print("-" * 60)
        
        for encoder_def in self.ENCODER_DEFINITIONS:
            try:
                module = importlib.import_module(encoder_def["module"])
                encoder_class = getattr(module, encoder_def["class"])
                
                # Verify it's a valid encoder class
                if inspect.isclass(encoder_class):
                    self.encoders[encoder_def["name"]] = {
                        "class": encoder_class,
                        "module": encoder_def["module"],
                        "default_params": encoder_def["params"]
                    }
                    print(f"  ✓ {encoder_def['name']:<30} [{encoder_def['module']}]")
            except (ImportError, AttributeError) as e:
                print(f"  ✗ {encoder_def['name']:<30} [Not available: {str(e)[:40]}]")
        
        print(f"\n  Total encoders available: {len(self.encoders)}")
        print("-" * 60)
    
    def get_encoder(self, name: str, n_bits: int = 128, max_features: int = 5000) -> Optional[Tuple[Any, Dict]]:
        """Get an encoder instance with configuration."""
        if name not in self.encoders:
            return None, None
        
        encoder_info = self.encoders[name]
        encoder_class = encoder_info["class"]
        
        # Build parameters
        params = {
            "n_bits": n_bits,
            "max_features": max_features,
            **encoder_info["default_params"]
        }
        
        # Remove unsupported parameters based on encoder type
        if "GoldenRatioEncoder" in encoder_class.__name__:
            params.pop("memory_limit_gb", None)
            params.pop("cache_size", None)
            params.pop("svd_n_iter", None)
            params.pop("itq_iterations", None)
        
        return encoder_class, params
    
    def list_encoders(self) -> List[str]:
        """Get list of available encoder names."""
        return list(self.encoders.keys())


# ============================================================================
# Resource Profiler
# ============================================================================

class ResourceProfiler:
    """Profile resource usage during execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        
    def start(self):
        """Start profiling."""
        gc.collect()
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = self.start_memory
        
    def sample(self) -> Dict:
        """Take a resource sample."""
        current_memory = self.process.memory_info().rss / (1024 * 1024)
        self.peak_memory = max(self.peak_memory, current_memory)
        
        return {
            'time': time.time() - self.start_time,
            'memory_mb': current_memory,
            'cpu_percent': self.process.cpu_percent()
        }
        
    def stop(self) -> Dict:
        """Stop profiling and return results."""
        elapsed = time.time() - self.start_time
        end_memory = self.process.memory_info().rss / (1024 * 1024)
        
        return {
            'elapsed_time': elapsed,
            'memory_start_mb': self.start_memory,
            'memory_end_mb': end_memory,
            'memory_peak_mb': self.peak_memory,
            'memory_delta_mb': end_memory - self.start_memory
        }


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_single_benchmark(
    dataset_name: str,
    encoder_name: str,
    encoder_class: Type,
    encoder_params: Dict,
    documents: List[str],
    run_number: int,
    test_size: int = 1000,
    n_queries: int = 100
) -> BenchmarkResult:
    """Run a single benchmark test."""
    profiler = ResourceProfiler()
    
    try:
        # Initialize encoder
        encoder = encoder_class(**encoder_params)
        
        # Training phase
        profiler.start()
        if hasattr(encoder, 'fit'):
            encoder.fit(documents)
        else:
            encoder.train(documents)
        train_profile = profiler.stop()
        
        # Encoding phase - adjust test size for small datasets
        actual_test_size = min(test_size, len(documents))
        test_docs = documents[:actual_test_size]
        
        profiler.start()
        if hasattr(encoder, 'transform'):
            fingerprints = encoder.transform(test_docs)
        elif hasattr(encoder, 'encode'):
            fingerprints = encoder.encode(test_docs)
        else:
            raise AttributeError("Encoder has no transform or encode method")
        encode_profile = profiler.stop()
        
        # Search phase - adjust queries for small datasets
        actual_queries = min(n_queries, len(test_docs))
        search_times = []
        recalls = {"at_1": [], "at_5": [], "at_10": []}
        
        query_indices = np.random.choice(len(test_docs), actual_queries, replace=False)
        
        for qi in query_indices:
            query_doc = [test_docs[qi]]
            query_start = time.time()
            
            # Encode query
            if hasattr(encoder, 'transform'):
                query_fp = encoder.transform(query_doc)
            else:
                query_fp = encoder.encode(query_doc)
            
            # Convert to numpy if needed
            if hasattr(fingerprints, 'numpy'):
                fingerprints_np = fingerprints.numpy()
            else:
                fingerprints_np = fingerprints
                
            if hasattr(query_fp, 'numpy'):
                query_fp_np = query_fp.numpy()
            else:
                query_fp_np = query_fp
            
            # Compute Hamming distances
            if query_fp_np.ndim > 1:
                query_fp_np = query_fp_np[0]
            
            if fingerprints_np.ndim == 1:
                distances = np.array([np.sum(query_fp_np != fingerprints_np)])
            else:
                distances = np.sum(query_fp_np != fingerprints_np, axis=1)
            
            # Get top-k
            k = min(10, len(distances))
            indices = np.argpartition(distances, min(k-1, len(distances)-1))[:k]
            indices = indices[np.argsort(distances[indices])]
            
            search_times.append(time.time() - query_start)
            
            # Calculate recall
            recalls["at_1"].append(1.0 if qi in indices[:1] else 0.0)
            recalls["at_5"].append(1.0 if qi in indices[:min(5, len(indices))] else 0.0)
            recalls["at_10"].append(1.0 if qi in indices[:min(10, len(indices))] else 0.0)
        
        # Calculate metrics
        train_time = train_profile['elapsed_time']
        encode_speed = len(test_docs) / encode_profile['elapsed_time'] if encode_profile['elapsed_time'] > 0 else 0
        search_speed = len(query_indices) / np.sum(search_times) if np.sum(search_times) > 0 else 0
        
        # Cleanup
        del encoder
        del fingerprints
        gc.collect()
        
        return BenchmarkResult(
            dataset_name=dataset_name,
            encoder_name=encoder_name,
            run_number=run_number,
            status=EncoderStatus.SUCCESS,
            train_time=train_time,
            encode_speed=encode_speed,
            search_speed=search_speed,
            memory_mb=train_profile['memory_peak_mb'],
            cpu_percent=profiler.process.cpu_percent(),
            recall_at_1=np.mean(recalls["at_1"]) * 100,
            recall_at_5=np.mean(recalls["at_5"]) * 100,
            recall_at_10=np.mean(recalls["at_10"]) * 100,
            doc_count=len(documents)
        )
        
    except MemoryError as e:
        return BenchmarkResult(
            dataset_name=dataset_name,
            encoder_name=encoder_name,
            run_number=run_number,
            status=EncoderStatus.MEMORY_ERROR,
            doc_count=len(documents),
            error=str(e)[:200]
        )
    except Exception as e:
        return BenchmarkResult(
            dataset_name=dataset_name,
            encoder_name=encoder_name,
            run_number=run_number,
            status=EncoderStatus.FAILED,
            doc_count=len(documents),
            error=f"{type(e).__name__}: {str(e)[:200]}"
        )


# ============================================================================
# Report Generator with Dataset-Specific Tables
# ============================================================================

class DatasetReportGenerator:
    """Generate dataset-specific benchmark reports."""
    
    def __init__(self, dataset_name: str, results: List[BenchmarkResult]):
        self.dataset_name = dataset_name
        self.results = [r for r in results if r.dataset_name == dataset_name]
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for r in self.results:
            if r.status == EncoderStatus.SUCCESS:
                data.append({
                    'Encoder': r.encoder_name,
                    'Run': r.run_number,
                    'Train(s)': r.train_time,
                    'Encode(d/s)': r.encode_speed,
                    'Search(q/s)': r.search_speed,
                    'Memory(MB)': r.memory_mb,
                    'CPU(%)': r.cpu_percent,
                    'R@1(%)': r.recall_at_1,
                    'R@5(%)': r.recall_at_5,
                    'R@10(%)': r.recall_at_10
                })
        return pd.DataFrame(data)
    
    def generate_summary_table(self) -> str:
        """Generate summary statistics table."""
        if self.df.empty:
            return f"No successful benchmark results for {self.dataset_name}."
        
        # Group by encoder and calculate statistics
        summary = self.df.groupby('Encoder').agg({
            'Train(s)': ['mean', 'std', 'min', 'max'],
            'Encode(d/s)': ['mean', 'std', 'min', 'max'],
            'Search(q/s)': ['mean', 'std', 'min', 'max'],
            'Memory(MB)': ['mean', 'std', 'min', 'max'],
            'R@10(%)': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        return summary.to_string()
    
    def generate_ranking_table(self) -> pd.DataFrame:
        """Generate performance ranking table."""
        if self.df.empty:
            return pd.DataFrame()
        
        # Calculate mean metrics per encoder
        rankings = self.df.groupby('Encoder').agg({
            'Train(s)': 'mean',
            'Encode(d/s)': 'mean',
            'Search(q/s)': 'mean',
            'Memory(MB)': 'mean',
            'R@10(%)': 'mean'
        }).round(2)
        
        # Sort by encoding speed (primary metric)
        rankings = rankings.sort_values('Encode(d/s)', ascending=False)
        
        # Add rank column
        rankings.insert(0, 'Rank', range(1, len(rankings) + 1))
        
        return rankings
    
    def generate_markdown_report(self, config: Dict) -> str:
        """Generate complete markdown report for dataset."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        doc_count = self.results[0].doc_count if self.results else 0
        
        # Get rankings
        rankings = self.generate_ranking_table()
        
        report = f"""# {self.dataset_name} Benchmark Report

## Configuration
- **Timestamp**: {timestamp}
- **Dataset**: {self.dataset_name}
- **Documents**: {doc_count:,}
- **Runs per encoder**: {config.get('n_runs', 'N/A')}
- **Bit size**: {config.get('n_bits', 'N/A')}
- **Max features**: {config.get('max_features', 'N/A')}

## Performance Rankings

"""
        
        if not rankings.empty:
            # Convert DataFrame to markdown table manually
            report += self._dataframe_to_markdown(rankings) + "\n\n"
        else:
            report += "No successful runs to report.\n\n"
        
        report += f"""## Summary Statistics

{self.generate_summary_table()}

## Winners by Category

{self._generate_winners_section()}

---
*Generated by TEJAS Benchmark Suite V3 - Multi-Dataset Edition*
"""
        return report
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to markdown table without tabulate dependency."""
        if df.empty:
            return "No data available."
        
        # Create header
        headers = list(df.columns)
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "|" + "---|" * len(headers)
        
        # Create rows
        rows = []
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(f"{row[col]:.2f}" if isinstance(row[col], float) else str(row[col]) for col in headers) + " |"
            rows.append(row_str)
        
        # Combine
        table = header_line + "\n" + separator + "\n" + "\n".join(rows)
        return table
    
    def _generate_winners_section(self) -> str:
        """Generate winners by category."""
        if self.df.empty:
            return "No data available for winner analysis."
        
        means = self.df.groupby('Encoder').mean()
        
        winners = []
        
        # Fastest training
        if 'Train(s)' in means.columns:
            fastest_train = means['Train(s)'].idxmin()
            winners.append(f"- **Fastest Training**: {fastest_train} ({means.loc[fastest_train, 'Train(s)']:.2f}s)")
        
        # Fastest encoding
        if 'Encode(d/s)' in means.columns:
            fastest_encode = means['Encode(d/s)'].idxmax()
            winners.append(f"- **Fastest Encoding**: {fastest_encode} ({means.loc[fastest_encode, 'Encode(d/s)']:.0f} docs/s)")
        
        # Fastest search
        if 'Search(q/s)' in means.columns:
            fastest_search = means['Search(q/s)'].idxmax()
            winners.append(f"- **Fastest Search**: {fastest_search} ({means.loc[fastest_search, 'Search(q/s)']:.0f} queries/s)")
        
        # Lowest memory
        if 'Memory(MB)' in means.columns:
            lowest_memory = means['Memory(MB)'].idxmin()
            winners.append(f"- **Lowest Memory**: {lowest_memory} ({means.loc[lowest_memory, 'Memory(MB)']:.0f} MB)")
        
        # Best recall
        if 'R@10(%)' in means.columns:
            best_recall = means['R@10(%)'].idxmax()
            winners.append(f"- **Best Recall@10**: {best_recall} ({means.loc[best_recall, 'R@10(%)']:.1f}%)")
        
        return '\n'.join(winners) if winners else "No winner data available."
    
    def save_results(self, output_dir: Path):
        """Save results to dataset-specific directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_file = output_dir / f"benchmark_results_{timestamp}.csv"
        self.df.to_csv(csv_file, index=False)
        print(f"    ✓ CSV saved: {csv_file}")
        
        # Save JSON
        json_file = output_dir / f"benchmark_results_{timestamp}.json"
        json_data = [r.to_dict() for r in self.results if r.dataset_name == self.dataset_name]
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"    ✓ JSON saved: {json_file}")
        
        return csv_file, json_file


# ============================================================================
# Combined Summary Generator
# ============================================================================

class CombinedSummaryGenerator:
    """Generate combined summary across all datasets."""
    
    def __init__(self, all_results: List[BenchmarkResult]):
        self.all_results = all_results
        self.datasets = list(set(r.dataset_name for r in all_results))
    
    def generate_cross_dataset_comparison(self) -> str:
        """Generate comparison table across all datasets."""
        
        # Create DataFrame with all successful results
        data = []
        for r in self.all_results:
            if r.status == EncoderStatus.SUCCESS:
                data.append({
                    'Dataset': r.dataset_name,
                    'Encoder': r.encoder_name,
                    'Train(s)': r.train_time,
                    'Encode(d/s)': r.encode_speed,
                    'Search(q/s)': r.search_speed,
                    'Memory(MB)': r.memory_mb,
                    'R@10(%)': r.recall_at_10
                })
        
        if not data:
            return "No successful runs across any dataset."
        
        df = pd.DataFrame(data)
        
        # Create pivot table: Encoders vs Datasets for key metrics
        report = "# Cross-Dataset Performance Comparison\n\n"
        
        for metric in ['Encode(d/s)', 'R@10(%)', 'Memory(MB)']:
            pivot = df.pivot_table(
                values=metric,
                index='Encoder',
                columns='Dataset',
                aggfunc='mean'
            ).round(2)
            
            report += f"\n## {metric} by Dataset\n\n"
            report += self._pivot_to_markdown(pivot) + "\n"
        
        # Overall best performers
        report += "\n## Overall Best Performers (averaged across datasets)\n\n"
        
        overall_means = df.groupby('Encoder').agg({
            'Encode(d/s)': 'mean',
            'Search(q/s)': 'mean',
            'R@10(%)': 'mean',
            'Memory(MB)': 'mean'
        }).round(2)
        
        overall_means = overall_means.sort_values('Encode(d/s)', ascending=False)
        report += self._dataframe_to_markdown_with_index(overall_means) + "\n"
        
        return report
    
    def _pivot_to_markdown(self, pivot: pd.DataFrame) -> str:
        """Convert pivot table to markdown."""
        if pivot.empty:
            return "No data available."
        
        # Create header with index name and column names
        headers = [pivot.index.name or "Encoder"] + list(pivot.columns)
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "|" + "---|" * len(headers)
        
        # Create rows
        rows = []
        for idx in pivot.index:
            row_data = [str(idx)] + [f"{pivot.loc[idx, col]:.2f}" if not pd.isna(pivot.loc[idx, col]) else "N/A" for col in pivot.columns]
            row_str = "| " + " | ".join(row_data) + " |"
            rows.append(row_str)
        
        # Combine
        table = header_line + "\n" + separator + "\n" + "\n".join(rows)
        return table
    
    def _dataframe_to_markdown_with_index(self, df: pd.DataFrame) -> str:
        """Convert DataFrame with index to markdown."""
        if df.empty:
            return "No data available."
        
        # Create header with index name and column names
        headers = [df.index.name or "Encoder"] + list(df.columns)
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "|" + "---|" * len(headers)
        
        # Create rows
        rows = []
        for idx in df.index:
            row_data = [str(idx)] + [f"{df.loc[idx, col]:.2f}" for col in df.columns]
            row_str = "| " + " | ".join(row_data) + " |"
            rows.append(row_str)
        
        # Combine
        table = header_line + "\n" + separator + "\n" + "\n".join(rows)
        return table
    
    def save_combined_summary(self, output_dir: Path):
        """Save combined summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = self.generate_cross_dataset_comparison()
        
        summary_file = output_dir / f"combined_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"  ✓ Combined summary saved: {summary_file}")
        return summary_file


# ============================================================================
# Main Multi-Dataset Benchmark Orchestrator
# ============================================================================

def run_multi_dataset_benchmark(
    n_runs: int = 10,
    n_bits: int = 128,
    max_features: int = 5000,
    test_size: int = 1000,
    n_queries: int = 100
):
    """Run comprehensive multi-dataset, multi-encoder benchmark."""
    
    print("\n" + "="*80)
    print(" TEJAS MULTI-DATASET BENCHMARK SUITE V3")
    print("="*80)
    print(f"Configuration:")
    print(f"  • Runs per encoder: {n_runs}")
    print(f"  • Bit size: {n_bits}")
    print(f"  • Max features: {max_features:,}")
    print(f"  • Test size: {test_size:,}")
    print(f"  • Queries: {n_queries}")
    print("="*80)
    
    # Define datasets
    datasets = [
        {
            'name': 'Wikipedia',
            'loader': WikipediaLoader(),
            'path': 'data/wikipedia/wikipedia_125000.txt',
            'limit': 100000
        },
        {
            'name': 'MS-MARCO',
            'loader': MSMARCOLoader(),
            'path': 'data/msmarco/collection.tsv',
            'limit': None  # Use all available
        },
        {
            'name': 'BEIR-SciFact',
            'loader': BEIRLoader(),
            'path': 'data/beir/scifact/corpus.jsonl',
            'limit': None  # Use all available
        }
    ]
    
    # Initialize encoder registry
    registry = EncoderRegistry()
    encoder_names = registry.list_encoders()
    
    if not encoder_names:
        print("  ❌ No encoders available!")
        return
    
    # Calculate total tests
    total_tests = len(datasets) * len(encoder_names) * n_runs
    print(f"\n📊 Total tests to run: {len(datasets)} datasets × {len(encoder_names)} encoders × {n_runs} runs = {total_tests} tests")
    print(f"⏱️  Estimated time: ~4 hours")
    
    # Run benchmarks for each dataset
    all_results = []
    
    for dataset_config in datasets:
        dataset_name = dataset_config['name']
        loader = dataset_config['loader']
        path = dataset_config['path']
        limit = dataset_config['limit']
        
        print(f"\n{'='*80}")
        print(f" DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        # Load dataset
        try:
            documents = loader.load_documents(path, limit)
            if not documents:
                print(f"  ❌ No documents loaded from {path}")
                continue
        except Exception as e:
            print(f"  ❌ Error loading dataset: {e}")
            continue
        
        print(f"  Documents loaded: {len(documents):,}")
        
        # Create dataset-specific output directory
        dataset_output_dir = Path("benchmark_results") / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run benchmarks for this dataset
        dataset_results = []
        
        for encoder_name in encoder_names:
            print(f"\n  📈 Testing: {encoder_name}")
            print("  " + "-" * 40)
            
            encoder_class, params = registry.get_encoder(encoder_name, n_bits, max_features)
            if encoder_class is None:
                print(f"    ✗ Encoder not available")
                continue
            
            for run_num in range(1, n_runs + 1):
                print(f"    Run {run_num}/{n_runs}...", end=' ', flush=True)
                
                result = run_single_benchmark(
                    dataset_name=dataset_name,
                    encoder_name=encoder_name,
                    encoder_class=encoder_class,
                    encoder_params=params,
                    documents=documents,
                    run_number=run_num,
                    test_size=test_size,
                    n_queries=n_queries
                )
                
                dataset_results.append(result)
                all_results.append(result)
                
                if result.status == EncoderStatus.SUCCESS:
                    print(f"✓ Train: {result.train_time:.2f}s, "
                          f"Encode: {result.encode_speed:.0f} d/s, "
                          f"Search: {result.search_speed:.0f} q/s, "
                          f"R@10: {result.recall_at_10:.1f}%")
                else:
                    print(f"✗ {result.status.value}")
                
                gc.collect()
                time.sleep(0.5)
        
        # Generate dataset-specific reports
        print(f"\n  📊 Generating {dataset_name} Reports...")
        generator = DatasetReportGenerator(dataset_name, dataset_results)
        
        # Save dataset results
        generator.save_results(dataset_output_dir)
        
        # Generate and save markdown report
        config = {
            'n_runs': n_runs,
            'n_bits': n_bits,
            'max_features': max_features,
            'test_size': test_size,
            'n_queries': n_queries
        }
        
        markdown_report = generator.generate_markdown_report(config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = dataset_output_dir / f"benchmark_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(markdown_report)
        print(f"    ✓ Markdown report: {md_file}")
        
        # Print dataset summary
        print(f"\n  {dataset_name} Performance Rankings:")
        rankings = generator.generate_ranking_table()
        if not rankings.empty:
            print(rankings.to_string())
    
    # Generate combined summary across all datasets
    print("\n" + "="*80)
    print(" GENERATING COMBINED SUMMARY")
    print("="*80)
    
    combined_generator = CombinedSummaryGenerator(all_results)
    combined_generator.save_combined_summary(Path("benchmark_results"))
    
    # Final summary
    print("\n" + "="*80)
    print(" MULTI-DATASET BENCHMARK COMPLETE!")
    print("="*80)
    print(f"Results saved in: benchmark_results/")
    print(f"  • Wikipedia results: benchmark_results/Wikipedia/")
    print(f"  • MS-MARCO results: benchmark_results/MS-MARCO/")
    print(f"  • BEIR-SciFact results: benchmark_results/BEIR-SciFact/")
    print(f"  • Combined summary: benchmark_results/combined_summary_*.md")
    
    # Statistics
    successful_runs = sum(1 for r in all_results if r.status == EncoderStatus.SUCCESS)
    failed_runs = sum(1 for r in all_results if r.status != EncoderStatus.SUCCESS)
    
    print(f"\nRun Statistics:")
    print(f"  • Total runs: {len(all_results)}")
    print(f"  • Successful: {successful_runs} ({successful_runs/len(all_results)*100:.1f}%)")
    print(f"  • Failed: {failed_runs} ({failed_runs/len(all_results)*100:.1f}%)")
    
    return all_results


# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TEJAS Multi-Dataset Benchmark Suite V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 2 runs
  python unified_benchmark_v3.py --runs 2
  
  # Standard benchmark with 10 runs
  python unified_benchmark_v3.py --runs 10
  
  # Custom configuration
  python unified_benchmark_v3.py --runs 10 --bits 256 --features 10000
        """
    )
    
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs per encoder per dataset (default: 10)')
    parser.add_argument('--bits', type=int, default=128,
                       help='Number of bits for fingerprinting (default: 128)')
    parser.add_argument('--features', type=int, default=5000,
                       help='Maximum features for TF-IDF (default: 5000)')
    parser.add_argument('--test-size', type=int, default=1000,
                       help='Number of documents for encoding test (default: 1000)')
    parser.add_argument('--queries', type=int, default=100,
                       help='Number of search queries to test (default: 100)')
    
    args = parser.parse_args()
    
    run_multi_dataset_benchmark(
        n_runs=args.runs,
        n_bits=args.bits,
        max_features=args.features,
        test_size=args.test_size,
        n_queries=args.queries
    )