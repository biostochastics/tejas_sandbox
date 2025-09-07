#!/usr/bin/env python3
"""
Unified TEJAS Benchmark Suite V2
=================================
Comprehensive multi-encoder benchmarking with automatic table generation.
Tests all encoder implementations with configurable parameters and rich reporting.
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
from typing import Dict, List, Optional, Tuple, Any, Type
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import importlib
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['status'] = self.status.value
        return d


@dataclass
class EncoderConfig:
    """Configuration for an encoder."""
    name: str
    class_path: str
    module_path: str
    n_bits: int
    max_features: int
    extra_params: Dict[str, Any]


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
        },
        {
            "name": "Fused V1",
            "module": "core.fused_encoder",
            "class": "FusedEncoder",
            "params": {}
        },
        {
            "name": "Fused V2",
            "module": "core.fused_encoder_v2",
            "class": "FusedEncoderV2",
            "params": {}
        },
        {
            "name": "Fused V2 Optimized",
            "module": "core.fused_encoder_v2_optimized",
            "class": "FusedEncoderV2Optimized",
            "params": {}
        },
        {
            "name": "Fused V2 ByteBPE",
            "module": "core.fused_encoder_v2_bytebpe",
            "class": "FusedEncoderV2ByteBPE",
            "params": {}
        },
        {
            "name": "Fused NoPack",
            "module": "core.fused_encoder_nopack",
            "class": "FusedEncoderNoPack",
            "params": {}
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
        
        # Encoding phase
        test_docs = documents[:min(test_size, len(documents))]
        profiler.start()
        if hasattr(encoder, 'transform'):
            fingerprints = encoder.transform(test_docs)
        elif hasattr(encoder, 'encode'):
            fingerprints = encoder.encode(test_docs)
        else:
            raise AttributeError("Encoder has no transform or encode method")
        encode_profile = profiler.stop()
        
        # Search phase (simplified)
        search_times = []
        recalls = {"at_1": [], "at_5": [], "at_10": []}
        
        query_indices = np.random.choice(len(test_docs), min(n_queries, len(test_docs)), replace=False)
        
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
            recalls["at_5"].append(1.0 if qi in indices[:5] else 0.0)
            recalls["at_10"].append(1.0 if qi in indices[:10] else 0.0)
        
        # Calculate metrics
        train_time = train_profile['elapsed_time']
        encode_speed = len(test_docs) / encode_profile['elapsed_time'] if encode_profile['elapsed_time'] > 0 else 0
        search_speed = len(query_indices) / np.sum(search_times) if np.sum(search_times) > 0 else 0
        
        # Cleanup
        del encoder
        gc.collect()
        
        return BenchmarkResult(
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
            recall_at_10=np.mean(recalls["at_10"]) * 100
        )
        
    except MemoryError as e:
        return BenchmarkResult(
            encoder_name=encoder_name,
            run_number=run_number,
            status=EncoderStatus.MEMORY_ERROR,
            error=str(e)[:200]
        )
    except Exception as e:
        return BenchmarkResult(
            encoder_name=encoder_name,
            run_number=run_number,
            status=EncoderStatus.FAILED,
            error=f"{type(e).__name__}: {str(e)[:200]}"
        )


# ============================================================================
# Report Generator with Automatic Tables
# ============================================================================

class ReportGenerator:
    """Generate comprehensive benchmark reports with automatic tables."""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
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
            return "No successful benchmark results to summarize."
        
        # Group by encoder and calculate statistics
        summary = self.df.groupby('Encoder').agg({
            'Train(s)': ['mean', 'std', 'min', 'max'],
            'Encode(d/s)': ['mean', 'std', 'min', 'max'],
            'Search(q/s)': ['mean', 'std', 'min', 'max'],
            'Memory(MB)': ['mean', 'std', 'min', 'max'],
            'R@10(%)': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        return summary.to_string()
    
    def generate_ranking_table(self) -> str:
        """Generate performance ranking table."""
        if self.df.empty:
            return "No successful benchmark results to rank."
        
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
        
        return rankings.to_string()
    
    def generate_markdown_report(self, config: Dict) -> str:
        """Generate complete markdown report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# TEJAS Multi-Encoder Benchmark Report

## Configuration
- **Timestamp**: {timestamp}
- **Documents**: {config.get('n_docs', 'N/A'):,}
- **Runs per encoder**: {config.get('n_runs', 'N/A')}
- **Bit size**: {config.get('n_bits', 'N/A')}
- **Max features**: {config.get('max_features', 'N/A')}

## Performance Rankings

{self._generate_markdown_table(self.generate_ranking_table())}

## Summary Statistics

{self._generate_markdown_table(self.generate_summary_table())}

## Winners by Category

{self._generate_winners_section()}

## Configuration Matrix Tested

{self._generate_config_matrix()}

## System Information
- **CPUs**: {psutil.cpu_count()} cores
- **Memory**: {psutil.virtual_memory().total / (1024**3):.1f} GB
- **Python**: {sys.version.split()[0]}

---
*Generated by TEJAS Benchmark Suite V2*
"""
        return report
    
    def _generate_markdown_table(self, table_str: str) -> str:
        """Convert string table to markdown format."""
        lines = table_str.split('\n')
        if len(lines) < 2:
            return table_str
        
        # Create markdown table
        md_lines = []
        for i, line in enumerate(lines):
            if i == 1:  # Add separator after header
                cols = len(line.split())
                md_lines.append('|' + '---|' * cols)
            md_lines.append('| ' + ' | '.join(line.split()) + ' |')
        
        return '\n'.join(md_lines)
    
    def _generate_winners_section(self) -> str:
        """Generate winners by category."""
        if self.df.empty:
            return "No data available for winner analysis."
        
        means = self.df.groupby('Encoder').mean()
        
        winners = []
        
        # Fastest training
        fastest_train = means['Train(s)'].idxmin()
        winners.append(f"- **Fastest Training**: {fastest_train} ({means.loc[fastest_train, 'Train(s)']:.2f}s)")
        
        # Fastest encoding
        fastest_encode = means['Encode(d/s)'].idxmax()
        winners.append(f"- **Fastest Encoding**: {fastest_encode} ({means.loc[fastest_encode, 'Encode(d/s)']:.0f} docs/s)")
        
        # Fastest search
        fastest_search = means['Search(q/s)'].idxmax()
        winners.append(f"- **Fastest Search**: {fastest_search} ({means.loc[fastest_search, 'Search(q/s)']:.0f} queries/s)")
        
        # Lowest memory
        lowest_memory = means['Memory(MB)'].idxmin()
        winners.append(f"- **Lowest Memory**: {lowest_memory} ({means.loc[lowest_memory, 'Memory(MB)']:.0f} MB)")
        
        # Best recall
        best_recall = means['R@10(%)'].idxmax()
        winners.append(f"- **Best Recall@10**: {best_recall} ({means.loc[best_recall, 'R@10(%)']:.1f}%)")
        
        return '\n'.join(winners)
    
    def _generate_config_matrix(self) -> str:
        """Generate configuration matrix summary."""
        unique_encoders = self.df['Encoder'].nunique()
        total_runs = len(self.df)
        successful_runs = len(self.df[self.df['R@10(%)'] > 0])
        
        return f"""- **Encoders tested**: {unique_encoders}
- **Total runs**: {total_runs}
- **Successful runs**: {successful_runs}
- **Success rate**: {(successful_runs/total_runs*100):.1f}%"""
    
    def save_results(self, output_dir: Path):
        """Save results to multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_file = output_dir / f"benchmark_results_{timestamp}.csv"
        self.df.to_csv(csv_file, index=False)
        print(f"  ✓ CSV saved: {csv_file}")
        
        # Save JSON
        json_file = output_dir / f"benchmark_results_{timestamp}.json"
        json_data = [r.to_dict() for r in self.results]
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"  ✓ JSON saved: {json_file}")
        
        return csv_file, json_file


# ============================================================================
# Main Benchmark Orchestrator
# ============================================================================

def run_comprehensive_benchmark(
    n_docs: int = 10000,
    n_runs: int = 3,
    n_bits: int = 128,
    max_features: int = 5000,
    test_size: int = 1000,
    n_queries: int = 100,
    parallel: bool = False
):
    """Run comprehensive multi-encoder benchmark."""
    
    print("\n" + "="*80)
    print(" TEJAS MULTI-ENCODER BENCHMARK SUITE V2")
    print("="*80)
    print(f"Configuration:")
    print(f"  • Documents: {n_docs:,}")
    print(f"  • Runs per encoder: {n_runs}")
    print(f"  • Bit size: {n_bits}")
    print(f"  • Max features: {max_features:,}")
    print(f"  • Test size: {test_size:,}")
    print(f"  • Queries: {n_queries}")
    print(f"  • Parallel execution: {parallel}")
    print("="*80)
    
    # Load dataset
    print("\n📚 Loading Dataset...")
    data_paths = [
        Path(f"data/wikipedia/wikipedia_{n_docs}.txt"),
        Path("data/wikipedia/wikipedia_10000.txt"),
        Path("data/wikipedia/wikipedia_en_20231101_titles.txt")
    ]
    
    documents = None
    for path in data_paths:
        if path.exists():
            print(f"  Found: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                all_docs = [line.strip() for line in f if line.strip()]
            documents = all_docs[:n_docs]
            print(f"  Loaded: {len(documents):,} documents")
            break
    
    if documents is None:
        print("  ❌ No dataset found!")
        return
    
    # Initialize encoder registry
    registry = EncoderRegistry()
    encoder_names = registry.list_encoders()
    
    if not encoder_names:
        print("  ❌ No encoders available!")
        return
    
    # Run benchmarks
    print(f"\n🚀 Running Benchmarks...")
    print(f"  Total tests: {len(encoder_names)} encoders × {n_runs} runs = {len(encoder_names) * n_runs} tests")
    
    all_results = []
    
    for encoder_name in encoder_names:
        print(f"\n  Testing: {encoder_name}")
        print("  " + "-" * 40)
        
        encoder_class, params = registry.get_encoder(encoder_name, n_bits, max_features)
        if encoder_class is None:
            print(f"    ✗ Encoder not available")
            continue
        
        for run_num in range(1, n_runs + 1):
            print(f"    Run {run_num}/{n_runs}...", end=' ', flush=True)
            
            result = run_single_benchmark(
                encoder_name=encoder_name,
                encoder_class=encoder_class,
                encoder_params=params,
                documents=documents,
                run_number=run_num,
                test_size=test_size,
                n_queries=n_queries
            )
            
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
    
    # Generate reports
    print("\n📊 Generating Reports...")
    generator = ReportGenerator(all_results)
    
    # Print summary tables
    print("\n" + "="*80)
    print(" PERFORMANCE RANKINGS")
    print("="*80)
    print(generator.generate_ranking_table())
    
    print("\n" + "="*80)
    print(" SUMMARY STATISTICS")
    print("="*80)
    print(generator.generate_summary_table())
    
    # Save results
    print("\n💾 Saving Results...")
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    config = {
        'n_docs': n_docs,
        'n_runs': n_runs,
        'n_bits': n_bits,
        'max_features': max_features,
        'test_size': test_size,
        'n_queries': n_queries
    }
    
    # Generate and save markdown report
    markdown_report = generator.generate_markdown_report(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_file = output_dir / f"benchmark_report_{timestamp}.md"
    with open(md_file, 'w') as f:
        f.write(markdown_report)
    print(f"  ✓ Markdown report: {md_file}")
    
    # Save CSV and JSON
    csv_file, json_file = generator.save_results(output_dir)
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE!")
    print("="*80)
    print(f"Results saved in: {output_dir}/")
    
    return all_results


# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TEJAS Multi-Encoder Benchmark Suite V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small dataset
  python unified_benchmark_v2.py --docs 1000 --runs 2
  
  # Standard benchmark
  python unified_benchmark_v2.py --docs 10000 --runs 5
  
  # Large scale test
  python unified_benchmark_v2.py --docs 100000 --runs 10 --parallel
  
  # Custom configuration
  python unified_benchmark_v2.py --docs 5000 --bits 256 --features 10000
        """
    )
    
    parser.add_argument('--docs', type=int, default=10000,
                       help='Number of documents to test (default: 10000)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per encoder (default: 3)')
    parser.add_argument('--bits', type=int, default=128,
                       help='Number of bits for fingerprinting (default: 128)')
    parser.add_argument('--features', type=int, default=5000,
                       help='Maximum features for TF-IDF (default: 5000)')
    parser.add_argument('--test-size', type=int, default=1000,
                       help='Number of documents for encoding test (default: 1000)')
    parser.add_argument('--queries', type=int, default=100,
                       help='Number of search queries to test (default: 100)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel execution')
    
    args = parser.parse_args()
    
    run_comprehensive_benchmark(
        n_docs=args.docs,
        n_runs=args.runs,
        n_bits=args.bits,
        max_features=args.features,
        test_size=args.test_size,
        n_queries=args.queries,
        parallel=args.parallel
    )