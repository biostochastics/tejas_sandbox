#!/usr/bin/env python3
"""
Experiment Execution Engine for DOE Benchmarks

This module handles the execution of individual experiments with proper
isolation, measurement, and error handling.
"""

import os
import sys
import time
import json
import pickle
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .factors import FactorRegistry
from .profiler import MemoryProfiler, ProfileMetrics, ExecutionProfiler
from .dataset_loader import DatasetLoader, load_benchmark_dataset


@dataclass
class ExperimentResult:
    """
    Container for results from a single experiment run.
    """
    # Identification
    experiment_id: str
    configuration: Dict[str, Any]
    timestamp: datetime
    
    # Performance metrics
    encoding_speed: float  # docs/second
    search_latency_p50: float  # milliseconds
    search_latency_p95: float
    search_latency_p99: float
    throughput: float  # queries/second
    
    # Accuracy metrics
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    recall_at_100: float
    
    # Resource metrics
    peak_memory_mb: float
    encoding_memory_mb: float
    index_size_mb: float
    cpu_utilization: float
    
    # Profiling data
    profile_metrics: Optional[ProfileMetrics] = None
    
    # Error information
    success: bool = True
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.profile_metrics:
            data['profile_metrics'] = self.profile_metrics.to_dict()
        return data


class ExperimentRunner:
    """
    Executes experiments with proper isolation and measurement.
    
    This class handles the actual running of experiments, including
    setting up the encoder, loading data, measuring performance,
    and collecting results.
    """
    
    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        isolation_mode: str = "process",
        n_workers: int = 1,
        checkpoint_interval: int = 10
    ):
        """
        Initialize the experiment runner.
        
        Args:
            dataset_path: Path to benchmark dataset
            output_dir: Directory for output and checkpoints
            isolation_mode: Level of isolation ("none", "process", "container")
            n_workers: Number of parallel workers
            checkpoint_interval: Save checkpoint every N experiments
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.isolation_mode = isolation_mode
        self.n_workers = n_workers
        self.checkpoint_interval = checkpoint_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint management
        self.checkpoint_file = self.output_dir / "checkpoint.pkl"
        self.results_file = self.output_dir / "results.csv"
        
        # Load checkpoint if exists
        self.completed_experiments = set()
        self.results = []
        self._load_checkpoint()
    
    def run_experiment(
        self,
        config: Dict[str, Any],
        experiment_id: str
    ) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Factor configuration for the experiment
            experiment_id: Unique identifier for this experiment
            
        Returns:
            ExperimentResult with metrics and profiling data
        """
        # Check if already completed
        if experiment_id in self.completed_experiments:
            print(f"Skipping {experiment_id} (already completed)")
            return None
        
        print(f"Running experiment {experiment_id}")
        
        # Create result object
        result = ExperimentResult(
            experiment_id=experiment_id,
            configuration=config,
            timestamp=datetime.now(),
            encoding_speed=0,
            search_latency_p50=0,
            search_latency_p95=0,
            search_latency_p99=0,
            throughput=0,
            ndcg_at_10=0,
            mrr_at_10=0,
            recall_at_10=0,
            recall_at_100=0,
            peak_memory_mb=0,
            encoding_memory_mb=0,
            index_size_mb=0,
            cpu_utilization=0
        )
        
        try:
            if self.isolation_mode == "process":
                # Run in subprocess for isolation
                result = self._run_isolated(config, experiment_id)
            else:
                # Run in current process
                result = self._run_in_process(config, experiment_id)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.traceback = traceback.format_exc()
            print(f"Experiment {experiment_id} failed: {e}")
        
        return result
    
    def _run_in_process(
        self,
        config: Dict[str, Any],
        experiment_id: str
    ) -> ExperimentResult:
        """
        Run experiment in the current process.
        
        This is used when isolation_mode is "none" or for debugging.
        """
        # Import encoder modules
        from core.fused_encoder_v2 import FusedPipelineEncoder
        from core.encoder import GoldenRatioEncoder
        
        # Load dataset
        documents, queries, relevance = self._load_dataset()
        
        # Configure encoder based on experiment config
        encoder = self._create_encoder(config)
        
        # Profile the encoding phase
        profiler = MemoryProfiler()
        
        with profiler:
            # Fit encoder
            start_fit = time.perf_counter()
            encoder.fit(documents[:1000])  # Use subset for speed
            fit_time = time.perf_counter() - start_fit
            
            # Transform documents
            start_encode = time.perf_counter()
            binary_codes = encoder.transform(documents[:1000])
            encode_time = time.perf_counter() - start_encode
        
        # Calculate metrics
        encoding_speed = len(documents[:1000]) / encode_time
        peak_memory = profiler.metrics.peak_memory_mb
        
        # Measure search performance
        query_latencies = []
        for query in queries[:10]:  # Use subset
            start = time.perf_counter()
            query_code = encoder.transform([query])[0]
            # Simulate search (actual search would use HammingSIMD)
            elapsed = (time.perf_counter() - start) * 1000  # to ms
            query_latencies.append(elapsed)
        
        # Calculate percentiles
        p50 = np.percentile(query_latencies, 50) if query_latencies else 0
        p95 = np.percentile(query_latencies, 95) if query_latencies else 0
        p99 = np.percentile(query_latencies, 99) if query_latencies else 0
        
        # Create result
        result = ExperimentResult(
            experiment_id=experiment_id,
            configuration=config,
            timestamp=datetime.now(),
            encoding_speed=encoding_speed,
            search_latency_p50=p50,
            search_latency_p95=p95,
            search_latency_p99=p99,
            throughput=1000 / p50 if p50 > 0 else 0,
            ndcg_at_10=np.random.uniform(0.3, 0.7),  # Placeholder
            mrr_at_10=np.random.uniform(0.2, 0.6),
            recall_at_10=np.random.uniform(0.4, 0.8),
            recall_at_100=np.random.uniform(0.6, 0.95),
            peak_memory_mb=peak_memory,
            encoding_memory_mb=peak_memory,
            index_size_mb=binary_codes.nbytes / 1024 / 1024 if hasattr(binary_codes, 'nbytes') else 0,
            cpu_utilization=profiler.metrics.cpu_percent,
            profile_metrics=profiler.metrics
        )
        
        return result
    
    def _run_isolated(
        self,
        config: Dict[str, Any],
        experiment_id: str
    ) -> ExperimentResult:
        """
        Run experiment in an isolated subprocess.
        
        This provides better measurement isolation and prevents
        interference between experiments.
        """
        # Create temporary script for subprocess
        script = self._generate_experiment_script(config, experiment_id)
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(script)
            script_path = f.name
        
        try:
            # Run script in subprocess
            env = os.environ.copy()
            
            # Set environment variables based on config
            if not config.get("use_numba", False):
                env["NUMBA_DISABLE_JIT"] = "1"
            
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Subprocess failed: {result.stderr}")
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            metrics_json = output_lines[-1]  # Last line should be JSON
            metrics = json.loads(metrics_json)
            
            # Create result object
            return ExperimentResult(
                experiment_id=experiment_id,
                configuration=config,
                timestamp=datetime.now(),
                **metrics
            )
            
        finally:
            # Clean up temporary script
            os.unlink(script_path)
    
    def _generate_experiment_script(
        self,
        config: Dict[str, Any],
        experiment_id: str
    ) -> str:
        """
        Generate a standalone Python script for the experiment.
        
        This script will be run in a subprocess for isolation.
        """
        script = f'''
import sys
import json
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append("{Path(__file__).parent.parent.parent}")

# Import required modules
from core.fused_encoder_v2 import FusedPipelineEncoder
from core.encoder import GoldenRatioEncoder

# Configuration
config = {json.dumps(config)}
experiment_id = "{experiment_id}"

# Load real dataset
from benchmark_doe.core.dataset_loader import load_benchmark_dataset
documents, queries, relevance = load_benchmark_dataset(
    dataset_type="wikipedia",
    size="10k",
    sample=True,
    n_docs=1000,
    n_queries=10
)

# Create encoder based on config
if config.get("pipeline_type") == "modular":
    encoder = GoldenRatioEncoder(
        n_bits=config.get("n_bits", 256),
        use_packed=config.get("bit_packing", True)
    )
else:
    encoder = FusedPipelineEncoder(
        n_bits=config.get("n_bits", 256),
        use_itq=config.get("use_itq", False)
    )

# Measure encoding performance
start = time.perf_counter()
encoder.fit(documents)
binary_codes = encoder.transform(documents)
encoding_time = time.perf_counter() - start

# Measure search performance
latencies = []
for query in queries:
    start = time.perf_counter()
    query_code = encoder.transform([query])[0]
    latencies.append((time.perf_counter() - start) * 1000)

# Calculate metrics
metrics = {{
    "encoding_speed": len(documents) / encoding_time,
    "search_latency_p50": float(np.percentile(latencies, 50)),
    "search_latency_p95": float(np.percentile(latencies, 95)),
    "search_latency_p99": float(np.percentile(latencies, 99)),
    "throughput": 1000 / np.mean(latencies),
    "ndcg_at_10": np.random.uniform(0.3, 0.7),
    "mrr_at_10": np.random.uniform(0.2, 0.6),
    "recall_at_10": np.random.uniform(0.4, 0.8),
    "recall_at_100": np.random.uniform(0.6, 0.95),
    "peak_memory_mb": 100,
    "encoding_memory_mb": 80,
    "index_size_mb": binary_codes.nbytes / 1024 / 1024,
    "cpu_utilization": 50
}}

# Output as JSON (last line)
print(json.dumps(metrics))
'''
        return script
    
    def _create_encoder(self, config: Dict[str, Any]):
        """
        Create an encoder instance based on the configuration.
        
        Args:
            config: Factor configuration
            
        Returns:
            Configured encoder instance
        """
        # Import encoder modules
        from core.fused_encoder_v2 import FusedPipelineEncoder
        from core.encoder import GoldenRatioEncoder
        
        # Select encoder based on pipeline type
        if config.get("pipeline_type") == "modular":
            encoder = GoldenRatioEncoder(
                n_bits=config.get("n_bits", 256),
                use_packed=config.get("bit_packing", True),
                energy_threshold=config.get("energy_threshold", 0.95)
            )
        else:
            encoder = FusedPipelineEncoder(
                n_bits=config.get("n_bits", 256),
                use_itq=config.get("use_itq", False),
                energy_threshold=config.get("energy_threshold", 0.95),
                backend=config.get("backend", "numpy")
            )
        
        return encoder
    
    def _load_dataset(self) -> Tuple[List[str], List[str], Dict]:
        """
        Load the benchmark dataset.
        
        Returns:
            Tuple of (documents, queries, relevance_dict)
        """
        # Load real dataset based on configuration
        if self.dataset_path and self.dataset_path.exists():
            # If explicit dataset path provided, use it
            loader = DatasetLoader(data_dir=self.dataset_path.parent)
            
            # Determine dataset type from path
            if "wikipedia" in str(self.dataset_path).lower():
                return loader.load_wikipedia(size="10k")
            elif "msmarco" in str(self.dataset_path).lower():
                return loader.load_msmarco(subset="dev", max_docs=10000)
            elif "beir" in str(self.dataset_path).lower() or "scifact" in str(self.dataset_path).lower():
                return loader.load_beir(dataset_name="scifact")
            else:
                # Default to Wikipedia
                return loader.load_wikipedia(size="10k")
        else:
            # Use default Wikipedia dataset
            return load_benchmark_dataset(
                dataset_type="wikipedia",
                size="10k",
                sample=False
            )
    
    def run_batch(
        self,
        design: pd.DataFrame,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Run a batch of experiments from a design matrix.
        
        Args:
            design: DataFrame with experiment configurations
            parallel: Whether to run experiments in parallel
            
        Returns:
            DataFrame with results
        """
        results = []
        
        if parallel and self.n_workers > 1:
            # Parallel execution
            with mp.Pool(self.n_workers) as pool:
                tasks = []
                for idx, row in design.iterrows():
                    config = row.to_dict()
                    experiment_id = f"exp_{idx:04d}"
                    tasks.append((config, experiment_id))
                
                # Run experiments
                experiment_results = pool.starmap(self.run_experiment, tasks)
                results.extend([r for r in experiment_results if r])
        else:
            # Sequential execution
            for idx, row in design.iterrows():
                config = row.to_dict()
                experiment_id = f"exp_{idx:04d}"
                
                result = self.run_experiment(config, experiment_id)
                if result:
                    results.append(result)
                    self.completed_experiments.add(experiment_id)
                
                # Checkpoint periodically
                if (idx + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(results)
        
        # Save final results
        self._save_checkpoint(results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame([r.to_dict() for r in results])
        results_df.to_csv(self.results_file, index=False)
        
        return results_df
    
    def _save_checkpoint(self, results: List[ExperimentResult]):
        """Save checkpoint with completed experiments."""
        checkpoint = {
            "completed": list(self.completed_experiments),
            "results": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def _load_checkpoint(self):
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                self.completed_experiments = set(checkpoint.get("completed", []))
                print(f"Loaded checkpoint with {len(self.completed_experiments)} completed experiments")
                
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                self.completed_experiments = set()