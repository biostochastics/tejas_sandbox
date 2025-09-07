# DOE-Based Benchmarking Framework for TEJAS v2

A comprehensive Design of Experiments (DOE) framework for systematically evaluating optimization strategies in binary semantic search.

## Overview

This framework implements a rigorous DOE methodology to explore the 11-dimensional optimization space of TEJAS, reducing the experimental burden from 13,824 full factorial combinations to ~100 strategically selected experiments while maintaining statistical validity.

## Latest Updates (v2)

### Critical Improvements
- **Wikipedia Dataset**: Now uses 250k documents as primary size (previously 125k)
- **BEIR Datasets**: Configured to use exactly 2 datasets (SciFact and NFCorpus)
- **Query Generation**: Advanced refinement system with entity, context, and question-based queries
- **Error Handling**: Comprehensive retry logic with exponential backoff and graceful degradation
- **Metrics Fixes**: Corrected docs_per_second calculation (excludes fit time) and memory measurement

### Key Issues Fixed
1. **Synthetic Data**: Removed fallback to fake queries/relevance that invalidated benchmarks
2. **Metrics Accuracy**: Fixed incorrect inclusion of fit_time in throughput calculation
3. **Memory Measurement**: Now uses array.nbytes instead of sys.getsizeof for accurate readings
4. **Dataset Support**: BEIR no longer hardcoded to single dataset, supports configurable selection
5. **Error Recovery**: Added specific error types with appropriate retry strategies

## Key Features

### Core Benchmark Script (`run_doe_benchmark.py`)

The enhanced benchmark runner includes:

- **Advanced Query Generation**
  - Entity-based queries from document titles
  - Context-aware domain-specific queries
  - Natural language question generation
  - Query diversity (short/medium/long variants)

- **Comprehensive Error Handling**
  ```
  Error Recovery Chain:
  ├── Network Error → Retry (3x with backoff) → Fallback → Skip
  ├── Memory Error → Reduce Batch Size → Clear Cache → Retry
  ├── Data Error → Try Smaller Dataset → Use Backup → Warning
  └── Timeout → Increase Timeout → Simplify → Retry
  ```

- **Dataset Configuration**
  - Wikipedia: 250k (primary), 125k, 50k, 10k (fallback chain)
  - BEIR: SciFact and NFCorpus (exactly 2 datasets)
  - MS MARCO: Dev subset with real queries
  - Internal: Dense and sparse retrieval benchmarks

### Experimental Design

- **Factorial design** across pipelines, bit sizes, and batch sizes
- **Pipeline configurations**:
  - `original_tejas`: Original TEJAS with sklearn SVD
  - `fused_char`: Fused pipeline with character n-grams
  - `fused_byte`: Fused pipeline with byte-level BPE

- **Multiple run support** with median aggregation for statistical robustness
- **Automatic configuration validation** and compatibility checking

### Advanced Profiling & Analysis

- **Memory profiling** with automatic limit enforcement
- **Process isolation** with timeout protection (ResourceGuard)
- **Component-level timing** and resource tracking
- **Statistical aggregation** (median, mean, std, min, max) across runs
- **Error tracking** with categorized error types and counts

## Installation

```bash
# Required dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install psutil  # For memory profiling and resource limits
pip install plotly  # For interactive visualizations

# Optional but recommended
pip install pyDOE2  # For advanced DOE designs
pip install statsmodels  # For statistical analysis

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Usage

### Running the Enhanced Benchmark

```bash
# Run with default configuration (Wikipedia 250k, BEIR 2 datasets, all pipelines)
python benchmark_doe/run_doe_benchmark.py

# Specify datasets (Wikipedia uses 250k by default)
python benchmark_doe/run_doe_benchmark.py --datasets wikipedia beir msmarco

# Specify pipelines
python benchmark_doe/run_doe_benchmark.py --pipelines original_tejas fused_char

# Run with multiple repetitions for statistical significance
python benchmark_doe/run_doe_benchmark.py --runs 5

# Set random seed for reproducibility
python benchmark_doe/run_doe_benchmark.py --seed 42
```

### Original Benchmark (backup)

```bash
# Original DOE benchmark (backup version with known issues)
python benchmark_doe/run_doe_benchmark_original.py --datasets --runs 3
```

## Dataset Requirements

### Wikipedia
- Primary: `wikipedia_250000.txt` (250k documents)
- Fallbacks: `wikipedia_125000.txt`, `wikipedia_50000.txt`, `wikipedia_10000.txt`
- Location: `data/wikipedia/`
- Query generation: Automatic refined query creation from titles

### BEIR
- Datasets: SciFact, NFCorpus (exactly 2 used)
- Location: `data/beir/{dataset_name}/`
- Files required: `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`

### MS MARCO
- Subset: Dev
- Location: `data/msmarco/`
- Files required: `collection.tsv`, `queries.dev.small.tsv`, `qrels.dev.small.tsv`

## Output Format

Results are saved as JSON with comprehensive metadata:

```json
{
  "schema_version": "2.1",
  "timestamp": "20240101_120000",
  "configuration": {
    "datasets": ["wikipedia", "beir", "msmarco"],
    "pipelines": ["original_tejas", "fused_char", "fused_byte"],
    "features": {
      "wikipedia_size": "250k",
      "beir_datasets": ["scifact", "nfcorpus"],
      "query_generation": "refined",
      "error_handling": "comprehensive"
    }
  },
  "results": [...],
  "summary": {
    "total_experiments": 54,
    "successful_results": 52,
    "success_rate": 0.963,
    "error_handler_summary": {...}
  },
  "highlights": {
    "best_throughput": {...},
    "best_latency": {...}
  }
}
```

## Metrics Calculation

### Corrected Metrics (v2)

- **docs_per_second**: `n_documents / encode_time` (excludes fit_time)
- **queries_per_second**: `valid_queries / query_time`
- **avg_query_latency_ms**: `(query_time * 1000) / valid_queries`
- **index_size_mb**: Uses `array.nbytes` for accurate memory measurement

### Performance Targets

- Throughput: >100,000 docs/second
- Query latency: <2ms average
- Memory usage: <100MB for 250k documents
- Success rate: >95% across all experiments

## Error Handling

The framework includes comprehensive error handling:

### Error Types
- `DATA_NOT_FOUND`: Missing dataset files
- `ENCODER_INIT`: Encoder initialization failures
- `MEMORY_ERROR`: Out of memory conditions
- `TIMEOUT_ERROR`: Operation timeouts
- `VALIDATION_ERROR`: Data validation failures

### Recovery Strategies
1. **Retry with exponential backoff** (network, timeout errors)
2. **Resource reduction** (batch size halving for memory errors)
3. **Graceful degradation** (fallback to smaller datasets)
4. **Skip and continue** (validation errors)

## Analysis Tools

### Analyze Results

```bash
# Analyze DOE results with statistical summaries
python benchmark_doe/analyze_doe_results.py results/enhanced_benchmark_*.json

# Export to CSV for external analysis
python benchmark_doe/analyze_doe_results.py --export-csv results.csv
```

### Visualizations

The analysis generates:
- Performance comparison across pipelines
- Dataset-specific performance profiles
- Error rate analysis
- Statistical significance tests

## Troubleshooting

### Common Issues

1. **FileNotFoundError for datasets**
   - Ensure Wikipedia 250k dataset exists or allow fallback to smaller sizes
   - BEIR datasets must have proper structure (corpus.jsonl, queries.jsonl)

2. **Memory errors**
   - Reduce batch size with `--batch-size 500`
   - The system automatically retries with 50% batch size reduction

3. **Timeout errors**
   - Increase timeout in configuration
   - System automatically doubles timeout on retry

### Debug Mode

```bash
# Run with debug logging
PYTHONPATH=. python -u benchmark_doe/run_doe_benchmark.py 2>&1 | tee debug.log
```

## Security Considerations

- Uses `SafeEvaluator` instead of `eval()` for configuration conditions
- Resource guards prevent runaway processes
- Input validation on all dataset loading
- No dynamic code execution

## Contributing

When adding new features:
1. Maintain backward compatibility with schema version
2. Add comprehensive error handling
3. Include retry logic for transient failures
4. Document all configuration options
5. Add tests for new components

## Version History

### v2.1 (Current)
- Wikipedia 250k as primary dataset size
- BEIR limited to exactly 2 datasets
- Advanced query generation system
- Comprehensive error handling with retry logic
- Fixed metrics calculation bugs

### v2.0
- Added SafeEvaluator to replace eval()
- Initial DOE framework implementation
- Basic dataset support

### v1.0
- Original benchmark implementation
- Basic pipeline comparison

## License

MIT License - See LICENSE file for details

## Citation

If using this benchmark framework, please cite:
```bibtex
@software{tejas_doe_benchmark,
  title = {DOE-Based Benchmarking Framework for TEJAS},
  version = {2.1},
  year = {2024},
  note = {Enhanced with 250k Wikipedia, refined queries, and comprehensive error handling}
}
```