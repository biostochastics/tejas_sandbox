# DOE Benchmark Suite - Enhanced Design of Experiments

## Overview

This enhanced DOE (Design of Experiments) benchmark suite provides systematic testing of TEJAS optimization configurations using statistical experimental design principles. The suite has been refactored to support comprehensive factor analysis across all pipeline architectures.

## Key Features

### 🔬 Complete Factor Coverage (13 Factors)

The suite tests all critical TEJAS optimization parameters:

**Binary Factors (5):**
- `bit_packing` - Pack bits into uint32 arrays
- `use_numba` - Enable Numba JIT compilation  
- `use_itq` - Apply Iterative Quantization
- `use_simd` - SIMD acceleration
- `use_reranker` - Post-processing reranking

**Categorical Factors (5):**
- `tokenizer` - char_ngram, byte_bpe, word, hybrid
- `svd_method` - truncated, randomized, randomized_downsampled
- `backend` - numpy, pytorch, numba
- `pipeline_type` - All 5 architectures (original_tejas, goldenratio, fused_char, fused_byte, optimized_fused)
- `itq_variant` - none, standard, optimized, similarity_preserving

**Ordinal/Continuous Factors (3):**
- `n_bits` - 64, 128, 256, 512
- `downsample_ratio` - 0.1 to 1.0
- `energy_threshold` - 0.80 to 0.99

### 🎯 Statistical Design Types

**1. Screening Design (Plackett-Burman)**
- Identifies most impactful factors with minimal runs
- Tests main effects only
- Efficient for initial exploration

**2. Optimization Design (Central Composite)**
- Response surface methodology
- Finds optimal factor settings
- Includes quadratic effects and interactions

**3. Full Factorial Design**
- Complete analysis of selected factors
- All interactions examined
- Most thorough but computationally intensive

**4. Fractional Factorial Design**
- Balanced coverage with fewer runs
- Practical for production use
- Good trade-off between coverage and efficiency

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python benchmark_doe/test_refactoring.py
```

## Quick Start

### Run All Pipelines (5 runs each, all datasets)

```bash
# Comprehensive benchmark with 5 runs per configuration
python benchmark_doe/run_comprehensive_benchmark.py \
    --datasets wikipedia msmarco beir \
    --runs 5 \
    --output ./benchmark_results/comprehensive
```

### Use Pre-configured DOE Designs

```bash
# Quick test - validates all pipelines work
python benchmark_doe/run_doe_benchmark.py --pipelines all --runs 1

# Screening design - identify important factors
python benchmark_doe/run_with_config.py configs/screening_design.yaml

# Optimization - find best settings
python benchmark_doe/run_with_config.py configs/optimization_design.yaml

# Full factorial - detailed analysis
python benchmark_doe/run_with_config.py configs/full_factorial_small.yaml
```

## Configuration Files

### Available Templates

```
configs/
├── screening_design.yaml        # Factor identification
├── optimization_design.yaml     # Performance optimization
├── full_factorial_small.yaml    # Detailed 3-4 factor analysis
├── complete_factorial_design.yaml # All factors (reference)
└── quick_test.yaml              # Rapid validation
```

### Custom Configuration Example

```yaml
design_type: fractional_factorial
factors:
  pipeline_type:
    levels: ["fused_char", "optimized_fused"]
  n_bits:
    levels: [128, 256, 512]
  use_simd:
    levels: [true, false]
    
experiment_config:
  n_runs_per_config: 5
  datasets: ["wikipedia", "msmarco"]
  random_seeds: [42, 123, 456, 789, 1001]
```

## Metrics Collected

### Performance Metrics
- `encoding_speed` - Documents per second
- `search_latency_p50/p95/p99` - Query latency percentiles
- `throughput_qps` - Queries per second
- `indexing_time` - Time to build index

### Accuracy Metrics  
- `ndcg_at_k` - Normalized Discounted Cumulative Gain
- `mrr_at_k` - Mean Reciprocal Rank
- `recall_at_k` - Recall at various cutoffs
- `precision_at_k` - Precision metrics

### Resource Metrics
- `peak_memory_mb` - Maximum memory usage
- `index_size_mb` - Index storage size
- `cpu_utilization_percent` - CPU usage

### Stability Metrics
- `variance_across_runs` - Consistency measure
- `failure_rate` - Reliability indicator

## Unified Validation System

The suite includes automatic validation and correction of factor combinations:

```python
from benchmark_doe.core.factors import FactorRegistry

registry = FactorRegistry()

# Validate configuration
config = {'pipeline_type': 'fused_char', 'use_simd': True, 'bit_packing': False}
is_valid, issues = registry.validate_configuration(config)

# Auto-fix invalid combinations
fixed_config = registry.auto_fix_configuration(config)
```

## Enhanced Metrics System

### Distance-Aware Similarity Computation
- **Hamming Distance**: Proper metric for binary codes (TEJAS)
- **Cosine Similarity**: For dense vectors (BERT)
- **Automatic Selection**: Based on encoding type

### Comprehensive IR Metrics
**Multi-Cutoff Evaluation:**
- Precision @ {1, 5, 10, 20, 50, 100}
- Recall @ {10, 20, 50, 100, 500, 1000}
- NDCG @ {1, 5, 10, 20}
- MAP @ {10, 100}
- MRR (with and without cutoff)
- Success @ {1, 5, 10} (critical for sparse relevance)

**Statistical Analysis:**
- Bootstrap confidence intervals (95% CI)
- Wilcoxon signed-rank test (paired comparisons)
- Mann-Whitney U test (independent samples)
- Median and percentile reporting

### Dataset-Specific Considerations
- **MS MARCO**: Typically 1 relevant doc per query
- **BEIR**: Varies from dense to sparse relevance
- **Wikipedia**: Custom relevance judgments

## Analysis & Reporting

### Generated Reports Include:
- Main effects analysis (Pareto plots)
- Interaction effects (2-way and 3-way)
- Response surface plots
- Optimal configuration identification
- Statistical significance testing (ANOVA, Tukey HSD)
- Performance profiles by pipeline and dataset
- **NEW**: Multi-metric comparison tables with confidence intervals
- **NEW**: Success rate analysis for sparse relevance scenarios

### Output Formats:
- JSON - Raw experimental data with full metric suite
- CSV - Tabular results including all cutoff points
- Parquet - Efficient storage for large experiments
- HTML/PDF - Visual reports with enhanced metric plots

## Running Comprehensive Benchmarks

### Full Experimental Suite

```bash
# Run complete DOE with all factors
python benchmark_doe/run_comprehensive_benchmark.py \
    --datasets wikipedia msmarco beir internal_dense internal_sparse \
    --runs 5 \
    --output ./results/full_doe
```

This will:
1. Generate valid experiment configurations
2. Test each configuration 5 times 
3. Run across all specified datasets
4. Save checkpoints every 10 experiments
5. Generate comprehensive analysis reports

### Expected Runtime

- Quick test: ~5 minutes
- Screening design: ~2 hours  
- Full factorial (small): ~9 hours
- Comprehensive (all datasets): ~24-48 hours

## Results Structure

```
benchmark_results/
├── experiment_matrix_*.csv      # Configuration matrix
├── results_*.json               # Complete results
├── results_*.csv                # Tabular format
├── results_*.parquet            # Efficient storage
├── checkpoint_*.json            # Intermediate saves
└── analysis_report.html         # Visual report
```

## Pipeline Compatibility Matrix

| Pipeline | Tokenizers | Backends | SVD Methods | SIMD | ITQ |
|----------|------------|----------|-------------|------|-----|
| original_tejas | char_ngram, word | numpy, pytorch | truncated | ❌ | ✓ |
| goldenratio | char_ngram | numpy, numba | truncated, randomized | ❌ | ✓ |
| fused_char | char_ngram | numpy, numba | all | ✓ | ✓ |
| fused_byte | byte_bpe | numpy | truncated, randomized | ✓ | ❌ |
| optimized_fused | char_ngram, byte_bpe | numpy, numba | all | ✓ | ✓ |

## Troubleshooting

### Common Issues

**Memory Errors:**
- Reduce `n_bits` or `batch_size`
- Enable `downsample_ratio < 1.0`
- Use `backend: numpy` instead of pytorch

**Validation Failures:**
- Check pipeline compatibility matrix
- Use `auto_fix_configuration()` for automatic correction
- Review constraint warnings in logs

**Slow Performance:**
- Enable `use_simd` and `bit_packing`
- Use `optimized_fused` pipeline
- Reduce dataset size for initial testing

## Citation

If you use this DOE benchmark suite, please cite:

```bibtex
@software{tejas_doe_benchmark,
  title={TEJAS DOE Benchmark Suite},
  author={TEJAS Team},
  year={2024},
  url={https://github.com/tejas/benchmark_doe}
}
```

## License

MIT License - See LICENSE file for details