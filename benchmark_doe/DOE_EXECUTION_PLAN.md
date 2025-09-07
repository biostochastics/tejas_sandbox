# DOE Benchmark Execution Plan

## Overview
This document outlines the comprehensive execution plan for the DOE benchmark suite, including all 7 pipelines (5 TEJAS + 2 BERT) and factor analysis capabilities.

## Fixed Issues
1. ✅ Import path issues resolved in `run_tejas_vs_bert.py`
2. ✅ Created monitoring wrapper with timeout and checkpointing
3. ✅ Created quick test harness for validation
4. ✅ Created factor analysis test scripts

## Pipeline Configurations

### TEJAS Pipelines (5 variants)
1. **TEJAS-Original**: Base implementation with truncated SVD
2. **TEJAS-GoldenRatio**: Golden ratio sampling with randomized SVD
3. **TEJAS-FusedChar**: Character-level fusion with SIMD optimizations
4. **TEJAS-FusedByte**: Byte-level tokenization with BPE
5. **TEJAS-Optimized**: Full optimizations with Numba JIT

### BERT Baselines (2 models)
6. **BERT-MiniLM**: all-MiniLM-L6-v2 (384 dims, lightweight)
7. **BERT-MPNet**: all-mpnet-base-v2 (768 dims, full-size)

## Execution Scripts

### 1. Quick Validation
```bash
# Test all 7 pipelines with minimal data
python3 benchmark_doe/test_single_pipeline.py

# Test with monitoring and timeout
python3 benchmark_doe/run_with_monitoring.py --test-all --timeout 300
```

### 2. Head-to-Head Comparison
```bash
# Quick test (2 runs)
python3 benchmark_doe/run_tejas_vs_bert.py --quick

# Full benchmark (10 runs)
python3 benchmark_doe/run_with_monitoring.py \
  --script benchmark_doe/run_tejas_vs_bert.py \
  --timeout 7200
```

### 3. Factor Analysis

#### Single Factor Analysis
```bash
# Test n_bits effect
python3 benchmark_doe/run_factor_analysis.py \
  --factor n_bits \
  --values 64,128,256,512 \
  --runs 10

# Test batch_size effect  
python3 benchmark_doe/run_factor_analysis.py \
  --factor batch_size \
  --values 500,1000,2000 \
  --runs 10
```

#### Interaction Analysis
```bash
# SIMD × Bit Packing interaction
python3 benchmark_doe/run_factor_analysis.py \
  --factors use_simd,bit_packing \
  --interaction \
  --runs 5

# Backend × Numba interaction
python3 benchmark_doe/run_factor_analysis.py \
  --factors backend,use_numba \
  --interaction \
  --runs 5
```

## Complete Factor List (13 factors)

1. **n_bits**: [64, 128, 256, 512] - Hash dimension
2. **batch_size**: [500, 1000, 2000] - Processing batch size
3. **backend**: ['numpy', 'numba'] - Computation backend
4. **use_simd**: [False, True] - SIMD optimizations
5. **use_numba**: [False, True] - Numba JIT compilation
6. **bit_packing**: [False, True] - Bit packing optimization
7. **tokenizer**: ['char_ngram', 'byte_bpe'] - Tokenization method
8. **svd_method**: ['truncated', 'randomized'] - SVD algorithm
9. **use_itq**: [False, True] - ITQ optimization
10. **use_reranker**: [False, True] - Reranking post-processing
11. **downsample_ratio**: [0.5, 0.75, 1.0] - Data downsampling
12. **energy_threshold**: [0.90, 0.95, 0.99] - SVD energy threshold
13. **max_features**: [5000, 10000, 20000] - Vocabulary size

## Datasets

### Primary Datasets
1. **Wikipedia**: 125k documents (fallback from 250k)
2. **MS MARCO**: Dev subset with real queries
3. **BEIR**: SciFact and NFCorpus subsets

### Dataset Sizes for Testing
- Quick test: 10k documents
- Standard: 50k-125k documents  
- Full: 250k documents (when available)

## Metrics Collected

### Primary Metrics (with 95% CI)
- **Encoding Speed**: Documents per second
- **Query Latency (P50)**: Median search latency in ms
- **Peak Memory**: Maximum memory usage in MB
- **NDCG@10**: Normalized Discounted Cumulative Gain

### Secondary Metrics
- Query Latency P95
- Index Size (MB)
- MRR@10 (Mean Reciprocal Rank)
- Recall@100

## Statistical Analysis

### Methods
- **Aggregation**: Median across runs
- **Confidence Intervals**: 95% CI using percentile method
- **Significance Tests**: Mann-Whitney U test between pipelines
- **Visualization**: Box plots and performance profiles

### Output Formats
- JSON: Complete results with all metrics
- CSV: Tabular format for analysis
- HTML: Interactive comparison tables
- LaTeX: Publication-ready tables

## Recommended Execution Order

1. **Validation Phase** (30 minutes)
   ```bash
   python3 benchmark_doe/test_single_pipeline.py
   python3 benchmark_doe/test_factor_analysis.py
   ```

2. **Quick Comparison** (1 hour)
   ```bash
   python3 benchmark_doe/run_tejas_vs_bert.py --quick
   ```

3. **Full Comparison** (4-6 hours)
   ```bash
   python3 benchmark_doe/run_with_monitoring.py \
     --script benchmark_doe/run_tejas_vs_bert.py \
     --timeout 21600
   ```

4. **Factor Analysis** (2-3 hours per factor)
   ```bash
   # Run for each critical factor
   for factor in n_bits batch_size use_simd use_numba; do
     python3 benchmark_doe/run_factor_analysis.py \
       --factor $factor --runs 10
   done
   ```

5. **Interaction Analysis** (1-2 hours per pair)
   ```bash
   # Key interactions to test
   python3 benchmark_doe/run_factor_analysis.py \
     --factors use_simd,bit_packing --interaction --runs 5
   ```

## Monitoring and Recovery

### Checkpointing
- Automatic checkpoints every 10 experiments
- Resume from checkpoint on failure
- Checkpoint location: `benchmark_results/checkpoints/`

### Timeout Management
- Default: 300s per experiment
- Long runs: 7200s for full comparison
- Adjustable via `--timeout` flag

### Error Recovery
- Automatic retry on network errors
- Skip and continue on encoder failures
- Final report includes success/failure stats

## Next Steps

1. Run validation tests to ensure all components work
2. Execute quick comparison for baseline results
3. Run full comparison with all datasets
4. Perform factor analysis for optimization insights
5. Generate final report with statistical significance

## Troubleshooting

### Common Issues and Solutions

1. **Import Errors**: Ensure running from project root directory
2. **Timeout Issues**: Increase timeout or reduce dataset size
3. **Memory Errors**: Reduce batch_size or n_bits
4. **Missing Data**: Check data/ directory for required datasets
5. **BERT Model Downloads**: Ensure internet connectivity for first run

### Debug Commands
```bash
# Test dataset loading
python3 -c "from benchmark_doe.core.dataset_loader import load_benchmark_dataset; d,q,r = load_benchmark_dataset(); print(f'Loaded {len(d)} docs')"

# Test encoder factory
python3 -c "from benchmark_doe.core.encoder_factory import EncoderFactory; print(EncoderFactory.list_available_encoders())"

# Check factor registry
python3 -c "from benchmark_doe.core.factors import FactorRegistry; r = FactorRegistry(); print(r.list_factors())"
```

## Performance Expectations

### Approximate Runtimes
- Quick test (1 pipeline, 10k docs): ~1 minute
- Single pipeline (125k docs): ~5-10 minutes  
- Full comparison (7 pipelines × 3 datasets × 10 runs): 4-6 hours
- Single factor analysis (4 values × 10 runs): 2-3 hours
- Interaction analysis (4 combinations × 5 runs): 1-2 hours

### Resource Requirements
- CPU: 4+ cores recommended
- Memory: 8GB minimum, 16GB recommended
- Storage: 10GB for datasets and results
- Network: Required for BERT model downloads

## Results Location
All results are saved to `benchmark_results/` with timestamps:
- `tejas_vs_bert/`: Head-to-head comparison results
- `factor_analysis/`: Factor analysis results
- `checkpoints/`: Recovery checkpoints
- `logs/`: Execution logs and debug information