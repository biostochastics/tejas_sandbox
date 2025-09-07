# DOE Benchmark v2 Changelog

## Summary of Changes

The original `run_doe_benchmark.py` has been replaced with an enhanced version (v2) that includes critical fixes and improvements. The original version is backed up as `run_doe_benchmark_original.py`.

## Critical Fixes

### 1. Data Validity Issues (FIXED)
- **Problem**: Synthetic/fake queries and relevance data were being generated when real data was missing
- **Solution**: Removed all synthetic data fallbacks; now requires real data or uses proper fallback datasets
- **Impact**: Benchmark results are now valid and meaningful

### 2. Metrics Calculation Errors (FIXED)
- **Problem**: `docs_per_second` incorrectly included `fit_time`, making throughput measurements inaccurate
- **Solution**: Now calculates `docs_per_second = n_documents / encode_time` (excluding fit time)
- **Impact**: Throughput metrics are now accurate and comparable

### 3. Memory Measurement (FIXED)
- **Problem**: Using `sys.getsizeof()` for numpy arrays gave incorrect memory usage
- **Solution**: Now uses `array.nbytes` for accurate memory measurement
- **Impact**: Memory usage metrics are now reliable

## Major Enhancements

### 1. Dataset Configuration
- **Wikipedia**: Now uses 250k documents as primary size (was 125k)
  - Fallback chain: 250k → 125k → 50k → 10k
  - Added support for `wikipedia_250000.txt` in dataset loader
- **BEIR**: Limited to exactly 2 datasets as required (SciFact and NFCorpus)
- **Internal**: Support for two custom datasets (dense and sparse retrieval)

### 2. Advanced Query Generation
New `QueryGenerator` class provides:
- **Entity-based queries**: Extract main entities from titles
- **Context-based queries**: Domain-specific queries with modifiers
- **Question-based queries**: Natural language questions
- **Query diversity**: Short (2-3 words), medium (4-6), long (7+)

### 3. Comprehensive Error Handling
New `ErrorHandler` class with:
- **Retry logic**: Exponential backoff for transient failures
- **Resource reduction**: Automatic batch size reduction on memory errors
- **Graceful degradation**: Fallback to smaller datasets when needed
- **Specific error types**: DATA_NOT_FOUND, MEMORY_ERROR, TIMEOUT_ERROR, etc.

### 4. Error Recovery Strategies
```
Network Error → Retry (3x) → Fallback → Skip
Memory Error  → Reduce Batch → Clear Cache → Retry
Data Error    → Try Smaller Dataset → Use Backup → Warning
Timeout       → Increase Timeout → Simplify → Retry
```

## File Changes

- `run_doe_benchmark.py` - Enhanced v2 implementation (new main script)
- `run_doe_benchmark_original.py` - Original version (backup)
- `run_core_benchmark.py` - Removed (intermediate version)
- `run_core_benchmark_v2.py` - Removed (renamed to run_doe_benchmark.py)
- `core/dataset_loader.py` - Updated to support 250k Wikipedia dataset
- `README.md` - Updated documentation for v2

## Usage

```bash
# New enhanced version (default)
python benchmark_doe/run_doe_benchmark.py

# With specific options
python benchmark_doe/run_doe_benchmark.py \
    --datasets wikipedia beir msmarco \
    --pipelines original_tejas fused_char fused_byte \
    --runs 3 \
    --seed 42

# Original version (if needed for comparison)
python benchmark_doe/run_doe_benchmark_original.py
```

## Migration Notes

1. The new script is backward compatible with existing command-line arguments
2. Results now include error summaries and degradation tracking
3. Output schema version updated to 2.1
4. Wikipedia 250k dataset file needs to be created or will fallback to 125k

## Performance Improvements

- Better query quality leads to more meaningful benchmark results
- Retry logic reduces failed experiments
- Resource management prevents out-of-memory crashes
- Proper metrics calculation enables accurate performance comparison

## Testing Recommendations

1. Test with small dataset first: `--datasets wikipedia --runs 1`
2. Verify 250k Wikipedia dataset exists or prepare for fallback
3. Monitor error_summary in results for degradation tracking
4. Compare results with original version to validate improvements