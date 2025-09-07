# DOE Framework Reproducibility Guide

## Overview

This guide documents the reproducibility features implemented in the DOE Framework to ensure experiments can be exactly reproduced and validated.

## Key Features

### 1. Global Seed Management

The framework now provides comprehensive seed management to ensure deterministic results:

```python
from benchmark_doe.core.reproducibility import set_global_seed

# Set a specific seed
set_global_seed(42)

# Or let it generate one (will be logged)
seed = set_global_seed(None)
print(f"Using seed: {seed}")
```

**Supported RNGs:**
- Python's `random` module
- NumPy's random module
- PyTorch (if installed)
- TensorFlow (if installed)
- Scikit-learn (via NumPy)

**Environment Setup:**
For complete reproducibility, set these environment variables before starting Python:
```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
```

### 2. Running Reproducible Experiments

```bash
# Run with a specific seed
python run_doe_benchmark.py --seed 42

# Run with auto-generated seed (will be logged and saved)
python run_doe_benchmark.py

# Multiple runs with same seed for validation
python run_doe_benchmark.py --seed 42 --runs 5
```

### 3. Individual Run Data Preservation

All individual run data is now preserved in results, not just aggregated statistics:

```python
# Results structure with preserved data
{
    "schema_version": "1.0",
    "seed_used": 42,
    "results": [
        {
            "experiment_id": "exp_0001",
            "metrics": {...},  # Aggregated metrics
            "individual_runs": [...],  # ALL run data preserved
            "failed_runs": [...],  # Failed attempts tracked
        }
    ]
}
```

### 4. CSV Export

Export all results and analysis to CSV format for use in Excel, R, or other tools:

```bash
# Export results to CSV
python analyze_doe_results.py --export-csv

# Output files created:
# - benchmark_results/csv_exports/doe_results_YYYYMMDD_HHMMSS.csv
# - benchmark_results/csv_exports/main_effects_YYYYMMDD_HHMMSS.csv
```

### 5. Output Manifest System

The framework now tracks all generated outputs with checksums for validation:

```python
# Manifest structure
{
    "schema_version": "1.0",
    "experiment_id": "doe_benchmark_20240115_143022",
    "configuration": {
        "seed": 42,
        "n_runs": 5,
        "n_experiments": 27
    },
    "outputs": {
        "results_json": ["path/to/results.json"],
        "csv_files": ["path/to/exports.csv"],
        "graphs": {
            "plotly_html": ["path/to/interactive.html"],
            "matplotlib_png": ["path/to/plot.png"]
        }
    },
    "checksums": {
        "path/to/results.json": "sha256_hash..."
    },
    "environment": {
        "python_version": "3.10.1",
        "packages": ["numpy==1.21.0", ...],
    },
    "git_info": {
        "commit_hash": "abc123...",
        "branch": "main",
        "has_uncommitted_changes": false
    }
}
```

## Usage Examples

### Complete Reproducible Workflow

```bash
# 1. Set environment for hash reproducibility
export PYTHONHASHSEED=0

# 2. Run experiments with specific seed
python run_doe_benchmark.py --seed 42 --runs 3 --datasets

# 3. Analyze and export results
python analyze_doe_results.py --export-csv

# 4. Check the manifest
cat benchmark_results/experiment_manifest.json
```

### Verifying Reproducibility

```python
# Run the reproducibility test suite
python test_reproducibility.py

# Expected output:
# ✅ Seed Management: PASS
# ✅ Data Preservation: PASS  
# ✅ Manifest Tracking: PASS
# ✅ CSV Export: PASS
```

### Reproducing Previous Experiments

1. Find the seed from previous results:
```bash
grep "seed_used" benchmark_results/doe_results_*.json
```

2. Re-run with same seed:
```bash
python run_doe_benchmark.py --seed <previous_seed>
```

3. Compare results:
```python
import json

# Load both results
with open('original_results.json') as f:
    original = json.load(f)
with open('reproduced_results.json') as f:
    reproduced = json.load(f)

# Compare metrics
for i, (orig, repro) in enumerate(zip(original['results'], reproduced['results'])):
    if orig['metrics'] != repro['metrics']:
        print(f"Difference in experiment {i}")
```

## API Reference

### core.reproducibility

- `set_global_seed(seed: Optional[int] = None) -> int`
  - Sets seeds for all available RNGs
  - Returns the seed used

- `get_reproducibility_info() -> Dict[str, Any]`
  - Returns information about available RNGs and settings

- `ReproducibilityContext(seed: Optional[int] = None)`
  - Context manager for reproducible code blocks

### core.manifest

- `ManifestTracker.get_instance() -> ManifestTracker`
  - Get singleton tracker instance

- `tracker.add_output(filepath: str, category: str = 'other')`
  - Track an output file

- `tracker.save_manifest(filepath: Optional[str] = None) -> str`
  - Save manifest to JSON file

- `tracker.validate_outputs() -> Dict[str, List[str]]`
  - Validate all tracked outputs exist and match checksums

## Troubleshooting

### Issue: Results still vary between runs

**Solutions:**
1. Ensure PYTHONHASHSEED is set before Python starts
2. Check for uncontrolled randomness in external libraries
3. Verify all RNGs are being seeded (check logs)
4. Set OMP_NUM_THREADS=1 for deterministic parallel execution

### Issue: CSV files have encoding issues

**Solution:** Files are saved with UTF-8 encoding. Open with:
```python
pd.read_csv('file.csv', encoding='utf-8')
```

### Issue: Manifest shows checksum mismatches

**Causes:**
1. Files were modified after experiment
2. Concurrent access during write
3. Filesystem corruption

**Solution:** Check file timestamps and re-run if necessary

## Best Practices

1. **Always log seeds**: Even with auto-generated seeds, they're logged for reproduction
2. **Use manifest validation**: Check for missing outputs before sharing results
3. **Version control configs**: Commit configuration files alongside code
4. **Document environment**: Include requirements.txt or environment.yml
5. **Test reproducibility**: Run test suite after changes

## Migration Guide

### Updating Existing Scripts

1. Add seed parameter:
```python
# Old
def run_experiment():
    ...

# New
def run_experiment(seed=None):
    from core.reproducibility import set_global_seed
    actual_seed = set_global_seed(seed)
    ...
```

2. Preserve individual data:
```python
# Old
result = {
    'metrics': aggregated_metrics
}

# New  
result = {
    'metrics': aggregated_metrics,
    'individual_runs': raw_runs,
    'schema_version': '1.0'
}
```

3. Add CSV export:
```python
# After creating DataFrame
if export_csv:
    df.to_csv('results.csv', index=False, encoding='utf-8')
```

## Performance Considerations

- **Storage**: Individual run preservation increases storage by ~10MB per 10,000 runs
- **Overhead**: Seed management adds <0.1% runtime overhead
- **Checksum**: SHA256 calculation adds ~1ms per file
- **CSV Export**: Scales linearly with data size

## Future Enhancements

Planned improvements:
- Parquet format support for large datasets
- Automatic experiment comparison tools
- Integration with MLflow/Weights & Biases
- Distributed experiment tracking
- Checkpointing for long-running experiments

## Support

For issues or questions:
1. Check this guide first
2. Run `test_reproducibility.py` to verify setup
3. Check logs for seed and RNG information
4. Open an issue with manifest file attached