#!/bin/bash
# Comprehensive DOE Benchmark Execution Script
# Runs all benchmarks with proper error handling and logging

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="benchmark_results/full_run_${TIMESTAMP}"
LOG_FILE="${RESULTS_DIR}/execution.log"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Function to run benchmark with timeout and logging
run_benchmark() {
    local name=$1
    local cmd=$2
    local timeout=${3:-3600}  # Default 1 hour timeout
    
    log "Starting: ${name}"
    log "Command: ${cmd}"
    
    # Run with timeout and capture output
    if timeout ${timeout} bash -c "${cmd}" >> "${LOG_FILE}" 2>&1; then
        log "✓ SUCCESS: ${name}"
        return 0
    else
        log "✗ FAILED: ${name} (exit code: $?)"
        return 1
    fi
}

# Main execution
log "="
log "DOE BENCHMARK SUITE - FULL EXECUTION"
log "="
log "Results directory: ${RESULTS_DIR}"
log "Starting comprehensive benchmark run..."

# Phase 1: Validation Tests
log ""
log "PHASE 1: VALIDATION TESTS"
log "-"

run_benchmark \
    "Single Pipeline Test" \
    "python3 benchmark_doe/test_single_pipeline.py" \
    300

run_benchmark \
    "Factor Analysis Test" \
    "python3 benchmark_doe/test_factor_analysis.py" \
    600

# Phase 2: Quick Comparison
log ""
log "PHASE 2: QUICK COMPARISON (2 runs)"
log "-"

run_benchmark \
    "TEJAS vs BERT Quick Test" \
    "python3 benchmark_doe/run_tejas_vs_bert.py --quick" \
    1800

# Phase 3: Full Comparison (if quick test passes)
if [ $? -eq 0 ]; then
    log ""
    log "PHASE 3: FULL COMPARISON (10 runs)"
    log "-"
    
    run_benchmark \
        "TEJAS vs BERT Full Benchmark" \
        "python3 benchmark_doe/run_with_monitoring.py --script benchmark_doe/run_tejas_vs_bert.py --timeout 7200" \
        28800  # 8 hours
fi

# Phase 4: Factor Analysis
log ""
log "PHASE 4: FACTOR ANALYSIS"
log "-"

# Critical factors to analyze
FACTORS=("n_bits:64,128,256,512" "batch_size:500,1000,2000" "use_simd:false,true" "use_numba:false,true")

for factor_spec in "${FACTORS[@]}"; do
    IFS=':' read -r factor values <<< "$factor_spec"
    run_benchmark \
        "Factor Analysis: ${factor}" \
        "python3 benchmark_doe/run_factor_analysis.py --factor ${factor} --values ${values} --runs 5" \
        3600
done

# Phase 5: Interaction Analysis
log ""
log "PHASE 5: INTERACTION ANALYSIS"  
log "-"

run_benchmark \
    "Interaction: SIMD × Bit Packing" \
    "python3 benchmark_doe/run_factor_analysis.py --factors use_simd,bit_packing --interaction --runs 3" \
    1800

# Generate summary report
log ""
log "GENERATING SUMMARY REPORT"
log "-"

python3 -c "
import json
import glob
from pathlib import Path

results_dir = Path('${RESULTS_DIR}')
summary = {
    'timestamp': '${TIMESTAMP}',
    'phases_completed': [],
    'total_experiments': 0,
    'files_generated': []
}

# Find all result files
for pattern in ['*.json', '*.csv', '*.html']:
    files = list(results_dir.glob(f'**/{pattern}'))
    summary['files_generated'].extend([str(f) for f in files])

# Save summary
summary_file = results_dir / 'execution_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to: {summary_file}')
" | tee -a "${LOG_FILE}"

# Final status
log ""
log "="
log "BENCHMARK EXECUTION COMPLETE"
log "="
log "Results directory: ${RESULTS_DIR}"
log "Log file: ${LOG_FILE}"
log ""

# Show summary statistics
echo "Summary Statistics:" | tee -a "${LOG_FILE}"
echo "  Total runtime: $SECONDS seconds" | tee -a "${LOG_FILE}"
echo "  Log lines: $(wc -l < "${LOG_FILE}")" | tee -a "${LOG_FILE}"
echo "  Result files: $(find "${RESULTS_DIR}" -name "*.json" -o -name "*.csv" | wc -l)" | tee -a "${LOG_FILE}"

exit 0