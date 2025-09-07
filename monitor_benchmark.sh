#!/bin/bash

# Monitor benchmark progress
echo "📊 Monitoring TEJAS Benchmark Progress..."
echo "=========================================="

while true; do
    # Check if benchmark is still running
    if pgrep -f "unified_benchmark_v3.py" > /dev/null; then
        # Count result files
        WIKI_COUNT=$(ls -1 benchmark_results/Wikipedia/*.json 2>/dev/null | wc -l)
        MARCO_COUNT=$(ls -1 benchmark_results/MS-MARCO/*.json 2>/dev/null | wc -l) 
        BEIR_COUNT=$(ls -1 benchmark_results/BEIR-SciFact/*.json 2>/dev/null | wc -l)
        
        echo -ne "\r⏱️  $(date '+%H:%M:%S') | Wikipedia: $WIKI_COUNT | MS-MARCO: $MARCO_COUNT | BEIR: $BEIR_COUNT | Status: Running..."
        
        sleep 10
    else
        echo -e "\n✅ Benchmark completed!"
        break
    fi
done

echo -e "\n\n📁 Final Results:"
echo "=========================================="
ls -la benchmark_results/*/benchmark_report_*.md 2>/dev/null