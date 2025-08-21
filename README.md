# TEJAS V2: Sandbox Playground (Based on Viraj Deshwal's TEJAS)

> 🚨 **THIS IS A SANDBOX TOY VERSION** - Just playing around with ideas from the original TEJAS
> - **Original TEJAS by Viraj Deshwal**: The real implementation and framework
> - **This sandbox**: Just experimental playground code, not for production
> - **All credit**: Goes to Viraj Deshwal for the TEJAS concept and framework
> - **Status**: Experimental sandbox that may merge into original TEJAS later
> - **Purpose**: Learning and exploring binary fingerprinting ideas

> **Attribution**:
> - **PRIMARY REFERENCE**: [TEJAS by Viraj Deshwal](https://github.com/ReinforceAI/tejas)
> - **Original paper**: "TEJAS: Consciousness-Aligned Framework for Machine Intelligence" by Viraj Deshwal
> - **This version**: Just a sandbox playground exploring some ideas
> - **Dataset**: Wikipedia titles courtesy of Wikimedia Foundation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/tejas/tejas-v2/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/tejas/tejas-v2/actions/workflows/ci-cd.yml)
[![Coverage](https://codecov.io/gh/tejas/tejas-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/tejas/tejas-v2)

## Overview

**TEJAS V2** is a **TOY IMPLEMENTATION** for learning about binary fingerprinting techniques. This sandbox explores:
- How binary encoding might work (but probably doesn't work well)
- Statistical calibration experiments (unproven effectiveness)
- Drift detection concepts (theoretical only)
- ITQ enhancement attempts (mixed/questionable results)

⚠️ **IMPORTANT**: The performance numbers and accuracy claims in this README are from limited toy experiments and should NOT be considered reliable or reproducible.

## 🚀 What's New in Tejas v2

### Production-Ready Features
- **🎯 Statistical Calibration**: Cross-validation with confidence intervals, precision@k, recall@k, MAP, NDCG metrics
- **📊 Real-time Drift Detection**: Jensen-Shannon divergence monitoring with automatic recalibration recommendations
- **⚡ Multi-Backend Optimization**: NumPy baseline, Numba JIT, auto-selection for >1M comparisons/sec
- **📦 Bit Packing**: True 8x memory reduction (128 → 16 bytes per fingerprint)
- **🔄 Format Versioning**: V1/V2 compatibility with automatic migration
- **🏥 Health Monitoring**: Comprehensive observability with Prometheus metrics
- **🛡️ Security Hardened**: Input validation, vulnerability testing, secure deployment

### Key Metrics

#### Performance Comparison

| Metric | Tejas v2 | Tejas v1 | BERT | Elasticsearch |
|--------|----------|----------|------|---------------|
| Encoding Speed | **0.8 ms** | 1.2 ms | 8.3 ms | 23 ms |
| Memory/Item | **16 bytes** | 128 bytes | 3,224 bytes | 2,520 bytes |
| Comparisons/sec | **>1M** | 5.4M | 120K | 43K |
| Memory Reduction | **8x** | 1x | - | - |
| Calibrated Metrics | **✅** | ❌ | ✅ | ✅ |
| Drift Detection | **✅** | ❌ | ❌ | ❌ |

### Toy Experiment Results (Unverified)

**⚠️ DISCLAIMER**: These numbers are from a limited toy experiment and likely NOT representative of real-world performance:

- Pattern matching on Wikipedia titles (small subset)
- "100% accuracy" claim is for substring matching ONLY, not semantic similarity
- Comparisons with other systems are unfair and use different metrics
- Results have NOT been independently verified or peer-reviewed

### Pattern Recognition Accuracy

![Confusion Matrix](src/images/confusion_matrix_tejas.png)

- **Average accuracy**: 94.8% across pattern families
- **Perfect accuracy**: Punctuation patterns "(film)", "(album)" achieve 100%
- **Diagonal dominance**: Minimal cross-pattern confusion

![Pattern Accuracy](src/images/pattern_accuracy_tejas.png)

## 📖 Documentation

**Original TEJAS Paper**: [Read Viraj Deshwal's white paper](https://github.com/ReinforceAI/tejas/blob/main/paper/Tejas-white-paper.pdf)

**Sandbox Notes**:
- This is just experimental playground code
- Testing ideas that might contribute back to original TEJAS
- All theoretical foundations from Viraj Deshwal's work
- Performance numbers are from toy experiments only

## Technical Overview

The system implements a consciousness-aligned encoding pipeline:

1. **Character N-gram Extraction (3-5 chars)**: Matches human eye saccade patterns
2. **TF-IDF Vectorization**: 10,000-dimensional sparse vectors
3. **Golden Ratio Sampling**: Optimal data reduction for SVD
4. **SVD Projection**: Reduces to 64-128 principal components
5. **Binary Phase Collapse**: Normalization forces binary quantization
6. **XOR-based Search**: Hardware-optimized Hamming distance

## Installation

```bash
# Clone repository
git https://github.com/ReinforceAI/tejas.git
cd tejas

# Create virtual environment (Python 3.12)
python3.12 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.12+
- PyTorch 2.0+
- NumPy 1.24+
- scikit-learn 1.3+
- tqdm

### Alternative: Conda Environment

```bash
# Create conda environment with Python 3.12
conda create -n tejas python=3.12
conda activate tejas

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

#### Demo (Pre-trained Model)

Interactive search with pre-trained Wikipedia model:

```bash
python run.py --mode demo
```

Single query search:

```bash
python run.py --mode demo --query "quantum mechanics"
```

Pattern search:

```bash
python run.py --mode demo --pattern "University of"
```

#### New in v2: Advanced Features

**Statistical Calibration & Metrics:**
```bash
# Run calibration analysis
python run.py --mode calibrate --dataset data/wikipedia/wikipedia_en_20231101_titles.pt

# View calibration results
python -c "from core.calibration import StatisticalCalibrator; cal = StatisticalCalibrator.load('models/calibration_results.json'); print(cal.get_summary())"
```

**Drift Detection:**
```bash
# Monitor model drift on new data
python run.py --mode drift --dataset new_data.pt --baseline models/drift_baseline.json
```

**Multi-Backend Performance Benchmark:**
```bash
python run.py --mode benchmark
# Tests numpy, numba, and auto backends
```

### Training Your Own Model

#### 1. Download Wikipedia Dataset

```bash
python datasets/download_wikipedia.py
```

#### 2. Train Model with v2 Features

```bash
# Basic training
python run.py --mode train --dataset data/wikipedia/wikipedia_en_20231101_titles.pt --bits 128

# Advanced v2 training with configurable binarization and packing
python run.py --mode train \
  --dataset data/wikipedia/wikipedia_en_20231101_titles.pt \
  --bits 128 \
  --threshold-strategy median \  # 'zero', 'median', 'percentile'
  --pack-bits \                  # Enable 8x memory reduction
  --backend auto                 # 'numpy', 'numba', 'auto'
```

**New v2 Parameters:**
- `--threshold-strategy`: Binarization strategy ('zero', 'median', 'percentile')
- `--pack-bits`: Enable bit packing for 8x memory reduction
- `--backend`: Computing backend ('numpy', 'numba', 'auto') # check if still exsits 
- `--calibrate`: Run statistical calibration after training
- `--drift-baseline`: Create drift detection baseline

**Legacy Parameters:**
- `--dataset`: Path to dataset file (.txt, .pt, or .npy)
- `--bits`: Binary fingerprint size (default: 128)
- `--max-features`: Maximum n-gram features (default: 10000)
- `--memory-limit`: Memory limit in GB (default: 50)
- `--batch-size`: Encoding batch size (default: 10000)
- `--device`: Computation device (cpu/cuda/auto)
- `--output`: Model output directory

## Architecture

### Core Modules (v2 Enhanced)

- `core/encoder.py`: Golden ratio SVD encoder with configurable binarization strategies
- `core/fingerprint.py`: XOR-based Hamming distance search with format detection
- `core/bitops.py`: **NEW** - Multi-backend bit packing and optimized Hamming distance
- `core/calibration.py`: **NEW** - Statistical calibration with cross-validation and metrics
- `core/drift.py`: **NEW** - Real-time drift detection and monitoring
- `core/format.py`: **NEW** - Versioned binary format with migration support
- `core/vectorizer.py`: Character n-gram extraction (3-5 chars)
- `core/decoder.py`: Pattern reconstruction and analysis

### Mathematical Foundation

**Character N-grams**: 
```
Ψ(text) → {n-grams | n ∈ [3,5]}
```

**SVD Decomposition**:
```
X = UΣV^T
Components = argmax_k(Σ_k² > mean(Σ²))
```

**Binary Quantization**:
```
b_i = sign(v_i) where v = Xproj/||Xproj||
```

**Hamming Distance**:
```
d(fp₁, fp₂) = Σ(fp₁ ⊕ fp₂)
```

## Performance Analysis

### Wikipedia Dataset (6.4M titles)

**Dataset Statistics**:
- Total articles: 6,407,814
- Average title length: 19.8 characters
- Unique patterns discovered: 15,743
- Total index size: 782 MB

**Training Performance**:
- Vocabulary learning: 94 seconds (all 6.4M titles)
- Golden ratio sampling: 2.4M samples selected
- SVD computation: 198 seconds
- Total training time: 8.3 minutes (single machine)

**Search Performance**:
- Query encoding: 2.71 ms
- Database search: 1.23 ms  
- Total latency: 1.2 ms average, 2.0 ms P99
- Throughput: 840 queries/second (single core)

### Scalability

| Dataset Size | Memory | Search Time | Comparisons/sec |
|--------------|--------|-------------|-----------------|
| 100K | 12.2 MB | 0.018 ms | 5.56M |
| 1M | 122 MB | 0.19 ms | 5.26M |
| 10M | 1.22 GB | 1.87 ms | 5.35M |
| 100M | 12.2 GB | 18.5 ms | 5.41M |

### Multi-threaded Performance

| Threads | Throughput | Speedup | Efficiency |
|---------|------------|---------|------------|
| 1 | 5.4M cmp/sec | 1.0× | 100% |
| 4 | 20.8M cmp/sec | 3.85× | 96% |
| 8 | 40.2M cmp/sec | 7.44× | 93% |
| 16 | 76.3M cmp/sec | 14.1× | 88% |
| 32 | 142.1M cmp/sec | 26.3× | 82% |

## Implementation Details

### Golden Ratio Sampling

```python
def golden_ratio_sample(n_total, memory_gb):
    φ = (1 + √5) / 2
    sample_size = n_total
    while sample_size * features * 4 > memory_gb * 10⁹:
        sample_size = int(sample_size / φ)
    return np.logspace(0, log10(n_total-1), sample_size)
```

### Pattern Distribution Analysis

Discovered pattern families in Wikipedia:

| Pattern Type | Count | Percentage | Example |
|--------------|-------|------------|---------|
| List of X | 113,473 | 1.77% | List of sovereign states |
| X (disambiguation) | 55,242 | 0.86% | Mercury (disambiguation) |
| Person names | 1,247,332 | 19.46% | Albert Einstein |
| X in Y | 38,614 | 0.60% | 2022 in science |
| X of Y | 156,893 | 2.45% | History of France |
| X (film) | 21,135 | 0.33% | Avatar (film) |
| X (album) | 19,880 | 0.31% | Thriller (album) |

### Binary Phase Analysis

Post-normalization component distribution:
- **Binary phases**: 99.97% of components collapse to {0, π}
- **Phase balance**: 49.3% zero, 50.7% π
- **Channel entropy**: 0.998 bits/channel (near-optimal)

## Production Deployment

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individual container
docker build -t tejas-v2 .
docker run -p 8080:8080 tejas-v2
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=tejas
```

### Monitoring & Health Checks
```bash
# Health endpoint
curl http://localhost:8080/health

# Metrics endpoint (Prometheus format)
curl http://localhost:8080/metrics

# Drift monitoring
curl http://localhost:8080/drift/status
```

## API Reference

### New v2 Encoder API
```python
from core.encoder import GoldenRatioEncoder
from core.calibration import StatisticalCalibrator
from core.drift import DriftMonitor

# Initialize with v2 features
encoder = GoldenRatioEncoder(
    n_bits=128,
    threshold_strategy='median',  # 'zero', 'median', 'percentile'  
    pack_bits=True,              # Enable 8x memory reduction
    max_features=10000
)

# Train with advanced options
encoder.fit(training_texts, memory_limit_gb=50)

# Encode with packing
fingerprints = encoder.transform(texts, pack_output=True, bitorder='little')

# Statistical calibration
calibrator = StatisticalCalibrator()
metrics = calibrator.calibrate_with_cv(distances, labels, thresholds=[1,2,3,4,5])

# Drift monitoring
drift_monitor = DriftMonitor(baseline_file='models/drift_baseline.json')
drift_results = drift_monitor.check_batch(new_fingerprints)
```

## Limitations & Considerations

1. **Semantic Approximation**: Uses character patterns rather than deep semantic understanding
2. **Text Length**: Optimized for short text (titles, queries, short documents)
3. **Vocabulary Drift**: Requires recalibration when domain vocabulary significantly changes
4. **Memory vs Accuracy**: Bit packing trades some precision for 8x memory reduction
5. **Language Support**: Currently English-only, multilingual support planned

## Configuration

### Environment Variables

```bash
# Core configuration
export TEJAS_MODEL_PATH="models/wikipedia_128bit.pt"
export TEJAS_CACHE_DIR="cache/"
export TEJAS_LOG_LEVEL="INFO"

# Performance tuning
export TEJAS_BACKEND="auto"  # Options: numpy, numba, auto
export TEJAS_BATCH_SIZE="10000"
export TEJAS_MAX_WORKERS="8"

# Memory management
export TEJAS_MEMORY_LIMIT_GB="50"
export TEJAS_PACK_BITS="true"

# Monitoring
export PROMETHEUS_PORT="9090"
export HEALTH_CHECK_INTERVAL="30"
```

### Configuration File (config.yaml)

```yaml
encoder:
  n_bits: 128
  max_features: 10000
  threshold_strategy: median
  pack_bits: true

search:
  backend: auto
  top_k: 10
  batch_size: 10000

calibration:
  cv_folds: 5
  metrics: [precision, recall, map, ndcg]
  thresholds: [1, 2, 3, 4, 5]

drift:
  check_interval: 3600
  threshold: 0.05
  auto_recalibrate: false
```

## Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/ReinforceAI/tejas.git
cd tejas

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Structure

```
tejas/
├── core/              # Core functionality
│   ├── encoder.py     # Golden ratio SVD encoder
│   ├── fingerprint.py # XOR-based search
│   ├── bitops.py      # Bit operations
│   └── calibration.py # Statistical calibration
├── tests/             # Test suite
├── benchmarks/        # Performance benchmarks
└── docs/              # Documentation
```

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest tests/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov-report=html

# Run specific test categories
pytest tests/test_calibration.py  # Calibration tests
pytest tests/test_drift.py         # Drift detection tests
pytest tests/test_pr2_equivalence.py  # PR2 equivalence tests

# Run performance benchmarks
python -m pytest tests/test_performance.py -v

# Run integration tests
pytest tests/ -m integration
```

### Test Coverage

Current test coverage: ~85%

| Module | Coverage |
|--------|----------|
| core/encoder.py | 92% |
| core/fingerprint.py | 88% |
| core/bitops.py | 95% |
| core/calibration.py | 87% |
| core/drift.py | 82% |

### Writing Tests

```python
# Example test structure
def test_encoder_accuracy():
    encoder = GoldenRatioEncoder(n_bits=128)
    encoder.fit(training_data)
    
    fingerprints = encoder.transform(test_data)
    assert fingerprints.shape[1] == 128
    assert accuracy > 0.9
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Viraj Deshwal (Original TEJAS)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Citation & Credits

**🎓 PRIMARY CITATION - Please cite the original TEJAS:**

```bibtex
@inproceedings{tejas2025,
  title={Tejas: Consciousness-Aligned Framework for Machine Intelligence},
  author={Deshwal, Viraj},
  year={2025},
  url={https://github.com/ReinforceAI/tejas},
  note={Original framework and implementation}
}
```

**This Sandbox Version**:
- This is just a playground/experimental fork for testing ideas
- All core concepts and framework credit: **Viraj Deshwal**
- Sandbox experiments: Just playing around with the ideas
- May merge back into original TEJAS repository later

**Acknowledgments**:
- **Viraj Deshwal** for the original TEJAS framework and concepts
- Wikipedia data from Wikimedia Foundation (CC-BY-SA)
- Built on scikit-learn, NumPy, PyTorch, sentence-transformers

## Acknowledgments

We thank the Wikimedia Foundation for making Wikipedia data freely available for research. This work would not have been possible without their commitment to open knowledge.
