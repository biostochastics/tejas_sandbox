# Test Suite Constraints and API Documentation

## Test Results Summary
- **Total Tests**: 542
- **Passing**: 252 (46%)
- **Failing**: 142 (26%)
- **Skipped**: 6
- **Warnings**: 1133 (numerical overflow/underflow - non-critical)

## Core API Constraints

### 1. Encoder Constraints

#### GoldenRatioEncoder
- **Minimum samples**: Must have more samples than `max_features` parameter
- **Parameters**: 
  - `n_bits`: Output fingerprint size
  - `max_features`: TF-IDF vocabulary size
  - Does NOT accept `random_state` parameter
- **Usage**:
  ```python
  encoder = GoldenRatioEncoder(n_bits=64, max_features=1000)
  encoder.fit(texts)  # texts must be list of strings
  fingerprints = encoder.transform(texts)  # returns np.ndarray
  ```

#### TejasSEncoder
- **Minimum samples**: Must have `n_samples > n_bits` for SVD to work
- **SVD requirement**: Uses RandomizedSVD internally, needs sufficient data
- **Usage**:
  ```python
  encoder = TejasSEncoder(n_bits=128)
  encoder.fit(texts)  # Need 128+ samples
  fps = encoder.transform(texts)  # or encoder.encode(texts)
  ```

#### TejasFEncoder/OptimizedFusedEncoder
- **Output shape**: May return packed format (n_bits/8) instead of full n_bits
- **Methods**: Uses `fit()` and `transform()`, not `encode_batch()`

### 2. Algorithm Constraints

#### RandomizedSVD
- **Dimension constraint**: `n_components` cannot exceed `min(n_samples, n_features)`
- **No transform method**: Only has `fit()` and `fit_transform()`
- **Usage**:
  ```python
  svd = RandomizedSVD(n_components=10, n_iter=3)
  U = svd.fit_transform(X)  # Returns U or (U, S, Vt) if return_components=True
  ```

#### ITQOptimizer
- **Input dimensions**: Expects data with `n_features == n_bits`
- **Config parameters**: Use `max_iterations` not `n_iterations`
- **Pre-processing**: Data should be pre-reduced to target dimensions
- **Usage**:
  ```python
  config = ITQConfig(n_bits=16, max_iterations=10)
  itq = ITQOptimizer(config)
  itq.fit(X)  # X must have 16 features
  binary = itq.transform(X)
  ```

### 3. Search Constraints

#### BinaryFingerprintSearch
- **Return format**: Returns list of tuples `(title, distance)` not dicts
- **Usage**:
  ```python
  search = BinaryFingerprintSearch(database, titles)
  results = search.search(query, k=5)
  # results = [(title1, dist1), (title2, dist2), ...]
  ```

#### ParallelSearchOptimized
- **Constructor**: Takes `n_workers` not `n_threads`
- **Initialization**: Requires fingerprints and titles at construction

### 4. Hamming Distance Constraints

#### hamming_distance_packed (bitops.py)
- **Input format**:
  - `query`: Must be 1D array (single query)
  - `database`: Must be 2D array (multiple targets)
- **Usage**:
  ```python
  query = np.array([...], dtype=np.uint8)  # 1D
  database = np.array([[...], [...]], dtype=np.uint8)  # 2D
  distances = hamming_distance_packed(query, database)
  ```

#### hamming_simd.py functions
- Different API - not directly compatible with bitops.py
- Has separate functions for different backends

### 5. Memory and Performance Constraints

#### Memory Limits
- Golden ratio sampling activates when data exceeds memory limits
- Default memory limit: 2-4 GB depending on encoder
- Automatic subsampling for large datasets

#### Numerical Stability
- Warnings about overflow/underflow are expected with random data
- Use scaled inputs to avoid numerical issues:
  ```python
  X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize
  ```

### 6. Backend Dependencies

#### Optional Dependencies
- **numba**: Required for parallel search and optimized Hamming distance
- **torch**: Required for some encoder variants
- **scipy**: Required for sparse matrix support

#### Skip Decorators
Tests properly skip when dependencies missing:
- `@pytest.mark.skipif(not HAS_NUMBA, reason="...")`
- `@pytest.mark.skipif(not HAS_SCIPY, reason="...")`

## Common Test Failures and Solutions

### 1. "n_components cannot exceed min(n_samples, n_features)"
**Cause**: Too few samples for requested dimensions
**Solution**: Ensure `n_samples > n_components` and `n_features > n_components`

### 2. "Expected N features, got M"
**Cause**: ITQ expects pre-reduced data
**Solution**: Use PCA or SVD to reduce to target dimensions first

### 3. "query_packed must be 1D"
**Cause**: Wrong input shape for Hamming distance
**Solution**: Reshape query to 1D, database to 2D

### 4. "tuple indices must be integers or slices"
**Cause**: Search returns tuples not dicts
**Solution**: Access as `result[0]` (title) and `result[1]` (distance)

### 5. Numerical warnings
**Cause**: Random test data causing overflow
**Solution**: Scale inputs, use float32 instead of float64

## Test Organization

### Core Tests (Working)
- `test_minimal.py` - Basic functionality tests
- `test_core_encoders.py` - Encoder tests (needs sample size fixes)
- `test_core_algorithms.py` - Algorithm tests (needs parameter fixes)

### Integration Tests
- `test_integration_core.py` - End-to-end workflows
- Various encoder-specific tests

### Performance Tests
- Marked with `@pytest.mark.slow`
- Can be skipped with `pytest -m "not slow"`

## Recommendations

1. **Fix sample sizes**: Ensure all tests provide sufficient data for algorithms
2. **Update API calls**: Match actual method signatures
3. **Handle backends**: Properly skip tests when dependencies missing
4. **Scale inputs**: Prevent numerical issues with normalized data
5. **Document constraints**: Make API requirements clear in docstrings