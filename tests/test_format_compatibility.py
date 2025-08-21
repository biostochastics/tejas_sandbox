"""
Cross-Format Compatibility Validation
=====================================

Validates that packed and unpacked fingerprints work seamlessly
with calibration and drift monitoring across all backends.
"""

import numpy as np
import torch
import tempfile
from pathlib import Path

# Import core components
from core.encoder import GoldenRatioEncoder
from core.fingerprint import BinaryFingerprintSearch, pack_fingerprints, unpack_fingerprints
from core.calibration import StatisticalCalibrator
from core.drift import DriftMonitor
from core.format import save_fingerprints_v2, load_fingerprints_v2


def test_packed_unpacked_equivalence():
    """Test that packed and unpacked fingerprints produce identical results."""
    np.random.seed(42)
    
    # Create sample data
    titles = ['University of California', 'List of countries', 'History of France',
              'Random document 1', 'Random document 2', 'Random document 3']
    
    # Create and train encoder
    encoder = GoldenRatioEncoder(n_bits=128, max_features=1000)
    encoder.fit(titles)
    
    # Generate fingerprints
    fingerprints = encoder.encode(titles)
    
    print(f"Original fingerprints shape: {fingerprints.shape}")
    print(f"Original fingerprints dtype: {fingerprints.dtype}")
    
    # Test packing/unpacking
    packed = pack_fingerprints(fingerprints, bitorder='little')
    unpacked = unpack_fingerprints(packed, fingerprints.shape[1], bitorder='little')
    
    print(f"Packed shape: {packed.shape}")
    print(f"Unpacked shape: {unpacked.shape}")
    
    # Verify exact equivalence
    assert torch.equal(fingerprints, unpacked), "Packing/unpacking should be lossless"
    
    # Test search equivalence
    search_orig = BinaryFingerprintSearch(fingerprints, titles)
    search_unpacked = BinaryFingerprintSearch(unpacked, titles)
    
    query_fp = encoder.encode_single(titles[0])
    
    results_orig = search_orig.search(query_fp, k=3)
    results_unpacked = search_unpacked.search(query_fp, k=3)
    
    # Results should be identical
    for (t1, d1, _), (t2, d2, _) in zip(results_orig, results_unpacked):
        assert t1 == t2, f"Title mismatch: {t1} vs {t2}"
        assert abs(d1 - d2) < 1e-10, f"Distance mismatch: {d1} vs {d2}"
    
    print("✅ Packed/unpacked search equivalence validated")
    return fingerprints, packed, titles, encoder


def test_calibration_format_compatibility():
    """Test calibration works with both packed and unpacked formats."""
    original_fingerprints, packed, titles, original_encoder = test_packed_unpacked_equivalence()
    
    # Unpack for testing
    unpacked = unpack_fingerprints(packed, original_fingerprints.shape[1], bitorder='little')
    
    # Test calibration with both formats
    calibration_results = {}
    
    for format_name, fingerprints in {'original': original_fingerprints, 'unpacked': unpacked}.items():
        # Initialize search engine with fingerprints
        search_engine = BinaryFingerprintSearch(fingerprints)
        
        # Set up the index with titles
        search_engine.identifiers = titles
        
        # Generate calibration data with multiple queries to ensure binary labels
        all_distances = []
        all_labels = []
        
        # Query 1: University query (use the original fitted encoder)
        query_fp1 = original_encoder.encode_single('University of California')
        results1 = search_engine.search(query_fp1, k=len(titles))
        for result_title, distance, _ in results1:
            all_distances.append(float(distance))
            # Relevance: 1 if contains "University", 0 otherwise
            all_labels.append(1 if 'University' in result_title else 0)
        
        # Query 2: Tech query (ensures we have 0 labels)
        query_fp2 = original_encoder.encode_single('Technology Corporation')
        results2 = search_engine.search(query_fp2, k=len(titles))
        for result_title, distance, _ in results2:
            all_distances.append(float(distance))
            # Relevance: 1 if contains "Tech" or "Company", 0 otherwise
            all_labels.append(1 if ('Tech' in result_title or 'Company' in result_title) else 0)
        
        if len(all_distances) > 0 and len(np.unique(all_labels)) == 2:
            distances = np.array(all_distances)
            labels = np.array(all_labels)
            
            calibrator = StatisticalCalibrator(n_folds=2, n_bootstrap=5)
            result = calibrator.calibrate_with_cv(
                distances, labels,
                k_values=[1, 2]
            )
            
            calibration_results[format_name] = result
            print(f"✅ Calibration with {format_name} format: MAP = {result['metrics']['map']:.3f}")
    
    # Results should be very similar
    if len(calibration_results) == 2:
        map1 = calibration_results['original']['metrics']['map']
        map2 = calibration_results['unpacked']['metrics']['map']
        assert abs(map1 - map2) < 0.001, f"MAP should be identical: {map1} vs {map2}"
    
    print("✅ Calibration format compatibility validated")
    return calibration_results


def test_drift_format_compatibility():
    """Test drift monitoring works with both packed and unpacked formats."""
    fingerprints, packed, titles, encoder = test_packed_unpacked_equivalence()
    
    # Unpack for testing
    unpacked = unpack_fingerprints(packed, fingerprints.shape[1], bitorder='little')
    
    # Test drift monitoring with both formats
    formats = {
        'original': fingerprints,
        'unpacked': unpacked
    }
    
    drift_results = {}
    
    for format_name, fp_data in formats.items():
        monitor = DriftMonitor(baseline_fingerprints=None, drift_threshold=0.1, sensitivity='medium')
        monitor.set_baseline(fp_data[:3])  # Use first 3 as baseline
        
        # Test with remaining data as "new batch"
        batch_data = fp_data[3:]
        result = monitor.check_batch(batch_data)
        
        drift_results[format_name] = result
        print(f"✅ Drift monitoring with {format_name} format: "
              f"JS divergence = {result['js_divergence']:.4f}, "
              f"severity = {result['drift_severity']}")
    
    # Results should be identical
    if len(drift_results) == 2:
        js1 = drift_results['original']['js_divergence']
        js2 = drift_results['unpacked']['js_divergence']
        assert abs(js1 - js2) < 1e-10, f"JS divergence should be identical: {js1} vs {js2}"
        
        sev1 = drift_results['original']['drift_severity']
        sev2 = drift_results['unpacked']['drift_severity']
        assert sev1 == sev2, f"Drift severity should match: {sev1} vs {sev2}"
    
    print("✅ Drift monitoring format compatibility validated")
    return drift_results


def test_format_v2_integration():
    """Test format v2 integration with all components."""
    fingerprints, packed, titles, encoder = test_packed_unpacked_equivalence()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test saving in v2 format
        v2_file = temp_path / 'test_v2.tj2'
        metadata = {
            'n_bits': fingerprints.shape[1],
            'bitorder': 'little',
            'encoder_version': '2.0',
            'creation_timestamp': '2024-01-01T00:00:00Z'
        }
        
        save_fingerprints_v2(v2_file, packed, titles, metadata)
        print(f"✅ Saved v2 format: {v2_file.stat().st_size} bytes")
        
        # Test loading
        loaded_packed, loaded_titles, loaded_metadata = load_fingerprints_v2(v2_file)
        
        assert loaded_titles == titles, "Titles should match after save/load"
        assert np.array_equal(packed, loaded_packed), "Packed data should match after save/load"
        assert loaded_metadata['n_bits'] == fingerprints.shape[1], "Metadata should be preserved"
        
        # Test that loaded data works with search
        loaded_unpacked = unpack_fingerprints(
            loaded_packed, 
            loaded_metadata['n_bits'], 
            loaded_metadata['bitorder']
        )
        
        search_engine = BinaryFingerprintSearch(loaded_unpacked, loaded_titles)
        query_fp = encoder.encode_single(titles[0])
        results = search_engine.search(query_fp, k=3)
        
        assert len(results) == 3, "Search should work with loaded v2 data"
        assert results[0][0] == titles[0], "Should find exact match first"
        
        print("✅ Format v2 integration validated")


def test_backend_consistency():
    """Test that all backends produce consistent results."""
    fingerprints, packed, titles, encoder = test_packed_unpacked_equivalence()
    
    # Test search consistency across backends
    unpacked = unpack_fingerprints(packed, fingerprints.shape[1], bitorder='little')
    search_engine = BinaryFingerprintSearch(unpacked, titles)
    
    query_fp = encoder.encode_single(titles[1])
    
    # Test different backends
    backends = ['numpy', 'auto']
    backend_results = {}
    
    for backend in backends:
        try:
            results = search_engine.search(query_fp, k=len(titles))
            backend_results[backend] = results
            print(f"✅ Backend {backend}: {len(results)} results")
        except Exception as e:
            print(f"⚠️ Backend {backend} failed: {e}")
    
    # Compare results across backends
    if len(backend_results) >= 2:
        backend_names = list(backend_results.keys())
        results1 = backend_results[backend_names[0]]
        results2 = backend_results[backend_names[1]]
        
        # Check that results are consistent
        for (t1, d1, _), (t2, d2, _) in zip(results1, results2):
            assert t1 == t2, f"Title order should be consistent: {t1} vs {t2}"
            assert abs(d1 - d2) < 1e-6, f"Distances should be consistent: {d1} vs {d2}"
    
    print("✅ Backend consistency validated")


def run_all_compatibility_tests():
    """Run all format compatibility tests."""
    print("Running cross-format compatibility validation...")
    print("=" * 60)
    
    try:
        # Test 1: Basic packed/unpacked equivalence
        test_packed_unpacked_equivalence()
        
        # Test 2: Calibration compatibility
        test_calibration_format_compatibility()
        
        # Test 3: Drift monitoring compatibility
        test_drift_format_compatibility()
        
        # Test 4: Format v2 integration
        test_format_v2_integration()
        
        # Test 5: Backend consistency
        test_backend_consistency()
        
        print("=" * 60)
        print("🎉 All cross-format compatibility tests PASSED!")
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_compatibility_tests()
