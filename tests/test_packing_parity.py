import os
import math
import torch
import numpy as np
import pytest

from core.fingerprint import pack_fingerprints, unpack_fingerprints, BinaryFingerprintSearch
from utils.benchmark import BenchmarkSuite


def _random_binary(n_items: int, n_bits: int) -> torch.Tensor:
    return torch.randint(0, 2, (n_items, n_bits), dtype=torch.uint8)


def _titles(n: int):
    return [f"Title {i}" for i in range(n)]


@pytest.mark.parametrize("bitorder", ["little", "big"])
@pytest.mark.parametrize("n_bits", [1, 7, 8, 9, 15, 16, 31, 32, 63, 64, 127, 128])
def test_pack_unpack_roundtrip_bitorder(bitorder, n_bits):
    # Random small matrix
    torch.manual_seed(42)
    X = _random_binary(n_items=37, n_bits=n_bits)

    packed = pack_fingerprints(X, bitorder=bitorder)
    assert packed.dtype == np.uint8
    # Shape check: ceil(n_bits/8)
    n_bytes = (n_bits + 7) // 8
    assert packed.shape == (X.shape[0], n_bytes)

    X_rt = unpack_fingerprints(packed, n_bits=n_bits, bitorder=bitorder)
    assert X_rt.shape == X.shape
    assert torch.equal(X_rt, X)


def test_search_parity_packed_vs_unpacked():
    torch.manual_seed(0)
    n_items, n_bits = 200, 128
    X = _random_binary(n_items, n_bits)
    titles = _titles(n_items)

    # Baseline search on unpacked
    search_unpacked = BinaryFingerprintSearch(X, titles)

    # Pack then unpack on-the-fly for search (simulating loader behavior)
    packed = pack_fingerprints(X, bitorder='little')
    X_unpacked_again = unpack_fingerprints(packed, n_bits=n_bits, bitorder='little')
    search_packed_path = BinaryFingerprintSearch(X_unpacked_again, titles)

    # Compare top-k results for a handful of random queries
    k = 10
    for q_idx in torch.randint(0, n_items, (10,)):
        q = X[q_idx]
        res_a = search_unpacked.search(q, k=k, show_pattern_analysis=False)
        res_b = search_packed_path.search(q, k=k, show_pattern_analysis=False)
        # Compare titles in order
        titles_a = [t for t, _, _ in res_a]
        titles_b = [t for t, _, _ in res_b]
        assert titles_a == titles_b


def test_benchmark_loader_handles_both(tmp_path):
    torch.manual_seed(1)
    n_items, n_bits = 50, 130  # non-multiple-of-8 to stress packing
    titles = _titles(n_items)
    X = _random_binary(n_items, n_bits)

    # 1) Unpacked format
    model_dir_unpacked = tmp_path / "model_unpacked"
    model_dir_unpacked.mkdir()
    fp_path_unpacked = model_dir_unpacked / "fingerprints.pt"
    torch.save({
        'fingerprints': X,
        'titles': titles,
        'metadata': {
            'n_titles': n_items,
            'n_bits': n_bits,
        }
    }, fp_path_unpacked)

    bs_unpacked = BenchmarkSuite(model_dir=str(model_dir_unpacked))
    X_loaded_u, titles_u, mem_mb_u = bs_unpacked._load_fingerprints_for_search()
    assert X_loaded_u.shape == (n_items, n_bits)
    assert titles_u == titles
    expected_mb_u = X.numel() * X.element_size() / 1024**2
    assert math.isclose(mem_mb_u, expected_mb_u, rel_tol=1e-6, abs_tol=1e-9)

    # 2) Packed format
    model_dir_packed = tmp_path / "model_packed"
    model_dir_packed.mkdir()
    fp_path_packed = model_dir_packed / "fingerprints.pt"
    packed = pack_fingerprints(X, bitorder='big')
    torch.save({
        'fingerprints_packed': packed,
        'titles': titles,
        'packed': True,
        'bitorder': 'big',
        'n_bits': n_bits,
        'metadata': {
            'n_titles': n_items,
            'n_bits': n_bits,
        }
    }, fp_path_packed)

    bs_packed = BenchmarkSuite(model_dir=str(model_dir_packed))
    X_loaded_p, titles_p, mem_mb_p = bs_packed._load_fingerprints_for_search()
    assert X_loaded_p.shape == (n_items, n_bits)
    assert titles_p == titles
    assert torch.equal(X_loaded_p, X)
    if isinstance(packed, np.ndarray):
        expected_mb_p = packed.size * packed.itemsize / 1024**2
    else:
        expected_mb_p = packed.numel() * packed.element_size() / 1024**2
    assert math.isclose(mem_mb_p, expected_mb_p, rel_tol=1e-6, abs_tol=1e-9)
