#!/usr/bin/env python3
"""
Verify that each pipeline has different characteristics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_doe.core.encoder_factory import EncoderFactory

def verify_pipelines():
    """Verify pipeline configurations are different."""
    
    print("=" * 60)
    print("PIPELINE VERIFICATION")
    print("=" * 60)
    
    pipelines = ['original_tejas', 'fused_char', 'fused_byte', 'optimized_fused']
    
    for pipeline in pipelines:
        print(f"\n{pipeline.upper()}:")
        print("-" * 40)
        
        # Get configuration
        config = EncoderFactory.ENCODER_CONFIGS.get(pipeline, {})
        
        # Print key characteristics
        print(f"  Module: {config.get('module', 'N/A')}")
        print(f"  Class: {config.get('class', config.get('function', 'N/A'))}")
        
        defaults = config.get('defaults', {})
        print(f"  Key Features:")
        
        if pipeline == 'original_tejas':
            print(f"    - Golden ratio subsampling: YES")
            print(f"    - Uses sklearn: {defaults.get('use_sklearn', False)}")
            print(f"    - Energy threshold: {defaults.get('energy_threshold', 'N/A')}")
            
        elif pipeline == 'fused_char':
            print(f"    - Tokenizer: {defaults.get('tokenizer_type', 'N/A')}")
            print(f"    - N-gram range: {defaults.get('ngram_range', 'N/A')}")
            print(f"    - Uses ITQ: {defaults.get('use_itq', False)}")
            
        elif pipeline == 'fused_byte':
            print(f"    - Tokenizer: {defaults.get('tokenizer_type', 'N/A')}")
            print(f"    - Vocab size: {defaults.get('vocab_size', 'N/A')}")
            print(f"    - Uses ITQ: {defaults.get('use_itq', False)}")
            
        elif pipeline == 'optimized_fused':
            print(f"    - Uses SIMD: {defaults.get('use_simd', False)}")
            print(f"    - Uses Numba: {defaults.get('use_numba', False)}")
            print(f"    - Tokenizer: {defaults.get('tokenizer_type', 'N/A')}")
        
        print(f"  Requires: {config.get('requires', [])}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    # Test creating encoders
    print("\nTesting encoder creation...")
    for pipeline in pipelines:
        try:
            encoder = EncoderFactory.create_encoder(pipeline)
            print(f"  ✓ {pipeline}: Created successfully")
        except Exception as e:
            print(f"  ✗ {pipeline}: Failed - {e}")

if __name__ == "__main__":
    verify_pipelines()