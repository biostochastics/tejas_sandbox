#!/usr/bin/env python3
"""
Prepare multi-scale Wikipedia datasets for benchmarking
Creates: 10k, 125k, 250k, 500k, 1m, 5m title subsets
"""

import os
import random
from pathlib import Path

def create_scale_datasets():
    """Create Wikipedia datasets at various scales."""
    
    # Setup paths
    data_dir = Path("data/wikipedia")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Read full dataset
    full_file = data_dir / "wikipedia_en_20231101_titles.txt"
    if not full_file.exists():
        print(f"Error: {full_file} not found!")
        return
    
    print(f"Reading full dataset from {full_file}...")
    with open(full_file, 'r', encoding='utf-8') as f:
        all_titles = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(all_titles):,} titles")
    
    # Define scales to create
    scales = [
        10_000,
        125_000,
        250_000,
        500_000,
        1_000_000,
        5_000_000
    ]
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Shuffle titles once
    shuffled_titles = all_titles.copy()
    random.shuffle(shuffled_titles)
    
    # Create datasets at each scale
    for scale in scales:
        if scale > len(all_titles):
            print(f"Skipping {scale:,} (larger than dataset)")
            continue
            
        output_file = data_dir / f"wikipedia_{scale}.txt"
        
        # Check if already exists
        if output_file.exists():
            print(f"File exists: {output_file.name}")
            continue
        
        print(f"Creating {output_file.name}...")
        
        # Select first N titles from shuffled list (ensures consistency)
        subset = shuffled_titles[:scale]
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for title in subset:
                f.write(title + '\n')
        
        # Report file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  Created: {scale:,} titles, {size_mb:.1f} MB")
    
    # List all datasets
    print("\nAvailable datasets:")
    for file in sorted(data_dir.glob("wikipedia_*.txt")):
        with open(file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name}: {count:,} titles, {size_mb:.1f} MB")

if __name__ == "__main__":
    create_scale_datasets()