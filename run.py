#!/usr/bin/env python3
"""
Tejas: Quantum Semantic Fingerprint Framework
============================================

Unified entry point for training and searching with consciousness-aligned fingerprints.

Usage:
    Training:
        python run.py --mode train --dataset path/to/data.pt --output models/my_model
    
    Demo (Interactive Search):
        python run.py --mode demo --model models/my_model

Author: Quantum Semantic Framework
"""

import argparse
import sys
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Tejas: Quantum Semantic Fingerprint Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train on Wikipedia dataset:
    python run.py --mode train --dataset data/wikipedia/wikipedia_en_20231101_titles.pt --bits 128
    
  Run interactive search demo:
    python run.py --mode demo --model models/fingerprint_encoder
    
  Run demo with specific query:
    python run.py --mode demo --model models/fingerprint_encoder --query "quantum mechanics"
    
  Benchmark search performance:
    python run.py --mode benchmark --model models/fingerprint_encoder
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True,
        choices=['train', 'demo', 'benchmark'],
        help='Operation mode: train, demo, or benchmark'
    )
    
    # Global arguments (used by multiple modes)
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'auto'],
        help='Device for computation (default: auto)'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training options')
    train_group.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset file (required for training)'
    )
    train_group.add_argument(
        '--output',
        type=str,
        default='models/fingerprint_encoder',
        help='Output directory for trained model (default: models/fingerprint_encoder)'
    )
    train_group.add_argument(
        '--bits',
        type=int,
        default=128,
        help='Number of bits in fingerprint (default: 128)'
    )
    train_group.add_argument(
        '--max-features',
        type=int,
        default=10000,
        help='Maximum number of n-gram features (default: 10000)'
    )
    train_group.add_argument(
        '--memory-limit',
        type=int,
        default=50,
        help='Memory limit in GB for training (default: 50)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for encoding (default: 10000)'
    )
    train_group.add_argument(
        '--max-titles',
        type=int,
        default=None,
        help='Maximum titles to use (for testing, default: use all)'
    )
    
    # Demo arguments
    demo_group = parser.add_argument_group('Demo options')
    demo_group.add_argument(
        '--model',
        type=str,
        default='models/fingerprint_encoder',
        help='Path to trained model directory (default: models/fingerprint_encoder)'
    )
    demo_group.add_argument(
        '--query',
        type=str,
        help='Search query (for non-interactive demo)'
    )
    demo_group.add_argument(
        '--pattern',
        type=str,
        help='Pattern to search for (e.g., "List of")'
    )
    demo_group.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    
    try:
        args = parser.parse_args()
        
        # Validate arguments based on mode
        if args.mode == 'train':
            if not args.dataset:
                parser.error("--dataset is required for training mode")
            
            # Import and run training
            from train.wikipedia_train import WikipediaTrainer
            
            # Handle 'auto' device selection for training
            device = args.device
            if device == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Auto-selected device: {device}")
            
            trainer = WikipediaTrainer(
                n_bits=args.bits,
                max_features=args.max_features,
                output_dir=args.output,
                device=device
            )
            
            logger.info(f"Starting training with dataset: {args.dataset}")
            trainer.train(
                dataset_path=args.dataset,
                memory_limit_gb=args.memory_limit,
                batch_size=args.batch_size,
                max_titles=args.max_titles
            )
            
        elif args.mode == 'demo':
            # Import and run demo
            from demo.wikipedia_demo import WikipediaDemo
            
            demo = WikipediaDemo(
                model_dir=args.model,
                device=args.device
            )
            
            if args.query:
                # Single query mode
                results = demo.search(args.query, k=args.top_k)
                demo.display_results(args.query, results)
            elif args.pattern:
                # Pattern search mode
                results = demo.search_pattern(args.pattern)
                demo.display_pattern_results(args.pattern, results)
            else:
                # Interactive mode
                demo.interactive()
                
        elif args.mode == 'benchmark':
            # Import and run benchmark
            from demo.wikipedia_demo import WikipediaDemo
            
            demo = WikipediaDemo(
                model_dir=args.model,
                device=args.device
            )
            
            demo.benchmark(n_queries=100)
        
        else:
            parser.error(f"Unknown mode: {args.mode}")
            
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()