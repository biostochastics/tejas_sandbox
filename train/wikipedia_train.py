"""
Wikipedia Dataset Training Module
=================================

Trains consciousness-aligned fingerprint encoder on Wikipedia titles.
Uses golden ratio sampling for optimal pattern capture.

Author: Quantum Semantic Framework
"""

import time
import logging
import torch
import numpy as np
import traceback
from pathlib import Path
from typing import Union, List
from datetime import datetime

# Import core modules
from core.encoder import GoldenRatioEncoder
from core.decoder import SemanticDecoder

logger = logging.getLogger(__name__)


class WikipediaTrainer:
    """
    Trainer for Wikipedia fingerprint encoder.
    Encapsulates the complete training pipeline.
    """
    
    def __init__(self, 
                 n_bits: int = 128,
                 max_features: int = 10000,
                 output_dir: str = "models/fingerprint_encoder",
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            n_bits: Number of bits in fingerprints
            max_features: Maximum n-gram features
            output_dir: Directory to save trained model
            device: Device for computation
        """
        try:
            self.n_bits = n_bits
            self.max_features = max_features
            self.output_dir = Path(output_dir)
            self.device = device
            
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Initialized WikipediaTrainer")
            logger.info(f"  Bits: {n_bits}")
            logger.info(f"  Max features: {max_features}")
            logger.info(f"  Output: {output_dir}")
            logger.info(f"  Device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WikipediaTrainer: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def load_dataset(self, dataset_path: Union[str, Path]) -> List[str]:
        """
        Load dataset from file.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            List of titles
        """
        try:
            dataset_path = Path(dataset_path)
            
            logger.info(f"Loading dataset from {dataset_path}")
            
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            if dataset_path.suffix == '.txt':
                # Text file with one title per line
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    titles = [line.strip() for line in f if line.strip()]
                    
            elif dataset_path.suffix == '.npy':
                # NumPy array
                titles = np.load(dataset_path, allow_pickle=True).tolist()
                
            elif dataset_path.suffix == '.pt':
                # PyTorch file
                data = torch.load(dataset_path)
                if isinstance(data, dict) and 'titles' in data:
                    titles = data['titles']
                else:
                    titles = data
                    
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
            
            logger.info(f"Loaded {len(titles):,} titles")
            
            # Basic validation
            if len(titles) == 0:
                raise ValueError("No titles found in dataset")
                
            # Show sample
            logger.info("Sample titles:")
            for i, title in enumerate(titles[:5]):
                logger.info(f"  {i+1}. {title}")
            
            return titles
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def train(self,
              dataset_path: Union[str, Path],
              memory_limit_gb: int = 50,
              batch_size: int = 10000,
              max_titles: int = None):
        """
        Train the encoder on dataset.
        
        Args:
            dataset_path: Path to dataset
            memory_limit_gb: Memory limit for training
            batch_size: Batch size for encoding
            max_titles: Maximum number of titles to use (None = use all)
        """
        start_time = time.time()
        
        try:
            # Load dataset
            titles = self.load_dataset(dataset_path)
            
            # Limit titles if requested (useful for testing)
            if max_titles is not None and max_titles < len(titles):
                logger.info(f"Limiting dataset to {max_titles:,} titles (from {len(titles):,})")
                titles = titles[:max_titles]
            
            # Create encoder using our consciousness-aligned architecture
            logger.info("\nCreating consciousness-aligned encoder...")
            encoder = GoldenRatioEncoder(
                n_bits=self.n_bits,
                max_features=self.max_features,
                device=self.device
            )
            
            # Train encoder with golden ratio sampling
            logger.info("\nTraining encoder with golden ratio sampling...")
            encoder.fit(titles, memory_limit_gb=memory_limit_gb)
            
            # Encode all titles to binary fingerprints
            logger.info("\nEncoding all titles to binary fingerprints...")
            fingerprints = encoder.transform(titles, batch_size=batch_size)
            
            # Log statistics
            self._log_fingerprint_stats(fingerprints)
            
            # Save encoder
            logger.info("\nSaving encoder...")
            try:
                encoder.save(self.output_dir)
                logger.info("Encoder saved successfully")
            except Exception as e:
                logger.error(f"Failed to save encoder: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            # Save fingerprints
            logger.info("Saving fingerprints...")
            try:
                fingerprint_data = {
                    'fingerprints': fingerprints,
                    'titles': titles,
                    'metadata': {
                        'n_titles': len(titles),
                        'n_bits': self.n_bits,
                        'timestamp': datetime.now().isoformat(),
                        'training_time': time.time() - start_time
                    }
                }
                torch.save(fingerprint_data, self.output_dir / 'fingerprints.pt')
                logger.info("Fingerprints saved successfully")
            except Exception as e:
                logger.error(f"Failed to save fingerprints: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            # Create decoder
            logger.info("\nCreating decoder...")
            try:
                decoder = SemanticDecoder.from_encoder(self.output_dir)
                decoder.save(self.output_dir / 'decoder')
                logger.info("Decoder created and saved successfully")
            except Exception as e:
                logger.error(f"Failed to create/save decoder: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            # Final summary
            total_time = time.time() - start_time
            logger.info("\n" + "="*50)
            logger.info("Training Complete!")
            logger.info("="*50)
            logger.info(f"Total time: {total_time/60:.2f} minutes")
            logger.info(f"Titles encoded: {len(titles):,}")
            logger.info(f"Model saved to: {self.output_dir}")
            logger.info(f"Fingerprint size: {self.n_bits} bits")
            logger.info(f"Database size: {fingerprints.nbytes / 1e9:.2f} GB")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _log_fingerprint_stats(self, fingerprints: torch.Tensor):
        """Log statistics about the fingerprints."""
        try:
            logger.info("\nFingerprint Statistics:")
            
            # Channel activation rates
            activation_rates = fingerprints.float().mean(dim=0)
            
            logger.info(f"  Shape: {fingerprints.shape}")
            logger.info(f"  Mean activation: {activation_rates.mean():.3f}")
            logger.info(f"  Std activation: {activation_rates.std():.3f}")
            
            # Channel balance
            balanced = ((activation_rates > 0.4) & (activation_rates < 0.6)).sum()
            logger.info(f"  Balanced channels: {balanced}/{self.n_bits} ({balanced/self.n_bits*100:.1f}%)")
            
            # Entropy
            def entropy(p):
                if p == 0 or p == 1:
                    return 0
                return -p * np.log2(p) - (1-p) * np.log2(1-p)
            
            channel_entropies = [entropy(p.item()) for p in activation_rates]
            mean_entropy = np.mean(channel_entropies)
            logger.info(f"  Mean channel entropy: {mean_entropy:.3f} bits")
            
            # Sample diversity (using Hamming distances)
            if len(fingerprints) > 100:
                sample_indices = torch.randperm(len(fingerprints))[:100]
                sample = fingerprints[sample_indices]
                
                # Compute pairwise Hamming distances
                distances = []
                for i in range(len(sample)):
                    for j in range(i+1, len(sample)):
                        dist = (sample[i] ^ sample[j]).sum().item()
                        distances.append(dist)
                
                mean_dist = np.mean(distances)
                logger.info(f"  Mean pairwise distance: {mean_dist:.1f} bits")
                logger.info(f"  Distance/dimension: {mean_dist/self.n_bits:.3f}")
                
        except Exception as e:
            logger.error(f"Failed to log fingerprint stats: {str(e)}")
            logger.error(traceback.format_exc())
            # Don't raise - this is just logging


def main():
    """Standalone training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Wikipedia fingerprint encoder")
    parser.add_argument("dataset", help="Path to dataset file")
    parser.add_argument("--bits", type=int, default=128, help="Number of bits")
    parser.add_argument("--output", default="models/fingerprint_encoder", help="Output directory")
    parser.add_argument("--memory-limit", type=int, default=50, help="Memory limit in GB")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    try:
        trainer = WikipediaTrainer(
            n_bits=args.bits,
            output_dir=args.output,
            device=args.device
        )
        
        trainer.train(
            dataset_path=args.dataset,
            memory_limit_gb=args.memory_limit
        )
        
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()