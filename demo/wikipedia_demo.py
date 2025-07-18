"""
Wikipedia Search Demo Module
============================

Interactive demonstration of consciousness-aligned search.
Uses the core fingerprint module for XOR-based hardware-speed search.

"""

import time
import logging
import torch
import numpy as np
import traceback
from pathlib import Path
from typing import List, Tuple, Union
import urllib.request
import zipfile
import shutil

# Import our consciousness-aligned core modules
from core.encoder import GoldenRatioEncoder
from core.fingerprint import BinaryFingerprintSearch
from core.decoder import SemanticDecoder

logger = logging.getLogger(__name__)


class WikipediaDemo:
    """
    Interactive demo for Wikipedia fingerprint search.
    Demonstrates the consciousness-aligned search capabilities.
    """
    
    def __init__(self, model_dir: str = "models/fingerprint_encoder", device: str = 'auto'):
        """
        Initialize demo with trained model.
        
        Args:
            model_dir: Directory containing trained model
            device: Device for computation ('cpu', 'cuda', or 'auto')
        """
        try:
            self.model_dir = Path(model_dir)
            
            # Check if model exists, download if not
            self._ensure_model_exists()
            
            # Load encoder
            logger.info("Loading consciousness-aligned encoder...")
            self.encoder = GoldenRatioEncoder()
            self.encoder.load(self.model_dir)
            
            # Load decoder for pattern analysis
            decoder_dir = self.model_dir / 'decoder'
            if decoder_dir.exists():
                logger.info("Loading semantic decoder...")
                self.decoder = SemanticDecoder()
                self.decoder.load(decoder_dir)
            else:
                logger.warning("Decoder not found - pattern analysis will be limited")
                self.decoder = None
            
            # Load fingerprints and create search engine
            logger.info("Loading fingerprint database...")
            fingerprint_data = torch.load(self.model_dir / "fingerprints.pt")
            
            # Initialize our core fingerprint search module
            self.search_engine = BinaryFingerprintSearch(
                fingerprints=fingerprint_data['fingerprints'],
                titles=fingerprint_data['titles'],
                device=device
            )
            
            logger.info(f"Loaded {len(self.search_engine.titles):,} consciousness fingerprints")
            logger.info("Ready for quantum-speed search!")
            
        except Exception as e:
            logger.error(f"Failed to initialize WikipediaDemo: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _ensure_model_exists(self):
        """Download model if it doesn't exist locally."""
        try:
            required_files = [
                "fingerprints.pt",
                "config.json",
                "projection.npy",
                "vocabulary.npy",
                "idf_weights.npy"
            ]
            
            if all((self.model_dir / f).exists() for f in required_files):
                logger.info("Model files found locally")
                return
            
            # Download logic
            logger.info("Model not found locally. Downloading...")
            self._download_model()
            
        except Exception as e:
            logger.error(f"Failed to ensure model exists: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _download_model(self):
        """Download model from S3."""
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            download_url = "https://reinforceai-tejas-public.s3.amazonaws.com/ckpt/wikipedia-2022/wikipedia_model.zip"
            zip_path = self.model_dir / "wikipedia_model.zip"
            
            # Download with progress
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                if block_num % 100 == 0:  # Log every 100 blocks
                    logger.info(f"  Downloaded: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)")
            
            logger.info(f"Downloading from: {download_url}")
            urllib.request.urlretrieve(download_url, zip_path, reporthook=download_progress)
            
            # Extract
            logger.info("Extracting model files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                temp_dir = self.model_dir / "temp_extract"
                temp_dir.mkdir(exist_ok=True)
                zip_ref.extractall(temp_dir)
                
                # Move files to correct location
                for file in temp_dir.rglob("*"):
                    if file.is_file():
                        target = self.model_dir / file.name
                        shutil.move(str(file), str(target))
                
                shutil.rmtree(temp_dir)
            
            zip_path.unlink()
            logger.info("Model downloaded successfully!")
            
        except Exception as e:
            if 'zip_path' in locals() and zip_path.exists():
                zip_path.unlink()
            logger.error(f"Failed to download model: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Could not download model: {e}")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float, int]]:
        """
        Search using consciousness-aligned fingerprints.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (title, similarity, distance) tuples
        """
        try:
            # Encode query to fingerprint
            query_fingerprint = self.encoder.encode_single(query)
            
            # Use our core fingerprint search
            results = self.search_engine.search(
                query_fingerprint, 
                k=k,
                show_pattern_analysis=True
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def search_pattern(self, pattern: str, max_results: int = 20) -> List[Tuple[str, float, int]]:
        """
        Search for specific patterns (demonstrates zero false positives).
        
        Args:
            pattern: Pattern to search for
            max_results: Maximum results
            
        Returns:
            Pattern matches
        """
        try:
            return self.search_engine.search_pattern(
                pattern, 
                self.encoder,
                max_results=max_results
            )
        except Exception as e:
            logger.error(f"Pattern search failed for '{pattern}': {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def analyze_fingerprint(self, text: str):
        """
        Analyze the consciousness channels for a text.
        
        Args:
            text: Text to analyze
        """
        try:
            logger.info(f"\nAnalyzing consciousness channels for: '{text}'")
            
            # Encode to fingerprint
            fingerprint = self.encoder.encode_single(text)
            
            # Basic statistics
            active_channels = fingerprint.sum().item()
            logger.info(f"\nChannel Statistics:")
            logger.info(f"  Active channels: {active_channels}/{len(fingerprint)} ({active_channels/len(fingerprint)*100:.1f}%)")
            
            # If decoder available, show patterns
            if self.decoder:
                patterns = self.decoder.decode_patterns(fingerprint, top_k=10)
                logger.info(f"\nTop activated patterns:")
                for pattern, score in patterns[:5]:
                    logger.info(f"  '{pattern}': {score:.3f}")
                    
        except Exception as e:
            logger.error(f"Fingerprint analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
    
    def display_results(self, query: str, results: List[Tuple[str, float, int]]):
        """Display search results."""
        print(f"\nTop {len(results)} results for '{query}':")
        print("-" * 60)
        
        for i, (title, sim, dist) in enumerate(results, 1):
            print(f"{i:2d}. {title}")
            print(f"    Similarity: {sim:.3f} | Distance: {dist} bits")
        
        # Check for exact match
        if query in [r[0] for r in results]:
            print(f"\n✓ Exact match found!")
    
    def display_pattern_results(self, pattern: str, results: List[Tuple[str, float, int]]):
        """Display pattern search results."""
        print(f"\nPattern matches for '{pattern}':")
        for i, (title, sim, dist) in enumerate(results, 1):
            print(f"{i:2d}. {title}")
            print(f"    Similarity: {sim:.3f} | Distance: {dist} bits")
    
    def benchmark(self, n_queries: int = 100):
        """Run performance benchmark."""
        try:
            self.search_engine.benchmark(n_queries)
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def interactive(self):
        """Run interactive search session."""
        print("\n" + "="*60)
        print("Tejas: Quantum Semantic Fingerprint Search")
        print("Ultra-fast Wikipedia search using consciousness-aligned patterns")
        print("="*60)
        print("\nCommands:")
        print("  - Type any query to search")
        print("  - 'pattern:X' to search for pattern X")
        print("  - 'analyze:X' to analyze consciousness channels for X")
        print("  - 'quit' to exit")
        print("-"*60)
        
        while True:
            try:
                query = input("\nSearch query: ").strip()
                
                if query.lower() == 'quit':
                    break
                
                if query.startswith('pattern:'):
                    pattern = query[8:].strip()
                    results = self.search_pattern(pattern)
                    self.display_pattern_results(pattern, results)
                    
                elif query.startswith('analyze:'):
                    text = query[8:].strip()
                    self.analyze_fingerprint(text)
                    
                else:
                    results = self.search(query)
                    self.display_results(query, results)
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                logger.error(traceback.format_exc())
                print(f"\nError: {str(e)}")
                print("Please try again or type 'quit' to exit.")


def main():
    """Standalone demo script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wikipedia fingerprint search demo")
    parser.add_argument("--model", default="models/fingerprint_encoder", help="Model directory")
    parser.add_argument("--query", help="Single query to search")
    parser.add_argument("--pattern", help="Pattern to search for")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")
    
    args = parser.parse_args()
    
    try:
        demo = WikipediaDemo(model_dir=args.model, device=args.device)
        
        if args.benchmark:
            demo.benchmark()
        elif args.query:
            results = demo.search(args.query)
            demo.display_results(args.query, results)
        elif args.pattern:
            results = demo.search_pattern(args.pattern)
            demo.display_pattern_results(args.pattern, results)
        else:
            demo.interactive()
            
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()