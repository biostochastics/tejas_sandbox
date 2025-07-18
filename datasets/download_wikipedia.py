"""
Wikipedia Dataset Downloader
======================================================

Downloads Wikipedia titles directly from HuggingFace Hub parquet files.
Compatible with datasets library 3.0+

"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download


class WikipediaDownloaderV2:
    """
    Downloads Wikipedia data directly from HuggingFace Hub parquet files.
    Works with modern datasets library versions.
    """
    
    def __init__(self, 
                 output_dir: str = "data/wikipedia",
                 log_dir: str = "logs",
                 cache_dir: Optional[str] = None):
        """Initialize downloader with configurable paths."""
        # Setup directories
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # HuggingFace API
        self.api = HfApi()
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"
        
        # Performance tracking
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_titles': 0,
            'unique_titles': 0,
            'memory_peak_mb': 0,
            'download_time_sec': 0,
            'processing_time_sec': 0
        }
        
        self.logger.info(f"Initialized WikipediaDownloaderV2")
        self.logger.info(f"Using direct parquet file method")
    
    def _setup_logging(self):
        """Configure logging."""
        self.logger = logging.getLogger('WikipediaDownloaderV2')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler
        log_file = self.log_dir / f"download_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def find_wikipedia_datasets(self) -> Dict[str, List[str]]:
        """Find available Wikipedia datasets on HuggingFace Hub."""
        self.logger.info("Searching for Wikipedia datasets on HuggingFace Hub...")
        
        # Known Wikipedia dataset repositories
        wikipedia_repos = [
            "wikimedia/wikipedia",  # New official repo
            "wikipedia",  # Old repo (might not work)
            "graelo/wikipedia"  # Alternative
        ]
        
        available = {}
        
        for repo in wikipedia_repos:
            try:
                # List files in repository
                files = self.api.list_repo_files(repo, repo_type="dataset")
                
                # Find parquet files
                parquet_files = [f for f in files if f.endswith('.parquet')]
                
                if parquet_files:
                    available[repo] = parquet_files
                    self.logger.info(f"Found {len(parquet_files)} parquet files in {repo}")
                
            except Exception as e:
                self.logger.debug(f"Repository {repo} not accessible: {e}")
                continue
        
        return available
    
    def download_wikipedia_parquet(self, 
                                  language: str = "en",
                                  date: str = "20231101",
                                  max_titles: Optional[int] = None) -> Dict[str, any]:
        """
        Download Wikipedia using direct parquet file access.
        
        Args:
            language: Language code
            date: Date string (used for output naming)
            max_titles: Maximum number of titles
            
        Returns:
            Download results dictionary
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting Wikipedia download (Parquet method)")
        self.logger.info(f"Language: {language}, Max titles: {max_titles or 'all'}")
        self.logger.info("="*80)
        
        self.metrics['start_time'] = time.time()
        
        try:
            # Find the best repository
            repo_id = "wikimedia/wikipedia"  # Most reliable
            
            self.logger.info(f"Using repository: {repo_id}")
            
            # Download configuration
            config_name = f"{date}.{language}"
            
            # Alternative: List available configs
            try:
                from datasets import get_dataset_config_names
                configs = get_dataset_config_names(repo_id)
                
                # Find matching config
                matching = [c for c in configs if language in c]
                if matching:
                    config_name = matching[-1]  # Use most recent
                    self.logger.info(f"Found config: {config_name}")
                else:
                    self.logger.warning(f"No config found for {language}, trying default")
                    config_name = "20231101.en"  # Fallback
                    
            except:
                self.logger.info("Could not list configs, using direct download")
            
            # Download and process
            titles = self._download_and_extract_titles(repo_id, config_name, max_titles)
            
            # Save results
            output_path = self._save_titles(titles, language, date)
            
            # Metrics
            self.metrics['end_time'] = time.time()
            self.metrics['total_time_sec'] = self.metrics['end_time'] - self.metrics['start_time']
            self._save_metrics(language, date)
            self._log_summary()
            
            return {
                'success': True,
                'output_path': str(output_path),
                'metrics': self.metrics,
                'language': language,
                'date': date
            }
            
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _download_and_extract_titles(self, 
                                   repo_id: str, 
                                   config_name: str,
                                   max_titles: Optional[int]) -> List[str]:
        """Download parquet files and extract titles."""
        # Try using datasets library first (newer method)
        try:
            from datasets import load_dataset
            
            self.logger.info(f"Attempting to load dataset {repo_id} with config {config_name}")
            
            # Load with streaming for memory efficiency
            dataset = load_dataset(
                repo_id,
                config_name,
                split="train",
                streaming=True,
                trust_remote_code=True  # Allow new loading method
            )
            
            return self._extract_titles_streaming(dataset, max_titles)
            
        except Exception as e:
            self.logger.warning(f"Datasets library method failed: {e}")
            self.logger.info("Falling back to direct parquet download...")
            
            # Fallback: Direct parquet download
            return self._download_parquet_direct(repo_id, config_name, max_titles)
    
    def _extract_titles_streaming(self, dataset, max_titles: Optional[int]) -> List[str]:
        """Extract titles from streaming dataset."""
        self.logger.info("Extracting titles from streaming dataset...")
        
        titles = []
        seen_titles = set()
        
        pbar = tqdm(desc="Extracting titles", unit="articles")
        
        for i, article in enumerate(dataset):
            # Extract title (handle different field names)
            title = article.get('title', '') or article.get('name', '') or article.get('page_title', '')
            title = str(title).strip()
            
            if title and title not in seen_titles:
                titles.append(title)
                seen_titles.add(title)
            
            pbar.update(1)
            
            if max_titles and len(titles) >= max_titles:
                break
                
        pbar.close()
        
        self.metrics['total_titles'] = len(seen_titles)
        self.metrics['unique_titles'] = len(titles)
        
        return titles
    
    def _download_parquet_direct(self, repo_id: str, config_name: str, max_titles: Optional[int]) -> List[str]:
        """Direct parquet file download method."""
        self.logger.info("Using direct parquet download method...")
        
        # List parquet files
        try:
            files = self.api.list_repo_files(repo_id, repo_type="dataset")
            
            # Find parquet files for our config
            parquet_files = [f for f in files if '.parquet' in f and config_name in f]
            
            if not parquet_files:
                # Try without config name
                parquet_files = [f for f in files if '.parquet' in f and '/train/' in f]
            
            if not parquet_files:
                raise ValueError("No parquet files found")
                
            self.logger.info(f"Found {len(parquet_files)} parquet files")
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {e}")
            raise
        
        # Download and process parquet files
        titles = []
        seen_titles = set()
        
        for parquet_file in tqdm(parquet_files[:5], desc="Processing parquet files"):  # Limit to first 5 files
            try:
                # Download file
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=parquet_file,
                    repo_type="dataset",
                    cache_dir=self.cache_dir
                )
                
                # Read parquet
                df = pd.read_parquet(local_path, columns=['title'])
                
                # Extract unique titles
                for title in df['title'].dropna():
                    title = str(title).strip()
                    if title and title not in seen_titles:
                        titles.append(title)
                        seen_titles.add(title)
                
                if max_titles and len(titles) >= max_titles:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to process {parquet_file}: {e}")
                continue
        
        self.metrics['total_titles'] = len(seen_titles)
        self.metrics['unique_titles'] = len(titles)
        
        return titles
    
    def _save_titles(self, titles: List[str], language: str, date: str) -> Path:
        """Save titles to multiple formats."""
        self.logger.info(f"Saving {len(titles)} titles...")
        
        filename_base = f"wikipedia_{language}_{date}_titles"
        
        # Save as text file
        txt_path = self.output_dir / f"{filename_base}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for title in titles:
                f.write(f"{title}\n")
        
        # Save as numpy
        npy_path = self.output_dir / f"{filename_base}.npy"
        np.save(npy_path, np.array(titles, dtype=object))
        
        # Save as PyTorch
        pt_path = self.output_dir / f"{filename_base}.pt"
        torch.save({
            'titles': titles,
            'metadata': {
                'language': language,
                'date': date,
                'count': len(titles),
                'timestamp': datetime.now().isoformat()
            }
        }, pt_path)
        
        # Save sample as JSON
        json_path = self.output_dir / f"{filename_base}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'language': language,
                'date': date,
                'total_titles': len(titles),
                'titles_sample': titles[:1000]
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved all formats to {self.output_dir}")
        
        return txt_path
    
    def _save_metrics(self, language: str, date: str):
        """Save performance metrics."""
        metrics_path = self.output_dir / f"metrics_{language}_{date}.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def _log_summary(self):
        """Log summary of operation."""
        self.logger.info("="*80)
        self.logger.info("DOWNLOAD SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Total titles:      {self.metrics['total_titles']:,}")
        self.logger.info(f"Unique titles:     {self.metrics['unique_titles']:,}")
        self.logger.info(f"Total time:        {self.metrics['total_time_sec']:.2f} sec")
        self.logger.info("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Wikipedia titles (v2)")
    parser.add_argument("--language", "-l", default="en", help="Language code")
    parser.add_argument("--date", "-d", default="20231101", help="Date for naming")
    parser.add_argument("--max-titles", "-m", type=int, help="Maximum titles")
    parser.add_argument("--output-dir", "-o", default="data/wikipedia")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = WikipediaDownloaderV2(output_dir=args.output_dir)
    
    # Try modern parquet method
    result = downloader.download_wikipedia_parquet(
        language=args.language,
        date=args.date,
        max_titles=args.max_titles
    )
    
    # If that fails, suggest using old version
    if not result['success']:
        print("\n" + "="*80)
        print("SUGGESTION: If the modern method fails, try:")
        print("1. pip install datasets==2.14.0")
        print("2. python download_wikipedia.py (original version)")
        print("="*80)
    
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()