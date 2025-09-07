#!/usr/bin/env python3
"""
Output Manifest System for DOE Framework

This module tracks all generated artifacts during experiments,
creating a comprehensive manifest for reproducibility and auditing.
"""

import os
import json
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ManifestTracker:
    """
    Singleton class to track all outputs generated during experiments.
    
    Usage:
        tracker = ManifestTracker.get_instance()
        tracker.add_output('results.json', 'results')
        tracker.save_manifest()
    """
    
    _instance = None
    
    def __init__(self):
        if ManifestTracker._instance is not None:
            raise RuntimeError("Use ManifestTracker.get_instance() instead")
        
        self.manifest = {
            'schema_version': '1.0',
            'experiment_id': None,
            'timestamp': datetime.now().isoformat(),
            'configuration': {},
            'outputs': {
                'results_json': [],
                'csv_files': [],
                'graphs': {
                    'plotly_html': [],
                    'matplotlib_png': [],
                    'matplotlib_pdf': []
                },
                'logs': [],
                'other': []
            },
            'checksums': {},
            'environment': self._capture_environment(),
            'git_info': self._capture_git_info()
        }
        self.output_dir = None
    
    @classmethod
    def get_instance(cls) -> 'ManifestTracker':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None
    
    def set_experiment_id(self, exp_id: str):
        """Set the experiment ID."""
        self.manifest['experiment_id'] = exp_id
    
    def set_configuration(self, config: Dict[str, Any]):
        """Set the experiment configuration."""
        self.manifest['configuration'] = config
    
    def set_output_dir(self, output_dir: Path):
        """Set the output directory for tracking."""
        self.output_dir = Path(output_dir)
    
    def add_output(self, filepath: str, category: str = 'other'):
        """
        Add an output file to the manifest.
        
        Args:
            filepath: Path to the output file
            category: Category of output ('results_json', 'csv_files', 'graphs', etc.)
        """
        filepath = Path(filepath)
        
        # Determine category based on file extension if not specified
        if category == 'other':
            if filepath.suffix == '.json':
                category = 'results_json'
            elif filepath.suffix == '.csv':
                category = 'csv_files'
            elif filepath.suffix == '.html':
                category = 'graphs'
                sub_category = 'plotly_html'
            elif filepath.suffix in ['.png', '.jpg', '.jpeg']:
                category = 'graphs'
                sub_category = 'matplotlib_png'
            elif filepath.suffix == '.pdf':
                category = 'graphs'
                sub_category = 'matplotlib_pdf'
            elif filepath.suffix == '.log':
                category = 'logs'
        
        # Add to appropriate category
        if category == 'graphs' and 'sub_category' in locals():
            self.manifest['outputs']['graphs'][sub_category].append(str(filepath))
        elif category in self.manifest['outputs']:
            if isinstance(self.manifest['outputs'][category], list):
                self.manifest['outputs'][category].append(str(filepath))
        else:
            self.manifest['outputs']['other'].append(str(filepath))
        
        # Generate checksum if file exists
        if filepath.exists():
            checksum = self._generate_checksum(filepath)
            self.manifest['checksums'][str(filepath)] = checksum
            logger.debug(f"Added {filepath} to manifest with checksum {checksum[:8]}...")
    
    def track_graph(self, save_path: str, graph_type: str = 'unknown'):
        """
        Track a graph file with metadata.
        
        Args:
            save_path: Path where graph was saved
            graph_type: Type of graph (main_effects, interaction, pareto, etc.)
        """
        self.add_output(save_path, 'graphs')
        
        # Add metadata
        if 'graph_metadata' not in self.manifest:
            self.manifest['graph_metadata'] = {}
        
        self.manifest['graph_metadata'][save_path] = {
            'type': graph_type,
            'created': datetime.now().isoformat()
        }
    
    def _generate_checksum(self, filepath: Path) -> str:
        """Generate SHA256 checksum for a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate checksum for {filepath}: {e}")
            return "error"
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information."""
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_path': sys.executable,
            'cwd': os.getcwd()
        }
        
        # Capture installed packages
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                env_info['packages'] = result.stdout.strip().split('\n')
        except Exception as e:
            logger.warning(f"Failed to capture pip packages: {e}")
            env_info['packages'] = []
        
        return env_info
    
    def _capture_git_info(self) -> Dict[str, Any]:
        """Capture git repository information."""
        git_info = {}
        
        try:
            # Get current commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                git_info['has_uncommitted_changes'] = len(result.stdout.strip()) > 0
            
        except Exception as e:
            logger.warning(f"Failed to capture git info: {e}")
        
        return git_info
    
    def save_manifest(self, filepath: Optional[str] = None) -> str:
        """
        Save the manifest to a JSON file.
        
        Args:
            filepath: Path to save manifest (default: output_dir/experiment_manifest.json)
            
        Returns:
            Path where manifest was saved
        """
        if filepath is None:
            if self.output_dir:
                filepath = self.output_dir / 'experiment_manifest.json'
            else:
                filepath = Path('experiment_manifest.json')
        
        filepath = Path(filepath)
        
        # Update timestamp
        self.manifest['save_timestamp'] = datetime.now().isoformat()
        
        # Save manifest
        with open(filepath, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        
        logger.info(f"Manifest saved to {filepath}")
        return str(filepath)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of tracked outputs."""
        summary = {
            'experiment_id': self.manifest['experiment_id'],
            'total_outputs': 0,
            'by_category': {}
        }
        
        for category, items in self.manifest['outputs'].items():
            if isinstance(items, list):
                count = len(items)
            elif isinstance(items, dict):
                count = sum(len(v) for v in items.values())
            else:
                count = 0
            
            summary['by_category'][category] = count
            summary['total_outputs'] += count
        
        return summary
    
    def validate_outputs(self) -> Dict[str, List[str]]:
        """
        Validate that all tracked outputs exist and match checksums.
        
        Returns:
            Dictionary with 'missing' and 'checksum_mismatch' lists
        """
        validation = {
            'missing': [],
            'checksum_mismatch': []
        }
        
        for filepath, expected_checksum in self.manifest['checksums'].items():
            path = Path(filepath)
            if not path.exists():
                validation['missing'].append(filepath)
            elif expected_checksum != 'error':
                actual_checksum = self._generate_checksum(path)
                if actual_checksum != expected_checksum:
                    validation['checksum_mismatch'].append(filepath)
        
        return validation


def integrate_with_doe_analysis(original_func):
    """
    Decorator to automatically track graph outputs from DOE analysis functions.
    
    Usage:
        @integrate_with_doe_analysis
        def plot_main_effects(self, response, save_path=None):
            ...
    """
    def wrapper(*args, **kwargs):
        # Call original function
        result = original_func(*args, **kwargs)
        
        # Track save_path if provided
        if 'save_path' in kwargs and kwargs['save_path']:
            tracker = ManifestTracker.get_instance()
            graph_type = original_func.__name__.replace('plot_', '')
            tracker.track_graph(kwargs['save_path'], graph_type)
        
        return result
    
    wrapper.__name__ = original_func.__name__
    wrapper.__doc__ = original_func.__doc__
    return wrapper