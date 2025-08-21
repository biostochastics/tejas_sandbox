"""
Binary Semantic Encoder with Golden Ratio Sampling
=================================================

Transforms TF-IDF vectors into binary fingerprints using SVD and phase collapse.
Implements golden ratio sampling for optimal pattern capture.
"""

import time
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
from tqdm import tqdm
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class GoldenRatioEncoder:
    """
    Encodes text into binary fingerprints using quantum-inspired phase collapse.
    Based on quantum consciousness principles for optimal pattern capture.
    """
    
    def __init__(self, n_bits=128, max_features=10000, device='cpu', threshold_strategy='zero',
                 use_itq=False, itq_iterations=50, use_randomized_svd=False, 
                 svd_n_oversamples=20, svd_n_iter=5):
        self.n_bits = n_bits
        self.max_features = max_features
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.device = device
        self.threshold_strategy = 'itq' if use_itq else threshold_strategy  # 'zero', 'median', 'percentile', 'itq'
        self.use_itq = use_itq
        self.itq_iterations = itq_iterations
        self.use_randomized_svd = use_randomized_svd
        self.svd_n_oversamples = svd_n_oversamples
        self.svd_n_iter = svd_n_iter
        
        # Components to be learned
        self.vectorizer = None
        self.projection = None
        self.singular_values = None
        self.thresholds = None  # For storing learned thresholds
        self.sample_indices = None
        self.training_stats = {}
        self.itq_optimizer = None  # ITQ optimizer if enabled
        self.is_fitted = False  # Track if model is trained
        
        logger.info(f"Initialized GoldenRatioEncoder")
        logger.info(f"  n_bits: {n_bits}")
        logger.info(f"  max_features: {max_features}")
        logger.info(f"  golden_ratio: {self.golden_ratio:.6f}")
        
    def _golden_ratio_sample(self, n_total, target_memory_gb=50, max_memory_gb=100):
        """
        Sample using golden ratio until it fits in memory with hard limits.
        
        Args:
            n_total: Total number of items
            target_memory_gb: Target memory usage
            max_memory_gb: Maximum allowed memory (hard limit)
            
        Returns:
            sample_indices: Indices to sample
        """
        # Enforce maximum memory limit
        if target_memory_gb > max_memory_gb:
            logger.warning(f"Target memory {target_memory_gb}GB exceeds max {max_memory_gb}GB, capping")
            target_memory_gb = max_memory_gb
        
        # Calculate how many samples we can fit
        bytes_per_element = 4  # float32
        elements_per_sample = self.max_features
        bytes_per_sample = bytes_per_element * elements_per_sample
        
        max_samples = int(target_memory_gb * 1e9 / bytes_per_sample)
        
        # Check if even a single sample would exceed memory
        if bytes_per_sample > max_memory_gb * 1e9:
            raise MemoryError(f"Single sample ({bytes_per_sample/1e9:.2f}GB) exceeds max memory {max_memory_gb}GB")
        
        # Apply golden ratio reduction until it fits
        sample_size = n_total
        reduction_level = 0
        
        while sample_size > max_samples:
            sample_size = int(sample_size / self.golden_ratio)
            reduction_level += 1
            
            # Safety check: prevent infinite loop
            if sample_size < 100:
                logger.warning("Sample size too small, using minimum of 100 samples")
                sample_size = min(100, n_total, max_samples)
                break
                
        logger.info(f"Golden ratio sampling:")
        logger.info(f"  Original: {n_total:,} samples")
        logger.info(f"  Reduced: {sample_size:,} samples")
        logger.info(f"  Reduction levels: {reduction_level}")
        logger.info(f"  Coverage: {sample_size/n_total*100:.1f}%")
        logger.info(f"  Estimated memory: {sample_size * bytes_per_sample / 1e9:.2f}GB")
        
        # Use uniform sampling instead of logarithmic for better coverage
        if sample_size < n_total:
            # Uniform sampling provides better coverage than logarithmic
            indices = np.linspace(0, n_total-1, sample_size, dtype=int)
            indices = np.unique(indices)  # Remove any duplicates from rounding
        else:
            indices = np.arange(n_total)
            
        logger.info(f"  Selected {len(indices):,} unique indices")
        return indices
    
    def train(self, titles, memory_limit_gb=50, batch_size=10000):
        """
        Train encoder using golden ratio sampling.
        This is the method called by the training script.
        
        Args:
            titles: List of all titles
            memory_limit_gb: Memory limit for computation
            batch_size: Not used in fit, but kept for compatibility
        """
        self.fit(titles, memory_limit_gb)
    
    def fit(self, titles, memory_limit_gb=50):
        """
        Fit encoder using golden ratio sampling.
        
        Args:
            titles: List of all titles
            memory_limit_gb: Memory limit for computation
        """
        start_time = time.time()
        logger.info(f"Training encoder on {len(titles):,} titles...")
        
        # Step 1: Fit vectorizer on ALL titles (learns vocabulary)
        logger.info("Step 1: Learning vocabulary from all titles...")
        t0 = time.time()
        
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=self.max_features,
            lowercase=True,
            dtype=np.float32
        )
        self.vectorizer.fit(titles)
        
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"  Vocabulary size: {vocab_size:,}")
        logger.info(f"  Time: {time.time() - t0:.2f}s")
        
        # Step 2: Golden ratio sampling
        logger.info("Step 2: Golden ratio sampling...")
        t0 = time.time()
        
        self.sample_indices = self._golden_ratio_sample(
            len(titles), memory_limit_gb
        )
        sample_titles = [titles[i] for i in self.sample_indices]
        logger.info(f"  Time: {time.time() - t0:.2f}s")
        
        # Step 3: Transform sample and compute SVD
        logger.info(f"Step 3: Transforming {len(sample_titles):,} sampled titles...")
        t0 = time.time()
        
        X_sample = self.vectorizer.transform(sample_titles)
        X_dense = X_sample.toarray()
        logger.info(f"  Matrix shape: {X_dense.shape}")
        logger.info(f"  Matrix memory: {X_dense.nbytes / 1e9:.2f} GB")
        
        # Convert to PyTorch for SVD
        X_tensor = torch.from_numpy(X_dense).float()
        if self.device != 'cpu' and torch.cuda.is_available():
            X_tensor = X_tensor.to(self.device)
        
        logger.info(f"  Time: {time.time() - t0:.2f}s")
        
        # Step 4: SVD with energy analysis
        logger.info("Step 4: Computing SVD with energy analysis...")
        t0 = time.time()
        
        if self.use_randomized_svd and X_tensor.shape[1] > 5000:
            # Use randomized SVD for high-dimensional data
            logger.info("  Using randomized SVD for efficiency...")
            from randomized_svd import RandomizedSVD
            
            # Determine number of components to compute
            n_components = min(self.n_bits * 2, min(X_tensor.shape) - 1)
            
            svd_solver = RandomizedSVD(
                n_components=n_components,
                n_iter=self.svd_n_iter,
                n_oversamples=self.svd_n_oversamples,
                backend='torch' if self.device != 'cpu' else 'numpy',
                device=self.device,
                random_state=42
            )
            
            # Convert to numpy if needed
            X_for_svd = X_tensor.cpu().numpy() if self.device == 'cpu' else X_tensor
            U_np, S_np, Vh_np = svd_solver.fit_transform(X_for_svd)
            
            # Convert back to torch tensors
            U = torch.from_numpy(U_np).to(X_tensor.device) if isinstance(U_np, np.ndarray) else U_np
            S = torch.from_numpy(S_np).to(X_tensor.device) if isinstance(S_np, np.ndarray) else S_np
            Vh = torch.from_numpy(Vh_np).to(X_tensor.device) if isinstance(Vh_np, np.ndarray) else Vh_np
            
            logger.info(f"  Randomized SVD computed {n_components} components")
        else:
            # Use standard full SVD
            U, S, Vh = torch.linalg.svd(X_tensor, full_matrices=False)
        
        # Energy analysis
        energy = S ** 2
        total_energy = energy.sum()
        energy_threshold = energy.mean()
        
        # Find components above mean energy
        n_components = torch.sum(energy > energy_threshold).item()
        
        # Constrain to reasonable range
        n_components = np.clip(n_components, 64, min(self.n_bits, len(S)))
        
        # Calculate explained variance
        explained_variance = energy[:n_components].sum() / total_energy
        
        logger.info(f"  Total singular values: {len(S)}")
        logger.info(f"  Energy threshold: {energy_threshold:.2f}")
        logger.info(f"  Selected components: {n_components}")
        logger.info(f"  Explained variance: {explained_variance:.3f}")
        logger.info(f"  Top 5 singular values: {S[:5].cpu().numpy()}")
        logger.info(f"  Time: {time.time() - t0:.2f}s")
        
        # Step 5: Store projection matrix
        self.projection = Vh[:n_components].T.cpu().numpy()
        self.singular_values = S[:n_components].cpu().numpy()
        self.n_components = n_components
        
        # Step 6: Apply ITQ optimization if enabled
        if self.threshold_strategy == 'itq' or self.use_itq:
            logger.info("Step 5: Applying ITQ optimization for better binary codes...")
            t0 = time.time()
            
            try:
                from core.itq import ITQOptimizer
                
                # Project sample data
                X_projected = X_dense @ self.projection
                
                # Initialize and fit ITQ
                self.itq_optimizer = ITQOptimizer(
                    n_bits=self.n_components,
                    n_iterations=self.itq_iterations,
                    random_state=42
                )
                self.itq_optimizer.fit(X_projected)
                
                # Compute quantization error
                error = self.itq_optimizer.compute_quantization_error(X_projected)
                logger.info(f"  ITQ quantization error: {error:.4f}")
                logger.info(f"  Time: {time.time() - t0:.2f}s")
                
                # ITQ handles thresholding internally
                self.thresholds = np.zeros(self.n_components)
                
            except ImportError:
                logger.warning("  ITQ module not available, falling back to zero threshold")
                self.threshold_strategy = 'zero'
                self.thresholds = np.zeros(self.n_components)
                self.use_itq = False
                
        # Step 6b: Calculate thresholds if needed (non-ITQ)
        elif self.threshold_strategy in ['median', 'percentile']:
            logger.info(f"Step 5: Calculating {self.threshold_strategy} thresholds...")
            t0 = time.time()
            
            # Project sample data to get distribution
            X_projected = X_dense @ self.projection
            
            if self.threshold_strategy == 'median':
                # Use median of each dimension as threshold
                self.thresholds = np.median(X_projected, axis=0)
                logger.info(f"  Using median thresholds per dimension")
            elif self.threshold_strategy == 'percentile':
                # Use 50th percentile (can be made configurable)
                percentile = 50
                self.thresholds = np.percentile(X_projected, percentile, axis=0)
                logger.info(f"  Using {percentile}th percentile thresholds")
            
            logger.info(f"  Threshold range: [{self.thresholds.min():.4f}, {self.thresholds.max():.4f}]")
            logger.info(f"  Time: {time.time() - t0:.2f}s")
        else:
            # Zero threshold strategy
            self.thresholds = np.zeros(self.n_components)
            logger.info("Step 5: Using zero thresholds")
        
        # Step 7: Validate coherence
        logger.info("Step 6: Validating projection coherence...")
        t0 = time.time()
        
        coherence = self._validate_coherence()
        logger.info(f"  Projection coherence: {coherence:.4f}")
        logger.info(f"  Time: {time.time() - t0:.2f}s")
        
        # Store training statistics
        self.training_stats = {
            'n_titles': len(titles),
            'n_samples': len(sample_titles),
            'sample_ratio': len(sample_titles) / len(titles),
            'n_features': vocab_size,
            'n_components': n_components,
            'explained_variance': float(explained_variance),
            'coherence': float(coherence),
            'training_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mark as fitted
        self.is_fitted = True
        
        logger.info(f"Training complete in {self.training_stats['training_time']:.2f}s")
        
    def encode(self, titles, batch_size=10000, show_progress=True):
        """
        Transform titles to binary fingerprints.
        This method is called by the training script.
        
        Args:
            titles: Titles to encode
            batch_size: Processing batch size
            show_progress: Show progress bar
            
        Returns:
            Binary fingerprints tensor (n_titles, n_bits)
        """
        return self.transform(titles, batch_size, show_progress)
        
    def transform(self, titles, batch_size=10000, show_progress=True):
        """
        Transform titles to binary fingerprints.
        
        Args:
            titles: Titles to encode
            batch_size: Processing batch size
            show_progress: Show progress bar
            
        Returns:
            Binary fingerprints as torch tensor (n_titles, n_bits)
        """
        if self.vectorizer is None:
            raise ValueError("Encoder must be fitted first")
            
        n_titles = len(titles)
        fingerprints = np.zeros((n_titles, self.n_bits), dtype=np.uint8)
        
        # Process in batches
        iterator = range(0, n_titles, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding titles")
            
        for i in iterator:
            batch_end = min(i + batch_size, n_titles)
            batch = titles[i:batch_end]
            
            # Transform to TF-IDF
            X_batch = self.vectorizer.transform(batch)
            # Handle both sparse and dense matrices
            if hasattr(X_batch, 'toarray'):
                X_dense = X_batch.toarray()
            else:
                X_dense = X_batch  # Already dense
            
            # Project using learned components
            X_projected = X_dense @ self.projection
            
            # Apply ITQ or threshold-based binarization
            if self.threshold_strategy == 'itq' and self.itq_optimizer is not None:
                # Use ITQ to get binary codes
                binary = self.itq_optimizer.transform(X_projected).numpy().astype(np.uint8)
            else:
                # Normalize to unit sphere
                norms = np.linalg.norm(X_projected, axis=1, keepdims=True)
                X_normalized = X_projected / (norms + 1e-8)
                
                # Extract binary phases using threshold strategy
                if self.thresholds is not None and len(self.thresholds) > 0:
                    # Apply per-dimension thresholds
                    binary = (X_normalized > self.thresholds).astype(np.uint8)
                else:
                    # Fallback to zero threshold
                    binary = (X_normalized > 0).astype(np.uint8)
            
            # Store (handling case where n_components < n_bits)
            actual_bits = min(binary.shape[1], self.n_bits)
            fingerprints[i:batch_end, :actual_bits] = binary[:, :actual_bits]
            
        # Convert to PyTorch tensor for compatibility
        return torch.from_numpy(fingerprints)
    
    def encode_single(self, title):
        """Encode a single title."""
        return self.encode([title], show_progress=False)[0]
    
    def _validate_coherence(self):
        """Measure coherence of projection using quantum principle."""
        # Create random test vectors
        test_vectors = np.random.randn(100, self.projection.shape[0])
        
        # Project
        projected = test_vectors @ self.projection
        
        # Convert to complex for phase analysis
        projected_complex = projected.astype(np.complex64)
        
        # Measure phase coherence
        phases = np.angle(np.sum(projected_complex, axis=1))
        phase_factors = np.exp(1j * phases)
        coherence = np.abs(np.mean(phase_factors))
        
        return coherence
    
    def save(self, save_dir):
            """Save encoder to disk."""
            try:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Saving encoder to {save_path}")
                
                # Save vectorizer vocabulary and IDF as numpy arrays
                if self.vectorizer is None:
                    raise ValueError("Cannot save encoder: vectorizer is None")
                
                vocab_items = sorted(self.vectorizer.vocabulary_.items(), key=lambda x: x[1])
                vocab_array = np.array([item[0] for item in vocab_items], dtype=object)
                
                vocab_path = save_path / 'vocabulary.npy'
                logger.info(f"Saving vocabulary to {vocab_path}")
                np.save(vocab_path, vocab_array)
                
                idf_path = save_path / 'idf_weights.npy'
                logger.info(f"Saving IDF weights to {idf_path}")
                np.save(idf_path, self.vectorizer.idf_)
                
                # Save projection and parameters
                if self.projection is None:
                    raise ValueError("Cannot save encoder: projection matrix is None")
                
                projection_path = save_path / 'projection.npy'
                logger.info(f"Saving projection matrix to {projection_path}")
                np.save(projection_path, self.projection)
                
                if self.singular_values is None:
                    raise ValueError("Cannot save encoder: singular values are None")
                    
                singular_path = save_path / 'singular_values.npy'
                logger.info(f"Saving singular values to {singular_path}")
                np.save(singular_path, self.singular_values)
                
                # Save thresholds if they exist
                if self.thresholds is not None:
                    thresholds_path = save_path / 'thresholds.npy'
                    logger.info(f"Saving thresholds to {thresholds_path}")
                    np.save(thresholds_path, self.thresholds)
                
                # Save ITQ parameters if available
                if self.itq_optimizer is not None and self.itq_optimizer.is_fitted:
                    itq_path = save_path / 'itq_params.npy'
                    logger.info(f"Saving ITQ parameters to {itq_path}")
                    self.itq_optimizer.save(str(itq_path))
                
                # Save configuration
                config = {
                    'n_bits': int(self.n_bits),
                    'n_components': int(self.n_components),
                    'max_features': int(self.max_features),
                    'threshold_strategy': self.threshold_strategy,
                    'use_itq': self.use_itq,
                    'itq_iterations': self.itq_iterations,
                    'golden_ratio': float(self.golden_ratio),
                    'sample_indices': self.sample_indices.tolist() if self.sample_indices is not None else None,
                    'training_stats': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                                    for k, v in self.training_stats.items()}
                }
                
                config_path = save_path / 'config.json'
                logger.info(f"Saving config to {config_path}")
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Verify all files were created
                expected_files = ['vocabulary.npy', 'idf_weights.npy', 'projection.npy', 
                                'singular_values.npy', 'config.json']
                
                for file in expected_files:
                    file_path = save_path / file
                    if not file_path.exists():
                        raise FileNotFoundError(f"Failed to save {file} - file does not exist after save")
                    logger.info(f"  Verified: {file} ({file_path.stat().st_size} bytes)")
                    
                logger.info(f"Encoder saved successfully to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to save encoder: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                raise
    
    def load(self, save_dir):
        """Load encoder from disk."""
        save_path = Path(save_dir)
        
        # Load configuration
        with open(save_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        self.n_bits = config['n_bits']
        self.n_components = config['n_components']
        self.max_features = config['max_features']
        self.threshold_strategy = config.get('threshold_strategy', 'zero')
        self.use_itq = config.get('use_itq', False)
        self.itq_iterations = config.get('itq_iterations', 50)
        self.golden_ratio = config['golden_ratio']
        self.training_stats = config.get('training_stats', {})
        
        # Load projection and singular values
        self.projection = np.load(save_path / 'projection.npy')
        self.singular_values = np.load(save_path / 'singular_values.npy')
        
        # Load thresholds if they exist
        thresholds_path = save_path / 'thresholds.npy'
        if thresholds_path.exists():
            self.thresholds = np.load(thresholds_path)
        else:
            self.thresholds = None
        
        # Recreate vectorizer
        vocab_array = np.load(save_path / 'vocabulary.npy', allow_pickle=True)
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=self.max_features,
            lowercase=True,
            dtype=np.float32
        )
        
        # Restore vocabulary
        self.vectorizer.vocabulary_ = {word: idx for idx, word in enumerate(vocab_array)}
        self.vectorizer.idf_ = np.load(save_path / 'idf_weights.npy')
        
        # Load ITQ parameters if available
        itq_path = save_path / 'itq_params.npy'
        if itq_path.exists():
            try:
                from core.itq import ITQOptimizer
                self.itq_optimizer = ITQOptimizer(
                    n_bits=self.n_components,
                    n_iterations=self.itq_iterations
                )
                self.itq_optimizer.load(str(itq_path))
                logger.info("Loaded ITQ parameters")
            except ImportError:
                logger.warning("ITQ module not available, skipping ITQ load")
                self.itq_optimizer = None
        else:
            self.itq_optimizer = None
        
        # Mark as fitted since we loaded a trained model
        self.is_fitted = True
        
        logger.info(f"Encoder loaded from {save_path}")