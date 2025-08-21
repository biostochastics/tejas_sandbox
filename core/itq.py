"""
Iterative Quantization (ITQ) for learned binary codes.

Based on the paper:
"Iterative Quantization: A Procrustean Approach to Learning Binary Codes"
by Yunchao Gong and Svetlana Lazebnik

This module provides ITQ optimization to learn better binary codes
by finding an optimal rotation matrix that minimizes quantization error.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ITQOptimizer:
    """
    Iterative Quantization optimizer for learning binary codes.
    
    ITQ learns an orthogonal rotation matrix to minimize the quantization
    error between continuous embeddings and their binary approximations.
    """
    
    def __init__(self, n_bits: int = 128, n_iterations: int = 50, 
                 random_state: Optional[int] = None, use_randomized_svd: bool = False):
        """
        Initialize ITQ optimizer.
        
        Args:
            n_bits: Number of bits in binary code
            n_iterations: Maximum iterations for optimization
            random_state: Random seed for reproducibility
            use_randomized_svd: Use randomized SVD for large matrices
        """
        self.n_bits = n_bits
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.use_randomized_svd = use_randomized_svd
        self.rotation_matrix = None
        self.is_fitted = False
        
    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> 'ITQOptimizer':
        """
        Learn optimal rotation matrix from training data.
        
        Args:
            X: Input data (n_samples, n_bits) - should be PCA/SVD projected
        
        Returns:
            Self for chaining
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        n_samples, n_features = X.shape
        
        if n_features != self.n_bits:
            raise ValueError(f"Expected {self.n_bits} features, got {n_features}")
        
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # Initialize rotation matrix (random orthogonal matrix)
        np.random.seed(self.random_state)
        R = np.random.randn(self.n_bits, self.n_bits)
        if self.use_randomized_svd and self.n_bits > 100:
            from randomized_svd import RandomizedSVD
            svd_solver = RandomizedSVD(n_components=min(self.n_bits, R.shape[0]), n_iter=2, random_state=self.random_state)
            U, _, Vt = svd_solver.fit_transform(R)
        else:
            U, _, Vt = np.linalg.svd(R, full_matrices=False)
        R = U @ Vt  # Random orthogonal matrix
        
        # Iterative optimization
        for iteration in range(self.n_iterations):
            # Step 1: Fix R, update B (binary codes)
            Z = X_centered @ R
            B = np.sign(Z)
            B[B == 0] = 1  # Handle zero values
            
            # Step 2: Fix B, update R
            # Solve orthogonal Procrustes problem: minimize ||B - XR||_F
            # Solution: R = V * U^T where UΣV^T = SVD(X^T * B)
            if self.use_randomized_svd and min(X_centered.T.shape[0], B.shape[1]) > 100:
                from randomized_svd import RandomizedSVD
                svd_solver = RandomizedSVD(n_components=min(self.n_bits, min(X_centered.T.shape[0], B.shape[1])), 
                                          n_iter=2, random_state=self.random_state)
                U, _, Vt = svd_solver.fit_transform(X_centered.T @ B)
            else:
                U, _, Vt = np.linalg.svd(X_centered.T @ B, full_matrices=False)
            R_new = U @ Vt
            
            # Check convergence
            if iteration > 0:
                change = np.linalg.norm(R - R_new, 'fro')
                if change < 1e-6:
                    logger.info(f"ITQ converged at iteration {iteration}")
                    break
            
            R = R_new
        
        self.rotation_matrix = R
        self.is_fitted = True
        
        # Compute final quantization error
        Z_final = X_centered @ self.rotation_matrix
        B_final = np.sign(Z_final)
        B_final[B_final == 0] = 1
        
        quantization_error = np.linalg.norm(B_final - Z_final, 'fro') / n_samples
        logger.info(f"ITQ optimization complete. Quantization error: {quantization_error:.4f}")
        
        return self
    
    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Transform data to binary codes using learned rotation.
        
        Args:
            X: Input data (n_samples, n_bits)
        
        Returns:
            Binary codes as boolean tensor
        """
        if not self.is_fitted:
            raise ValueError("ITQOptimizer must be fitted before transform")
        
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        # Apply rotation and binarize
        X_centered = X - X.mean(axis=0)
        Z = X_centered @ self.rotation_matrix
        B = np.sign(Z)
        B[B == 0] = 1
        
        # Convert to boolean tensor
        binary_codes = (B > 0)
        
        return torch.from_numpy(binary_codes)
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Fit and transform in one step.
        
        Args:
            X: Input data (n_samples, n_bits)
        
        Returns:
            Binary codes as boolean tensor
        """
        self.fit(X)
        return self.transform(X)
    
    def save(self, path: str):
        """Save ITQ parameters."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ITQ optimizer")
        
        np.save(path, {
            'rotation_matrix': self.rotation_matrix,
            'n_bits': self.n_bits,
            'n_iterations': self.n_iterations
        })
    
    def load(self, path: str):
        """Load ITQ parameters."""
        data = np.load(path, allow_pickle=True).item()
        self.rotation_matrix = data['rotation_matrix']
        self.n_bits = data['n_bits']
        self.n_iterations = data.get('n_iterations', 50)
        self.is_fitted = True
    
    def compute_quantization_error(self, X: Union[np.ndarray, torch.Tensor],
                                  binary_codes: Optional[torch.Tensor] = None) -> float:
        """
        Compute quantization error for given data.
        
        Args:
            X: Continuous features
            binary_codes: Pre-computed binary codes (optional)
        
        Returns:
            Average quantization error per sample
        """
        if not self.is_fitted:
            raise ValueError("ITQOptimizer must be fitted first")
        
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        X_centered = X - X.mean(axis=0)
        Z = X_centered @ self.rotation_matrix
        
        if binary_codes is None:
            B = np.sign(Z)
            B[B == 0] = 1
        else:
            B = binary_codes.numpy() * 2 - 1  # Convert from {0,1} to {-1,1}
        
        error = np.linalg.norm(B - Z, 'fro') / len(X)
        return error


class ITQEncoder:
    """
    Encoder that combines dimensionality reduction with ITQ optimization.
    """
    
    def __init__(self, n_bits: int = 128, n_iterations: int = 50,
                 use_pca: bool = True, random_state: Optional[int] = None):
        """
        Initialize ITQ encoder with dimensionality reduction.
        
        Args:
            n_bits: Number of bits in binary code
            n_iterations: ITQ iterations
            use_pca: Whether to use PCA before ITQ
            random_state: Random seed
        """
        self.n_bits = n_bits
        self.n_iterations = n_iterations
        self.use_pca = use_pca
        self.random_state = random_state
        
        self.pca = None
        self.itq = ITQOptimizer(n_bits, n_iterations, random_state)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'ITQEncoder':
        """
        Fit PCA and ITQ on training data.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Self for chaining
        """
        if self.use_pca:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.n_bits, random_state=self.random_state)
            X_reduced = self.pca.fit_transform(X)
        else:
            # Assume X is already reduced
            X_reduced = X[:, :self.n_bits]
        
        # Fit ITQ on reduced data
        self.itq.fit(X_reduced)
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> torch.Tensor:
        """
        Transform features to binary codes.
        
        Args:
            X: Input features
        
        Returns:
            Binary codes
        """
        if not self.is_fitted:
            raise ValueError("ITQEncoder must be fitted first")
        
        if self.use_pca and self.pca is not None:
            X_reduced = self.pca.transform(X)
        else:
            X_reduced = X[:, :self.n_bits]
        
        return self.itq.transform(X_reduced)
    
    def fit_transform(self, X: np.ndarray) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)