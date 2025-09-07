"""
Optimized ITQ (Iterative Quantization) Implementation
=====================================================

This is the consolidated, optimized version of ITQ based on extensive testing.
Removes all unnecessary adaptive convergence complexity and implements proper
automatic convergence detection based on the original ITQ paper.

Key improvements:
1. Automatic convergence detection based on rotation matrix change
2. No dimension-based rules - treats all data equally
3. Proper minimum iterations to ensure optimization
4. Efficient SVD computation with optional caching
5. Clean, maintainable code structure

Based on: "Iterative Quantization: A Procrustean Approach to Learning Binary Codes"
by Yunchao Gong and Svetlana Lazebnik
"""

import numpy as np
import torch
from typing import Optional, Union, Dict, Any
import logging
import time
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ITQConfig:
    """Configuration for ITQ optimization."""

    # Core parameters
    n_bits: int = 128  # Number of bits for binary codes
    max_iterations: int = 50  # Maximum iterations (reduced from 150 based on SOTA)

    # Convergence criteria
    convergence_threshold: float = 1e-5  # Rotation change threshold (Frobenius norm)
    min_iterations: int = 5  # Minimum iterations before checking convergence

    # Early stopping
    patience: int = 5  # Stop if no improvement for N iterations
    patience_threshold: float = 1e-6  # Minimum improvement to reset patience

    # Performance options
    random_state: Optional[int] = None  # Random seed for reproducibility
    verbose: bool = True  # Enable logging

    # Advanced options
    check_convergence_every: int = 1  # Check convergence every N iterations
    track_history: bool = True  # Track convergence history

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Validate n_bits
        if self.n_bits <= 0:
            raise ValueError(f"n_bits must be positive, got {self.n_bits}")
        if self.n_bits > 2048:
            raise ValueError(
                f"n_bits must be <= 2048 to prevent memory issues, got {self.n_bits}"
            )

        # Validate iterations
        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )
        if self.min_iterations < 0:
            raise ValueError(
                f"min_iterations must be non-negative, got {self.min_iterations}"
            )
        if self.min_iterations > self.max_iterations:
            raise ValueError(
                f"min_iterations ({self.min_iterations}) cannot exceed max_iterations ({self.max_iterations})"
            )

        # Validate convergence parameters
        if self.convergence_threshold <= 0:
            raise ValueError(
                f"convergence_threshold must be positive, got {self.convergence_threshold}"
            )
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.patience_threshold < 0:
            raise ValueError(
                f"patience_threshold must be non-negative, got {self.patience_threshold}"
            )

        # Validate other parameters
        if self.check_convergence_every <= 0:
            raise ValueError(
                f"check_convergence_every must be positive, got {self.check_convergence_every}"
            )


class ITQOptimizer:
    """
    Optimized ITQ (Iterative Quantization) optimizer for learning binary codes.

    ITQ learns an orthogonal rotation matrix to minimize the quantization error
    between continuous embeddings and their binary approximations.
    """

    def __init__(self, config: Optional[ITQConfig] = None):
        """
        Initialize ITQ optimizer.

        Args:
            config: ITQ configuration object. If None, uses defaults.
        """
        self.config = config or ITQConfig()

        # Core parameters
        self.n_bits = self.config.n_bits
        self.max_iterations = self.config.max_iterations
        self.convergence_threshold = self.config.convergence_threshold
        self.min_iterations = self.config.min_iterations

        # State
        self.rotation_matrix = None
        self.data_mean = None
        self.is_fitted = False

        # Convergence tracking
        self.history = (
            {"rotation_changes": [], "quantization_errors": [], "iteration_times": []}
            if self.config.track_history
            else None
        )

        self.convergence_info = {
            "converged": False,
            "iterations": 0,
            "reason": None,
            "final_error": None,
            "final_change": None,
        }

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "ITQOptimizer":
        """
        Learn optimal rotation matrix from training data.

        The algorithm alternates between:
        1. Fixing R and updating B (binary codes)
        2. Fixing B and updating R (rotation matrix via Orthogonal Procrustes)

        Args:
            X: Input data (n_samples, n_bits) - should be PCA/reduced features

        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Input validation
        if not np.isfinite(X).all():
            raise ValueError("Input contains NaN or Inf values")

        n_samples, n_features = X.shape

        # Enforce float64 for high dimensions for numerical stability
        if n_features >= 256 and X.dtype != np.float64:
            logger.info(
                f"Converting to float64 for numerical stability (d={n_features})"
            )
            X = X.astype(np.float64)

        if n_features != self.n_bits:
            raise ValueError(f"Expected {self.n_bits} features, got {n_features}")

        # Center the data
        self.data_mean = X.mean(axis=0)
        X_centered = X - self.data_mean

        # Initialize random orthogonal rotation matrix
        np.random.seed(self.config.random_state)
        R = self._initialize_rotation_matrix(self.n_bits)

        # Track optimization
        best_error = float("inf")
        patience_counter = 0
        converged = False
        actual_iterations = 0

        if self.config.verbose:
            logger.info(
                f"Starting ITQ optimization: {n_samples} samples, {self.n_bits} bits"
            )

        # Main optimization loop
        R_old = None  # Will be set after first iteration
        for iteration in range(self.max_iterations):
            iter_start = time.time()

            # Step 1: Fix R, update B (binary codes)
            Z = X_centered @ R
            B = np.sign(Z)
            B[B == 0] = 1  # Handle zero values

            # Step 2: Fix B, update R (Orthogonal Procrustes problem)
            # Solve: minimize ||B - XR||_F  =>  R = UV^T where USV^T = X^T B
            R_new = self._solve_procrustes(X_centered, B)

            # Calculate metrics (without copying)
            rotation_change = (
                np.linalg.norm(R_new - R, "fro") if R_old is not None else float("inf")
            )

            # Update references
            R_old = R  # Save previous R
            R = R_new  # Update to new R

            # Calculate quantization error
            Z_new = X_centered @ R
            B_new = np.sign(Z_new)
            B_new[B_new == 0] = 1
            quantization_error = np.linalg.norm(B_new - Z_new, "fro") / n_samples

            # Track history
            if self.config.track_history:
                self.history["rotation_changes"].append(rotation_change)
                self.history["quantization_errors"].append(quantization_error)
                self.history["iteration_times"].append(time.time() - iter_start)

            # Check convergence (after minimum iterations)
            if iteration >= self.min_iterations:
                if iteration % self.config.check_convergence_every == 0:
                    # Primary convergence: rotation change
                    if rotation_change < self.convergence_threshold:
                        actual_iterations = iteration + 1  # Fix: update before break
                        converged = True
                        self.convergence_info["reason"] = "rotation_converged"
                        if self.config.verbose:
                            logger.info(
                                f"Converged at iteration {iteration}: "
                                f"rotation change {rotation_change:.2e} < {self.convergence_threshold:.2e}"
                            )
                        break

                    # Secondary convergence: patience-based early stopping
                    improvement = best_error - quantization_error
                    if improvement > self.config.patience_threshold:
                        best_error = quantization_error
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.patience:
                            actual_iterations = (
                                iteration + 1
                            )  # Fix: update before break
                            converged = True
                            self.convergence_info["reason"] = "early_stop_patience"
                            if self.config.verbose:
                                logger.info(
                                    f"Early stopping at iteration {iteration}: "
                                    f"no improvement for {self.config.patience} iterations"
                                )
                            break

            actual_iterations = iteration + 1

            # Log progress periodically
            if self.config.verbose and iteration % 10 == 0:
                logger.debug(
                    f"Iteration {iteration}: error={quantization_error:.4f}, "
                    f"change={rotation_change:.2e}"
                )

        # Store final results
        self.rotation_matrix = R
        self.is_fitted = True

        # Update convergence info
        self.convergence_info.update(
            {
                "converged": converged,
                "iterations": actual_iterations,
                "final_error": quantization_error,
                "final_change": rotation_change,
            }
        )

        if self.config.verbose:
            if not converged:
                logger.warning(
                    f"ITQ did not converge within {self.max_iterations} iterations"
                )
            logger.info(
                f"ITQ optimization complete: {actual_iterations} iterations, "
                f"final error={quantization_error:.4f}"
            )

        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transform data to binary codes using learned rotation.

        Args:
            X: Input data (n_samples, n_bits)

        Returns:
            Binary codes as uint8 array (0/1 encoding)
        """
        if not self.is_fitted:
            raise ValueError("ITQOptimizer must be fitted before transform")

        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Validate input dimensions
        if X.shape[1] != self.n_bits:
            raise ValueError(f"Expected {self.n_bits} features, got {X.shape[1]}")

        # Center and rotate
        X_centered = X - self.data_mean
        Z = X_centered @ self.rotation_matrix

        # Binarize
        B = np.sign(Z)
        B[B == 0] = 1

        # Convert to 0/1 encoding
        binary_codes = ((B + 1) / 2).astype(np.uint8)

        return binary_codes

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Input data (n_samples, n_bits)

        Returns:
            Binary codes as uint8 array
        """
        self.fit(X)
        return self.transform(X)

    def _initialize_rotation_matrix(self, n_bits: int) -> np.ndarray:
        """
        Initialize random orthogonal rotation matrix.

        Args:
            n_bits: Matrix dimension

        Returns:
            Random orthogonal matrix
        """
        # Random matrix
        R = np.random.randn(n_bits, n_bits)

        try:
            # Make orthogonal via SVD
            U, _, Vt = np.linalg.svd(R, full_matrices=False)
            return U @ Vt
        except np.linalg.LinAlgError as e:
            # Fallback to QR decomposition if SVD fails
            logger.warning(
                f"SVD failed during initialization, using QR decomposition: {e}"
            )
            Q, _ = np.linalg.qr(R)
            return Q

    def _solve_procrustes(self, X: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Solve orthogonal Procrustes problem: minimize ||B - XR||_F.

        Solution: R = UV^T where USV^T = SVD(X^T B)

        Args:
            X: Data matrix (centered)
            B: Binary codes matrix

        Returns:
            Optimal rotation matrix
        """
        # Compute X^T B for SVD
        M = X.T @ B

        try:
            # Compute SVD directly (no cache needed - matrix changes every iteration)
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError as e:
            # Handle singular matrix or convergence failure
            logger.warning(f"SVD failed in Procrustes solution: {e}")
            # Return identity as fallback (no rotation)
            return np.eye(M.shape[0], dtype=M.dtype)

        return U @ Vt

    def compute_quantization_error(self, X: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute quantization error for given data.

        Args:
            X: Input data

        Returns:
            Average quantization error per sample
        """
        if not self.is_fitted:
            raise ValueError("ITQOptimizer must be fitted first")

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        X_centered = X - self.data_mean
        Z = X_centered @ self.rotation_matrix
        B = np.sign(Z)
        B[B == 0] = 1

        error = np.linalg.norm(B - Z, "fro") / len(X)
        return error

    def get_convergence_summary(self) -> Dict[str, Any]:
        """
        Get summary of convergence behavior.

        Returns:
            Dictionary with convergence statistics
        """
        summary = self.convergence_info.copy()

        if self.config.track_history and self.history["rotation_changes"]:
            summary.update(
                {
                    "min_rotation_change": min(self.history["rotation_changes"]),
                    "max_rotation_change": max(self.history["rotation_changes"]),
                    "initial_error": self.history["quantization_errors"][0],
                    "error_reduction": 1
                    - (
                        self.convergence_info["final_error"]
                        / self.history["quantization_errors"][0]
                    ),
                    "avg_iteration_time": np.mean(self.history["iteration_times"]),
                    "total_time": sum(self.history["iteration_times"]),
                }
            )

        return summary

    def save(self, path: str):
        """Save ITQ parameters to file (secure, no pickle)."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ITQ optimizer")

        # Save numpy arrays
        np.savez(
            path,
            rotation_matrix=self.rotation_matrix,
            data_mean=self.data_mean,
            n_bits=self.n_bits,
        )

        # Save convergence info as JSON (safe alternative to pickle)
        json_path = path.replace(".npz", "_convergence.json")
        with open(json_path, "w") as f:
            json.dump(self.convergence_info, f)

    def load(self, path: str):
        """Load ITQ parameters from file (secure, no pickle)."""
        # Load numpy arrays (safe, no pickle needed)
        data = np.load(path)
        self.rotation_matrix = data["rotation_matrix"]
        self.data_mean = data["data_mean"]
        self.n_bits = int(data["n_bits"])

        # Load convergence info from JSON
        json_path = path.replace(".npz", "_convergence.json")
        try:
            with open(json_path, "r") as f:
                self.convergence_info = json.load(f)
        except FileNotFoundError:
            # Backward compatibility: if JSON doesn't exist, use defaults
            self.convergence_info = {
                "converged": True,
                "iterations": 0,
                "reason": "loaded_from_file",
                "final_error": None,
                "final_change": None,
            }

        self.is_fitted = True


def create_itq_optimizer(
    n_bits: int = 128,
    max_iterations: Optional[int] = None,
    convergence_threshold: Optional[float] = None,
    verbose: bool = True,
) -> ITQOptimizer:
    """
    Convenience function to create ITQ optimizer with common settings.

    Args:
        n_bits: Number of bits for binary codes
        max_iterations: Maximum iterations (default: 150)
        convergence_threshold: Convergence threshold (default: 1e-5)
        verbose: Enable logging

    Returns:
        Configured ITQ optimizer
    """
    config = ITQConfig(
        n_bits=n_bits,
        max_iterations=max_iterations or 50,
        convergence_threshold=convergence_threshold or 1e-5,
        verbose=verbose,
    )
    return ITQOptimizer(config)
