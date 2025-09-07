"""
Statistical Calibration Module for TEJAS Fingerprint System
============================================================

This module provides statistical calibration functionality for optimizing
binary fingerprint thresholds and evaluating retrieval performance.

Key Features:
- Cross-validation based calibration
- Bootstrap confidence intervals
- Multiple metric optimization (F1, precision, recall, ROC-AUC)
- Threshold optimization for retrieval tasks
- Integration with drift detection

DISCLAIMER: Experimental research code - not validated for production.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List, Union
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import warnings

# Try to import sklearn for advanced metrics
try:
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        precision_recall_curve,
        confusion_matrix,
        accuracy_score,
    )

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn(
        "scikit-learn not available. Some calibration features will be limited."
    )

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""

    threshold: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    roc_auc: float
    precision_at_k: float
    recall_at_k: float
    confidence_lower: float
    confidence_upper: float
    n_samples: int
    n_positive: int
    n_negative: int


class StatisticalCalibrator:
    """
    Statistical calibrator for binary fingerprint systems.

    Provides threshold optimization and performance evaluation using
    cross-validation and bootstrap methods.
    """

    def __init__(
        self,
        n_folds: int = 5,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the statistical calibrator.

        Args:
            n_folds: Number of cross-validation folds
            n_bootstrap: Number of bootstrap samples for confidence intervals
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state

        # Calibration results storage
        self.calibration_results = None
        self.optimal_threshold = None
        self.threshold_metrics = {}

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

        logger.info(
            f"Initialized StatisticalCalibrator with {n_folds} folds, "
            f"{n_bootstrap} bootstrap samples"
        )

    def calibrate_with_cv(
        self,
        distances: np.ndarray,
        labels: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        k_values: Optional[List[int]] = None,
        metric: str = "f1_score",
    ) -> pd.DataFrame:
        """
        Perform cross-validation based calibration.

        Args:
            distances: Array of distances/scores (lower = more similar)
            labels: Binary labels (1 = relevant, 0 = not relevant)
            thresholds: Thresholds to evaluate (auto-generated if None)
            k_values: Values of k for precision/recall@k (default: [5, 10, 20])
            metric: Primary metric to optimize

        Returns:
            DataFrame with calibration results for each threshold
        """
        if len(distances) != len(labels):
            raise ValueError("Distances and labels must have same length")

        # Convert to numpy arrays
        distances = np.asarray(distances)
        labels = np.asarray(labels)

        # Generate thresholds if not provided
        if thresholds is None:
            # Use percentiles of distances as thresholds
            percentiles = np.linspace(5, 95, 19)  # 19 thresholds
            thresholds = np.percentile(distances, percentiles)
            thresholds = np.unique(thresholds)  # Remove duplicates

        # Default k values for precision/recall@k
        if k_values is None:
            k_values = [5, 10, 20]

        # Ensure we use the minimum of provided k and available samples
        max_k = min(len(distances) // 2, max(k_values) if k_values else 20)
        k_values = [k for k in k_values if k <= max_k]
        if not k_values:
            k_values = [min(5, max_k)]

        results = []

        # Vectorized threshold evaluation for better performance
        if len(thresholds) > 10 and HAS_SKLEARN:
            # Batch evaluate multiple thresholds at once
            logger.info(
                f"  Evaluating {len(thresholds)} thresholds using vectorized operations"
            )
            results = self._evaluate_thresholds_vectorized(
                distances, labels, thresholds, k_values
            )
        else:
            # Original sequential evaluation for small threshold sets
            for threshold in thresholds:
                metrics = self._evaluate_threshold_cv(
                    distances, labels, threshold, k_values
                )
                results.append(metrics)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by primary metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)

        self.calibration_results = df

        # Find optimal threshold
        if metric in df.columns:
            self.optimal_threshold, _ = self.find_optimal_threshold(df, metric)
            logger.info(f"Optimal threshold for {metric}: {self.optimal_threshold:.4f}")

        return df

    def _evaluate_threshold_cv(
        self,
        distances: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        k_values: List[int],
    ) -> Dict[str, float]:
        """
        Evaluate a single threshold using cross-validation.

        Args:
            distances: Distance/score array
            labels: Binary labels
            threshold: Threshold to evaluate
            k_values: Values for precision/recall@k

        Returns:
            Dictionary of metrics
        """
        n_samples = len(distances)

        # Initialize metric collectors
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        specificities = []
        aucs = []
        precision_at_ks = defaultdict(list)
        recall_at_ks = defaultdict(list)

        # Use stratified k-fold if sklearn available
        if HAS_SKLEARN and self.n_folds > 1:
            # Ensure we have enough samples for stratification
            min_class_samples = min(np.sum(labels == 0), np.sum(labels == 1))
            if min_class_samples >= self.n_folds:
                kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True)
            else:
                kfold = KFold(
                    n_splits=min(self.n_folds, min_class_samples), shuffle=True
                )

            for train_idx, test_idx in kfold.split(distances, labels):
                test_distances = distances[test_idx]
                test_labels = labels[test_idx]

                # Skip if no positive samples in test set
                if np.sum(test_labels) == 0:
                    continue

                # Make predictions based on threshold
                predictions = (test_distances <= threshold).astype(int)

                # Calculate metrics
                fold_metrics = self._calculate_metrics(
                    test_labels, predictions, test_distances, k_values
                )

                # Collect metrics
                precisions.append(fold_metrics["precision"])
                recalls.append(fold_metrics["recall"])
                f1_scores.append(fold_metrics["f1_score"])
                accuracies.append(fold_metrics["accuracy"])
                specificities.append(fold_metrics["specificity"])
                if fold_metrics["roc_auc"] is not None:
                    aucs.append(fold_metrics["roc_auc"])

                for k in k_values:
                    precision_at_ks[k].append(fold_metrics[f"precision_at_{k}"])
                    recall_at_ks[k].append(fold_metrics[f"recall_at_{k}"])
        else:
            # Single evaluation without cross-validation
            predictions = (distances <= threshold).astype(int)
            fold_metrics = self._calculate_metrics(
                labels, predictions, distances, k_values
            )

            precisions = [fold_metrics["precision"]]
            recalls = [fold_metrics["recall"]]
            f1_scores = [fold_metrics["f1_score"]]
            accuracies = [fold_metrics["accuracy"]]
            specificities = [fold_metrics["specificity"]]
            if fold_metrics["roc_auc"] is not None:
                aucs = [fold_metrics["roc_auc"]]

            for k in k_values:
                precision_at_ks[k] = [fold_metrics[f"precision_at_{k}"]]
                recall_at_ks[k] = [fold_metrics[f"recall_at_{k}"]]

        # Aggregate results
        result = {
            "threshold": threshold,
            "precision": np.mean(precisions) if precisions else 0.0,
            "recall": np.mean(recalls) if recalls else 0.0,
            "f1_score": np.mean(f1_scores) if f1_scores else 0.0,
            "accuracy": np.mean(accuracies) if accuracies else 0.0,
            "specificity": np.mean(specificities) if specificities else 0.0,
            "roc_auc": np.mean(aucs) if aucs else 0.0,
        }

        # Add precision/recall@k metrics
        for k in k_values:
            if k in precision_at_ks:
                result[f"precision_at_{k}"] = np.mean(precision_at_ks[k])
                result[f"recall_at_{k}"] = np.mean(recall_at_ks[k])

        # Add default precision_at_k and recall_at_k for compatibility
        default_k = k_values[0] if k_values else 5
        result["precision_at_k"] = result.get(f"precision_at_{default_k}", 0.0)
        result["recall_at_k"] = result.get(f"recall_at_{default_k}", 0.0)

        # Add confidence intervals using bootstrap if requested
        if self.n_bootstrap > 0:
            ci_lower, ci_upper = self._bootstrap_confidence_interval(
                distances, labels, threshold, metric="f1_score"
            )
            result["ci_lower"] = ci_lower
            result["ci_upper"] = ci_upper

        return result

    def _evaluate_thresholds_vectorized(
        self,
        distances: np.ndarray,
        labels: np.ndarray,
        thresholds: np.ndarray,
        k_values: List[int],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple thresholds using vectorized operations.

        Args:
            distances: Distance/score array
            labels: Binary labels
            thresholds: Array of thresholds to evaluate
            k_values: Values for precision/recall@k

        Returns:
            List of metric dictionaries
        """
        n_samples = len(distances)
        n_thresholds = len(thresholds)

        # Expand dimensions for broadcasting
        distances_expanded = distances[:, np.newaxis]  # (n_samples, 1)
        thresholds_expanded = thresholds[np.newaxis, :]  # (1, n_thresholds)

        # Vectorized predictions for all thresholds at once
        all_predictions = (distances_expanded <= thresholds_expanded).astype(
            int
        )  # (n_samples, n_thresholds)

        results = []

        # Process each threshold's predictions
        for i, threshold in enumerate(thresholds):
            predictions = all_predictions[:, i]

            # Calculate metrics for this threshold
            if HAS_SKLEARN:
                # Basic metrics
                if len(np.unique(predictions)) == 1:
                    precision = (
                        1.0 if predictions[0] == 1 and np.any(labels == 1) else 0.0
                    )
                    recall = (
                        1.0
                        if np.sum(labels) == 0
                        or (predictions[0] == 1 and np.all(labels == 1))
                        else 0.0
                    )
                    f1 = 0.0
                else:
                    precision = precision_score(labels, predictions, zero_division=0)
                    recall = recall_score(labels, predictions, zero_division=0)
                    f1 = f1_score(labels, predictions, zero_division=0)

                accuracy = accuracy_score(labels, predictions)

                # Specificity
                tn = np.sum((predictions == 0) & (labels == 0))
                fp = np.sum((predictions == 1) & (labels == 0))
                specificity = tn / (tn + fp + 1e-10)

                # ROC-AUC
                try:
                    if len(np.unique(labels)) > 1:
                        scores = -distances
                        roc_auc = roc_auc_score(labels, scores)
                    else:
                        roc_auc = 0.0
                except:
                    roc_auc = 0.0
            else:
                # Manual calculation
                tp = np.sum((predictions == 1) & (labels == 1))
                fp = np.sum((predictions == 1) & (labels == 0))
                tn = np.sum((predictions == 0) & (labels == 0))
                fn = np.sum((predictions == 0) & (labels == 1))

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                accuracy = (tp + tn) / n_samples
                specificity = tn / (tn + fp + 1e-10)
                roc_auc = 0.0

            # Calculate precision/recall@k
            precision_at_k = {}
            recall_at_k = {}

            for k in k_values:
                if k <= n_samples:
                    # Get top k by distance
                    top_k_idx = np.argpartition(distances, min(k - 1, n_samples - 1))[
                        :k
                    ]
                    top_k_labels = labels[top_k_idx]
                    top_k_predictions = predictions[top_k_idx]

                    # Precision@k: fraction of top k that are relevant
                    precision_at_k[k] = np.mean(top_k_labels)

                    # Recall@k: fraction of all relevant items in top k
                    total_relevant = np.sum(labels)
                    if total_relevant > 0:
                        recall_at_k[k] = np.sum(top_k_labels) / total_relevant
                    else:
                        recall_at_k[k] = 0.0

            # Build result dictionary
            result = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "specificity": specificity,
                "roc_auc": roc_auc,
            }

            # Add precision/recall@k
            for k in k_values:
                if k in precision_at_k:
                    result[f"precision_at_{k}"] = precision_at_k[k]
                    result[f"recall_at_{k}"] = recall_at_k[k]

            # Add default metrics
            default_k = k_values[0] if k_values else 5
            result["precision_at_k"] = result.get(f"precision_at_{default_k}", 0.0)
            result["recall_at_k"] = result.get(f"recall_at_{default_k}", 0.0)

            results.append(result)

        return results

    def _calculate_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        distances: np.ndarray,
        k_values: List[int],
    ) -> Dict[str, float]:
        """
        Calculate various metrics for predictions.

        Args:
            labels: True labels
            predictions: Binary predictions
            distances: Distance scores
            k_values: Values for precision/recall@k

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        if HAS_SKLEARN:
            # Handle edge cases where all predictions are the same class
            if len(np.unique(predictions)) == 1:
                metrics["precision"] = (
                    1.0 if predictions[0] == 1 and labels[0] == 1 else 0.0
                )
                metrics["recall"] = (
                    1.0
                    if np.sum(labels) == 0
                    or (predictions[0] == 1 and np.all(labels == 1))
                    else 0.0
                )
                metrics["f1_score"] = 0.0
            else:
                metrics["precision"] = precision_score(
                    labels, predictions, zero_division=0
                )
                metrics["recall"] = recall_score(labels, predictions, zero_division=0)
                metrics["f1_score"] = f1_score(labels, predictions, zero_division=0)

            metrics["accuracy"] = accuracy_score(labels, predictions)

            # Specificity (true negative rate)
            tn, fp, fn, tp = confusion_matrix(
                labels, predictions, labels=[0, 1]
            ).ravel()
            epsilon = 1e-10  # Prevent division by zero
            metrics["specificity"] = tn / (tn + fp + epsilon)

            # ROC-AUC (using distances as scores)
            try:
                if len(np.unique(labels)) > 1:
                    # Invert distances so lower distance = higher score
                    scores = -distances
                    metrics["roc_auc"] = roc_auc_score(labels, scores)
                else:
                    metrics["roc_auc"] = None
            except:
                metrics["roc_auc"] = None
        else:
            # Manual calculation without sklearn
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))

            epsilon = 1e-10  # Prevent division by zero
            metrics["precision"] = tp / (tp + fp + epsilon)
            metrics["recall"] = tp / (tp + fn + epsilon)

            # F1 score with epsilon protection
            precision_recall_sum = metrics["precision"] + metrics["recall"]
            if precision_recall_sum > epsilon:
                metrics["f1_score"] = (
                    2
                    * (metrics["precision"] * metrics["recall"])
                    / precision_recall_sum
                )
            else:
                metrics["f1_score"] = 0.0

            metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn + epsilon)
            metrics["specificity"] = tn / (tn + fp + epsilon)
            metrics["roc_auc"] = None

        # Precision/Recall at K
        for k in k_values:
            if k <= len(distances):
                # Get top-k predictions by distance
                top_k_idx = np.argsort(distances)[:k]
                top_k_labels = labels[top_k_idx]

                # Precision@k: fraction of top-k that are relevant
                metrics[f"precision_at_{k}"] = np.mean(top_k_labels)

                # Recall@k: fraction of relevant items in top-k
                n_relevant = np.sum(labels)
                if n_relevant > 0:
                    metrics[f"recall_at_{k}"] = np.sum(top_k_labels) / n_relevant
                else:
                    metrics[f"recall_at_{k}"] = 0.0
            else:
                metrics[f"precision_at_{k}"] = 0.0
                metrics[f"recall_at_{k}"] = 0.0

        return metrics

    def _bootstrap_confidence_interval(
        self,
        distances: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        metric: str = "f1_score",
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a metric.

        Args:
            distances: Distance array
            labels: Label array
            threshold: Threshold value
            metric: Metric to calculate CI for

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n_samples = len(distances)
        bootstrap_metrics = []

        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            boot_distances = distances[idx]
            boot_labels = labels[idx]

            # Skip if no positive samples
            if np.sum(boot_labels) == 0:
                continue

            # Calculate predictions
            predictions = (boot_distances <= threshold).astype(int)

            # Calculate metric
            boot_metric = self._calculate_metrics(
                boot_labels, predictions, boot_distances, [5]
            )

            if metric in boot_metric:
                bootstrap_metrics.append(boot_metric[metric])

        if bootstrap_metrics:
            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
            upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
            return lower, upper
        else:
            return 0.0, 0.0

    def find_optimal_threshold(
        self, calibration_df: pd.DataFrame, metric: str = "f1_score"
    ) -> Tuple[float, float]:
        """
        Find optimal threshold based on specified metric.

        Args:
            calibration_df: DataFrame with calibration results
            metric: Metric to optimize

        Returns:
            Tuple of (optimal_threshold, optimal_metric_value)
        """
        if calibration_df is None or calibration_df.empty:
            raise ValueError("No calibration results available")

        if metric not in calibration_df.columns:
            raise ValueError(f"Metric '{metric}' not found in calibration results")

        # Find row with maximum metric value
        optimal_idx = calibration_df[metric].idxmax()
        optimal_row = calibration_df.loc[optimal_idx]

        optimal_threshold = optimal_row["threshold"]
        optimal_value = optimal_row[metric]

        self.optimal_threshold = optimal_threshold
        self.threshold_metrics[optimal_threshold] = optimal_row.to_dict()

        return optimal_threshold, optimal_value

    def predict_with_threshold(
        self, distances: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Make predictions using calibrated threshold.

        Args:
            distances: Distance scores
            threshold: Threshold to use (uses optimal if None)

        Returns:
            Binary predictions
        """
        if threshold is None:
            if self.optimal_threshold is None:
                raise ValueError(
                    "No optimal threshold available. Run calibration first."
                )
            threshold = self.optimal_threshold

        return (distances <= threshold).astype(int)

    def save_calibration_results(
        self, calibration_df: pd.DataFrame, filepath: Union[str, Path]
    ) -> None:
        """
        Save calibration results to JSON file.

        Args:
            calibration_df: Calibration results DataFrame
            filepath: Path to save file
        """
        filepath = Path(filepath)

        # Convert DataFrame to list of dicts for JSON serialization
        results = calibration_df.to_dict("records")

        # Convert any numpy types to Python types
        for result in results:
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    result[key] = int(value)
                elif isinstance(value, (np.floating, np.float32, np.float64)):
                    result[key] = float(value)

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved calibration results to {filepath}")

    def load_calibration_results(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load calibration results from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            Calibration results DataFrame
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            results = json.load(f)

        df = pd.DataFrame(results)
        self.calibration_results = df

        logger.info(f"Loaded calibration results from {filepath}")
        return df

    def get_threshold_curve(
        self, distances: np.ndarray, labels: np.ndarray, n_points: int = 100
    ) -> pd.DataFrame:
        """
        Generate threshold curve showing metrics across threshold range.

        Args:
            distances: Distance scores
            labels: Binary labels
            n_points: Number of points on curve

        Returns:
            DataFrame with threshold curve data
        """
        # Generate threshold range
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        thresholds = np.linspace(min_dist, max_dist, n_points)

        # Calculate metrics for each threshold
        results = []
        for threshold in thresholds:
            predictions = (distances <= threshold).astype(int)
            metrics = self._calculate_metrics(labels, predictions, distances, [5])
            metrics["threshold"] = threshold
            results.append(metrics)

        return pd.DataFrame(results)

    def analyze_threshold_stability(
        self,
        distances: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        perturbation: float = 0.01,
    ) -> Dict[str, float]:
        """
        Analyze stability of metrics around a threshold.

        Args:
            distances: Distance scores
            labels: Binary labels
            threshold: Threshold to analyze
            perturbation: Relative perturbation amount

        Returns:
            Dictionary with stability metrics
        """
        # Calculate metrics at threshold and perturbed values
        thresholds = [
            threshold * (1 - perturbation),
            threshold,
            threshold * (1 + perturbation),
        ]

        metrics_list = []
        for t in thresholds:
            predictions = (distances <= t).astype(int)
            metrics = self._calculate_metrics(labels, predictions, distances, [5])
            metrics_list.append(metrics)

        # Calculate stability (variation in metrics)
        stability = {}
        for key in ["precision", "recall", "f1_score"]:
            values = [m[key] for m in metrics_list]
            stability[f"{key}_std"] = np.std(values)
            stability[f"{key}_range"] = np.max(values) - np.min(values)

        return stability


def mean_average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Mean Average Precision (MAP).

    Args:
        y_true: Binary relevance labels (1 = relevant, 0 = not relevant)
        y_scores: Predicted scores (higher = more relevant)

    Returns:
        MAP score
    """
    # Sort by scores in descending order
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]

    # Calculate precision at each relevant item
    precisions = []
    n_relevant = 0

    for i, relevant in enumerate(y_true_sorted):
        if relevant:
            n_relevant += 1
            precision_at_i = n_relevant / (i + 1)
            precisions.append(precision_at_i)

    if len(precisions) == 0:
        return 0.0

    return np.mean(precisions)


def ndcg_score(
    y_true: np.ndarray, y_scores: np.ndarray, k: Optional[int] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).

    Args:
        y_true: Binary relevance labels (1 = relevant, 0 = not relevant)
        y_scores: Predicted scores (higher = more relevant)
        k: Consider only top k items (None = all items)

    Returns:
        NDCG@k score
    """
    # Sort by scores in descending order
    sorted_indices = np.argsort(-y_scores)

    # Limit to top k if specified
    if k is not None:
        sorted_indices = sorted_indices[:k]

    y_true_sorted = y_true[sorted_indices]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, rel in enumerate(y_true_sorted):
        # Using alternative formula: rel / log2(i+2)
        # Position is i+1, so denominator is log2(i+2)
        dcg += rel / np.log2(i + 2)

    # Calculate IDCG (Ideal DCG) - perfect ranking
    ideal_sorted = np.sort(y_true)[::-1]  # Sort relevance scores descending
    if k is not None:
        ideal_sorted = ideal_sorted[:k]

    idcg = 0.0
    for i, rel in enumerate(ideal_sorted):
        idcg += rel / np.log2(i + 2)

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    return dcg / idcg


def create_calibration_report(
    calibrator: StatisticalCalibrator, distances: np.ndarray, labels: np.ndarray
) -> str:
    """
    Create a detailed calibration report.

    Args:
        calibrator: Calibrated StatisticalCalibrator instance
        distances: Distance scores
        labels: Binary labels

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("CALIBRATION REPORT")
    report.append("=" * 60)

    if calibrator.calibration_results is not None:
        df = calibrator.calibration_results

        # Best thresholds for each metric
        report.append("\nOptimal Thresholds by Metric:")
        report.append("-" * 30)

        for metric in ["f1_score", "precision", "recall", "roc_auc"]:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_row = df.loc[best_idx]
                report.append(
                    f"  {metric:12s}: threshold={best_row['threshold']:.4f}, "
                    f"value={best_row[metric]:.4f}"
                )

        # Overall statistics
        report.append("\nDataset Statistics:")
        report.append("-" * 30)
        report.append(f"  Total samples: {len(labels)}")
        report.append(
            f"  Positive samples: {np.sum(labels)} ({100 * np.mean(labels):.1f}%)"
        )
        report.append(
            f"  Distance range: [{np.min(distances):.4f}, {np.max(distances):.4f}]"
        )

        # Stability analysis if optimal threshold exists
        if calibrator.optimal_threshold is not None:
            stability = calibrator.analyze_threshold_stability(
                distances, labels, calibrator.optimal_threshold
            )
            report.append("\nThreshold Stability Analysis:")
            report.append("-" * 30)
            for key, value in stability.items():
                report.append(f"  {key}: {value:.6f}")

    return "\n".join(report)


if __name__ == "__main__":
    # Demo of calibration functionality
    print("Statistical Calibration Demo")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_positive = 100

    # Create distances with known distribution
    distances = np.random.gamma(2, 2, n_samples)
    labels = np.zeros(n_samples)
    labels[:n_positive] = 1

    # Make positive samples have lower distances
    distances[:n_positive] *= 0.5

    # Initialize calibrator
    calibrator = StatisticalCalibrator(n_folds=5, n_bootstrap=50)

    # Perform calibration
    print("\nRunning calibration...")
    results = calibrator.calibrate_with_cv(distances, labels)

    # Find optimal threshold
    optimal_threshold, optimal_f1 = calibrator.find_optimal_threshold(
        results, "f1_score"
    )
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"Optimal F1 score: {optimal_f1:.4f}")

    # Generate report
    report = create_calibration_report(calibrator, distances, labels)
    print("\n" + report)

    print("\n✓ Calibration module demonstration complete")
