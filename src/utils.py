"""Utility functions: scoring, logging, submission formatting."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.config import SUBMISSION_PATH, ID_COL, TARGET


def compute_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute competition score: max(0, 100 * R²)."""
    r2 = r2_score(actual, predicted)
    return max(0.0, 100.0 * r2)


def print_phase_score(phase: int, score: float, fold_scores: list = None):
    """Print validation score for a phase."""
    print(f"\n{'='*60}")
    print(f"PHASE {phase} RESULTS")
    print(f"{'='*60}")
    if fold_scores:
        for i, fs in enumerate(fold_scores):
            print(f"  Fold {i+1} R² Score: {fs:.4f}")
        print(f"  Mean R² Score: {np.mean(fold_scores):.4f}")
        print(f"  Std R² Score:  {np.std(fold_scores):.4f}")
    print(f"  Competition Score: {score:.4f}")
    print(f"{'='*60}\n")


def create_submission(test_indices: np.ndarray, predictions: np.ndarray, path: str = None):
    """Create submission.csv with Index and demand columns."""
    if path is None:
        path = SUBMISSION_PATH
    predictions = np.clip(predictions, 0, None)
    submission = pd.DataFrame({
        ID_COL: test_indices,
        TARGET: predictions,
    })
    submission.to_csv(path, index=False)
    print(f"Submission saved to {path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Demand range: [{submission[TARGET].min():.6f}, {submission[TARGET].max():.6f}]")
    print(f"  Demand mean: {submission[TARGET].mean():.6f}")
    return submission
