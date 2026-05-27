"""Utility functions."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.config import SUBMISSION_PATH, ID_COL, TARGET


def compute_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    return max(0.0, 100.0 * r2_score(actual, predicted))


def print_phase(phase: str, score: float, fold_scores: list = None):
    print(f"\n{'='*60}")
    print(f"  {phase}")
    print(f"{'='*60}")
    if fold_scores:
        for i, fs in enumerate(fold_scores):
            print(f"  Fold {i+1}: {fs:.4f}")
        print(f"  Mean:   {np.mean(fold_scores):.4f}  Std: {np.std(fold_scores):.4f}")
    print(f"  Score:  {score:.4f}")
    print(f"{'='*60}")


def create_submission(test_indices: np.ndarray, predictions: np.ndarray, path: str = None):
    if path is None:
        path = SUBMISSION_PATH
    predictions = np.clip(predictions, 0, None)
    sub = pd.DataFrame({ID_COL: test_indices, TARGET: predictions})
    sub.to_csv(path, index=False)
    print(f"  Submission: {path}  Shape: {sub.shape}")
    print(f"  Demand: mean={sub[TARGET].mean():.6f}  range=[{sub[TARGET].min():.6f}, {sub[TARGET].max():.6f}]")
    return sub
