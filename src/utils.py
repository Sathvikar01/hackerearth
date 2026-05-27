"""Utility functions: scoring, submission."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.config import SUBMISSION_PATH, ID_COL, TARGET


def compute_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    return max(0.0, 100.0 * r2_score(actual, predicted))


def print_scores(model_a_score: float, model_b_score: float, blended_score: float):
    print(f"\n{'='*60}")
    print(f"  VALIDATION SCORES (Day 49 Holdout)")
    print(f"{'='*60}")
    print(f"  Model A (Global Learner):  {model_a_score:.4f}")
    print(f"  Model B (Lag Specialist):  {model_b_score:.4f}")
    print(f"  Final Blended:             {blended_score:.4f}")
    print(f"{'='*60}")


def create_submission(test_indices: np.ndarray, predictions: np.ndarray, path: str = None):
    if path is None:
        path = SUBMISSION_PATH
    predictions = np.clip(predictions, 0, None)
    sub = pd.DataFrame({ID_COL: test_indices, TARGET: predictions})
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}  Shape: {sub.shape}")
    print(f"  Demand: mean={sub[TARGET].mean():.6f} range=[{sub[TARGET].min():.6f}, {sub[TARGET].max():.6f}]")
    return sub
