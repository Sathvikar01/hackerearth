"""Stage 5: Dynamic Blending with Weight Optimization."""
import numpy as np
from sklearn.metrics import r2_score
from src.config import W_GRID_SIZE


def optimize_blend_weight(val_actual: np.ndarray, val_pred_a: np.ndarray,
                          val_pred_b: np.ndarray, has_lag_mask: np.ndarray) -> tuple:
    """Find optimal blend weight W on validation set.

    Blending rule:
        - If no lag: Final = Model_A
        - If has lag: Final = W * Model_B + (1 - W) * Model_A

    Args:
        val_actual: Actual validation targets
        val_pred_a: Model A predictions
        val_pred_b: Model B predictions
        has_lag_mask: Boolean mask for rows with lag feature

    Returns:
        (best_w, best_score, blended_predictions)
    """
    w_grid = np.linspace(0.5, 1.0, W_GRID_SIZE)
    best_w = 0.5
    best_score = -np.inf
    best_preds = val_pred_a.copy()

    for w in w_grid:
        # Blend: for rows with lag, use W*B + (1-W)*A; otherwise just A
        blended = val_pred_a.copy()
        blended[has_lag_mask] = w * val_pred_b[has_lag_mask] + (1 - w) * val_pred_a[has_lag_mask]

        score = max(0, 100 * r2_score(val_actual, blended))

        if score > best_score:
            best_score = score
            best_w = w
            best_preds = blended.copy()

    return best_w, best_score, best_preds


def blend_predictions(pred_a: np.ndarray, pred_b: np.ndarray,
                      has_lag_mask: np.ndarray, w: float) -> np.ndarray:
    """Apply blending with optimized weight.

    Args:
        pred_a: Model A predictions
        pred_b: Model B predictions
        has_lag_mask: Boolean mask for rows with lag feature
        w: Blend weight

    Returns:
        Blended predictions
    """
    blended = pred_a.copy()
    blended[has_lag_mask] = w * pred_b[has_lag_mask] + (1 - w) * pred_a[has_lag_mask]
    return blended
