"""Stage 4: Dual-Model CatBoost Training."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from src.config import CATBOOST_PARAMS, SEED


def get_cat_indices(cat_cols: list, all_cols: list) -> list:
    """Get integer indices of categorical features."""
    return [all_cols.index(c) for c in cat_cols if c in all_cols]


def train_model_a(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  features: dict, target: str = "demand") -> tuple:
    """Train Model A: Global Learner (no lag features).

    If val_df has target column: train with early stopping on val.
    If val_df has no target (test data): train on train_df only.

    Returns:
        (model, val_predictions, val_score)
    """
    from sklearn.metrics import r2_score

    cat_cols = features["cat"]
    num_cols = features["num"]
    all_features = cat_cols + num_cols

    X_train = train_df[all_features].copy()
    y_train = train_df[target].values
    X_val = val_df[all_features].copy()

    has_target = target in val_df.columns

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)

    cat_indices = get_cat_indices(cat_cols, all_features)

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)

    if has_target:
        y_val = val_df[target].values
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)
        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        val_pred = model.predict(val_pool)
        val_pred = np.clip(val_pred, 0, None)
        val_score = max(0, 100 * r2_score(y_val, val_pred))
    else:
        model = CatBoostRegressor(**{k: v for k, v in CATBOOST_PARAMS.items() if k != "early_stopping_rounds"})
        model.fit(train_pool)
        val_pool = Pool(X_val, cat_features=cat_indices)
        val_pred = model.predict(val_pool)
        val_pred = np.clip(val_pred, 0, None)
        val_score = 0.0

    return model, val_pred, val_score


def train_model_b(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  features: dict, target: str = "demand") -> tuple:
    """Train Model B: Lag Specialist (only rows with lag != NaN).

    Returns:
        (model, val_predictions, val_score)
    """
    from sklearn.metrics import r2_score

    cat_cols = features["cat"]
    num_cols = features["num"]
    all_features = cat_cols + num_cols

    # Filter: only rows where exact_lag_demand is NOT NaN
    train_mask = train_df["exact_lag_demand"].notna()
    val_mask = val_df["exact_lag_demand"].notna()

    train_filtered = train_df[train_mask].copy()
    val_filtered = val_df[val_mask].copy()

    if len(train_filtered) == 0 or len(val_filtered) == 0:
        print("    Model B: No rows with lag available, returning zeros")
        return None, np.zeros(len(val_df)), 0.0

    X_train = train_filtered[all_features].copy()
    y_train = train_filtered[target].values
    X_val = val_filtered[all_features].copy()
    y_val = val_filtered[target].values

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)

    cat_indices = get_cat_indices(cat_cols, all_features)

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Predict on ALL val rows (fill non-lag rows with 0)
    val_pred_full = np.zeros(len(val_df))
    val_pred_lag = model.predict(X_val)
    val_pred_lag = np.clip(val_pred_lag, 0, None)
    val_pred_full[val_mask] = val_pred_lag

    # Score only on lag rows
    val_score = max(0, 100 * r2_score(y_val, val_pred_lag))

    return model, val_pred_full, val_score


def predict_model_a(model, test_df: pd.DataFrame, features: dict) -> np.ndarray:
    """Generate Model A predictions on test data."""
    cat_cols = features["cat"]
    num_cols = features["num"]
    all_features = cat_cols + num_cols

    X_test = test_df[all_features].copy()
    for c in cat_cols:
        X_test[c] = X_test[c].astype(str)

    cat_indices = get_cat_indices(cat_cols, all_features)
    test_pool = Pool(X_test, cat_features=cat_indices)

    preds = model.predict(test_pool)
    return np.clip(preds, 0, None)


def predict_model_b(model, test_df: pd.DataFrame, features: dict) -> np.ndarray:
    """Generate Model B predictions on test data.

    Only predicts rows where exact_lag_demand is NOT NaN.
    Returns zeros for rows without lag.
    """
    if model is None:
        return np.zeros(len(test_df))

    cat_cols = features["cat"]
    num_cols = features["num"]
    all_features = cat_cols + num_cols

    test_mask = test_df["exact_lag_demand"].notna()
    test_filtered = test_df[test_mask].copy()

    if len(test_filtered) == 0:
        return np.zeros(len(test_df))

    X_test = test_filtered[all_features].copy()
    for c in cat_cols:
        X_test[c] = X_test[c].astype(str)

    cat_indices = get_cat_indices(cat_cols, all_features)
    test_pool = Pool(X_test, cat_features=cat_indices)

    preds_full = np.zeros(len(test_df))
    preds_lag = model.predict(test_pool)
    preds_lag = np.clip(preds_lag, 0, None)
    preds_full[test_mask] = preds_lag

    return preds_full
