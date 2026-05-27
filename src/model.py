"""CatBoost model training with 5-fold KFold CV."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from src.config import CATBOOST_PARAMS, CAT_FEATURES, SEED
from src.utils import compute_score


def get_cat_indices(cat_cols: list, all_cols: list) -> list:
    return [all_cols.index(c) for c in cat_cols if c in all_cols]


def train_evaluate(train_df: pd.DataFrame, features: list, cat_cols: list,
                   target: str = "demand", n_splits: int = 5) -> tuple:
    """Train CatBoost with KFold, return (oof_preds, mean_score, fold_scores, models)."""
    X = train_df[features].copy()
    y = train_df[target].values
    cat_indices = get_cat_indices(cat_cols, features)

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_df))
    fold_scores = []
    models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_pred = model.predict(X_val)
        oof[val_idx] = val_pred

        fs = compute_score(y_val, val_pred)
        fold_scores.append(fs)
        models.append(model)
        print(f"    Fold {fold+1}: {fs:.4f}")

    mean_score = compute_score(y, oof)
    return oof, mean_score, fold_scores, models


def predict_test(models: list, test_df: pd.DataFrame, features: list,
                 cat_cols: list) -> np.ndarray:
    """Average predictions across fold models."""
    X = test_df[features].copy()
    cat_indices = get_cat_indices(cat_cols, features)
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    pool = Pool(X, cat_features=cat_indices)
    preds = np.zeros(len(test_df))
    for m in models:
        preds += m.predict(pool)
    preds /= len(models)
    return np.clip(preds, 0, None)


def get_feature_importance(models: list, features: list) -> pd.DataFrame:
    """Average feature importance across fold models."""
    imp = np.zeros(len(features))
    for m in models:
        imp += m.get_feature_importance()
    imp /= len(models)
    return pd.DataFrame({"feature": features, "importance": imp}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
