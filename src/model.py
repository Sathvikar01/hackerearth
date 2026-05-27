"""CatBoost model training and prediction wrapper."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from src.config import CATBOOST_PARAMS, CAT_FEATURES, SEED
from src.utils import compute_score


def get_cat_feature_indices(cat_cols: list, all_cols: list) -> list:
    """Get integer indices of categorical features in the feature matrix."""
    return [all_cols.index(c) for c in cat_cols if c in all_cols]


def train_model_phase1(train_df: pd.DataFrame, features: list,
                        cat_cols: list, target: str = "demand",
                        n_splits: int = 5) -> tuple:
    """Train CatBoost with GroupKFold grouped by geohash.

    Returns:
        (models, oof_predictions, mean_score, fold_scores)
    """
    X = train_df[features].copy()
    y = train_df[target].values
    groups = train_df["geohash"].values

    cat_indices = get_cat_feature_indices(cat_cols, features)

    # Ensure categorical columns are strings
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    gkf = GroupKFold(n_splits=n_splits)
    models = []
    oof_preds = np.zeros(len(train_df))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"  Fold {fold + 1}/{n_splits}...", end=" ")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        params = dict(CATBOOST_PARAMS)
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred

        fold_score = compute_score(y_val, val_pred)
        fold_scores.append(fold_score)
        print(f"Score: {fold_score:.4f}")

        models.append(model)

    mean_score = compute_score(y, oof_preds)
    return models, oof_preds, mean_score, fold_scores


def train_model_final(train_df: pd.DataFrame, features: list,
                       cat_cols: list, target: str = "demand",
                       n_splits: int = 5) -> tuple:
    """Train final model using all training data (for prediction on test).

    Uses the best iteration from CV as the iteration count.

    Returns:
        (model, mean_cv_score)
    """
    X = train_df[features].copy()
    y = train_df[target].values
    groups = train_df["geohash"].values

    cat_indices = get_cat_feature_indices(cat_cols, features)

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # First do CV to get best iteration
    gkf = GroupKFold(n_splits=n_splits)
    best_iterations = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        params = dict(CATBOOST_PARAMS)
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        best_iterations.append(model.best_iteration_)

    # Train final model on all data with median best iteration
    median_iter = int(np.median(best_iterations))
    print(f"  Final model iterations: {median_iter}")

    full_pool = Pool(X, y, cat_features=cat_indices)
    params = dict(CATBOOST_PARAMS)
    params["iterations"] = median_iter
    params.pop("early_stopping_rounds", None)

    final_model = CatBoostRegressor(**params)
    final_model.fit(full_pool)

    # Compute CV score for reporting
    cv_score = 0.0
    fold_scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        params_cv = dict(CATBOOST_PARAMS)
        m = CatBoostRegressor(**params_cv)
        m.fit(train_pool, eval_set=val_pool, use_best_model=True)
        val_pred = m.predict(X_val)
        fs = compute_score(y_val, val_pred)
        fold_scores.append(fs)

    cv_score = np.mean(fold_scores)
    return final_model, cv_score, fold_scores


def predict_test(model, test_df: pd.DataFrame, features: list,
                  cat_cols: list) -> np.ndarray:
    """Generate predictions on test data."""
    X = test_df[features].copy()
    cat_indices = get_cat_feature_indices(cat_cols, features)
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    test_pool = Pool(X, cat_features=cat_indices)
    predictions = model.predict(test_pool)
    return np.clip(predictions, 0, None)


def get_feature_importance(model, features: list) -> pd.DataFrame:
    """Get feature importance from a trained CatBoost model."""
    importance = model.get_feature_importance()
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df
