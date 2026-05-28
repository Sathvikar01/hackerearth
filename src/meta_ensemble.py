"""Meta-Ensemble with Uncertainty Soft-Blending.

- Base Model 1: CatBoost (categorical interactions)
- Base Model 2: LightGBM (fast gradient boosting)
- Meta-Learner: Bayesian Ridge (stacked predictions)
- Soft-blending: W = sigmoid(-variance) for smooth uncertainty-weighted blend
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb


def get_cat_indices(cat_cols: list, all_cols: list) -> list:
    return [all_cols.index(c) for c in cat_cols if c in all_cols]


def train_catboost_base(X_train: pd.DataFrame, y_train: np.ndarray,
                         X_val: pd.DataFrame, y_val: np.ndarray,
                         cat_cols: list, params: dict,
                         sample_weights: np.ndarray = None) -> tuple:
    all_cols = list(X_train.columns)
    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)
    cat_indices = get_cat_indices(cat_cols, all_cols)

    train_pool = Pool(X_train, y_train, cat_features=cat_indices, weight=sample_weights)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = np.clip(model.predict(val_pool), 0, None)
    return model, val_pred


def train_lgbm_base(X_train: pd.DataFrame, y_train: np.ndarray,
                     X_val: pd.DataFrame, y_val: np.ndarray,
                     cat_cols: list, params: dict,
                     sample_weights: np.ndarray = None) -> tuple:
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_val[c] = X_val[c].astype("category")

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                              categorical_feature=cat_cols, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols,
                            free_raw_data=False, reference=train_data)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(params, train_data, valid_sets=[val_data],
                      num_boost_round=2000, callbacks=callbacks)

    val_pred = np.clip(model.predict(X_val), 0, None)
    return model, val_pred


def compute_inverse_variance_weights(imputation_variance: np.ndarray,
                                       alpha: float = 10.0,
                                       real_weight: float = 1.0) -> np.ndarray:
    weights = np.where(
        imputation_variance == 0,
        real_weight,
        1.0 / (1.0 + alpha * imputation_variance)
    )
    return weights


def compute_soft_blend_weight(imputed_lag_var: np.ndarray) -> np.ndarray:
    """Compute soft blend weight W using sigmoid of negative variance.

    W = 1 / (1 + variance) — normalized between 0.5 and 1.0
    - If variance = 0 (exact lag exists): W = 1.0 (100% Model B)
    - If variance is high: W approaches 0.5 (50/50 blend)

    This replaces the hard switch (W=1.0 for all lag rows) with a smooth,
    uncertainty-weighted blend that prevents discontinuous prediction curves.

    Args:
        imputed_lag_var: Variance from diffusion/KNN imputer (0 for real lag)

    Returns:
        Blend weight W for each row (between 0.5 and 1.0)
    """
    W = 1.0 / (1.0 + imputed_lag_var)
    # Normalize to [0.5, 1.0]
    W = 0.5 + 0.5 * W
    return W


def train_meta_ensemble(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         features: dict, catboost_params: dict,
                         lgbm_params: dict, target: str = "demand",
                         use_variance_weighting: bool = True) -> tuple:
    cat_cols = features["cat"]
    num_cols = features["num"]
    all_features = cat_cols + num_cols

    X_train = train_df[all_features].copy()
    y_train = train_df[target].values
    X_val = val_df[all_features].copy()
    y_val = val_df[target].values

    if use_variance_weighting and "imputed_lag_var" in train_df.columns:
        train_weights = compute_inverse_variance_weights(
            train_df["imputed_lag_var"].values
        )
    else:
        train_weights = None

    print("    Training CatBoost base model...")
    cb_model, cb_val_pred = train_catboost_base(
        X_train.copy(), y_train, X_val.copy(), y_val,
        cat_cols, catboost_params, train_weights
    )
    cb_score = max(0, 100 * r2_score(y_val, cb_val_pred))
    print(f"    CatBoost Score: {cb_score:.4f}")

    print("    Training LightGBM base model...")
    lgbm_model, lgbm_val_pred = train_lgbm_base(
        X_train.copy(), y_train, X_val.copy(), y_val,
        cat_cols, lgbm_params, train_weights
    )
    lgbm_score = max(0, 100 * r2_score(y_val, lgbm_val_pred))
    print(f"    LightGBM Score: {lgbm_score:.4f}")

    stacked_train = np.column_stack([cb_val_pred, lgbm_val_pred])

    if "imputed_lag_var" in val_df.columns:
        stacked_train = np.column_stack([
            stacked_train,
            val_df["imputed_lag_var"].values,
            val_df["is_lag_imputed"].values if "is_lag_imputed" in val_df.columns else np.zeros(len(val_df)),
        ])

    print("    Training Bayesian Ridge meta-learner...")
    meta_model = BayesianRidge()
    meta_model.fit(stacked_train, y_val)
    meta_pred = np.clip(meta_model.predict(stacked_train), 0, None)
    meta_score = max(0, 100 * r2_score(y_val, meta_pred))
    print(f"    Meta-Ensemble Score: {meta_score:.4f}")

    return cb_model, lgbm_model, meta_model, meta_pred, meta_score


def predict_meta_ensemble(cb_model, lgbm_model, meta_model,
                           test_df: pd.DataFrame, features: dict) -> np.ndarray:
    cat_cols = features["cat"]
    num_cols = features["num"]
    all_features = cat_cols + num_cols

    X_test = test_df[all_features].copy()
    for c in cat_cols:
        X_test[c] = X_test[c].astype(str)

    cat_indices = get_cat_indices(cat_cols, all_features)
    test_pool = Pool(X_test, cat_features=cat_indices)
    cb_pred = np.clip(cb_model.predict(test_pool), 0, None)

    X_test_lgb = test_df[all_features].copy()
    for c in cat_cols:
        X_test_lgb[c] = X_test_lgb[c].astype("category")
    lgbm_pred = np.clip(lgbm_model.predict(X_test_lgb), 0, None)

    stacked = np.column_stack([cb_pred, lgbm_pred])

    if "imputed_lag_var" in test_df.columns:
        stacked = np.column_stack([
            stacked,
            test_df["imputed_lag_var"].values,
            test_df["is_lag_imputed"].values if "is_lag_imputed" in test_df.columns else np.zeros(len(test_df)),
        ])

    final_pred = np.clip(meta_model.predict(stacked), 0, None)
    return final_pred


def soft_blend_predictions(meta_pred: np.ndarray, model_b_pred: np.ndarray,
                            imputed_lag_var: np.ndarray,
                            has_lag_mask: np.ndarray) -> np.ndarray:
    """Uncertainty soft-blending between Meta-Ensemble and Model B.

    Replaces the hard switch (W=1.0 for all lag rows) with a smooth,
    variance-weighted blend:
        W = 1 / (1 + variance), normalized to [0.5, 1.0]
        Final = W * Model_B + (1 - W) * Meta_Ensemble

    For rows with exact lag (variance=0): W=1.0 (100% Model B)
    For rows with imputed lag (high variance): W approaches 0.5

    Args:
        meta_pred: Meta-Ensemble predictions
        model_b_pred: Model B predictions
        imputed_lag_var: Variance from imputer (0 for real lag)
        has_lag_mask: Boolean mask for rows with any lag

    Returns:
        Blended predictions
    """
    final = meta_pred.copy()

    # Compute soft blend weight for lag rows
    W = compute_soft_blend_weight(imputed_lag_var)

    # Apply soft blend only to rows with lag
    lag_indices = np.where(has_lag_mask)[0]
    for i in lag_indices:
        w = W[i]
        final[i] = w * model_b_pred[i] + (1 - w) * meta_pred[i]

    return final


LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}
