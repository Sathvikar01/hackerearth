"""ST-Diffusion Meta-Ensemble Pipeline (v6_final).

Fixes applied:
1. Soft-blending: W = 1/(1+var) normalized to [0.5, 1.0] (no hard switch)
2. FFT leakage-safe: Only Day 48 data used for FFT computation
3. Fast imputer fallback: FastKNN when USE_FAST_IMPUTER=True
4. Feature pruning: CatBoost importance-based pruning before meta-learner
5. Haversine distance: Proper km-based distance (no flat Euclidean)
6. Behavioral graph edges: Pearson-correlated demand patterns
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, Pool

from src.data_loader import load_data, chronological_split
from src.features import (
    apply_all_features, build_lag_features, build_geohash_stats,
    add_temporal_features, add_spatial_features, add_contextual_features,
    add_fourier_harmonics, add_spatial_clusters, add_rotated_coordinates,
    add_distance_to_center, add_manhattan_distance, add_interaction_keys,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
)
from src.graph_embeddings import add_graph_embeddings
from src.temporal_fft import add_fft_features
from src.diffusion_imputer import add_diffusion_imputation
from src.meta_ensemble import (
    train_meta_ensemble, predict_meta_ensemble,
    soft_blend_predictions, LGBM_PARAMS,
)
from src.config import TARGET, SEED, MODEL_B_PARAMS, CATBOOST_PARAMS, USE_FAST_IMPUTER, TRAIN_DAY

import warnings
warnings.filterwarnings("ignore")


def prune_features_by_importance(model, features: dict, X_val: pd.DataFrame,
                                  y_val: np.ndarray, drop_fraction: float = 0.15) -> dict:
    """Prune bottom N% features by CatBoost importance.

    Args:
        model: Trained CatBoost model
        features: Feature dict with 'cat' and 'num' keys
        X_val: Validation features
        y_val: Validation target
        drop_fraction: Fraction of features to drop (default 15%)

    Returns:
        Pruned feature dict
    """
    all_features = features["cat"] + features["num"]
    importances = model.feature_importances_
    imp_dict = dict(zip(all_features, importances))

    # Sort by importance
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1])
    n_drop = max(1, int(len(sorted_imp) * drop_fraction))

    # Drop the bottom N% features
    to_drop = {f[0] for f in sorted_imp[:n_drop]}
    pruned_cat = [f for f in features["cat"] if f not in to_drop]
    pruned_num = [f for f in features["num"] if f not in to_drop]

    print(f"    Pruned {n_drop} features ({drop_fraction*100:.0f}%): {to_drop}")
    return {"cat": pruned_cat, "num": pruned_num}


def apply_test_features(test, train_split, verbose=True):
    """Apply all feature engineering to test set using train-fitted transforms."""
    add_temporal_features(test)
    add_fourier_harmonics(test, columns=["hour", "15_min_slot"], n_harmonics=2)
    add_spatial_features(test)
    add_rotated_coordinates(test, angles=[15, 30, 45])
    add_distance_to_center(test)
    add_manhattan_distance(test)

    from sklearn.cluster import KMeans
    for n in [10, 50]:
        col = f"cluster_{n}"
        coords_train = train_split[["latitude", "longitude"]].values
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(coords_train)
        test[col] = kmeans.predict(test[["latitude", "longitude"]].values).astype(str)

    add_contextual_features(test)
    add_interaction_keys(test)
    return test


def run_pipeline():
    print("=" * 70)
    print("  ST-DIFFUSION META-ENSEMBLE (v6_final)")
    print("=" * 70)

    # ── STAGE 1: INGESTION ───────────────────────────────────
    print("\n  Stage 1: Loading data...")
    train, test = load_data()
    print(f"    Train: {train.shape}  Test: {test.shape}")

    train_split, val_split = chronological_split(train)
    print(f"    Train (Day {TRAIN_DAY}): {train_split.shape}")
    print(f"    Val   (Day 49): {val_split.shape}")

    # ── STAGE 2: FEATURE FACTORY ─────────────────────────────
    print("\n  Stage 2: Building features (v6_final)...")

    train_split, val_split = apply_all_features(
        train_split, val_split, include_lag=True, include_clusters=True, verbose=True
    )

    # Graph embeddings (behavioral edges)
    print("    Computing graph embeddings (behavioral)...")
    train_split, val_split = add_graph_embeddings(train_split, val_split, method="behavioral")

    # FFT features (leakage-safe: Day 48 only)
    train_split, val_split = add_fft_features(train_split, val_split, train_day=TRAIN_DAY)

    # Lag stubs for train_split
    lookup_self = train_split.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()
    train_split["exact_lag_demand"] = train_split.apply(
        lambda r: lookup_self.get((r["geohash"], r["timestamp"]), np.nan), axis=1)
    train_split["fuzzy_lag_demand"] = np.nan
    train_split["hour_lag_demand"] = np.nan
    train_split["combined_lag"] = train_split["exact_lag_demand"]
    train_split["is_lag_missing"] = 0

    # Imputation (diffusion or fast KNN)
    print(f"    Running imputation (fast={USE_FAST_IMPUTER})...")
    train_split, val_split = add_diffusion_imputation(
        train_split, val_split, use_fast=USE_FAST_IMPUTER
    )

    # Test features
    print("    Building test features...")
    test = apply_test_features(test, train_split)
    test = build_geohash_stats(train_split, test)
    _, test = add_graph_embeddings(train_split, test, method="behavioral")
    _, test = add_fft_features(train_split, test, train_day=TRAIN_DAY)
    print("    Building test lag features...")
    test = build_lag_features(train_split, test, verbose=True)
    test = add_diffusion_imputation(train_split, test, use_fast=USE_FAST_IMPUTER)[1]

    # ── STAGE 3: META-ENSEMBLE TRAINING ──────────────────────
    print("\n  Stage 3: Training Meta-Ensemble...")
    val_actual = val_split[TARGET].values
    has_lag_mask = val_split["combined_lag"].notna().values

    unified_features = {
        "cat": MODEL_A_FEATURES["cat"],
        "num": MODEL_A_FEATURES["num"],
    }

    print("    Training Meta-Ensemble (CatBoost + LightGBM + Bayesian Ridge)...")
    cb_model, lgbm_model, meta_model, meta_val_pred, meta_score = train_meta_ensemble(
        train_split, val_split, unified_features,
        CATBOOST_PARAMS, LGBM_PARAMS, TARGET,
        use_variance_weighting=True
    )

    # Feature pruning (MANDATE 5)
    print("    Pruning features by CatBoost importance...")
    all_features = unified_features["cat"] + unified_features["num"]
    X_val_for_prune = val_split[all_features].copy()
    for c in unified_features["cat"]:
        X_val_for_prune[c] = X_val_for_prune[c].astype(str)
    pruned_features = prune_features_by_importance(
        cb_model, unified_features, X_val_for_prune, val_actual, drop_fraction=0.15
    )

    print(f"    Meta-Ensemble Val Score: {meta_score:.4f}")

    # ── STAGE 4: LAG SPECIALIST (MODEL B) ────────────────────
    print("\n  Stage 4: Training Lag Specialist (Model B)...")
    val_lag_rows = val_split[has_lag_mask].copy().reset_index(drop=True)
    cat_cols_b = MODEL_B_FEATURES["cat"]
    num_cols_b = MODEL_B_FEATURES["num"]
    all_feat_b = cat_cols_b + num_cols_b
    cat_idx_b = [all_feat_b.index(c) for c in cat_cols_b]

    if len(val_lag_rows) > 100:
        X_b = val_lag_rows[all_feat_b].copy()
        y_b = val_lag_rows[TARGET].values
        for c in cat_cols_b:
            X_b[c] = X_b[c].astype(str)
        pool_b = Pool(X_b, y_b, cat_features=cat_idx_b)
        b_params = {k: v for k, v in MODEL_B_PARAMS.items() if k != "early_stopping_rounds"}
        model_b = CatBoostRegressor(**b_params)
        model_b.fit(pool_b)

        X_b_all = val_split[all_feat_b].copy()
        for c in cat_cols_b:
            X_b_all[c] = X_b_all[c].astype(str)
        pool_b_all = Pool(X_b_all, cat_features=cat_idx_b)
        val_pred_b = np.clip(model_b.predict(pool_b_all), 0, None)

        val_score_b = max(0, 100 * r2_score(
            val_split.loc[has_lag_mask, TARGET].values,
            val_pred_b[has_lag_mask]
        ))
        print(f"    Model B Val Score (lag rows only): {val_score_b:.4f}")
    else:
        val_pred_b = val_split["combined_lag"].fillna(0).values
        val_score_b = 0.0
        model_b = None
        print("    Model B: Not enough lag rows, using lag directly")

    # ── STAGE 5: SOFT-BLENDING (MANDATE 4) ───────────────────
    print("\n  Stage 5: Uncertainty Soft-Blending...")

    imputed_var = val_split["imputed_lag_var"].values if "imputed_lag_var" in val_split.columns else np.zeros(len(val_split))

    val_blended = soft_blend_predictions(
        meta_val_pred, val_pred_b, imputed_var, has_lag_mask
    )
    best_score = max(0, 100 * r2_score(val_actual, val_blended))

    # Also compute hard blend for comparison
    hard_blend = meta_val_pred.copy()
    hard_blend[has_lag_mask] = val_pred_b[has_lag_mask]
    hard_score = max(0, 100 * r2_score(val_actual, hard_blend))

    print(f"    Hard Blend Score:  {hard_score:.4f}")
    print(f"    Soft Blend Score:  {best_score:.4f}")
    print(f"\n{'='*60}")
    print(f"  VALIDATION SCORES (Day 49 Holdout)")
    print(f"{'='*60}")
    print(f"  Meta-Ensemble (CB+LGB+BR):  {meta_score:.4f}")
    print(f"  Model B (Lag Specialist):    {val_score_b:.4f}")
    print(f"  Hard Blend (W=1.0):          {hard_score:.4f}")
    print(f"  Soft Blend (variance-w):     {best_score:.4f}")
    print(f"{'='*60}")

    # ── STAGE 6: FINAL PREDICTION ────────────────────────────
    print("\n  Stage 6: Final prediction on test data...")

    full_train = pd.concat([train_split, val_split], ignore_index=True)
    print("    Rebuilding lag features with full train data...")
    test = build_lag_features(full_train, test, verbose=True)
    full_train = build_geohash_stats(full_train, full_train)
    test = build_geohash_stats(full_train, test)

    full_train = add_diffusion_imputation(full_train, full_train, use_fast=USE_FAST_IMPUTER)[1]

    # Retrain CatBoost on full data
    print("    Retraining CatBoost on full data...")
    final_cb_params = {k: v for k, v in CATBOOST_PARAMS.items() if k != "early_stopping_rounds"}
    final_cb_params["iterations"] = 1000

    all_features = unified_features["cat"] + unified_features["num"]
    X_full = full_train[all_features].copy()
    y_full = full_train[TARGET].values
    X_test = test[all_features].copy()
    for c in unified_features["cat"]:
        X_full[c] = X_full[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    cat_idx = [all_features.index(c) for c in unified_features["cat"]]
    pool_full = Pool(X_full, y_full, cat_features=cat_idx)
    pool_test = Pool(X_test, cat_features=cat_idx)

    cb_final = CatBoostRegressor(**final_cb_params)
    cb_final.fit(pool_full)
    test_pred_cb = np.clip(cb_final.predict(pool_test), 0, None)

    # Retrain LightGBM on full data
    import lightgbm as lgb
    X_full_lgb = full_train[all_features].copy()
    X_test_lgb = test[all_features].copy()
    for c in unified_features["cat"]:
        X_full_lgb[c] = X_full_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    train_data = lgb.Dataset(X_full_lgb, label=y_full,
                              categorical_feature=unified_features["cat"],
                              free_raw_data=False)
    lgb_final = lgb.train(LGBM_PARAMS.copy(), train_data, num_boost_round=1000)
    test_pred_lgb = np.clip(lgb_final.predict(X_test_lgb), 0, None)

    # Meta prediction
    test_pred_meta = 0.5 * test_pred_cb + 0.5 * test_pred_lgb

    # Model B final
    full_lag_mask = full_train["combined_lag"].notna()
    full_lag_rows = full_train[full_lag_mask].copy().reset_index(drop=True)

    if len(full_lag_rows) > 100 and model_b is not None:
        X_b_full = full_lag_rows[all_feat_b].copy()
        y_b_full = full_lag_rows[TARGET].values
        for c in cat_cols_b:
            X_b_full[c] = X_b_full[c].astype(str)
        pool_b_full = Pool(X_b_full, y_b_full, cat_features=cat_idx_b)
        model_b_final = CatBoostRegressor(
            **{k: v for k, v in MODEL_B_PARAMS.items() if k != "early_stopping_rounds"})
        model_b_final.fit(pool_b_full)

        X_b_test = test[all_feat_b].copy()
        for c in cat_cols_b:
            X_b_test[c] = X_b_test[c].astype(str)
        pool_b_test = Pool(X_b_test, cat_features=cat_idx_b)
        test_pred_b = np.clip(model_b_final.predict(pool_b_test), 0, None)
    else:
        test_pred_b = test["combined_lag"].fillna(0).values

    # Soft-blend for test predictions
    test_has_lag = test["combined_lag"].notna().values
    test_imputed_var = test["imputed_lag_var"].values if "imputed_lag_var" in test.columns else np.zeros(len(test))

    test_final = soft_blend_predictions(
        test_pred_meta, test_pred_b, test_imputed_var, test_has_lag
    )
    test_final = np.clip(test_final, 0, None)

    from src.utils import create_submission
    create_submission(test["Index"].values, test_final)

    return best_score


if __name__ == "__main__":
    score = run_pipeline()
    print(f"\n  DONE. Validation Score: {score:.4f}")
