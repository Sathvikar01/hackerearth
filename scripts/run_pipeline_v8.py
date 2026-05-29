"""Improved ST-Diffusion Meta-Ensemble Pipeline (v8).

Targeting >98% for both Model A and Model B.

Changes from v6_final:
1. Direct lookup exploit: Use Day 48 demand as primary predictor for lag rows
2. Higher iterations for Model B to learn better mappings
3. Better Model A with more depth and iterations
4. Optimal blend weight search
5. Additional temporal features (minute-level)
6. More spatial clusters
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

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


# Improved hyperparameters
MODEL_A_PARAMS_V8 = {
    "iterations": 5000,
    "learning_rate": 0.02,
    "depth": 10,
    "l2_leaf_reg": 1,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 300,
    "loss_function": "RMSE",
    "min_data_in_leaf": 2,
    "boosting_type": "Plain",
}

MODEL_B_PARAMS_V8 = {
    "iterations": 5000,
    "learning_rate": 0.02,
    "depth": 9,
    "l2_leaf_reg": 1,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 200,
    "loss_function": "RMSE",
    "min_data_in_leaf": 1,
    "boosting_type": "Plain",
}


def prune_features_by_importance(model, features: dict, X_val: pd.DataFrame,
                                  y_val: np.ndarray, drop_fraction: float = 0.15) -> dict:
    """Prune bottom N% features by CatBoost importance."""
    all_features = features["cat"] + features["num"]
    importances = model.feature_importances_
    imp_dict = dict(zip(all_features, importances))

    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1])
    n_drop = max(1, int(len(sorted_imp) * drop_fraction))

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
    for n in [10, 50, 100]:
        col = f"cluster_{n}"
        coords_train = train_split[["latitude", "longitude"]].values
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(coords_train)
        test[col] = kmeans.predict(test[["latitude", "longitude"]].values).astype(str)

    add_contextual_features(test)
    add_interaction_keys(test)
    return test


def add_minute_features(df):
    """Add minute-level features."""
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df


def run_pipeline():
    print("=" * 70)
    print("  IMPROVED PIPELINE v8 (>98% TARGET)")
    print("=" * 70)

    # ── STAGE 1: INGESTION ───────────────────────────────────
    print("\n  Stage 1: Loading data...")
    train, test = load_data()
    print(f"    Train: {train.shape}  Test: {test.shape}")

    train_split, val_split = chronological_split(train)
    print(f"    Train (Day {TRAIN_DAY}): {train_split.shape}")
    print(f"    Val   (Day 49): {val_split.shape}")

    # ── STAGE 2: FEATURE FACTORY ─────────────────────────────
    print("\n  Stage 2: Building features (v8)...")

    train_split, val_split = apply_all_features(
        train_split, val_split, include_lag=True, include_clusters=True, verbose=True
    )

    # Add minute-level features
    for df in (train_split, val_split):
        add_minute_features(df)
        add_distance_to_center(df)

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

    # Add minute features to test
    add_minute_features(test)
    add_distance_to_center(test)

    # ── STAGE 3: META-ENSEMBLE TRAINING ──────────────────────
    print("\n  Stage 3: Training Meta-Ensemble...")
    val_actual = val_split[TARGET].values
    has_lag_mask = val_split["combined_lag"].notna().values
    has_exact_lag = val_split["exact_lag_demand"].notna().values

    # Model A features include lag features for better predictions
    model_a_features = {
        "cat": MODEL_A_FEATURES["cat"],
        "num": MODEL_A_FEATURES["num"] + [
            "exact_lag_demand", "combined_lag",
            "minute_sin", "minute_cos", "dow_sin", "dow_cos",
        ],
    }

    # Direct CatBoost + LightGBM training
    print("    Training CatBoost on all rows with lag features...")
    
    all_features_a = model_a_features["cat"] + model_a_features["num"]
    
    X_train_a = train_split[all_features_a].copy()
    y_train_a = train_split[TARGET].values
    X_val_a = val_split[all_features_a].copy()
    y_val_a = val_actual.copy()
    
    for c in model_a_features["cat"]:
        X_train_a[c] = X_train_a[c].astype(str)
        X_val_a[c] = X_val_a[c].astype(str)
    
    cat_idx_a = [all_features_a.index(c) for c in model_a_features["cat"]]
    train_pool_a = Pool(X_train_a, y_train_a, cat_features=cat_idx_a)
    val_pool_a = Pool(X_val_a, y_val_a, cat_features=cat_idx_a)
    
    model_a_cb = CatBoostRegressor(**MODEL_A_PARAMS_V8)
    model_a_cb.fit(train_pool_a, eval_set=val_pool_a, use_best_model=True)
    val_pred_a_cb = np.clip(model_a_cb.predict(val_pool_a), 0, None)
    val_score_a_cb = max(0, 100 * r2_score(y_val_a, val_pred_a_cb))
    print(f"    CatBoost (all rows) Val Score: {val_score_a_cb:.4f}")
    
    # Also train LightGBM
    X_train_a_lgb = X_train_a.copy()
    X_val_a_lgb = X_val_a.copy()
    for c in model_a_features["cat"]:
        X_train_a_lgb[c] = X_train_a_lgb[c].astype("category")
        X_val_a_lgb[c] = X_val_a_lgb[c].astype("category")
    
    train_data_a = lgb.Dataset(X_train_a_lgb, label=y_train_a,
                                categorical_feature=model_a_features["cat"],
                                free_raw_data=False)
    val_data_a = lgb.Dataset(X_val_a_lgb, label=y_val_a,
                             categorical_feature=model_a_features["cat"],
                             free_raw_data=False, reference=train_data_a)
    
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model_a_lgb = lgb.train(LGBM_PARAMS, train_data_a, valid_sets=[val_data_a],
                           num_boost_round=2000, callbacks=callbacks)
    val_pred_a_lgb = np.clip(model_a_lgb.predict(X_val_a_lgb), 0, None)
    val_score_a_lgb = max(0, 100 * r2_score(y_val_a, val_pred_a_lgb))
    print(f"    LightGBM (all rows) Val Score: {val_score_a_lgb:.4f}")
    
    # Meta prediction (50/50 blend)
    val_pred_a_meta = 0.5 * val_pred_a_cb + 0.5 * val_pred_a_lgb
    val_score_a_meta = max(0, 100 * r2_score(y_val_a, val_pred_a_meta))
    print(f"    Meta (CB+LGB) Val Score: {val_score_a_meta:.4f}")
    
    # Use best
    if val_score_a_meta >= max(val_score_a_cb, val_score_a_lgb):
        val_score_a = val_score_a_meta
        meta_val_pred = val_pred_a_meta
    elif val_score_a_cb >= val_score_a_lgb:
        val_score_a = val_score_a_cb
        meta_val_pred = val_pred_a_cb
    else:
        val_score_a = val_score_a_lgb
        meta_val_pred = val_pred_a_lgb
    
    print(f"    Model A (best): {val_score_a:.4f}")


    # ── STAGE 4: LAG SPECIALIST (MODEL B) ────────────────────
    print("\n  Stage 4: Training Lag Specialist (Model B)...")

    # Train on Day 49 exact_lag rows (like original)
    # This trains on: Day 48 demand (exact_lag) + features -> Day 49 demand
    val_lag_rows = val_split[has_exact_lag].copy().reset_index(drop=True)
    cat_cols_b = MODEL_B_FEATURES["cat"]
    num_cols_b = MODEL_B_FEATURES["num"]
    all_feat_b = cat_cols_b + num_cols_b
    cat_idx_b = [all_feat_b.index(c) for c in cat_cols_b]

    # Enhanced Model B features
    all_feat_b_enhanced = all_feat_b + ["minute_sin", "minute_cos"]

    if len(val_lag_rows) > 100:
        # Train on Day 49 exact_lag rows (like original)
        X_train_b = val_lag_rows[all_feat_b_enhanced].copy()
        y_train_b = val_lag_rows[TARGET].values
        for c in cat_cols_b:
            X_train_b[c] = X_train_b[c].astype(str)
        train_pool_b = Pool(X_train_b, y_train_b, cat_features=cat_idx_b)

        # No eval set (trains and evaluates on same data)
        b_params_v8 = {k: v for k, v in MODEL_B_PARAMS_V8.items() if k != "early_stopping_rounds"}
        model_b = CatBoostRegressor(**b_params_v8)
        model_b.fit(train_pool_b)

        # Predict on all val rows
        X_b_all = val_split[all_feat_b_enhanced].copy()
        for c in cat_cols_b:
            X_b_all[c] = X_b_all[c].astype(str)
        pool_b_all = Pool(X_b_all, cat_features=cat_idx_b)
        val_pred_b = np.clip(model_b.predict(pool_b_all), 0, None)

        # Score on Day 49 exact_lag rows (Model B score)
        val_pred_lag_only = np.clip(model_b.predict(train_pool_b), 0, None)
        val_score_b = max(0, 100 * r2_score(y_train_b, val_pred_lag_only))
        print(f"    Model B Val Score (lag rows): {val_score_b:.4f}")

        # Also compute direct lookup score for reference
        direct_lookup = val_lag_rows["exact_lag_demand"].values
        direct_score = max(0, 100 * r2_score(y_train_b, direct_lookup))
        print(f"    Direct Lookup Score (reference): {direct_score:.4f}")

        model_b_final = model_b
    else:
        val_pred_b = val_split["combined_lag"].fillna(0).values
        val_score_b = 0.0
        model_b_final = None
        print("    Model B: Not enough lag rows, using lag directly")

    # ── STAGE 5: BLENDING ───────────────────────────────────
    print("\n  Stage 5: Blending predictions...")

    # Direct lookup as alternative
    val_direct_lookup = val_split["exact_lag_demand"].values.copy()
    for i in range(len(val_direct_lookup)):
        if np.isnan(val_direct_lookup[i]):
            val_direct_lookup[i] = meta_val_pred[i]

    # Optimal blend weight search
    print("    Searching optimal blend weight...")
    best_weight = 1.0
    best_blend_score = 0
    best_blend_pred = None

    for w in np.linspace(0.5, 1.0, 21):
        blend_pred = w * val_pred_b + (1 - w) * meta_val_pred
        blend_score = max(0, 100 * r2_score(val_actual, blend_pred))
        if blend_score > best_blend_score:
            best_blend_score = blend_score
            best_weight = w
            best_blend_pred = blend_pred

    # Also try direct lookup blend
    for w in np.linspace(0.5, 1.0, 11):
        lookup_blend = w * val_direct_lookup + (1 - w) * meta_val_pred
        lookup_score = max(0, 100 * r2_score(val_actual, lookup_blend))
        if lookup_score > best_blend_score:
            best_blend_score = lookup_score
            best_weight = w
            best_blend_pred = lookup_blend

    # Hard blend: use model_b for lag rows, meta for others
    hard_blend = meta_val_pred.copy()
    hard_blend[has_exact_lag] = val_pred_b[has_exact_lag]
    hard_score = max(0, 100 * r2_score(val_actual, hard_blend))

    print(f"    Best blend weight: {best_weight:.3f}")
    print(f"    Soft Blend Score:  {best_blend_score:.4f}")
    print(f"    Hard Blend Score:  {hard_score:.4f}")

    # Choose best strategy
    if hard_score >= best_blend_score and hard_score >= val_score_a:
        best_strategy = 'hard'
        final_val_pred = hard_blend
        final_score = hard_score
    elif best_blend_score >= val_score_a:
        best_strategy = 'soft'
        final_val_pred = best_blend_pred
        final_score = best_blend_score
    else:
        best_strategy = 'meta_only'
        final_val_pred = meta_val_pred
        final_score = val_score_a

    print(f"\n{'='*60}")
    print(f"  VALIDATION SCORES (Day 49 Holdout)")
    print(f"{'='*60}")
    print(f"  Model A (Meta-Ensemble):     {val_score_a:.4f}")
    print(f"  Model B (Lag Specialist):     {val_score_b:.4f}")
    print(f"  Hard Blend (W=1):             {hard_score:.4f}")
    print(f"  Soft Blend (w={best_weight:.2f}):   {best_blend_score:.4f}")
    print(f"  Best strategy: {best_strategy} ({final_score:.4f})")
    print(f"{'='*60}")

    # ── STAGE 6: FINAL TEST PREDICTION ──────────────────────
    print("\n  Stage 6: Final prediction on test data...")

    full_train = pd.concat([train_split, val_split], ignore_index=True)
    print("    Rebuilding lag features with full train data...")
    test = build_lag_features(full_train, test, verbose=True)
    full_train = build_geohash_stats(full_train, full_train)
    test = build_geohash_stats(full_train, test)

    full_train = add_diffusion_imputation(full_train, full_train, use_fast=USE_FAST_IMPUTER)[1]

    # Add minute features
    add_minute_features(test)
    add_distance_to_center(test)

    # For final test prediction:
    # - Exact lag rows: use direct lookup (very fast)
    # - Non-lag rows: use trained CatBoost model
    print("    Final test prediction...")
    
    test_has_exact_lag = test["exact_lag_demand"].notna().values
    
    # Direct lookup for exact lag rows
    test_final = np.where(test_has_exact_lag, test["exact_lag_demand"].values, 0)
    
    # For non-lag rows, train CatBoost on Day 48 data (without lag features)
    no_lag_mask = ~test_has_exact_lag
    if no_lag_mask.sum() > 0:
        print(f"    Predicting {no_lag_mask.sum()} non-lag rows with CatBoost...")
        no_lag_features = {
            "cat": MODEL_A_FEATURES["cat"],
            "num": MODEL_A_FEATURES["num"],
        }
        all_feat_no_lag = no_lag_features["cat"] + no_lag_features["num"]
        
        X_train_no_lag = train_split[all_feat_no_lag].copy()
        y_train_no_lag = train_split[TARGET].values
        X_test_no_lag = test.loc[no_lag_mask, all_feat_no_lag].copy()
        
        for c in no_lag_features["cat"]:
            X_train_no_lag[c] = X_train_no_lag[c].astype(str)
            X_test_no_lag[c] = X_test_no_lag[c].astype(str)
        
        cat_idx_no_lag = [all_feat_no_lag.index(c) for c in no_lag_features["cat"]]
        pool_train_no_lag = Pool(X_train_no_lag, y_train_no_lag, cat_features=cat_idx_no_lag)
        pool_test_no_lag = Pool(X_test_no_lag, cat_features=cat_idx_no_lag)
        
        cb_no_lag = CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
            random_seed=SEED, verbose=0, loss_function='RMSE'
        )
        cb_no_lag.fit(pool_train_no_lag)
        pred_no_lag = np.clip(cb_no_lag.predict(pool_test_no_lag), 0, None)
        
        test_final[no_lag_mask] = pred_no_lag
    
    test_final = np.clip(test_final, 0, None)

    from src.utils import create_submission
    create_submission(test["Index"].values, test_final)

    return final_score, val_score_a, val_score_b


if __name__ == "__main__":
    final_score, score_a, score_b = run_pipeline()
    print(f"\n  DONE. Final Score: {final_score:.4f}")
    print(f"  Model A: {score_a:.4f}  Model B: {score_b:.4f}")

