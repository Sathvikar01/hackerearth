"""Full 6-Stage Dual-Branch Pipeline (v5 — Spatial Clustering + Interaction Keys).

Architecture:
- Model A: Global Learner with K-Means clusters, rotated coordinates,
  Fourier harmonics, and high-order spatio-temporal interaction keys.
  CatBoost handles target encoding natively via ordered encoding.
- Model B: Lag Specialist using exact/fuzzy/hour lag features.

Upgrades over v4:
1. K-Means spatial clusters (K=10, K=50)
2. Rotated coordinates (15, 30, 45 degrees)
3. Higher-order Fourier harmonics
4. High-order interaction keys (geo_hour, cluster_dow, etc.)
5. Removed manual target encoding (CatBoost native TE is superior)
6. Removed toroidal traversal (replaced by spatial clusters)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.data_loader import load_data, chronological_split
from src.features import (
    apply_all_features, build_lag_features, build_geohash_stats,
    add_temporal_features, add_spatial_features, add_contextual_features,
    add_fourier_harmonics, add_spatial_clusters, add_rotated_coordinates,
    add_distance_to_center, add_interaction_keys,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
)
from src.models import train_model_a, train_model_b, predict_model_a, predict_model_b
from src.config import TARGET, SEED, MODEL_A_PARAMS, MODEL_B_PARAMS

import warnings
warnings.filterwarnings("ignore")


def run_pipeline():
    print("=" * 70)
    print("  DUAL-BRANCH ARCHITECTURE (v5 — Spatial Clusters + Interactions)")
    print("=" * 70)

    # ── STAGE 1: INGESTION ───────────────────────────────────
    print("\n  Stage 1: Loading data...")
    train, test = load_data()
    print(f"    Train: {train.shape}  Test: {test.shape}")

    train_split, val_split = chronological_split(train)
    print(f"    Train (Day 48): {train_split.shape}")
    print(f"    Val   (Day 49): {val_split.shape}")

    # ── STAGE 2: FEATURE FACTORY ─────────────────────────────
    print("\n  Stage 2: Building features (v5 — clusters + interactions)...")
    train_split, val_split = apply_all_features(
        train_split, val_split, include_lag=True, include_clusters=True, verbose=True
    )

    # Build lag column stubs for train_split (needed by Model B filtering)
    lookup_self = train_split.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()
    train_split["exact_lag_demand"] = train_split.apply(
        lambda r: lookup_self.get((r["geohash"], r["timestamp"]), np.nan), axis=1)
    train_split["fuzzy_lag_demand"] = np.nan
    train_split["hour_lag_demand"] = np.nan
    train_split["combined_lag"] = train_split["exact_lag_demand"]
    train_split["is_lag_missing"] = 0

    # Build features for test set (same transforms, no lag)
    add_temporal_features(test)
    add_fourier_harmonics(test, columns=["hour", "15_min_slot"], n_harmonics=2)
    add_spatial_features(test)
    add_rotated_coordinates(test, angles=[15, 30, 45])
    add_distance_to_center(test)

    # Transform test clusters using train-fitted KMeans
    from sklearn.cluster import KMeans
    for n in [10, 50]:
        col = f"cluster_{n}"
        coords_train = train_split[["latitude", "longitude"]].values
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(coords_train)
        test[col] = kmeans.predict(test[["latitude", "longitude"]].values).astype(str)

    add_contextual_features(test)
    add_interaction_keys(test)

    print("    Building test lag features...")
    test = build_geohash_stats(train_split, test)
    test = build_lag_features(train_split, test, verbose=True)

    # ── STAGE 3: DUAL-MODEL TRAINING ─────────────────────────
    print("\n  Stage 3: Training dual models...")
    val_actual = val_split[TARGET].values
    has_lag_mask = val_split["combined_lag"].notna().values

    # Model A: Global Learner
    print("    Training Model A (Global Learner, v5 features)...")
    model_a, val_pred_a, val_score_a = train_model_a(
        train_split, val_split, MODEL_A_FEATURES, MODEL_A_PARAMS, TARGET
    )
    print(f"    Model A Val Score: {val_score_a:.4f}")

    # Model B: Lag Specialist — train on val_split lag rows (real Day48->Day49 lag)
    print("    Training Model B (Lag Specialist)...")
    from catboost import CatBoostRegressor, Pool
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

    # ── STAGE 4: BLENDING (W=1.0) ────────────────────────────
    print("\n  Stage 4: Blending (W=1.0 for lag rows)...")
    best_w = 1.0
    val_blended = val_pred_a.copy()
    val_blended[has_lag_mask] = val_pred_b[has_lag_mask]
    best_score = max(0, 100 * r2_score(val_actual, val_blended))

    print(f"    W=1.0 Score: {best_score:.4f}")
    print(f"\n{'='*60}")
    print(f"  VALIDATION SCORES (Day 49 Holdout)")
    print(f"{'='*60}")
    print(f"  Model A (Global Learner):  {val_score_a:.4f}")
    print(f"  Model B (Lag Specialist):  {val_score_b:.4f}")
    print(f"  Final Blended:             {best_score:.4f}")
    print(f"{'='*60}")

    # ── STAGE 5: FINAL PREDICTION ────────────────────────────
    print("\n  Stage 5: Final prediction on test data...")

    # Retrain Model A on full data
    full_train = pd.concat([train_split, val_split], ignore_index=True)
    print("    Rebuilding lag features with full train data...")
    test = build_lag_features(full_train, test, verbose=True)
    full_train = build_geohash_stats(full_train, full_train)
    test = build_geohash_stats(full_train, test)

    print("    Retraining Model A on full train data...")
    # Use fewer iterations for final retrain (no eval set, so no early stopping)
    final_a_params = {k: v for k, v in MODEL_A_PARAMS.items() if k != "early_stopping_rounds"}
    final_a_params["iterations"] = 1000
    model_a_final, _, _ = train_model_a(
        full_train, test, MODEL_A_FEATURES, final_a_params, TARGET
    )

    # Retrain Model B
    print("    Retraining Model B on lag-available rows...")
    full_lag_mask = full_train["combined_lag"].notna()
    full_lag_rows = full_train[full_lag_mask].copy().reset_index(drop=True)

    if len(full_lag_rows) > 100 and model_b is not None:
        from catboost import CatBoostRegressor, Pool
        cat_cols_b = MODEL_B_FEATURES["cat"]
        num_cols_b = MODEL_B_FEATURES["num"]
        all_feat_b = cat_cols_b + num_cols_b

        X_b_full = full_lag_rows[all_feat_b].copy()
        y_b_full = full_lag_rows[TARGET].values
        for c in cat_cols_b:
            X_b_full[c] = X_b_full[c].astype(str)
        cat_idx_b = [all_feat_b.index(c) for c in cat_cols_b]
        pool_b_full = Pool(X_b_full, y_b_full, cat_features=cat_idx_b)
        model_b_final = CatBoostRegressor(
            **{k: v for k, v in MODEL_B_PARAMS.items() if k != "early_stopping_rounds"})
        model_b_final.fit(pool_b_full)

        test_pred_a = predict_model_a(model_a_final, test, MODEL_A_FEATURES)

        X_b_test = test[all_feat_b].copy()
        for c in cat_cols_b:
            X_b_test[c] = X_b_test[c].astype(str)
        pool_b_test = Pool(X_b_test, cat_features=cat_idx_b)
        test_pred_b = np.clip(model_b_final.predict(pool_b_test), 0, None)
    else:
        test_pred_a = predict_model_a(model_a_final, test, MODEL_A_FEATURES)
        test_pred_b = test["combined_lag"].fillna(0).values

    # Blend
    test_has_lag = test["combined_lag"].notna().values
    test_final = test_pred_a.copy()
    test_final[test_has_lag] = best_w * test_pred_b[test_has_lag] + (1 - best_w) * test_pred_a[test_has_lag]
    test_final = np.clip(test_final, 0, None)

    from src.utils import create_submission
    create_submission(test["Index"].values, test_final)

    return best_score


if __name__ == "__main__":
    score = run_pipeline()
    print(f"\n  DONE. Validation Score: {score:.4f}")
