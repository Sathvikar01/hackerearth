"""ST-Diffusion Meta-Ensemble Pipeline (v6).

Architecture:
- Phase 1: Deep Representation Learning
  - Node2Vec graph embeddings for spatial topology
  - FFT spectral features for temporal periodicity
  - K-Means clusters, rotated coordinates, Fourier harmonics

- Phase 2: Generative Imputation (Diffusion Imputer)
  - Denoising MLP trained on rows with lag data
  - Generates N samples for missing lags -> Imputed Mean + Variance
  - Uncertainty-aware features: imputed_lag, imputed_lag_var, is_lag_imputed

- Phase 3: Meta-Ensemble Forecasting
  - Base Model 1: CatBoost (categorical interactions)
  - Base Model 2: LightGBM (fast gradient boosting)
  - Meta-Learner: Bayesian Ridge (stacked predictions)
  - Inverse-variance sample weighting for imputation uncertainty

- Phase 4: Lag Specialist (Model B)
  - Trained on rows with real lag data
  - Blended with meta-ensemble for final prediction
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
    add_distance_to_center, add_interaction_keys,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
)
from src.graph_embeddings import add_graph_embeddings
from src.temporal_fft import add_fft_features
from src.diffusion_imputer import add_diffusion_imputation
from src.meta_ensemble import train_meta_ensemble, predict_meta_ensemble, LGBM_PARAMS
from src.config import TARGET, SEED, MODEL_B_PARAMS, CATBOOST_PARAMS

import warnings
warnings.filterwarnings("ignore")


def apply_test_features(test, train_split, verbose=True):
    """Apply all feature engineering to test set using train-fitted transforms."""
    add_temporal_features(test)
    add_fourier_harmonics(test, columns=["hour", "15_min_slot"], n_harmonics=2)
    add_spatial_features(test)
    add_rotated_coordinates(test, angles=[15, 30, 45])
    add_distance_to_center(test)

    # Clusters using train-fitted KMeans
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
    print("  ST-DIFFUSION META-ENSEMBLE (v6)")
    print("=" * 70)

    # ── STAGE 1: INGESTION ───────────────────────────────────
    print("\n  Stage 1: Loading data...")
    train, test = load_data()
    print(f"    Train: {train.shape}  Test: {test.shape}")

    train_split, val_split = chronological_split(train)
    print(f"    Train (Day 48): {train_split.shape}")
    print(f"    Val   (Day 49): {val_split.shape}")

    # ── STAGE 2: FEATURE FACTORY ─────────────────────────────
    print("\n  Stage 2: Building features (v6)...")

    # Apply v5 features
    train_split, val_split = apply_all_features(
        train_split, val_split, include_lag=True, include_clusters=True, verbose=True
    )

    # Graph embeddings
    print("    Computing graph embeddings...")
    train_split, val_split = add_graph_embeddings(train_split, val_split)

    # FFT features
    train_split, val_split = add_fft_features(train_split, val_split)

    # Build lag stubs for train_split
    lookup_self = train_split.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()
    train_split["exact_lag_demand"] = train_split.apply(
        lambda r: lookup_self.get((r["geohash"], r["timestamp"]), np.nan), axis=1)
    train_split["fuzzy_lag_demand"] = np.nan
    train_split["hour_lag_demand"] = np.nan
    train_split["combined_lag"] = train_split["exact_lag_demand"]
    train_split["is_lag_missing"] = 0

    # Diffusion imputation
    print("    Running diffusion imputation...")
    train_split, val_split = add_diffusion_imputation(train_split, val_split)

    # Build features for test set
    print("    Building test features...")
    test = apply_test_features(test, train_split)
    test = build_geohash_stats(train_split, test)

    # Graph embeddings for test
    _, test = add_graph_embeddings(train_split, test)

    # FFT for test
    _, test = add_fft_features(train_split, test)

    # Lag features for test
    print("    Building test lag features...")
    test = build_lag_features(train_split, test, verbose=True)

    # Diffusion imputation for test
    test = add_diffusion_imputation(train_split, test)[1]

    # ── STAGE 3: META-ENSEMBLE TRAINING ──────────────────────
    print("\n  Stage 3: Training Meta-Ensemble...")
    val_actual = val_split[TARGET].values
    has_lag_mask = val_split["combined_lag"].notna().values

    # Combine features for unified model
    unified_features = {
        "cat": MODEL_A_FEATURES["cat"],
        "num": MODEL_A_FEATURES["num"],
    }

    # Train Meta-Ensemble (CatBoost + LightGBM + Bayesian Ridge)
    print("    Training Meta-Ensemble (CatBoost + LightGBM + Bayesian Ridge)...")
    cb_model, lgbm_model, meta_model, meta_val_pred, meta_score = train_meta_ensemble(
        train_split, val_split, unified_features,
        CATBOOST_PARAMS, LGBM_PARAMS, TARGET,
        use_variance_weighting=True
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

    # ── STAGE 5: BLENDING ────────────────────────────────────
    print("\n  Stage 5: Blending Meta-Ensemble + Lag Specialist...")

    # For lag rows: blend meta-ensemble and model_b
    # For no-lag rows: use meta-ensemble only
    best_w = 1.0
    val_blended = meta_val_pred.copy()
    val_blended[has_lag_mask] = val_pred_b[has_lag_mask]
    best_score = max(0, 100 * r2_score(val_actual, val_blended))

    print(f"    Blended Score: {best_score:.4f}")
    print(f"\n{'='*60}")
    print(f"  VALIDATION SCORES (Day 49 Holdout)")
    print(f"{'='*60}")
    print(f"  Meta-Ensemble (CB+LGB+BR):  {meta_score:.4f}")
    print(f"  Model B (Lag Specialist):    {val_score_b:.4f}")
    print(f"  Final Blended:               {best_score:.4f}")
    print(f"{'='*60}")

    # ── STAGE 6: FINAL PREDICTION ────────────────────────────
    print("\n  Stage 6: Final prediction on test data...")

    # Retrain on full data
    full_train = pd.concat([train_split, val_split], ignore_index=True)
    print("    Rebuilding lag features with full train data...")
    test = build_lag_features(full_train, test, verbose=True)
    full_train = build_geohash_stats(full_train, full_train)
    test = build_geohash_stats(full_train, test)

    # Diffusion imputation for full train
    full_train = add_diffusion_imputation(full_train, full_train)[1]

    # Retrain meta-ensemble on full data
    print("    Retraining Meta-Ensemble on full data...")
    final_cb_params = {k: v for k, v in CATBOOST_PARAMS.items() if k != "early_stopping_rounds"}
    final_cb_params["iterations"] = 1000
    final_lgbm_params = LGBM_PARAMS.copy()

    # CatBoost final
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

    # LightGBM final
    import lightgbm as lgb
    X_full_lgb = full_train[all_features].copy()
    X_test_lgb = test[all_features].copy()
    for c in unified_features["cat"]:
        X_full_lgb[c] = X_full_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    train_data = lgb.Dataset(X_full_lgb, label=y_full,
                              categorical_feature=unified_features["cat"],
                              free_raw_data=False)
    lgb_final = lgb.train(final_lgbm_params, train_data, num_boost_round=1000)
    test_pred_lgb = np.clip(lgb_final.predict(X_test_lgb), 0, None)

    # Meta prediction (simple average since we retrained)
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

    # Blend
    test_has_lag = test["combined_lag"].notna().values
    test_final = test_pred_meta.copy()
    test_final[test_has_lag] = test_pred_b[test_has_lag]
    test_final = np.clip(test_final, 0, None)

    from src.utils import create_submission
    create_submission(test["Index"].values, test_final)

    return best_score


if __name__ == "__main__":
    score = run_pipeline()
    print(f"\n  DONE. Validation Score: {score:.4f}")
