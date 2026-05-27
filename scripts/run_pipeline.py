"""Full 6-Stage Dual-Branch Pipeline.

Stage 1-2: Feature Factory (temporal, spatial, contextual, golden lag)
Stage 3: Leakage-safe Target Encoding (manual OOF with Bayesian smoothing)
Stage 4: Dual-Model Training (Model A: Global, Model B: Lag Specialist)
Stage 5: Dynamic Blending (optimize W on Day 49 validation)
Stage 6: Final prediction and submission
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.data_loader import load_data, chronological_split
from src.features import (
    add_temporal_features, add_spatial_features, add_contextual_features,
    add_combined_target_features,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
)
from src.target_encoder import BayesianTargetEncoder
from src.models import train_model_a, predict_model_a
from src.blending import optimize_blend_weight, blend_predictions
from src.utils import print_scores, create_submission
from src.config import TARGET

import warnings
warnings.filterwarnings("ignore")


def apply_features(df):
    """Apply all feature engineering in-place."""
    add_temporal_features(df)
    add_spatial_features(df)
    add_contextual_features(df)
    add_combined_target_features(df)


def run_pipeline():
    print("=" * 70)
    print("  DUAL-BRANCH SPATIO-TEMPORAL LAG ARCHITECTURE")
    print("=" * 70)

    # ── STAGE 1: INGESTION ───────────────────────────────────
    print("\n  Stage 1: Loading data...")
    train, test = load_data()
    print(f"    Train: {train.shape}  Test: {test.shape}")

    # Chronological split: Day 48 = train, Day 49 = validation
    train_split, val_split = chronological_split(train)
    print(f"    Train (Day 48): {train_split.shape}")
    print(f"    Val   (Day 49): {val_split.shape}")

    # ── STAGE 2: FEATURE FACTORY ─────────────────────────────
    print("\n  Stage 2: Building features...")

    for df in [train_split, val_split, test]:
        apply_features(df)

    # Golden Lag: map Day 48 demand → val/test via (geohash, timestamp)
    print("    Adding golden lag feature...")
    lookup_d48 = train_split.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()

    val_split["exact_lag_demand"] = val_split.apply(
        lambda r: lookup_d48.get((r["geohash"], r["timestamp"]), np.nan), axis=1)
    test["exact_lag_demand"] = test.apply(
        lambda r: lookup_d48.get((r["geohash"], r["timestamp"]), np.nan), axis=1)

    val_lag = val_split["exact_lag_demand"].notna().sum()
    test_lag = test["exact_lag_demand"].notna().sum()
    print(f"    Val  lag coverage: {val_lag}/{len(val_split)} ({val_lag/len(val_split)*100:.1f}%)")
    print(f"    Test lag coverage: {test_lag}/{len(test)} ({test_lag/len(test)*100:.1f}%)")

    # ── STAGE 3: TARGET ENCODING ─────────────────────────────
    print("\n  Stage 3: Leakage-safe Target Encoding (fit on Day 48 only)...")

    te_columns = ["geohash", "geo_slot", "geo_p4_hour"]
    encoder = BayesianTargetEncoder(columns=te_columns, target=TARGET, m=10)
    encoder.fit(train_split)

    train_split = encoder.transform(train_split)
    val_split = encoder.transform(val_split)
    test = encoder.transform(test)

    te_num_cols = [f"{c}_te" for c in te_columns]
    model_a_features = {
        "cat": MODEL_A_FEATURES["cat"],
        "num": MODEL_A_FEATURES["num"] + te_num_cols,
    }

    # ── STAGE 4: DUAL-MODEL TRAINING ─────────────────────────
    print("\n  Stage 4: Training dual models...")
    val_actual = val_split[TARGET].values
    has_lag_mask = val_split["exact_lag_demand"].notna().values

    # Model A: Global Learner — train on Day 48, predict Day 49
    print("    Training Model A (Global Learner on Day 48)...")
    model_a, val_pred_a, val_score_a = train_model_a(
        train_split, val_split, model_a_features, TARGET
    )
    print(f"    Model A Val Score: {val_score_a:.4f}")

    # Model B: Lag Specialist — train on Day 49 rows WITH lag, predict Day 49
    # This exploits the near-deterministic Day 48 → Day 49 continuity
    print("    Training Model B (Lag Specialist on Day 49 lag rows)...")
    val_lag_rows = val_split[has_lag_mask].copy().reset_index(drop=True)

    if len(val_lag_rows) > 100:
        from catboost import CatBoostRegressor, Pool
        from src.config import CATBOOST_PARAMS

        cat_cols_b = MODEL_B_FEATURES["cat"]
        num_cols_b = MODEL_B_FEATURES["num"]
        all_feat_b = cat_cols_b + num_cols_b

        X_b = val_lag_rows[all_feat_b].copy()
        y_b = val_lag_rows[TARGET].values
        for c in cat_cols_b:
            X_b[c] = X_b[c].astype(str)
        cat_idx_b = [all_feat_b.index(c) for c in cat_cols_b]

        pool_b = Pool(X_b, y_b, cat_features=cat_idx_b)
        model_b = CatBoostRegressor(**CATBOOST_PARAMS)
        model_b.fit(pool_b)

        # Predict on ALL val rows
        val_pred_b = np.zeros(len(val_split))
        X_b_all = val_split[all_feat_b].copy()
        for c in cat_cols_b:
            X_b_all[c] = X_b_all[c].astype(str)
        pool_b_all = Pool(X_b_all, cat_features=cat_idx_b)
        val_pred_b_raw = model_b.predict(pool_b_all)
        val_pred_b_raw = np.clip(val_pred_b_raw, 0, None)
        val_pred_b = val_pred_b_raw

        # Score Model B on lag rows only
        val_score_b = max(0, 100 * r2_score(
            val_split.loc[has_lag_mask, TARGET].values,
            val_pred_b[has_lag_mask]
        ))
        print(f"    Model B Val Score (lag rows only): {val_score_b:.4f}")
    else:
        val_pred_b = val_split["exact_lag_demand"].fillna(0).values
        val_score_b = 0.0
        model_b = None
        print("    Model B: Not enough lag rows, using lag directly")

    # ── STAGE 5: DYNAMIC BLENDING ────────────────────────────
    print("\n  Stage 5: Optimizing blend weight on Day 49 validation...")

    best_w, best_score, val_blended = optimize_blend_weight(
        val_actual, val_pred_a, val_pred_b, has_lag_mask
    )

    print(f"    Optimal W: {best_w:.2f}")
    print_scores(val_score_a, val_score_b, best_score)

    # ── STAGE 6: FINAL PREDICTION ────────────────────────────
    print("\n  Stage 6: Final prediction on test data...")

    # Retrain encoder on ALL train data
    full_train = pd.concat([train_split, val_split], ignore_index=True)
    encoder_final = BayesianTargetEncoder(columns=te_columns, target=TARGET, m=10)
    encoder_final.fit(full_train)
    full_train = encoder_final.transform(full_train)
    test = encoder_final.transform(test)

    # Retrain Model A on all train data
    print("    Retraining Model A on full train data...")
    model_a_final, _, _ = train_model_a(full_train, test, model_a_features, TARGET)

    # Model B: retrain on all rows with lag (Day 49 rows)
    print("    Retraining Model B on lag-available rows...")
    full_lag_mask = full_train["exact_lag_demand"].notna()
    full_lag_rows = full_train[full_lag_mask].copy().reset_index(drop=True)

    if len(full_lag_rows) > 100 and model_b is not None:
        X_b_full = full_lag_rows[all_feat_b].copy()
        y_b_full = full_lag_rows[TARGET].values
        for c in cat_cols_b:
            X_b_full[c] = X_b_full[c].astype(str)
        pool_b_full = Pool(X_b_full, y_b_full, cat_features=cat_idx_b)
        model_b_final = CatBoostRegressor(**CATBOOST_PARAMS)
        model_b_final.fit(pool_b_full)

        # Predict test
        test_pred_a = predict_model_a(model_a_final, test, model_a_features)

        X_b_test = test[all_feat_b].copy()
        for c in cat_cols_b:
            X_b_test[c] = X_b_test[c].astype(str)
        pool_b_test = Pool(X_b_test, cat_features=cat_idx_b)
        test_pred_b = model_b_final.predict(pool_b_test)
        test_pred_b = np.clip(test_pred_b, 0, None)
    else:
        test_pred_a = predict_model_a(model_a_final, test, model_a_features)
        test_pred_b = test["exact_lag_demand"].fillna(0).values

    # Blend
    test_has_lag = test["exact_lag_demand"].notna().values
    test_final = blend_predictions(test_pred_a, test_pred_b, test_has_lag, best_w)
    test_final = np.clip(test_final, 0, None)

    create_submission(test["Index"].values, test_final)

    return best_score


if __name__ == "__main__":
    score = run_pipeline()
    print(f"\n  DONE. Validation Score: {score:.4f}")
