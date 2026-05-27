"""Full pipeline orchestrator for Traffic Demand Prediction.

Runs all 3 phases sequentially:
  Phase 1: Robust baseline with local spatial features
  Phase 2: Add continuous toroidal signal features
  Phase 3: Add formal toroidal traversal features + final submission
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.data_loader import load_data
from src.feature_engineering import (
    engineer_features_phase1,
    engineer_features_phase2,
    engineer_features_phase3,
)
from src.model import train_model_phase1, predict_test, get_feature_importance
from src.toroidal import ToroidalTraversalGenerator
from src.utils import print_phase_score, create_submission
from src.config import CAT_FEATURES, SUBMISSION_PATH, TARGET


def run_full_pipeline():
    print("=" * 70)
    print("TRAFFIC DEMAND PREDICTION — FULL 3-PHASE PIPELINE")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train, test = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")

    results = {}

    # ── PHASE 1 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: ROBUST BASELINE (LOCAL SPATIAL BRANCH)")
    print("=" * 70)

    train_p1, test_p1 = engineer_features_phase1(train.copy(), test.copy())

    cat_cols = list(CAT_FEATURES)
    num_cols_p1 = [
        "NumberofLanes", "Temperature", "hour", "minute",
        "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "RoadType_x_Lanes", "Weather_x_Temp",
        "geohash_target_mean", "geohash_target_var",
        "geohash_prefix_4_target_mean", "geohash_prefix_4_target_var",
    ]
    features_p1 = cat_cols + num_cols_p1

    models_p1, oof_p1, score_p1, folds_p1 = train_model_phase1(
        train_p1, features_p1, cat_cols, target=TARGET, n_splits=5,
    )
    print_phase_score(1, score_p1, folds_p1)
    results["phase1"] = score_p1

    # ── PHASE 2 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: CONTINUOUS TOROIDAL SIGNALS")
    print("=" * 70)

    train_p2, test_p2 = engineer_features_phase2(train_p1.copy(), test_p1.copy())

    num_cols_p2 = num_cols_p1 + ["toroidal_dist_rush_hour", "toroidal_dist_weekly_peak"]
    features_p2 = cat_cols + num_cols_p2

    models_p2, oof_p2, score_p2, folds_p2 = train_model_phase1(
        train_p2, features_p2, cat_cols, target=TARGET, n_splits=5,
    )
    print_phase_score(2, score_p2, folds_p2)
    results["phase2"] = score_p2

    improvement_p2 = score_p2 - score_p1
    print(f"  Phase 2 improvement over Phase 1: {improvement_p2:+.4f}")

    # ── PHASE 3 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: FORMAL TOROIDAL TRAVERSAL SYSTEM")
    print("=" * 70)

    print("\n  Initializing ToroidalTraversalGenerator (N=16)...")
    toroidal_gen = ToroidalTraversalGenerator(n=16)
    summary = toroidal_gen.get_grid_summary()
    print(f"  Grid summary: {summary}")

    train_p3, test_p3 = engineer_features_phase3(
        train_p2.copy(), test_p2.copy(), toroidal_gen,
    )

    num_cols_p3 = num_cols_p2 + ["toroidal_phase", "toroidal_neighborhood_entropy", "toroidal_collision_frequency"]
    features_p3 = cat_cols + num_cols_p3

    # Phase 3 CatBoost with stronger regularization to control toroidal feature importance
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import GroupKFold
    from src.config import CATBOOST_PARAMS

    X = train_p3[features_p3].copy()
    y = train_p3[TARGET].values
    groups = train_p3["geohash"].values

    cat_indices = [features_p3.index(c) for c in cat_cols if c in features_p3]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    gkf = GroupKFold(n_splits=5)
    models_p3 = []
    oof_p3 = np.zeros(len(train_p3))
    fold_scores_p3 = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"  Fold {fold + 1}/5...", end=" ")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)

        params = dict(CATBOOST_PARAMS)
        # Stronger regularization for Phase 3
        params["l2_leaf_reg"] = 10
        params["random_strength"] = 2.0

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        val_pred = model.predict(X_val)
        oof_p3[val_idx] = val_pred

        fold_score = max(0, 100 * (1 - np.sum((y_val - val_pred)**2) / np.sum((y_val - np.mean(y_val))**2)))
        fold_scores_p3.append(fold_score)
        print(f"Score: {fold_score:.4f}")
        models_p3.append(model)

    score_p3 = max(0, 100 * (1 - np.sum((y - oof_p3)**2) / np.sum((y - np.mean(y))**2)))
    print_phase_score(3, score_p3, fold_scores_p3)
    results["phase3"] = score_p3

    improvement_p3 = score_p3 - score_p2
    print(f"  Phase 3 improvement over Phase 2: {improvement_p3:+.4f}")

    # Check toroidal feature importance
    imp = get_feature_importance(models_p3[-1], features_p3)
    toroidal_features = ["toroidal_phase", "toroidal_neighborhood_entropy",
                         "toroidal_collision_frequency", "toroidal_dist_rush_hour",
                         "toroidal_dist_weekly_peak"]
    toroidal_imp = imp[imp["feature"].isin(toroidal_features)]["importance"].sum()
    total_imp = imp["importance"].sum()
    toroidal_pct = (toroidal_imp / total_imp) * 100
    print(f"\n  Toroidal feature importance: {toroidal_pct:.1f}% of total")
    print("\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    # Generate test predictions using best model (Phase 3)
    print("\n  Generating test predictions...")
    X_test = test_p3[features_p3].copy()
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
    test_pool = Pool(X_test, cat_features=cat_indices)

    # Average predictions across all folds
    test_preds = np.zeros(len(test_p3))
    for m in models_p3:
        test_preds += m.predict(test_pool)
    test_preds /= len(models_p3)
    test_preds = np.clip(test_preds, 0, None)

    create_submission(test_p3["Index"].values, test_preds, SUBMISSION_PATH)

    # ── SUMMARY ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    for phase, score in results.items():
        print(f"  {phase}: {score:.4f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_full_pipeline()
