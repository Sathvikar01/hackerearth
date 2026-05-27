"""Phase 3: Formal Toroidal Traversal System.

Maps 168 temporal states onto a 16x16 toroidal grid using algebraic
traversal. Extracts phase, neighborhood entropy, and collision frequency
features. Trains the final model and generates submission.csv.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data_loader import load_data
from src.feature_engineering import (
    engineer_features_phase1,
    engineer_features_phase2,
    engineer_features_phase3,
)
from src.toroidal import ToroidalTraversalGenerator
from src.model import train_model_phase1, get_feature_importance
from src.utils import print_phase_score, create_submission
from src.config import CAT_FEATURES, SUBMISSION_PATH, TARGET


PHASE = 3


def run_phase3():
    print("=" * 60)
    print("PHASE 3: FORMAL TOROIDAL TRAVERSAL SYSTEM")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    train, test = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")

    # Phase 1 + Phase 2 features
    print("\n[2/6] Engineering Phase 1 + Phase 2 features...")
    train, test = engineer_features_phase1(train, test)
    train, test = engineer_features_phase2(train, test)

    # Initialize toroidal generator
    print("\n[3/6] Initializing ToroidalTraversalGenerator (N=16)...")
    toroidal_gen = ToroidalTraversalGenerator(n=16)
    summary = toroidal_gen.get_grid_summary()
    print(f"  Grid size: {summary['grid_size']}")
    print(f"  Real states placed: {summary['real_states_placed']}")
    print(f"  Phantom states: {summary['phantom_states']}")
    print(f"  Total collision events: {summary['total_collision_events']}")
    print(f"  Collision rate: {summary['collision_rate']:.4f}")

    # Phase 3 features
    print("\n[4/6] Adding Phase 3 toroidal traversal features...")
    train, test = engineer_features_phase3(train, test, toroidal_gen)

    # Build feature lists
    cat_cols = list(CAT_FEATURES)
    num_cols = [
        "NumberofLanes", "Temperature", "hour", "minute",
        "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "RoadType_x_Lanes", "Weather_x_Temp",
        "geohash_target_mean", "geohash_target_var",
        "geohash_prefix_4_target_mean", "geohash_prefix_4_target_var",
        "toroidal_dist_rush_hour", "toroidal_dist_weekly_peak",
        "toroidal_phase", "toroidal_neighborhood_entropy", "toroidal_collision_frequency",
    ]
    features = cat_cols + num_cols
    print(f"  Total features: {len(features)}")

    # Train model with stronger regularization
    print("\n[5/6] Training CatBoost with Phase 3 features...")
    models, oof_preds, mean_score, fold_scores = train_model_phase1(
        train, features, cat_cols, target=TARGET, n_splits=5,
    )

    # Report
    print_phase_score(PHASE, mean_score, fold_scores)

    # Check toroidal feature importance
    imp = get_feature_importance(models[-1], features)
    toroidal_feats = [
        "toroidal_phase", "toroidal_neighborhood_entropy",
        "toroidal_collision_frequency", "toroidal_dist_rush_hour",
        "toroidal_dist_weekly_peak",
    ]
    toroidal_imp = imp[imp["feature"].isin(toroidal_feats)]["importance"].sum()
    total_imp = imp["importance"].sum()
    toroidal_pct = (toroidal_imp / total_imp) * 100
    print(f"\n  Toroidal feature importance: {toroidal_pct:.1f}% of total")
    print("\n  Top 15 features:")
    print(imp.head(15).to_string(index=False))

    # Generate submission
    print(f"\n[6/6] Generating submission...")
    from catboost import Pool
    cat_indices = [features.index(c) for c in cat_cols if c in features]
    X_test = test[features].copy()
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)
    test_pool = Pool(X_test, cat_features=cat_indices)

    # Average predictions across all folds
    test_preds = np.zeros(len(test))
    for m in models:
        test_preds += m.predict(test_pool)
    test_preds /= len(models)
    test_preds = np.clip(test_preds, 0, None)

    submission = create_submission(test["Index"].values, test_preds, SUBMISSION_PATH)

    return mean_score, submission


if __name__ == "__main__":
    score, submission = run_phase3()
    print(f"\nPhase 3 complete. Competition Score: {score:.4f}")
    print(f"Submission saved with {len(submission)} rows.")
