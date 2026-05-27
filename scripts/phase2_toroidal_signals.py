"""Phase 2: Continuous Toroidal Signals.

Adds behavior-driven temporal distance metrics to Phase 1 features.
Prints validation R² improvement over Phase 1.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.feature_engineering import (
    engineer_features_phase1,
    engineer_features_phase2,
)
from src.model import train_model_phase1, get_feature_importance
from src.utils import print_phase_score
from src.config import CAT_FEATURES


PHASE = 2


def run_phase2():
    print("=" * 60)
    print("PHASE 2: CONTINUOUS TOROIDAL SIGNALS")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    train, test = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")

    # Phase 1 features first
    print("\n[2/5] Engineering Phase 1 features...")
    train, test = engineer_features_phase1(train, test)

    # Phase 2 features
    print("\n[3/5] Adding Phase 2 toroidal signal features...")
    train, test = engineer_features_phase2(train, test)

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
    ]
    features = cat_cols + num_cols
    print(f"  Total features: {len(features)}")

    # Train model
    print("\n[4/5] Training CatBoost with Phase 2 features...")
    models, oof_preds, mean_score, fold_scores = train_model_phase1(
        train, features, cat_cols, target="demand", n_splits=5,
    )

    # Report
    print_phase_score(PHASE, mean_score, fold_scores)

    # Feature importance
    print("[5/5] Feature importance (last fold):")
    imp = get_feature_importance(models[-1], features)
    print(imp.head(15).to_string(index=False))

    return mean_score, models, train, test, features, cat_cols


if __name__ == "__main__":
    score, models, train, test, features, cat_cols = run_phase2()
    print(f"\nPhase 2 complete. Competition Score: {score:.4f}")
