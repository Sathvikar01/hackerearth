"""Phase 1: Robust Baseline (Local Spatial Branch).

Trains CatBoostRegressor with Phase 1 features only.
Prints validation R² scores for ablation testing.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, get_feature_columns
from src.feature_engineering import engineer_features_phase1
from src.model import train_model_phase1, get_feature_importance
from src.utils import print_phase_score, compute_score
from src.config import CAT_FEATURES

PHASE = 1


def run_phase1():
    print("=" * 60)
    print("PHASE 1: ROBUST BASELINE (LOCAL SPATIAL BRANCH)")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    train, test = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")

    # Feature engineering
    print("\n[2/4] Engineering Phase 1 features...")
    train, test = engineer_features_phase1(train, test)

    # Build feature lists
    cat_cols = list(CAT_FEATURES)
    num_cols = [
        "NumberofLanes", "Temperature", "hour", "minute",
        "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "RoadType_x_Lanes", "Weather_x_Temp",
        "geohash_target_mean", "geohash_target_var",
        "geohash_prefix_4_target_mean", "geohash_prefix_4_target_var",
    ]
    features = cat_cols + num_cols
    print(f"  Total features: {len(features)} ({len(cat_cols)} cat + {len(num_cols)} num)")

    # Train model
    print("\n[3/4] Training CatBoost with 5-Fold GroupKFold...")
    models, oof_preds, mean_score, fold_scores = train_model_phase1(
        train, features, cat_cols, target="demand", n_splits=5,
    )

    # Report
    print_phase_score(PHASE, mean_score, fold_scores)

    # Feature importance (from last fold)
    print("[4/4] Feature importance (last fold):")
    imp = get_feature_importance(models[-1], features)
    print(imp.head(15).to_string(index=False))

    return mean_score, models, train, test, features, cat_cols


if __name__ == "__main__":
    score, models, train, test, features, cat_cols = run_phase1()
    print(f"\nPhase 1 complete. Competition Score: {score:.4f}")
