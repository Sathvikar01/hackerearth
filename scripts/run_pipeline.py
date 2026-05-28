"""Full 6-Stage Dual-Branch Pipeline (v4 — 4 Upgrades).

Upgrades:
1. Geohash demand statistics (mean, std, median, count)
2. Fuzzy lag (+/- 30min rolling window)
3. Missingness indicator (is_lag_missing)
4. Model tuning + day_of_week categorical
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
    add_temporal_features, add_spatial_features, add_contextual_features,
    add_combined_target_features,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
)
from src.target_encoder import BayesianTargetEncoder
from src.utils import print_scores, create_submission
from src.config import TARGET, SEED

import warnings
warnings.filterwarnings("ignore")


# Upgraded CatBoost params for Model A
MODEL_A_PARAMS = {
    "iterations": 2500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 200,
    "loss_function": "RMSE",
}

MODEL_B_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 50,
    "loss_function": "RMSE",
}


def apply_features(df):
    """Apply all feature engineering in-place."""
    add_temporal_features(df)
    add_spatial_features(df)
    add_contextual_features(df)
    add_combined_target_features(df)


def build_geohash_stats(train_split, val_or_test):
    """Upgrade 1: Geohash demand statistics from Day 48."""
    stats = train_split.groupby("geohash")["demand"].agg(
        ["mean", "std", "median", "count"]
    ).reset_index()
    stats.columns = ["geohash", "geo_demand_mean", "geo_demand_std",
                     "geo_demand_median", "geo_demand_count"]

    global_mean = train_split["demand"].mean()
    global_std = train_split["demand"].std()
    global_median = train_split["demand"].median()

    val_or_test = val_or_test.merge(stats, on="geohash", how="left")
    val_or_test["geo_demand_mean"] = val_or_test["geo_demand_mean"].fillna(global_mean)
    val_or_test["geo_demand_std"] = val_or_test["geo_demand_std"].fillna(global_std)
    val_or_test["geo_demand_median"] = val_or_test["geo_demand_median"].fillna(global_median)
    val_or_test["geo_demand_count"] = val_or_test["geo_demand_count"].fillna(0)

    return val_or_test


def build_lag_features(train_split, val_or_test, verbose=True):
    """Build primary (exact), fuzzy (+/-30min), and secondary (hour) lag features."""
    # Convert timestamp to minutes for fuzzy matching
    def ts_to_min(ts):
        h, m = ts.split(":")
        return int(h) * 60 + int(m)

    train_split = train_split.copy()
    train_split["_minutes"] = train_split["timestamp"].apply(ts_to_min)
    val_or_test = val_or_test.copy()
    val_or_test["_minutes"] = val_or_test["timestamp"].apply(ts_to_min)

    # Primary lag: exact (geohash, timestamp)
    lookup_exact = train_split.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()
    val_or_test["exact_lag_demand"] = val_or_test.apply(
        lambda r: lookup_exact.get((r["geohash"], r["timestamp"]), np.nan), axis=1)

    # Upgrade 2: Fuzzy lag (+/- 30 minutes)
    # For each row, find Day 48 rows with same geohash within +/- 30 min window
    fuzzy_lookup = {}
    for gh in train_split["geohash"].unique():
        gh_data = train_split[train_split["geohash"] == gh]
        for _, row in gh_data.iterrows():
            key = (gh, row["_minutes"])
            if key not in fuzzy_lookup:
                fuzzy_lookup[key] = []
            fuzzy_lookup[key].append(row["demand"])

    # Build fuzzy lag: average demand within +/- 30 min window
    fuzzy_results = []
    for _, row in val_or_test.iterrows():
        gh = row["geohash"]
        target_min = row["_minutes"]
        window_demands = []
        for offset in range(-30, 31):
            key = (gh, target_min + offset)
            if key in fuzzy_lookup:
                window_demands.extend(fuzzy_lookup[key])
        if window_demands:
            fuzzy_results.append(np.mean(window_demands))
        else:
            fuzzy_results.append(np.nan)
    val_or_test["fuzzy_lag_demand"] = fuzzy_results

    # Secondary fallback: (geohash, hour) average
    lookup_hour = train_split.groupby(["geohash", "hour"])["demand"].mean().to_dict()
    val_or_test["hour_lag_demand"] = val_or_test.apply(
        lambda r: lookup_hour.get((r["geohash"], r["hour"]), np.nan), axis=1)

    # Combined lag cascade: exact -> fuzzy -> hour
    val_or_test["combined_lag"] = val_or_test["exact_lag_demand"].fillna(
        val_or_test["fuzzy_lag_demand"]
    ).fillna(val_or_test["hour_lag_demand"])

    # Upgrade 3: Missingness indicator
    val_or_test["is_lag_missing"] = val_or_test["exact_lag_demand"].isna().astype(int)

    # Cleanup
    val_or_test.drop(columns=["_minutes"], inplace=True, errors="ignore")

    if verbose:
        exact_cov = val_or_test["exact_lag_demand"].notna().sum()
        fuzzy_cov = val_or_test["fuzzy_lag_demand"].notna().sum()
        combined_cov = val_or_test["combined_lag"].notna().sum()
        total = len(val_or_test)
        print(f"    Exact lag:    {exact_cov}/{total} ({exact_cov/total*100:.1f}%)")
        print(f"    Fuzzy lag:    {fuzzy_cov}/{total} ({fuzzy_cov/total*100:.1f}%)")
        print(f"    Combined:     {combined_cov}/{total} ({combined_cov/total*100:.1f}%)")
        print(f"    Missing flag: {val_or_test['is_lag_missing'].sum()} rows marked missing")

    return val_or_test


def train_model_a(train_df, val_df, features, cat_cols, params, target="demand"):
    """Train Model A with early stopping on val set."""
    all_features = features["cat"] + features["num"]
    X_train = train_df[all_features].copy()
    y_train = train_df[target].values
    X_val = val_df[all_features].copy()

    has_target = target in val_df.columns

    for c in features["cat"]:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)

    cat_indices = [all_features.index(c) for c in features["cat"] if c in all_features]
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)

    if has_target:
        y_val = val_df[target].values
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        val_pred = np.clip(model.predict(val_pool), 0, None)
        val_score = max(0, 100 * r2_score(y_val, val_pred))
    else:
        final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        model = CatBoostRegressor(**final_params)
        model.fit(train_pool)
        val_pool = Pool(X_val, cat_features=cat_indices)
        val_pred = np.clip(model.predict(val_pool), 0, None)
        val_score = 0.0

    return model, val_pred, val_score


def predict_model(model, df, features):
    """Generate predictions."""
    all_features = features["cat"] + features["num"]
    X = df[all_features].copy()
    for c in features["cat"]:
        X[c] = X[c].astype(str)
    cat_indices = [all_features.index(c) for c in features["cat"] if c in all_features]
    pool = Pool(X, cat_features=cat_indices)
    return np.clip(model.predict(pool), 0, None)


def run_pipeline():
    print("=" * 70)
    print("  DUAL-BRANCH ARCHITECTURE (v4 — 4 UPGRADES)")
    print("=" * 70)

    # ── STAGE 1: INGESTION ───────────────────────────────────
    print("\n  Stage 1: Loading data...")
    train, test = load_data()
    print(f"    Train: {train.shape}  Test: {test.shape}")

    train_split, val_split = chronological_split(train)
    print(f"    Train (Day 48): {train_split.shape}")
    print(f"    Val   (Day 49): {val_split.shape}")

    # ── STAGE 2: FEATURE FACTORY ─────────────────────────────
    print("\n  Stage 2: Building features...")

    for df in [train_split, val_split, test]:
        apply_features(df)

    # Upgrade 1: Geohash demand statistics
    print("    Adding geohash demand statistics...")
    train_split = build_geohash_stats(train_split, train_split)
    val_split = build_geohash_stats(train_split, val_split)
    test = build_geohash_stats(train_split, test)

    # Lag features with fuzzy + missingness indicator
    print("    Building lag features (exact + fuzzy + hour)...")
    # For train_split, build lag from self (no leakage since it's the training data)
    lookup_self = train_split.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()
    train_split["exact_lag_demand"] = train_split["geohash"].map(
        lambda g: np.nan)  # Not used for training, just need column to exist
    train_split["exact_lag_demand"] = train_split.apply(
        lambda r: lookup_self.get((r["geohash"], r["timestamp"]), np.nan), axis=1)
    train_split["fuzzy_lag_demand"] = np.nan
    train_split["hour_lag_demand"] = np.nan
    train_split["combined_lag"] = train_split["exact_lag_demand"]
    train_split["is_lag_missing"] = 0

    print("    Validation set:")
    val_split = build_lag_features(train_split, val_split)
    print("    Test set:")
    test = build_lag_features(train_split, test)

    # ── STAGE 3: TARGET ENCODING ─────────────────────────────
    print("\n  Stage 3: Leakage-safe Target Encoding (fit on Day 48 only)...")
    te_columns = ["geohash", "geo_slot", "geo_p4_hour"]
    encoder = BayesianTargetEncoder(columns=te_columns, target=TARGET, m=10)
    encoder.fit(train_split)

    train_split = encoder.transform(train_split)
    val_split = encoder.transform(val_split)
    test = encoder.transform(test)

    te_num_cols = [f"{c}_te" for c in te_columns]

    # Upgrade 4: Move day_of_week from num to cat, plus new numeric features
    model_a_cat = [c for c in MODEL_A_FEATURES["cat"]] + ["day_of_week"]
    model_a_num = [n for n in MODEL_A_FEATURES["num"] if n != "day_of_week"] + te_num_cols + [
        "geo_demand_mean", "geo_demand_std", "geo_demand_median", "geo_demand_count",
        "is_lag_missing",
    ]
    model_a_features = {
        "cat": model_a_cat,
        "num": model_a_num,
    }

    # ── STAGE 4: DUAL-MODEL TRAINING ─────────────────────────
    print("\n  Stage 4: Training dual models...")
    val_actual = val_split[TARGET].values
    has_lag_mask = val_split["combined_lag"].notna().values

    # Model A: Global Learner (upgraded)
    print("    Training Model A (Global Learner, upgraded params)...")
    model_a, val_pred_a, val_score_a = train_model_a(
        train_split, val_split, model_a_features, model_a_features["cat"],
        MODEL_A_PARAMS, TARGET
    )
    print(f"    Model A Val Score: {val_score_a:.4f}")

    # Model B: Lag Specialist
    print("    Training Model B (Lag Specialist)...")
    val_lag_rows = val_split[has_lag_mask].copy().reset_index(drop=True)

    if len(val_lag_rows) > 100:
        cat_cols_b = MODEL_B_FEATURES["cat"]
        num_cols_b = MODEL_B_FEATURES["num"]
        all_feat_b = cat_cols_b + num_cols_b

        X_b = val_lag_rows[all_feat_b].copy()
        y_b = val_lag_rows[TARGET].values
        for c in cat_cols_b:
            X_b[c] = X_b[c].astype(str)
        cat_idx_b = [all_feat_b.index(c) for c in cat_cols_b]

        pool_b = Pool(X_b, y_b, cat_features=cat_idx_b)
        model_b = CatBoostRegressor(**MODEL_B_PARAMS)
        model_b.fit(pool_b)

        # Predict on ALL val rows
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
        cat_cols_b = MODEL_B_FEATURES["cat"]
        all_feat_b = cat_cols_b + MODEL_B_FEATURES["num"]
        cat_idx_b = [all_feat_b.index(c) for c in cat_cols_b]
        print("    Model B: Not enough lag rows, using lag directly")

    # ── STAGE 5: BLENDING (W=1.0) ────────────────────────────
    print("\n  Stage 5: Blending (W=1.0 for lag rows)...")

    best_w = 1.0
    val_blended = val_pred_a.copy()
    val_blended[has_lag_mask] = val_pred_b[has_lag_mask]
    best_score = max(0, 100 * r2_score(val_actual, val_blended))

    print(f"    W=1.0 Score: {best_score:.4f}")
    print_scores(val_score_a, val_score_b, best_score)

    # ── STAGE 6: FINAL PREDICTION ────────────────────────────
    print("\n  Stage 6: Final prediction on test data...")

    # Retrain encoder on ALL train data
    full_train = pd.concat([train_split, val_split], ignore_index=True)
    encoder_final = BayesianTargetEncoder(columns=te_columns, target=TARGET, m=10)
    encoder_final.fit(full_train)
    full_train = encoder_final.transform(full_train)
    test = encoder_final.transform(test)

    # Rebuild lag features for test using full train data
    print("    Rebuilding lag features with full train data...")
    test = build_lag_features(full_train, test, verbose=True)

    # Rebuild geohash stats with full train data
    test = test.drop(columns=["geo_demand_mean", "geo_demand_std",
                               "geo_demand_median", "geo_demand_count"], errors="ignore")
    test = build_geohash_stats(full_train, test)

    # Retrain Model A on full data
    print("    Retraining Model A on full train data...")
    model_a_final_params = {k: v for k, v in MODEL_A_PARAMS.items() if k != "early_stopping_rounds"}
    model_a_final, _, _ = train_model_a(
        full_train, test, model_a_features, model_a_features["cat"],
        model_a_final_params, TARGET
    )

    # Retrain Model B on all lag-available rows
    print("    Retraining Model B on lag-available rows...")
    full_lag_mask = full_train["combined_lag"].notna()
    full_lag_rows = full_train[full_lag_mask].copy().reset_index(drop=True)

    if len(full_lag_rows) > 100 and model_b is not None:
        X_b_full = full_lag_rows[all_feat_b].copy()
        y_b_full = full_lag_rows[TARGET].values
        for c in cat_cols_b:
            X_b_full[c] = X_b_full[c].astype(str)
        pool_b_full = Pool(X_b_full, y_b_full, cat_features=cat_idx_b)
        model_b_final = CatBoostRegressor(**{k: v for k, v in MODEL_B_PARAMS.items() if k != "early_stopping_rounds"})
        model_b_final.fit(pool_b_full)

        test_pred_a = predict_model(model_a_final, test, model_a_features)

        X_b_test = test[all_feat_b].copy()
        for c in cat_cols_b:
            X_b_test[c] = X_b_test[c].astype(str)
        pool_b_test = Pool(X_b_test, cat_features=cat_idx_b)
        test_pred_b = np.clip(model_b_final.predict(pool_b_test), 0, None)
    else:
        test_pred_a = predict_model(model_a_final, test, model_a_features)
        test_pred_b = test["combined_lag"].fillna(0).values

    # Blend
    test_has_lag = test["combined_lag"].notna().values
    test_final = test_pred_a.copy()
    test_final[test_has_lag] = best_w * test_pred_b[test_has_lag] + (1 - best_w) * test_pred_a[test_has_lag]
    test_final = np.clip(test_final, 0, None)

    create_submission(test["Index"].values, test_final)

    return best_score


if __name__ == "__main__":
    score = run_pipeline()
    print(f"\n  DONE. Validation Score: {score:.4f}")
