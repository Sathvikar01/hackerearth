"""Full pipeline with rollback logic.

Phase 1: Bare-bones baseline (geohash + basic features)
Phase 2: Exploits (2A lat/lon, 2B OOF lookup, 2C lag) with rollback
Phase 3: Toroidal features with rollback
Final: submission.csv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pygeohash
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from src.toroidal import ToroidalTraversalGenerator
from src.config import SEED, TARGET, SUBMISSION_PATH


# ── Helpers ──────────────────────────────────────────────────
def load_data():
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    for df in (train, test):
        df["hour"] = df["timestamp"].apply(lambda x: int(x.split(":")[0]))
        df["minute"] = df["timestamp"].apply(lambda x: int(x.split(":")[1]))
        df["day_of_week"] = df["day"] % 7
        df["RoadType"] = df["RoadType"].fillna("Unknown")
        df["Weather"] = df["Weather"].fillna("Unknown")
        df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())
    return train, test


def vectorized_lookup(target_df, lookup_df, global_mean):
    stats = lookup_df.groupby(["geohash", "timestamp"])["demand"].mean().reset_index()
    stats.columns = ["geohash", "timestamp", "_v"]
    m = target_df[["geohash", "timestamp"]].merge(stats, on=["geohash", "timestamp"], how="left")
    result = m["_v"].values
    nan_mask = np.isnan(result)
    if nan_mask.any():
        s2 = lookup_df.groupby(["geohash", "hour"])["demand"].mean().reset_index()
        s2.columns = ["geohash", "hour", "_v2"]
        m2 = target_df.loc[nan_mask, ["geohash", "hour"]].merge(s2, on=["geohash", "hour"], how="left")
        result[nan_mask] = m2["_v2"].values
    nan_mask = np.isnan(result)
    if nan_mask.any():
        s3 = lookup_df.groupby("geohash")["demand"].mean().reset_index()
        s3.columns = ["geohash", "_v3"]
        m3 = target_df.loc[nan_mask, ["geohash"]].merge(s3, on="geohash", how="left")
        result[nan_mask] = m3["_v3"].values
    return np.nan_to_num(result, nan=global_mean)


def train_evaluate(X, y, cat_indices, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    fold_scores = []
    models = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        tp = Pool(X_tr, y_tr, cat_features=cat_indices)
        vp = Pool(X_val, y_val, cat_features=cat_indices)
        model = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=5,
            verbose=0, early_stopping_rounds=50, random_seed=SEED,
        )
        model.fit(tp, eval_set=vp, use_best_model=True)
        val_pred = model.predict(X_val)
        oof[val_idx] = val_pred
        score = max(0, 100 * r2_score(y_val, val_pred))
        fold_scores.append(score)
        models.append(model)
        print(f"    Fold {fold+1}: {score:.4f}")
    return oof, fold_scores, models


def evaluate_score(y, oof):
    return max(0, 100 * r2_score(y, oof))


def print_result(label, score, folds):
    print(f"\n  {label}")
    print(f"    Folds: {[f'{s:.2f}' for s in folds]}")
    print(f"    Mean:  {np.mean(folds):.4f}")
    print(f"    Score: {score:.4f}")


# ── Main Pipeline ────────────────────────────────────────────
def run():
    print("=" * 70)
    print("  TRAFFIC DEMAND PREDICTION — EXPLOIT-DRIVEN PIPELINE")
    print("=" * 70)

    train, test = load_data()
    print(f"  Train: {train.shape}  Test: {test.shape}")

    best_score = -np.inf
    best_features = None
    best_models = None
    best_train = train.copy()
    best_test = test.copy()
    best_cat_indices = None

    # ── PHASE 1: Bare-bones ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: BARE-BONES BASELINE")
    print("=" * 70)

    cat_cols = ["geohash", "RoadType", "Weather", "LargeVehicles", "Landmarks"]
    num_cols = ["hour", "minute", "NumberofLanes", "Temperature"]
    features = cat_cols + num_cols

    X = best_train[features].copy()
    y = best_train["demand"].values
    for c in cat_cols:
        X[c] = X[c].astype(str)
    cat_indices = [features.index(c) for c in cat_cols]

    oof, folds, models = train_evaluate(X, y, cat_indices)
    score = evaluate_score(y, oof)
    print_result("PHASE 1: BARE-BONES", score, folds)

    best_score = score
    best_features = list(features)
    best_models = models
    best_train["_oof"] = oof
    best_cat_indices = list(cat_indices)

    # ── PHASE 2: Exploits ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: EXPLOIT HUNTERS")
    print("=" * 70)

    # Exploit 2A: geohash lat/lon
    print("\n  --- 2A: Geohash Lat/Lon ---")
    t_train = best_train.copy()
    t_test = best_test.copy()
    for df in (t_train, t_test):
        coords = df["geohash"].apply(lambda g: pygeohash.decode(g))
        df["geo_lat"] = coords.apply(lambda x: x[0])
        df["geo_lon"] = coords.apply(lambda x: x[1])

    t_features = best_features + ["geo_lat", "geo_lon"]
    X = t_train[t_features].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    ci = [t_features.index(c) for c in cat_cols]

    oof_new, folds_new, models_new = train_evaluate(X, y, ci)
    score_new = evaluate_score(y, oof_new)
    print_result("2A: Geohash Lat/Lon", score_new, folds_new)

    if score_new > best_score:
        print(f"  KEPT (delta: {score_new - best_score:+.4f})")
        best_score = score_new
        best_features = t_features
        best_models = models_new
        best_train = t_train.copy()
        best_train["_oof"] = oof_new
        best_test = t_test.copy()
        best_cat_indices = ci
    else:
        print(f"  DROPPED (delta: {score_new - best_score:+.4f})")

    # Exploit 2B: OOF demand lookup
    print("\n  --- 2B: OOF Demand Lookup ---")
    t_train = best_train.copy()
    t_test = best_test.copy()
    t_train["demand_lookup"] = np.nan
    gm = t_train["demand"].mean()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(t_train)):
        tr_fold = t_train.iloc[tr_idx]
        val_fold = t_train.iloc[val_idx]
        t_train.loc[t_train.index[val_idx], "demand_lookup"] = vectorized_lookup(val_fold, tr_fold, gm)
    t_test["demand_lookup"] = vectorized_lookup(t_test, t_train, gm)

    t_features = best_features + ["demand_lookup"]
    X = t_train[t_features].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    ci = [t_features.index(c) for c in cat_cols]

    oof_new, folds_new, models_new = train_evaluate(X, y, ci)
    score_new = evaluate_score(y, oof_new)
    print_result("2B: OOF Demand Lookup", score_new, folds_new)

    if score_new > best_score:
        print(f"  KEPT (delta: {score_new - best_score:+.4f})")
        best_score = score_new
        best_features = t_features
        best_models = models_new
        best_train = t_train.copy()
        best_train["_oof"] = oof_new
        best_test = t_test.copy()
        best_cat_indices = ci
    else:
        print(f"  DROPPED (delta: {score_new - best_score:+.4f})")

    # Exploit 2C: Chronological lag
    print("\n  --- 2C: Chronological Lag ---")
    t_train = best_train.copy()
    t_test = best_test.copy()

    def ts_to_min(ts):
        h, m = ts.split(":")
        return int(h) * 60 + int(m)

    for df in (t_train, t_test):
        df["_min"] = df["timestamp"].apply(ts_to_min)
        df["_tk"] = df["day"] * 1440 + df["_min"]
        df["_lk"] = df["_tk"] - 60

    lag_ref = t_train[["geohash", "_tk", "demand"]].rename(columns={"demand": "demand_1_hour_ago", "_tk": "_lk"})
    t_train = t_train.merge(lag_ref, on=["geohash", "_lk"], how="left")
    t_train["demand_1_hour_ago"] = t_train["demand_1_hour_ago"].fillna(0)

    lag_ref2 = t_train[["geohash", "_tk", "demand"]].rename(columns={"demand": "_lag_tmp", "_tk": "_lk"})
    t_test = t_test.merge(lag_ref2, on=["geohash", "_lk"], how="left")
    t_test["demand_1_hour_ago"] = t_test["_lag_tmp"].fillna(0)
    t_test.drop(columns=["_lag_tmp"], inplace=True, errors="ignore")

    for df in (t_train, t_test):
        df.drop(columns=["_min", "_tk", "_lk"], inplace=True, errors="ignore")

    t_features = best_features + ["demand_1_hour_ago"]
    X = t_train[t_features].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    ci = [t_features.index(c) for c in cat_cols]

    oof_new, folds_new, models_new = train_evaluate(X, y, ci)
    score_new = evaluate_score(y, oof_new)
    print_result("2C: Chronological Lag", score_new, folds_new)

    if score_new > best_score:
        print(f"  KEPT (delta: {score_new - best_score:+.4f})")
        best_score = score_new
        best_features = t_features
        best_models = models_new
        best_train = t_train.copy()
        best_train["_oof"] = oof_new
        best_test = t_test.copy()
        best_cat_indices = ci
    else:
        print(f"  DROPPED (delta: {score_new - best_score:+.4f})")

    # ── PHASE 3: Toroidal ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3: ALGEBRAIC TORUS")
    print("=" * 70)

    gen = ToroidalTraversalGenerator(n=16)
    summary = gen.get_grid_summary()
    for k, v in summary.items():
        print(f"    {k}: {v}")

    t_train = best_train.copy()
    t_test = best_test.copy()
    demand_map = t_train.groupby(["day_of_week", "hour"])["demand"].mean().to_dict()

    for df in (t_train, t_test):
        df["toroidal_phase"] = df.apply(
            lambda r: gen.get_toroidal_phase(int(r["day_of_week"]), int(r["hour"])), axis=1)
        df["toroidal_entropy"] = df.apply(
            lambda r: gen.get_neighborhood_entropy(int(r["day_of_week"]), int(r["hour"]), demand_map), axis=1)
        df["toroidal_collision"] = df.apply(
            lambda r: gen.get_collision_frequency(int(r["day_of_week"]), int(r["hour"])), axis=1)

    t_features = best_features + ["toroidal_phase", "toroidal_entropy", "toroidal_collision"]
    X = t_train[t_features].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    ci = [t_features.index(c) for c in cat_cols]

    oof_new, folds_new, models_new = train_evaluate(X, y, ci)
    score_new = evaluate_score(y, oof_new)
    print_result("PHASE 3: TOROIDAL", score_new, folds_new)

    if score_new > best_score:
        print(f"  KEPT (delta: {score_new - best_score:+.4f})")
        best_score = score_new
        best_features = t_features
        best_models = models_new
        best_train = t_train.copy()
        best_test = t_test.copy()
        best_cat_indices = ci
    else:
        print(f"  DROPPED (delta: {score_new - best_score:+.4f})")

    # ── FINAL ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Best Score: {best_score:.4f}")
    print(f"  Features ({len(best_features)}): {best_features}")

    # Importance
    imp = np.zeros(len(best_features))
    for m in best_models:
        imp += m.get_feature_importance()
    imp /= len(best_models)
    imp_df = pd.DataFrame({"feature": best_features, "importance": imp}).sort_values("importance", ascending=False)
    print(f"\n  Feature Importance:")
    print(imp_df.to_string(index=False))

    # Predict
    print("\n  Generating submission...")
    X_test = best_test[best_features].copy()
    for c in cat_cols:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype(str)
    test_pool = Pool(X_test, cat_features=best_cat_indices)
    test_preds = np.zeros(len(best_test))
    for m in best_models:
        test_preds += m.predict(test_pool)
    test_preds /= len(best_models)
    test_preds = np.clip(test_preds, 0, None)

    sub = pd.DataFrame({"Index": best_test["Index"], "demand": test_preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Saved: {SUBMISSION_PATH}  Shape: {sub.shape}")
    print(f"  Demand: mean={sub['demand'].mean():.6f} range=[{sub['demand'].min():.6f}, {sub['demand'].max():.6f}]")

    return best_score


if __name__ == "__main__":
    score = run()
    print(f"\n  DONE. Final Score: {score:.4f}")
