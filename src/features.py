"""Modular feature engineering with rollback support.

Each feature function takes (train, test) and returns (train, test) with new columns.
The OOF lookup uses strict out-of-fold calculation to prevent leakage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.config import SEED, TEMPORAL_STATES


# ---------------------------------------------------------------------------
# Phase 1: Bare-bones (no extra features needed, just raw columns)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Exploit 2A: Geohash → lat/lon
# ---------------------------------------------------------------------------
def add_geohash_latlon(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Decode geohash to latitude and longitude."""
    import pygeohash

    def decode_geo(gh):
        lat, lon = pygeohash.decode(gh)
        return lat, lon

    for df in (train, test):
        coords = df["geohash"].apply(decode_geo)
        df["geo_lat"] = coords.apply(lambda x: x[0])
        df["geo_lon"] = coords.apply(lambda x: x[1])

    return train, test


# ---------------------------------------------------------------------------
# Exploit 2B: "Perfect Memory" OOF Lookup (vectorized)
# ---------------------------------------------------------------------------
def _vectorized_lookup(df_target: pd.DataFrame, lookup_df: pd.DataFrame,
                       global_mean: float) -> np.ndarray:
    """Apply cascading fallback lookup using vectorized merge.

    Priority: (geohash, timestamp) → (geohash, hour) → geohash → global_mean
    """
    # Level 1: (geohash, timestamp)
    stats_ts = lookup_df.groupby(["geohash", "timestamp"])["demand"].mean().reset_index()
    stats_ts.columns = ["geohash", "timestamp", "_lookup_val"]
    merged = df_target[["geohash", "timestamp"]].merge(
        stats_ts, on=["geohash", "timestamp"], how="left"
    )
    result = merged["_lookup_val"].values

    # Level 2: (geohash, hour) for NaN
    nan_mask = np.isnan(result)
    if nan_mask.any():
        stats_hour = lookup_df.groupby(["geohash", "hour"])["demand"].mean().reset_index()
        stats_hour.columns = ["geohash", "hour", "_lookup_hour"]
        merged_h = df_target.loc[nan_mask, ["geohash", "hour"]].merge(
            stats_hour, on=["geohash", "hour"], how="left"
        )
        result[nan_mask] = merged_h["_lookup_hour"].values

    # Level 3: geohash for remaining NaN
    nan_mask = np.isnan(result)
    if nan_mask.any():
        stats_geo = lookup_df.groupby("geohash")["demand"].mean().reset_index()
        stats_geo.columns = ["geohash", "_lookup_geo"]
        merged_g = df_target.loc[nan_mask, ["geohash"]].merge(
            stats_geo, on="geohash", how="left"
        )
        result[nan_mask] = merged_g["_lookup_geo"].values

    # Level 4: global mean
    result = np.nan_to_num(result, nan=global_mean)
    return result


def add_demand_lookup_oof(train: pd.DataFrame, test: pd.DataFrame,
                          n_splits: int = 5) -> tuple:
    """Add OOF demand lookup feature using strict out-of-fold calculation."""
    train["demand_lookup"] = np.nan
    global_mean = train["demand"].mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
        tr_fold = train.iloc[tr_idx]
        val_fold = train.iloc[val_idx]
        train.loc[train.index[val_idx], "demand_lookup"] = _vectorized_lookup(
            val_fold, tr_fold, global_mean
        )

    # For test: use full train lookup
    test["demand_lookup"] = _vectorized_lookup(test, train, global_mean)

    return train, test


# ---------------------------------------------------------------------------
# Exploit 2C: Chronological Lag (vectorized)
# ---------------------------------------------------------------------------
def add_chronological_lag(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Create demand_1_hour_ago by shifting within geohash, sorted by time.

    Uses merge_asof for efficient temporal join. Only uses past data.
    """
    # Build time index: map timestamp to minutes since midnight
    def ts_to_minutes(ts):
        h, m = ts.split(":")
        return int(h) * 60 + int(m)

    for df in (train, test):
        df["_minutes"] = df["timestamp"].apply(ts_to_minutes)
        df["_time_key"] = df["day"] * 1440 + df["_minutes"]

    # Target time = current time - 60 minutes
    train["_lag_key"] = train["_time_key"] - 60
    test["_lag_key"] = test["_time_key"] - 60

    # Build lookup: (geohash, _time_key) -> demand
    lag_ref = train[["geohash", "_time_key", "demand"]].copy()
    lag_ref = lag_ref.rename(columns={"demand": "demand_1_hour_ago", "_time_key": "_lag_key"})

    # Merge train
    train = train.merge(lag_ref, on=["geohash", "_lag_key"], how="left")
    train["demand_1_hour_ago"] = train["demand_1_hour_ago"].fillna(0)

    # Merge test (use full train as reference)
    lag_ref_test = train[["geohash", "_time_key", "demand"]].copy()
    lag_ref_test = lag_ref_test.rename(columns={"demand": "demand_1_hour_ago_tmp", "_time_key": "_lag_key"})
    test = test.merge(lag_ref_test, on=["geohash", "_lag_key"], how="left")
    test["demand_1_hour_ago"] = test.get("demand_1_hour_ago_tmp", pd.Series(dtype=float))
    if "demand_1_hour_ago_tmp" in test.columns:
        test.drop(columns=["demand_1_hour_ago_tmp"], inplace=True)
    test["demand_1_hour_ago"] = test["demand_1_hour_ago"].fillna(0)

    # Cleanup temp columns
    for df in (train, test):
        df.drop(columns=["_minutes", "_time_key", "_lag_key"], inplace=True, errors="ignore")

    return train, test


# ---------------------------------------------------------------------------
# Phase 3: Toroidal Features
# ---------------------------------------------------------------------------
def add_toroidal_features(train: pd.DataFrame, test: pd.DataFrame,
                           toroidal_gen) -> tuple:
    """Add ToroidalPhase, NeighborhoodEntropy, CollisionFrequency."""
    demand_map = train.groupby(["day_of_week", "hour"])["demand"].mean().to_dict()

    for df in (train, test):
        df["toroidal_phase"] = df.apply(
            lambda r: toroidal_gen.get_toroidal_phase(int(r["day_of_week"]), int(r["hour"])),
            axis=1,
        )
        df["toroidal_neighborhood_entropy"] = df.apply(
            lambda r: toroidal_gen.get_neighborhood_entropy(
                int(r["day_of_week"]), int(r["hour"]), demand_map),
            axis=1,
        )
        df["toroidal_collision_frequency"] = df.apply(
            lambda r: toroidal_gen.get_collision_frequency(
                int(r["day_of_week"]), int(r["hour"])),
            axis=1,
        )

    return train, test


# ---------------------------------------------------------------------------
# Feature registry for rollback system
# ---------------------------------------------------------------------------
PHASE1_FEATURES = {
    "num": ["hour", "minute", "NumberofLanes", "Temperature"],
    "cat": ["geohash", "RoadType", "Weather", "LargeVehicles", "Landmarks"],
}

EXPLOITS = {
    "2A_geohash_latlon": {
        "func": add_geohash_latlon,
        "num": ["geo_lat", "geo_lon"],
        "cat": [],
    },
    "2B_demand_lookup": {
        "func": add_demand_lookup_oof,
        "num": ["demand_lookup"],
        "cat": [],
    },
    "2C_chronological_lag": {
        "func": add_chronological_lag,
        "num": ["demand_1_hour_ago"],
        "cat": [],
    },
}

TOROIDAL_FEATURES = {
    "num": ["toroidal_phase", "toroidal_neighborhood_entropy", "toroidal_collision_frequency"],
    "cat": [],
}
