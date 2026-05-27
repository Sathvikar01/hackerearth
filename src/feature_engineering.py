"""Feature engineering for all three phases of the pipeline."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.config import (
    RUSH_HOURS, TEMPORAL_STATES, SEED,
    BASE_NUMERIC_FEATURES, PHASE2_FEATURES, PHASE3_FEATURES,
)


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine encodings for hour (24) and day_of_week (7)."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_geohash_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create geohash prefix features for spatial granularity."""
    df["geohash_prefix_3"] = df["geohash"].str[:3]
    df["geohash_prefix_4"] = df["geohash"].str[:4]
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features: RoadType x Lanes, Weather x Temp."""
    # Label encode for interaction
    road_map = {v: i for i, v in enumerate(df["RoadType"].unique())}
    weather_map = {v: i for i, v in enumerate(df["Weather"].unique())}

    df["RoadType_x_Lanes"] = df["RoadType"].map(road_map).fillna(0) * df["NumberofLanes"]
    df["Weather_x_Temp"] = df["Weather"].map(weather_map).fillna(0) * df["Temperature"]
    return df


def add_target_encoding_oof(train: pd.DataFrame, test: pd.DataFrame,
                             group_col: str, target: str = "demand",
                             n_splits: int = 5) -> tuple:
    """Leakage-safe target encoding using GroupKFold out-of-fold predictions.

    Groups by geohash to prevent spatial leakage.

    Returns:
        (train_df, test_df) with new columns: {group_col}_target_mean, {group_col}_target_var
    """
    mean_col = f"{group_col}_target_mean"
    var_col = f"{group_col}_target_var"

    train[mean_col] = np.nan
    train[var_col] = np.nan

    gkf = GroupKFold(n_splits=n_splits)
    groups = train["geohash"].values

    for train_idx, val_idx in gkf.split(train, train[target], groups):
        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]

        stats = train_fold.groupby(group_col)[target].agg(["mean", "var"]).reset_index()
        stats.columns = [group_col, mean_col, var_col]

        merged = val_fold[[group_col]].merge(stats, on=group_col, how="left")
        train.loc[train.index[val_idx], mean_col] = merged[mean_col].values
        train.loc[train.index[val_idx], var_col] = merged[var_col].values

    # Fill any remaining NaN with global stats
    global_mean = train[target].mean()
    global_var = train[target].var()
    train[mean_col] = train[mean_col].fillna(global_mean)
    train[var_col] = train[var_col].fillna(global_var)

    # For test: compute from full train
    stats_full = train.groupby(group_col)[target].agg(["mean", "var"]).reset_index()
    stats_full.columns = [group_col, mean_col, var_col]
    test = test.merge(stats_full, on=group_col, how="left")
    test[mean_col] = test[mean_col].fillna(global_mean)
    test[var_col] = test[var_col].fillna(global_var)

    return train, test


def add_all_target_encodings(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Apply target encoding for geohash and geohash_prefix_4."""
    train, test = add_target_encoding_oof(train, test, "geohash")
    train, test = add_target_encoding_oof(train, test, "geohash_prefix_4")
    return train, test


def compute_toroidal_distance_to_rush_hour(hour: np.ndarray) -> np.ndarray:
    """Compute wrapped cyclic distance to nearest rush hour (8 or 18).

    d_hour = min(|H - 8|, 24 - |H - 8|, |H - 18|, 24 - |H - 18|)
    """
    distances = []
    for rh in RUSH_HOURS:
        diff = np.abs(hour - rh)
        wrapped = np.minimum(diff, 24 - diff)
        distances.append(wrapped)
    return np.minimum(distances[0], distances[1])


def compute_weekly_peak(train: pd.DataFrame) -> tuple:
    """Find the (day_of_week, hour) with highest average demand in training data.

    Returns:
        (peak_dow, peak_hour) tuple
    """
    grouped = train.groupby(["day_of_week", "hour"])["demand"].mean()
    peak_idx = grouped.idxmax()
    return peak_idx


def compute_toroidal_distance_to_weekly_peak(df: pd.DataFrame,
                                              peak_dow: int,
                                              peak_hour: int) -> np.ndarray:
    """Compute wrapped 168-hour cyclic distance to the weekly peak.

    t = day_of_week * 24 + hour
    d = min(|t1 - t2|, 168 - |t1 - t2|)
    """
    t = df["day_of_week"].values * 24 + df["hour"].values
    t_peak = peak_dow * 24 + peak_hour
    diff = np.abs(t - t_peak)
    return np.minimum(diff, TEMPORAL_STATES - diff)


def add_phase2_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Add Phase 2: Continuous Toroidal Signals."""
    # Toroidal distance to rush hour
    train["toroidal_dist_rush_hour"] = compute_toroidal_distance_to_rush_hour(train["hour"].values)
    test["toroidal_dist_rush_hour"] = compute_toroidal_distance_to_rush_hour(test["hour"].values)

    # Toroidal distance to weekly peak
    peak_dow, peak_hour = compute_weekly_peak(train)
    print(f"  Weekly peak demand at: day_of_week={peak_dow}, hour={peak_hour}")
    train["toroidal_dist_weekly_peak"] = compute_toroidal_distance_to_weekly_peak(train, peak_dow, peak_hour)
    test["toroidal_dist_weekly_peak"] = compute_toroidal_distance_to_weekly_peak(test, peak_dow, peak_hour)

    return train, test


def add_phase3_features(train: pd.DataFrame, test: pd.DataFrame,
                         toroidal_gen) -> tuple:
    """Add Phase 3: Formal Toroidal Traversal features.

    Args:
        toroidal_gen: ToroidalTraversalGenerator instance
    """
    # Build demand map from training data: (dow, hour) -> mean demand
    demand_map = train.groupby(["day_of_week", "hour"])["demand"].mean().to_dict()

    for df in [train, test]:
        # ToroidalPhase
        df["toroidal_phase"] = df.apply(
            lambda row: toroidal_gen.get_toroidal_phase(int(row["day_of_week"]), int(row["hour"])),
            axis=1,
        )

        # ToroidalNeighborhoodEntropy
        df["toroidal_neighborhood_entropy"] = df.apply(
            lambda row: toroidal_gen.get_neighborhood_entropy(
                int(row["day_of_week"]), int(row["hour"]), demand_map
            ),
            axis=1,
        )

        # ToroidalCollisionFrequency
        df["toroidal_collision_frequency"] = df.apply(
            lambda row: toroidal_gen.get_collision_frequency(
                int(row["day_of_week"]), int(row["hour"])
            ),
            axis=1,
        )

    return train, test


def engineer_features_phase1(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Full Phase 1 feature engineering pipeline."""
    print("Phase 1: Feature Engineering")
    print("  Adding cyclic features...")
    train = add_cyclic_features(train)
    test = add_cyclic_features(test)

    print("  Adding geohash features...")
    train = add_geohash_features(train)
    test = add_geohash_features(test)

    print("  Adding interaction features...")
    train = add_interaction_features(train)
    test = add_interaction_features(test)

    print("  Adding target encodings (leakage-safe OOF)...")
    train, test = add_all_target_encodings(train, test)

    return train, test


def engineer_features_phase2(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Phase 2: Add toroidal signal features."""
    print("Phase 2: Adding toroidal signal features...")
    train, test = add_phase2_features(train, test)
    return train, test


def engineer_features_phase3(train: pd.DataFrame, test: pd.DataFrame,
                              toroidal_gen) -> tuple:
    """Phase 3: Add formal toroidal traversal features."""
    print("Phase 3: Adding toroidal traversal features...")
    train, test = add_phase3_features(train, test, toroidal_gen)
    return train, test
