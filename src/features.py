"""Stage 1-2: Feature Factory.

Constructs temporal, spatial, contextual, and golden lag features.
"""
import numpy as np
import pandas as pd
import pygeohash


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encodings for hour and 15_min_slot."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["slot_sin"] = np.sin(2 * np.pi * df["15_min_slot"] / 96)
    df["slot_cos"] = np.cos(2 * np.pi * df["15_min_slot"] / 96)
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decode geohash to lat/lon, extract prefix features."""
    coords = df["geohash"].apply(lambda g: pygeohash.decode(g))
    df["latitude"] = coords.apply(lambda x: x[0])
    df["longitude"] = coords.apply(lambda x: x[1])
    df["geohash_prefix_3"] = df["geohash"].str[:3]
    df["geohash_prefix_4"] = df["geohash"].str[:4]
    return df


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features: RoadType x hour, Weather x Temperature."""
    df["RoadType_x_hour"] = df["RoadType"].astype(str) + "_" + df["hour"].astype(str)
    df["Weather_x_Temp"] = df["Weather"].astype(str) + "_" + df["Temperature"].round(0).astype(int).astype(str)
    return df


def add_golden_lag(train_split: pd.DataFrame, val_or_test: pd.DataFrame) -> pd.DataFrame:
    """Create exact_lag_demand from Day 48 based on (geohash, timestamp).

    Maps demand from train_split onto val_or_test using exact (geohash, timestamp) match.
    Rows without a match get NaN (handled by the blending logic).
    """
    # Build lookup from train_split: (geohash, timestamp) -> demand
    lookup = train_split.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()

    # Map to val_or_test
    val_or_test["exact_lag_demand"] = val_or_test.apply(
        lambda r: lookup.get((r["geohash"], r["timestamp"]), np.nan), axis=1
    )

    # Print coverage
    coverage = val_or_test["exact_lag_demand"].notna().sum()
    total = len(val_or_test)
    print(f"    Lag coverage: {coverage}/{total} ({coverage/total*100:.1f}%)")

    return val_or_test


def add_combined_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create combined string features for target encoding."""
    df["geo_slot"] = df["geohash"] + "_" + df["15_min_slot"].astype(str)
    df["geo_p4_hour"] = df["geohash_prefix_4"] + "_" + df["hour"].astype(str)
    return df


def build_features(train_split: pd.DataFrame, val_or_test: pd.DataFrame,
                   include_lag: bool = True) -> tuple:
    """Full feature factory pipeline.

    Args:
        train_split: Training data (Day 48) for lag lookup
        val_or_test: Validation or test data to add features to
        include_lag: Whether to add the golden lag feature

    Returns:
        (train_features, val_features) DataFrames
    """
    print("  Building features...")

    # Apply to both
    for df in (train_split, val_or_test):
        df = add_temporal_features(df)
        df = add_spatial_features(df)
        df = add_contextual_features(df)
        df = add_combined_target_features(df)

    # Lag feature (only from train_split to val_or_test)
    if include_lag:
        val_or_test = add_golden_lag(train_split, val_or_test)

    return train_split, val_or_test


# Feature lists for different models
MODEL_A_FEATURES = {
    "cat": ["geohash", "RoadType", "Weather", "LargeVehicles", "Landmarks",
            "geohash_prefix_3", "geohash_prefix_4", "RoadType_x_hour", "Weather_x_Temp"],
    "num": ["hour", "minute", "minute_of_day", "15_min_slot", "day_of_week",
            "hour_sin", "hour_cos", "slot_sin", "slot_cos",
            "latitude", "longitude", "Temperature"],
}

MODEL_B_FEATURES = {
    "cat": ["geohash", "geohash_prefix_4"],
    "num": ["exact_lag_demand", "Temperature", "hour", "minute",
            "latitude", "longitude", "hour_sin", "hour_cos"],
}
