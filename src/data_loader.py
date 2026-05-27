"""Data loading, parsing, and basic preprocessing."""
import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH, TARGET


def parse_timestamp(ts: str) -> tuple:
    """Parse timestamp string 'H:M' into (hour, minute) integers."""
    parts = ts.split(":")
    return int(parts[0]), int(parts[1])


def load_data() -> tuple:
    """Load train and test DataFrames with parsed time features."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for df in (train, test):
        # Parse timestamp -> hour, minute
        parsed = df["timestamp"].apply(parse_timestamp)
        df["hour"] = parsed.apply(lambda x: x[0])
        df["minute"] = parsed.apply(lambda x: x[1])

        # Day of week (synthetic from day number)
        df["day_of_week"] = df["day"] % 7
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Compute minute_of_day for later use
        df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # Fill NaN values
    for df in (train, test):
        df["RoadType"] = df["RoadType"].fillna("Unknown")
        df["Weather"] = df["Weather"].fillna("Unknown")
        df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

    return train, test


def get_feature_columns(phase: int = 1) -> tuple:
    """Return (cat_features, num_features) for the given phase."""
    from src.config import CAT_FEATURES, BASE_NUMERIC_FEATURES, PHASE2_FEATURES, PHASE3_FEATURES

    cat_cols = CAT_FEATURES
    num_cols = list(BASE_NUMERIC_FEATURES)

    if phase >= 2:
        num_cols += PHASE2_FEATURES
    if phase >= 3:
        num_cols += PHASE3_FEATURES

    return cat_cols, num_cols
