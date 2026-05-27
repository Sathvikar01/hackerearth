"""Data loading and preprocessing."""
import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH, TRAIN_DAY, VAL_DAY


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features and fill NaN."""
    df["hour"] = df["timestamp"].apply(lambda x: int(x.split(":")[0]))
    df["minute"] = df["timestamp"].apply(lambda x: int(x.split(":")[1]))
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["15_min_slot"] = df["minute_of_day"] // 15
    df["day_of_week"] = df["day"] % 7

    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    return df


def load_data() -> tuple:
    """Load and preprocess train/test data."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = preprocess(train)
    test = preprocess(test)

    return train, test


def chronological_split(train: pd.DataFrame) -> tuple:
    """Split train into Day 48 (train) and Day 49 (validation)."""
    train_split = train[train["day"] == TRAIN_DAY].copy().reset_index(drop=True)
    val_split = train[train["day"] == VAL_DAY].copy().reset_index(drop=True)
    return train_split, val_split
