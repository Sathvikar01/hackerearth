"""Data loading and basic preprocessing."""
import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH


def parse_timestamp(ts: str) -> tuple:
    parts = ts.split(":")
    return int(parts[0]), int(parts[1])


def load_data() -> tuple:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for df in (train, test):
        parsed = df["timestamp"].apply(parse_timestamp)
        df["hour"] = parsed.apply(lambda x: x[0])
        df["minute"] = parsed.apply(lambda x: x[1])
        df["day_of_week"] = df["day"] % 7

        df["RoadType"] = df["RoadType"].fillna("Unknown")
        df["Weather"] = df["Weather"].fillna("Unknown")
        df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

    return train, test
