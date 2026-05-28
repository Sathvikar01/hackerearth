"""Temporal FFT Features (Leakage-Safe).

Extracts dominant frequencies, amplitudes, and phases from historical
demand patterns per geohash using STRICTLY Day 48 data only.

FFT is computed once on Day 48 and mapped forward to Day 49/Test.
This prevents any future data from leaking into training features.
"""
import numpy as np
import pandas as pd
from scipy.fft import fft


def compute_fft_features(demand_series: np.ndarray, n_dominant: int = 3) -> dict:
    """Extract dominant frequency features from a demand time series."""
    n = len(demand_series)
    if n < 4:
        return {f"fft_amp_{i}": 0.0 for i in range(n_dominant)} | \
               {f"fft_phase_{i}": 0.0 for i in range(n_dominant)} | \
               {"fft_dominant_freq": 0.0, "fft_spectral_energy": 0.0}

    fft_vals = fft(demand_series)
    fft_magnitude = np.abs(fft_vals[:n // 2]) / n
    fft_phase = np.angle(fft_vals[:n // 2])

    if len(fft_magnitude) < 2:
        return {f"fft_amp_{i}": 0.0 for i in range(n_dominant)} | \
               {f"fft_phase_{i}": 0.0 for i in range(n_dominant)} | \
               {"fft_dominant_freq": 0.0, "fft_spectral_energy": 0.0}

    magnitudes = fft_magnitude[1:]
    phases = fft_phase[1:]
    sorted_indices = np.argsort(magnitudes)[::-1]

    features = {}
    for i in range(n_dominant):
        if i < len(sorted_indices):
            idx = sorted_indices[i]
            features[f"fft_amp_{i}"] = float(magnitudes[idx])
            features[f"fft_phase_{i}"] = float(phases[idx])
        else:
            features[f"fft_amp_{i}"] = 0.0
            features[f"fft_phase_{i}"] = 0.0

    features["fft_dominant_freq"] = float(sorted_indices[0] + 1) / n if len(sorted_indices) > 0 else 0.0
    features["fft_spectral_energy"] = float(np.sum(magnitudes ** 2))
    return features


def add_fft_features(train_df: pd.DataFrame, val_or_test_df: pd.DataFrame,
                      n_dominant: int = 3,
                      train_day: int = 48) -> tuple:
    """Add FFT features based on STRICTLY Day 48 demand patterns.

    Leakage-safe: Only uses data where day <= train_day to compute FFT.
    The computed features are then mapped to both DataFrames via geohash merge.

    Args:
        train_df: Training DataFrame with 'geohash', 'demand', 'day' columns
        val_or_test_df: Validation or test DataFrame
        n_dominant: Number of dominant frequencies to extract
        train_day: Maximum day to use for FFT computation (default 48)

    Returns:
        (train_df, val_or_test_df) with FFT features added
    """
    print("    Computing FFT spectral features (Day 48 only, leakage-safe)...")

    # Strict chronological cutoff: only use Day 48 data
    train_day48 = train_df[train_df["day"] <= train_day].copy()

    fft_features = {}
    for gh, group in train_day48.groupby("geohash"):
        demand = group.sort_values("minute_of_day")["demand"].values
        fft_features[gh] = compute_fft_features(demand, n_dominant=n_dominant)

    fft_df = pd.DataFrame.from_dict(fft_features, orient="index")
    fft_df.index.name = "geohash"
    fft_df = fft_df.reset_index()

    # Drop existing fft columns to allow re-calling
    fft_cols = [c for c in fft_df.columns if c.startswith("fft_")]
    for col in fft_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in val_or_test_df.columns:
            val_or_test_df = val_or_test_df.drop(columns=[col])

    train_df = train_df.merge(fft_df, on="geohash", how="left")
    val_or_test_df = val_or_test_df.merge(fft_df, on="geohash", how="left")

    for col in fft_cols:
        train_df[col] = train_df[col].fillna(0.0)
        val_or_test_df[col] = val_or_test_df[col].fillna(0.0)

    print(f"    Added {len(fft_cols)} FFT features (leakage-safe)")
    return train_df, val_or_test_df
