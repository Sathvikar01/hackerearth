"""Temporal FFT Features.

Extracts dominant frequencies, amplitudes, and phases from historical
demand patterns per geohash, providing the model with frequency-domain
insights into traffic periodicity.
"""
import numpy as np
import pandas as pd
from scipy.fft import fft


def compute_fft_features(demand_series: np.ndarray, n_dominant: int = 3) -> dict:
    """Extract dominant frequency features from a demand time series.

    Args:
        demand_series: 1D array of demand values
        n_dominant: Number of dominant frequencies to extract

    Returns:
        Dictionary with amplitude and phase features
    """
    n = len(demand_series)
    if n < 4:
        return {f"fft_amp_{i}": 0.0 for i in range(n_dominant)} | \
               {f"fft_phase_{i}": 0.0 for i in range(n_dominant)} | \
               {"fft_dominant_freq": 0.0, "fft_spectral_energy": 0.0}

    # Apply FFT
    fft_vals = fft(demand_series)
    # Take only positive frequencies
    fft_magnitude = np.abs(fft_vals[:n // 2]) / n
    fft_phase = np.angle(fft_vals[:n // 2])

    # Find dominant frequencies (excluding DC component at index 0)
    if len(fft_magnitude) < 2:
        return {f"fft_amp_{i}": 0.0 for i in range(n_dominant)} | \
               {f"fft_phase_{i}": 0.0 for i in range(n_dominant)} | \
               {"fft_dominant_freq": 0.0, "fft_spectral_energy": 0.0}

    # Sort by magnitude (skip DC component)
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

    # Dominant frequency (normalized)
    features["fft_dominant_freq"] = float(sorted_indices[0] + 1) / n if len(sorted_indices) > 0 else 0.0

    # Total spectral energy (sum of squared magnitudes)
    features["fft_spectral_energy"] = float(np.sum(magnitudes ** 2))

    return features


def add_fft_features(train_df: pd.DataFrame, val_or_test_df: pd.DataFrame,
                      n_dominant: int = 3) -> tuple:
    """Add FFT features based on historical demand patterns per geohash.

    Computes FFT on the training data demand grouped by geohash,
    then maps the features to both DataFrames.

    Args:
        train_df: Training DataFrame with 'geohash' and 'demand' columns
        val_or_test_df: Validation or test DataFrame
        n_dominant: Number of dominant frequencies to extract

    Returns:
        (train_df, val_or_test_df) with FFT features added
    """
    print("    Computing FFT spectral features per geohash...")

    # Compute FFT features per geohash from training data
    fft_features = {}
    for gh, group in train_df.groupby("geohash"):
        demand = group.sort_values("minute_of_day")["demand"].values
        fft_features[gh] = compute_fft_features(demand, n_dominant=n_dominant)

    # Convert to DataFrame
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

    # Merge
    train_df = train_df.merge(fft_df, on="geohash", how="left")
    val_or_test_df = val_or_test_df.merge(fft_df, on="geohash", how="left")

    # Fill missing
    for col in fft_cols:
        train_df[col] = train_df[col].fillna(0.0)
        val_or_test_df[col] = val_or_test_df[col].fillna(0.0)

    print(f"    Added {len(fft_cols)} FFT features")
    return train_df, val_or_test_df
