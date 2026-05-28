"""Feature Factory: Spatial Clustering, Temporal Harmonics, Interaction Keys.

Replaces the old toroidal traversal with K-Means spatial clusters,
rotated coordinates, Fourier harmonics, and high-order spatio-temporal
interaction features for CatBoost's native categorical encoding.
"""
import numpy as np
import pandas as pd
import pygeohash
from sklearn.cluster import KMeans


# ── Temporal ─────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encodings for hour and 15_min_slot."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["slot_sin"] = np.sin(2 * np.pi * df["15_min_slot"] / 96)
    df["slot_cos"] = np.cos(2 * np.pi * df["15_min_slot"] / 96)
    return df


# ── Spatial ──────────────────────────────────────────────────

def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decode geohash to lat/lon, extract prefix features."""
    coords = df["geohash"].apply(lambda g: pygeohash.decode(g))
    df["latitude"] = coords.apply(lambda x: x[0])
    df["longitude"] = coords.apply(lambda x: x[1])
    df["geohash_prefix_4"] = df["geohash"].str[:4]
    return df


def add_spatial_clusters(train_df: pd.DataFrame, val_or_test_df: pd.DataFrame,
                         n_clusters_list: list = None) -> tuple:
    """K-Means spatial clusters on latitude/longitude.

    Fits on train_df, transforms both. Creates cluster ID columns
    (e.g., 'cluster_10', 'cluster_50') as categorical features.

    Args:
        train_df: Training DataFrame (must have 'latitude', 'longitude')
        val_or_test_df: Validation or test DataFrame
        n_clusters_list: List of K values (default [10, 50])

    Returns:
        (train_df, val_or_test_df) with new cluster columns added
    """
    if n_clusters_list is None:
        n_clusters_list = [10, 50]

    coords_train = train_df[["latitude", "longitude"]].values
    coords_val = val_or_test_df[["latitude", "longitude"]].values

    for n in n_clusters_list:
        col_name = f"cluster_{n}"
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        train_df[col_name] = kmeans.fit_predict(coords_train).astype(str)
        val_or_test_df[col_name] = kmeans.predict(coords_val).astype(str)

    return train_df, val_or_test_df


def add_rotated_coordinates(df: pd.DataFrame, angles: list = None) -> pd.DataFrame:
    """Rotated lat/lon for diagonal spatial boundaries.

    Trees split orthogonally. Rotating coordinates by multiple angles
    lets CatBoost carve diagonal spatial boundaries more easily.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns
        angles: List of rotation angles in degrees (default [15, 30, 45, 60])

    Returns:
        DataFrame with new rotated coordinate columns
    """
    if angles is None:
        angles = [15, 30, 45, 60]

    lat = df["latitude"].values
    lon = df["longitude"].values

    for angle in angles:
        rad = np.radians(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        df[f"lat_rot_{angle}"] = lat * cos_a - lon * sin_a
        df[f"lon_rot_{angle}"] = lat * sin_a + lon * cos_a

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers between two lat/lon points."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def add_distance_to_center(df: pd.DataFrame,
                           center_lat: float = None,
                           center_lon: float = None) -> pd.DataFrame:
    """Haversine distance from geographic center of the dataset (in km).

    Replaces the old flat-Euclidean approximation with proper haversine.
    """
    if center_lat is None:
        center_lat = df["latitude"].mean()
    if center_lon is None:
        center_lon = df["longitude"].mean()

    df["dist_to_center"] = haversine_distance(
        df["latitude"].values, df["longitude"].values,
        center_lat, center_lon
    )
    return df


def add_manhattan_distance(df: pd.DataFrame,
                           center_lat: float = None,
                           center_lon: float = None) -> pd.DataFrame:
    """Manhattan distance approximation using haversine (in km).

    Computes haversine distance for lat and lon separately and sums them.
    This better represents actual driving distance on a city grid.
    """
    if center_lat is None:
        center_lat = df["latitude"].mean()
    if center_lon is None:
        center_lon = df["longitude"].mean()

    # Haversine along latitude only (same longitude)
    lat_component = haversine_distance(
        df["latitude"].values, np.full(len(df), center_lon),
        np.full(len(df), center_lat), np.full(len(df), center_lon)
    )
    # Haversine along longitude only (same latitude)
    lon_component = haversine_distance(
        np.full(len(df), center_lat), df["longitude"].values,
        np.full(len(df), center_lat), np.full(len(df), center_lon)
    )
    df["manhattan_dist_to_center"] = lat_component + lon_component
    return df


# ── Fourier Harmonics ────────────────────────────────────────

def add_fourier_harmonics(df: pd.DataFrame, columns: list = None,
                          n_harmonics: int = 2) -> pd.DataFrame:
    """Higher-order Fourier harmonics for complex periodic patterns.

    Beyond basic sin/cos, harmonics capture 12-hour, 8-hour, 6-hour cycles.

    Args:
        df: DataFrame
        columns: Column names to harmonize (default ['hour', '15_min_slot'])
        n_harmonics: Number of harmonic orders (default 2)

    Returns:
        DataFrame with harmonic columns added
    """
    if columns is None:
        columns = ["hour", "15_min_slot"]

    periods = {"hour": 24, "15_min_slot": 96}

    for col in columns:
        period = periods.get(col, 24)
        for h in range(2, n_harmonics + 1):
            df[f"{col}_sin_{h}"] = np.sin(2 * np.pi * h * df[col] / period)
            df[f"{col}_cos_{h}"] = np.cos(2 * np.pi * h * df[col] / period)

    return df


# ── Contextual ───────────────────────────────────────────────

def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features: RoadType x hour, Weather x Temperature."""
    df["RoadType_x_hour"] = df["RoadType"].astype(str) + "_" + df["hour"].astype(str)
    df["Weather_x_Temp"] = (
        df["Weather"].astype(str) + "_" + df["Temperature"].round(0).astype(int).astype(str)
    )
    return df


# ── High-Order Spatio-Temporal Interactions ──────────────────

def add_interaction_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Combined categorical keys for CatBoost's native target encoding.

    These capture localized historical averages without explicit lag data:
    - geohash + hour: "How busy is this location at this hour?"
    - geohash + day_of_week: "How busy is this location on Mondays?"
    - cluster + hour: "How busy is this neighborhood at rush hour?"
    - RoadType + is_weekend: "How busy are highways on weekends?"

    Args:
        df: DataFrame with geohash, hour, day_of_week, cluster columns

    Returns:
        DataFrame with interaction key columns added
    """
    df["geo_hour"] = df["geohash"] + "_h" + df["hour"].astype(str)
    df["geo_dow"] = df["geohash"] + "_d" + df["day_of_week"].astype(str)
    df["geo_slot"] = df["geohash"] + "_s" + df["15_min_slot"].astype(str)
    df["geo_p4_hour"] = df["geohash_prefix_4"] + "_h" + df["hour"].astype(str)

    # Cluster interactions (if cluster columns exist)
    for n in [10, 50]:
        col = f"cluster_{n}"
        if col in df.columns:
            df[f"cl{n}_hour"] = df[col] + "_h" + df["hour"].astype(str)
            df[f"cl{n}_dow"] = df[col] + "_d" + df["day_of_week"].astype(str)
            df[f"cl{n}_slot"] = df[col] + "_s" + df["15_min_slot"].astype(str)

    # Contextual interactions
    df["rt_dow"] = df["RoadType"].astype(str) + "_d" + df["day_of_week"].astype(str)
    df["rt_hour"] = df["RoadType"].astype(str) + "_h" + df["hour"].astype(str)
    df["wx_hour"] = df["Weather"].astype(str) + "_h" + df["hour"].astype(str)

    return df


# ── Lag Features (for Model B only) ─────────────────────────

def build_lag_features(train_split: pd.DataFrame, val_or_test: pd.DataFrame,
                       verbose: bool = True) -> pd.DataFrame:
    """Build primary (exact), fuzzy (+/-30min), and secondary (hour) lag features."""
    import time as _time

    def ts_to_min(ts):
        h, m = ts.split(":")
        return int(h) * 60 + int(m)

    if verbose:
        t0 = _time.time()
        print("      Computing exact lag...")
    train_split = train_split.copy()
    train_split["_minutes"] = train_split["timestamp"].apply(ts_to_min)
    val_or_test = val_or_test.copy()
    val_or_test["_minutes"] = val_or_test["timestamp"].apply(ts_to_min)

    # Primary lag: exact (geohash, timestamp)
    lookup_exact = train_split.groupby(["geohash", "timestamp"])["demand"].mean().to_dict()
    val_or_test["exact_lag_demand"] = val_or_test.apply(
        lambda r: lookup_exact.get((r["geohash"], r["timestamp"]), np.nan), axis=1)
    if verbose:
        print(f"      Exact lag done ({_time.time()-t0:.1f}s)")

    # Fuzzy lag: vectorized approach using pre-computed minute averages per geohash
    if verbose:
        t1 = _time.time()
        print("      Computing fuzzy lag...")
    gh_minute_avg = (
        train_split.groupby(["geohash", "_minutes"])["demand"]
        .mean()
        .reset_index()
    )
    # Build a dict for fast lookup: (geohash, minute) -> avg_demand
    fuzzy_dict = {}
    for _, row in gh_minute_avg.iterrows():
        fuzzy_dict[(row["geohash"], int(row["_minutes"]))] = row["demand"]

    fuzzy_results = []
    for _, row in val_or_test.iterrows():
        gh = row["geohash"]
        target_min = int(row["_minutes"])
        window_vals = []
        for offset in range(-30, 31):
            key = (gh, target_min + offset)
            if key in fuzzy_dict:
                window_vals.append(fuzzy_dict[key])
        fuzzy_results.append(np.mean(window_vals) if window_vals else np.nan)
    val_or_test["fuzzy_lag_demand"] = fuzzy_results
    if verbose:
        print(f"      Fuzzy lag done ({_time.time()-t1:.1f}s)")

    # Secondary fallback: (geohash, hour) average
    if verbose:
        t2 = _time.time()
        print("      Computing hour lag...")
    lookup_hour = train_split.groupby(["geohash", "hour"])["demand"].mean().to_dict()
    val_or_test["hour_lag_demand"] = val_or_test.apply(
        lambda r: lookup_hour.get((r["geohash"], r["hour"]), np.nan), axis=1)
    if verbose:
        print(f"      Hour lag done ({_time.time()-t2:.1f}s)")

    # Combined lag cascade
    val_or_test["combined_lag"] = val_or_test["exact_lag_demand"].fillna(
        val_or_test["fuzzy_lag_demand"]
    ).fillna(val_or_test["hour_lag_demand"])

    # Missingness indicator
    val_or_test["is_lag_missing"] = val_or_test["exact_lag_demand"].isna().astype(int)

    val_or_test.drop(columns=["_minutes"], inplace=True, errors="ignore")

    if verbose:
        exact_cov = val_or_test["exact_lag_demand"].notna().sum()
        combined_cov = val_or_test["combined_lag"].notna().sum()
        total = len(val_or_test)
        print(f"      Total lag time: {_time.time()-t0:.1f}s")
        print(f"    Exact lag:    {exact_cov}/{total} ({exact_cov/total*100:.1f}%)")
        print(f"    Combined:     {combined_cov}/{total} ({combined_cov/total*100:.1f}%)")

    return val_or_test


def build_geohash_stats(train_split: pd.DataFrame, val_or_test: pd.DataFrame) -> pd.DataFrame:
    """Geohash demand statistics from training data."""
    stats = train_split.groupby("geohash")["demand"].agg(
        ["mean", "std", "median", "count"]
    ).reset_index()
    stats.columns = ["geohash", "geo_demand_mean", "geo_demand_std",
                     "geo_demand_median", "geo_demand_count"]

    global_mean = train_split["demand"].mean()
    global_std = train_split["demand"].std()
    global_median = train_split["demand"].median()

    # Drop existing geo_demand columns to allow re-calling
    for col in ["geo_demand_mean", "geo_demand_std", "geo_demand_median", "geo_demand_count"]:
        if col in val_or_test.columns:
            val_or_test = val_or_test.drop(columns=[col])

    val_or_test = val_or_test.merge(stats, on="geohash", how="left")
    val_or_test["geo_demand_mean"] = val_or_test["geo_demand_mean"].fillna(global_mean)
    val_or_test["geo_demand_std"] = val_or_test["geo_demand_std"].fillna(global_std)
    val_or_test["geo_demand_median"] = val_or_test["geo_demand_median"].fillna(global_median)
    val_or_test["geo_demand_count"] = val_or_test["geo_demand_count"].fillna(0)

    return val_or_test


# ── Full Pipeline ────────────────────────────────────────────

def apply_all_features(train_split: pd.DataFrame, val_or_test: pd.DataFrame,
                       include_lag: bool = True, include_clusters: bool = True,
                       verbose: bool = True) -> tuple:
    """Complete feature engineering pipeline.

    Applies temporal, spatial, clustering, rotation, Fourier, contextual,
    and interaction features to both DataFrames. Optionally adds lag features.

    Args:
        train_split: Training data (Day 48)
        val_or_test: Validation or test data
        include_lag: Whether to add lag features (for Model B)
        include_clusters: Whether to fit K-Means clusters
        verbose: Print progress

    Returns:
        (train_split, val_or_test) with all features added
    """
    if verbose:
        print("  Building features...")

    # Temporal
    for df in (train_split, val_or_test):
        add_temporal_features(df)
        add_fourier_harmonics(df, columns=["hour", "15_min_slot"], n_harmonics=2)

    # Spatial
    for df in (train_split, val_or_test):
        add_spatial_features(df)
        add_rotated_coordinates(df, angles=[15, 30, 45])
        add_distance_to_center(df)
        add_manhattan_distance(df)

    # Clustering (fit on train, transform both)
    if include_clusters:
        train_split, val_or_test = add_spatial_clusters(
            train_split, val_or_test, n_clusters_list=[10, 50]
        )

    # Contextual
    for df in (train_split, val_or_test):
        add_contextual_features(df)

    # Interaction keys
    for df in (train_split, val_or_test):
        add_interaction_keys(df)

    # Geohash statistics (fit on train, apply to both)
    if verbose:
        print("    Adding geohash demand statistics...")
    train_split = build_geohash_stats(train_split, train_split)
    val_or_test = build_geohash_stats(train_split, val_or_test)

    # Lag features (Model B only)
    if include_lag:
        if verbose:
            print("    Building lag features...")
        val_or_test = build_lag_features(train_split, val_or_test, verbose=verbose)

    return train_split, val_or_test


# ── Feature Lists ────────────────────────────────────────────

MODEL_A_FEATURES = {
    "cat": [
        # Core spatial
        "geohash", "geohash_prefix_4", "RoadType", "Weather", "LargeVehicles", "Landmarks",
        # Clusters
        "cluster_10", "cluster_50",
        # Temporal categoricals
        "day_of_week",
        # Contextual interactions
        "RoadType_x_hour", "Weather_x_Temp",
        # High-order spatio-temporal interaction keys
        "geo_hour", "geo_dow", "geo_slot", "geo_p4_hour",
        "cl10_hour", "cl10_dow", "cl10_slot",
        "cl50_hour", "cl50_dow", "cl50_slot",
        "rt_dow", "rt_hour", "wx_hour",
    ],
    "num": [
        # Time
        "hour", "minute", "minute_of_day", "15_min_slot",
        # Cyclical
        "hour_sin", "hour_cos", "slot_sin", "slot_cos",
        # Fourier harmonics
        "hour_sin_2", "hour_cos_2", "15_min_slot_sin_2", "15_min_slot_cos_2",
        # Spatial coordinates
        "latitude", "longitude",
        # Rotated coordinates
        "lat_rot_15", "lon_rot_15", "lat_rot_30", "lon_rot_30",
        "lat_rot_45", "lon_rot_45",
        # Distance (haversine in km)
        "dist_to_center", "manhattan_dist_to_center",
        # Contextual
        "Temperature",
        # Geohash statistics
        "geo_demand_mean", "geo_demand_std", "geo_demand_median", "geo_demand_count",
        # Graph embeddings (Node2Vec PCA)
        "n2v_pca_0", "n2v_pca_1", "n2v_pca_2", "n2v_pca_3",
        "n2v_pca_4", "n2v_pca_5", "n2v_pca_6", "n2v_pca_7",
        # FFT spectral features
        "fft_amp_0", "fft_amp_1", "fft_amp_2",
        "fft_phase_0", "fft_phase_1", "fft_phase_2",
        "fft_dominant_freq", "fft_spectral_energy",
        # Diffusion imputation features
        "imputed_lag", "imputed_lag_var",
    ],
}

MODEL_B_FEATURES = {
    "cat": ["geohash", "geohash_prefix_4"],
    "num": [
        "exact_lag_demand", "Temperature", "hour", "minute",
        "latitude", "longitude", "hour_sin", "hour_cos",
        # Graph embeddings
        "n2v_pca_0", "n2v_pca_1", "n2v_pca_2", "n2v_pca_3",
        # FFT features
        "fft_amp_0", "fft_spectral_energy",
        # Imputation uncertainty
        "imputed_lag_var",
    ],
}
