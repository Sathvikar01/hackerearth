"""Unit tests for the new feature engineering pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from src.features import (
    add_temporal_features, add_spatial_features, add_spatial_clusters,
    add_rotated_coordinates, add_distance_to_center, add_fourier_harmonics,
    add_contextual_features, add_interaction_keys,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking the dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "geohash": [f"dr5ru{j//10}" for j in range(n)],
        "hour": np.random.randint(0, 24, n),
        "minute": np.random.randint(0, 60, n),
        "minute_of_day": np.random.randint(0, 1440, n),
        "15_min_slot": np.random.randint(0, 96, n),
        "day_of_week": np.random.randint(0, 7, n),
        "RoadType": np.random.choice(["Highway", "Local", "Unknown"], n),
        "Weather": np.random.choice(["Clear", "Rain", "Cloudy"], n),
        "Temperature": np.random.uniform(10, 40, n),
        "LargeVehicles": np.random.randint(0, 5, n).astype(str),
        "Landmarks": np.random.randint(0, 3, n).astype(str),
    })


class TestTemporalFeatures:
    def test_hour_sin_cos_range(self, sample_df):
        df = add_temporal_features(sample_df.copy())
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_slot_sin_cos_range(self, sample_df):
        df = add_temporal_features(sample_df.copy())
        assert df["slot_sin"].between(-1, 1).all()
        assert df["slot_cos"].between(-1, 1).all()


class TestSpatialFeatures:
    def test_geohash_decoding(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert df["latitude"].between(-90, 90).all()
        assert df["longitude"].between(-180, 180).all()

    def test_geohash_prefix(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        assert "geohash_prefix_4" in df.columns
        assert all(len(str(x)) == 4 for x in df["geohash_prefix_4"])


class TestSpatialClusters:
    def test_cluster_columns_created(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        train = df.iloc[:80].copy()
        test = df.iloc[80:].copy()
        train, test = add_spatial_clusters(train, test, n_clusters_list=[10, 50])
        assert "cluster_10" in train.columns
        assert "cluster_50" in train.columns
        assert "cluster_10" in test.columns
        assert "cluster_50" in test.columns

    def test_cluster_values_are_strings(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        train = df.iloc[:80].copy()
        test = df.iloc[80:].copy()
        train, test = add_spatial_clusters(train, test, n_clusters_list=[10])
        assert train["cluster_10"].dtype == object
        assert test["cluster_10"].dtype == object

    def test_cluster_range(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        train = df.iloc[:80].copy()
        test = df.iloc[80:].copy()
        train, test = add_spatial_clusters(train, test, n_clusters_list=[10])
        assert all(0 <= int(x) < 10 for x in train["cluster_10"])
        assert all(0 <= int(x) < 10 for x in test["cluster_10"])


class TestRotatedCoordinates:
    def test_rotated_columns_created(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        df = add_rotated_coordinates(df, angles=[15, 30, 45])
        assert "lat_rot_15" in df.columns
        assert "lon_rot_15" in df.columns
        assert "lat_rot_30" in df.columns
        assert "lat_rot_45" in df.columns

    def test_rotated_values_finite(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        df = add_rotated_coordinates(df, angles=[45])
        assert np.isfinite(df["lat_rot_45"]).all()
        assert np.isfinite(df["lon_rot_45"]).all()


class TestDistanceToCenter:
    def test_distance_non_negative(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        df = add_distance_to_center(df)
        assert "dist_to_center" in df.columns
        assert (df["dist_to_center"] >= 0).all()

    def test_center_point_has_zero_distance(self):
        df = pd.DataFrame({"latitude": [10.0, 10.0], "longitude": [20.0, 20.0]})
        df = add_distance_to_center(df, center_lat=10.0, center_lon=20.0)
        assert all(abs(df["dist_to_center"]) < 1e-10)


class TestFourierHarmonics:
    def test_harmonic_columns_created(self, sample_df):
        df = add_fourier_harmonics(sample_df.copy(), columns=["hour"], n_harmonics=2)
        assert "hour_sin_2" in df.columns
        assert "hour_cos_2" in df.columns

    def test_harmonic_range(self, sample_df):
        df = add_fourier_harmonics(sample_df.copy(), columns=["hour", "15_min_slot"], n_harmonics=3)
        for col in ["hour_sin_2", "hour_cos_2", "15_min_slot_sin_2", "15_min_slot_cos_2"]:
            assert df[col].between(-1, 1).all(), f"{col} out of range"


class TestContextualFeatures:
    def test_contextual_columns(self, sample_df):
        df = add_contextual_features(sample_df.copy())
        assert "RoadType_x_hour" in df.columns
        assert "Weather_x_Temp" in df.columns


class TestInteractionKeys:
    def test_interaction_columns_created(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        df = add_spatial_clusters(df.iloc[:80].copy(), df.iloc[80:].copy(), [10, 50])[0]
        df = add_interaction_keys(df)
        assert "geo_hour" in df.columns
        assert "geo_dow" in df.columns
        assert "geo_slot" in df.columns
        assert "geo_p4_hour" in df.columns
        assert "cl10_hour" in df.columns
        assert "cl50_dow" in df.columns
        assert "rt_dow" in df.columns
        assert "wx_hour" in df.columns

    def test_interaction_values_are_strings(self, sample_df):
        df = add_spatial_features(sample_df.copy())
        df = add_spatial_clusters(df.iloc[:80].copy(), df.iloc[80:].copy(), [10])[0]
        df = add_interaction_keys(df)
        assert df["geo_hour"].dtype == object
        assert df["cl10_hour"].dtype == object


class TestFeatureLists:
    def test_model_a_features_exist(self):
        assert "cat" in MODEL_A_FEATURES
        assert "num" in MODEL_A_FEATURES
        assert len(MODEL_A_FEATURES["cat"]) > 0
        assert len(MODEL_A_FEATURES["num"]) > 0

    def test_model_b_features_exist(self):
        assert "cat" in MODEL_B_FEATURES
        assert "num" in MODEL_B_FEATURES

    def test_no_toroidal_features(self):
        all_features = MODEL_A_FEATURES["cat"] + MODEL_A_FEATURES["num"]
        for f in all_features:
            assert "toroid" not in f.lower()
            assert "grid" not in f.lower()
