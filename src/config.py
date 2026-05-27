"""Configuration and constants for the Traffic Demand Prediction pipeline."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

# Target
TARGET = "demand"
ID_COL = "Index"

# Categorical features for CatBoost
CAT_FEATURES = [
    "geohash",
    "geohash_prefix_3",
    "geohash_prefix_4",
    "RoadType",
    "Weather",
    "LargeVehicles",
    "Landmarks",
]

# Numeric features that are always included
BASE_NUMERIC_FEATURES = [
    "NumberofLanes",
    "Temperature",
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "RoadType_x_Lanes",
    "Weather_x_Temp",
]

# Phase 2 features
PHASE2_FEATURES = [
    "toroidal_dist_rush_hour",
    "toroidal_dist_weekly_peak",
]

# Phase 3 features
PHASE3_FEATURES = [
    "toroidal_phase",
    "toroidal_neighborhood_entropy",
    "toroidal_collision_frequency",
]

# CatBoost hyperparameters
CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5,
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 100,
    "loss_function": "RMSE",
    "eval_metric": "R2",
}

# Toroidal grid
TOROIDAL_N = 16
TOROIDAL_GRID_SIZE = TOROIDAL_N * TOROIDAL_N  # 256
TEMPORAL_STATES = 7 * 24  # 168

# Rush hours
RUSH_HOURS = [8, 18]

# Random seed
SEED = 42
