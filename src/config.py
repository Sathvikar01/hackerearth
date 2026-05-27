"""Configuration and constants."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

TARGET = "demand"
ID_COL = "Index"
SEED = 42

CAT_FEATURES = ["geohash", "RoadType", "Weather", "LargeVehicles", "Landmarks"]

CATBOOST_PARAMS = {
    "iterations": 800,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 50,
    "loss_function": "RMSE",
}

TOROIDAL_N = 16
TOROIDAL_GRID_SIZE = TOROIDAL_N * TOROIDAL_N
TEMPORAL_STATES = 7 * 24
