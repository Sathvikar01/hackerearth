"""Configuration and constants for Dual-Branch Architecture."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

TARGET = "demand"
ID_COL = "Index"
SEED = 42

# Validation: Day 48 = train, Day 49 = validation
TRAIN_DAY = 48
VAL_DAY = 49

# Blending weight search
W_GRID_SIZE = 51  # np.linspace(0.5, 1.0, 51)

# ── Model A: Global Learner (Upgraded) ──────────────────────
# Deep trees + interaction features + native CatBoost target encoding
MODEL_A_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 200,
    "loss_function": "RMSE",
}

# ── Model B: Lag Specialist ─────────────────────────────────
MODEL_B_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 50,
    "loss_function": "RMSE",
}
