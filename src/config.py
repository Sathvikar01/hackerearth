"""Configuration and constants for ST-Diffusion Meta-Ensemble Architecture."""
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
W_GRID_SIZE = 51

# ── CatBoost: Base Model (for meta-ensemble) ────────────────
CATBOOST_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": SEED,
    "verbose": 0,
    "early_stopping_rounds": 200,
    "loss_function": "RMSE",
}

# ── LightGBM: Base Model (for meta-ensemble) ───────────────
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
}

# ── Model B: Lag Specialist (CatBoost, separate) ───────────
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

# ── Diffusion Imputer ──────────────────────────────────────
DIFFUSION_PARAMS = {
    "n_samples": 10,
    "epochs": 30,
    "lr": 1e-3,
    "batch_size": 1024,
    "noise_levels": [0.1, 0.3, 0.5, 0.8, 1.0],
}

# ── Imputer Mode ────────────────────────────────────────────
USE_FAST_IMPUTER = False  # True = FastKNN (deterministic), False = Diffusion (stochastic)

# ── Graph Embeddings ───────────────────────────────────────
GRAPH_PARAMS = {
    "dimensions": 16,
    "n_pca": 8,
    "walk_length": 20,
    "num_walks": 50,
    "p": 1.0,
    "q": 1.0,
}
