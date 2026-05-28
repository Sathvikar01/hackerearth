# Project Log

## 2026-05-28 — Model A Architecture Overhaul (v4 → v5)

### Problem Statement
Model A (Global Learner) scored 52.75 on validation — far below the target of 95. The old architecture relied on toroidal traversal for spatial features, which was inefficient and introduced false spatial neighborhoods.

### Root Cause Analysis
1. **Toroidal traversal** created artificial grid mappings with edge artifacts
2. **Simple geohash prefixes** (3/4 chars) were too coarse for spatial differentiation
3. **No spatial clustering** — the model couldn't learn neighborhood-level patterns
4. **No rotated coordinates** — tree splits on lat/lon can't carve diagonal boundaries
5. **No Fourier harmonics** — only basic sin/cos for hour/slot
6. **No interaction keys** — CatBoost had to discover all spatio-temporal interactions from scratch
7. **Manual target encoding** — less effective than CatBoost's native ordered encoding

### Solution: v5 Architecture
Complete feature engineering overhaul with 6 key upgrades:

1. **K-Means Spatial Clusters (K=10, K=50)**: Groups geohashes into macro-regions and micro-neighborhoods
2. **Rotated Coordinates (15°, 30°, 45°)**: Enables diagonal spatial boundary splits
3. **Higher-Order Fourier Harmonics**: Captures 12-hour and 8-hour traffic cycles
4. **High-Order Interaction Keys**: 15+ combined categorical features (geo_hour, cluster_dow, rt_hour, wx_hour, etc.)
5. **CatBoost Native Target Encoding**: Removed manual Bayesian TE — CatBoost's ordered encoding is superior
6. **Removed Toroidal Traversal**: Replaced with spatial clusters

### Model A Hyperparameter Changes
- iterations: 1000 → 3000
- learning_rate: 0.05 → 0.03
- depth: 6 → 8
- l2_leaf_reg: 5 → 3

### Results

| Metric | v4 (Before) | v5 (After) | Change |
|--------|-------------|------------|--------|
| Model A | 52.75 | **72.51** | +37.5% |
| Model B | 95.25 | 95.25 | — |
| **Blended** | 93.74 | **95.18** | **+1.5%** |

### Verification
- [x] 19/19 pytest tests pass (test_features.py)
- [x] Pipeline runs end-to-end successfully
- [x] Submission.csv generated (41778 predictions)
- [x] Git committed and pushed

### Files Changed
- `src/features.py` — Complete rewrite: spatial clusters, rotated coords, Fourier harmonics, interaction keys
- `src/config.py` — New MODEL_A_PARAMS, MODEL_B_PARAMS, removed toroidal constants
- `src/models.py` — Updated to use new params, removed toroidal imports
- `scripts/run_pipeline.py` — v5 pipeline with new feature engineering
- `tests/test_features.py` — New test suite (replaced test_toroidal.py)
- `src/toroidal.py` — **Deleted** (replaced by spatial clusters)

---

## 2026-05-28 — Initial Model A Failure Analysis & Fixes

### Root Cause Analysis

**Issue 1: Test Suite Blocked (ImportError)**
- `src/toroidal.py` imported `TOROIDAL_N`, `TOROIDAL_GRID_SIZE`, `TEMPORAL_STATES` from `src/config.py`
- These constants were completely missing from config.py

**Issue 2: CatBoost Pool Predict Bug (Model A Crash)**
- `model.predict(X_val)` called on raw Pandas DataFrame without Pool wrapping
- CatBoost requires categorical features wrapped in Pool objects

### Fixes Applied
1. Fixed CatBoost Pool predict bug in `src/models.py` and `scripts/run_pipeline.py`
2. All 22 pytest tests passed
