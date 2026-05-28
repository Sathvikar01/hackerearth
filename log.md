# Project Log

## 2026-05-28 — Model A Failure Analysis & Fixes

### Root Cause Analysis

**Issue 1: Test Suite Blocked (ImportError)**
- `src/toroidal.py` imports `TOROIDAL_N`, `TOROIDAL_GRID_SIZE`, `TEMPORAL_STATES` from `src/config.py`
- These constants were completely missing from config.py
- Result: `pytest tests/` fails at collection time with ImportError

**Issue 2: CatBoost Pool Predict Bug (Model A Crash)**
- In `src/models.py` lines 48 and 54: `model.predict(X_val)` called on raw Pandas DataFrame
- CatBoost requires categorical features to be wrapped in a `Pool` object
- Passing a DataFrame with string categorical columns directly to `predict()` throws `CatBoostError: categorical features must be specified`
- Same bug existed in `scripts/run_pipeline.py` lines 179 and 185

**Issue 3: Code Duplication**
- `scripts/run_pipeline.py` completely re-implements `train_model_a` instead of importing from `src/models.py`
- This caused configuration drift (different hyperparams, feature handling)

### Fixes Applied

1. **config.py**: Added `TOROIDAL_N=16`, `TOROIDAL_GRID_SIZE=256`, `TEMPORAL_STATES=168`
2. **models.py**: Changed `model.predict(X_val)` → `model.predict(val_pool)` in both has_target and no-target branches
3. **run_pipeline.py**: Same Pool fix applied to the duplicated `train_model_a`

### Verification
- [x] `pytest tests/` — 22/22 passed (0.89s)
- [x] All imports verified (models.py, config.py, toroidal.py)
- [x] Git committed and pushed to GitHub (commits c1c3269, 53bfcd2)
- [x] `python scripts/run_pipeline.py` — runs end-to-end successfully

### Pipeline Results (Post-Fix)
- Model A (Global Learner): **52.75** (was 52.41 before fix)
- Model B (Lag Specialist): **95.25** (was 95.13 before fix)
- Final Blended: **93.74** (was 93.14 before fix)
- The Pool fix eliminated the CatBoost predict crash and slightly improved scores
