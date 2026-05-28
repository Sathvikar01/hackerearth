# Project Log

## 2026-05-28 — v6_final Refactor (6 Mandates)

### Mandates Implemented
1. **Graph Optimization**: BallTree O(N log N) haversine edges + Pearson behavioral edges (12,720 edges vs 278K)
2. **FFT Leakage Fix**: Strict Day 48 cutoff — FFT only computed on train_day <= 48
3. **Imputation Fallback**: FastKNN deterministic imputer as alternative to diffusion
4. **Soft-Blending**: W = 1/(1+var) normalized to [0.5, 1.0] — smooth uncertainty-weighted blend
5. **Feature Pruning**: CatBoost importance-based pruning (drop bottom 15%)
6. **Haversine Distance**: Replaced flat Euclidean with proper haversine (km) + Manhattan distance

### Results
| Metric | v6 | v6_final | Change |
|--------|-----|----------|--------|
| Meta-Ensemble | 81.48 | 79.72 | -1.76 |
| Model B | 96.70 | **96.84** | +0.14 |
| Blended | 96.67 | **96.81** | **+0.14** |

Note: Soft-blend = Hard-blend because diffusion variance is tiny (0.000078). The imputer is highly confident.

### New Features
- `manhattan_dist_to_center` — Manhattan distance via haversine
- `behavioral` graph method — Pearson-correlated demand patterns
- `FastKNNImputer` — deterministic fallback
- `soft_blend_predictions()` — variance-weighted blending

---

## 2026-05-28 — ST-Diffusion Meta-Ensemble (v5 → v6)

### Problem Statement
Model A scored 72.51 and Model B scored 95.25 in v5. The blended score was 95.18. To push beyond 98%, we needed:
1. Deep spatial representation learning (graph embeddings)
2. Frequency-domain temporal features (FFT)
3. Generative imputation for missing lags (diffusion)
4. Meta-ensemble stacking (CatBoost + LightGBM + Bayesian Ridge)

### Architecture: ST-Diffusion Meta-Ensemble (STD-ME)

**Phase 1: Deep Representation Learning**
- Node2Vec graph embeddings: Built geohash adjacency graph (1241 nodes, 278K edges), learned 16D embeddings, reduced to 8D via PCA
- FFT spectral features: Extracted dominant frequencies, amplitudes, phases per geohash
- 8 graph embedding features + 8 FFT features added

**Phase 2: Generative Imputation**
- Denoising MLP trained on rows with lag data
- Generates 10 samples for missing lags -> Imputed Mean + Variance
- Uncertainty-aware features: imputed_lag, imputed_lag_var, is_lag_imputed

**Phase 3: Meta-Ensemble Forecasting**
- Base Model 1: CatBoost (categorical interactions) -> 55.80
- Base Model 2: LightGBM (fast gradient boosting) -> 61.36
- Meta-Learner: Bayesian Ridge (stacked predictions) -> 81.48
- Inverse-variance sample weighting for imputation uncertainty

**Phase 4: Lag Specialist + Blending**
- Model B (CatBoost on lag rows) -> 96.70
- Blended: W=1.0 for lag rows -> **96.67**

### Results

| Metric | v5 | v6 | Change |
|--------|-----|-----|--------|
| Meta-Ensemble | — | 81.48 | New |
| Model B | 95.25 | 96.70 | +1.5% |
| **Blended** | 95.18 | **96.67** | **+1.5%** |

### New Modules
- `src/graph_embeddings.py` — Node2Vec spatial embeddings
- `src/temporal_fft.py` — FFT spectral features
- `src/diffusion_imputer.py` — Denoising MLP imputation + uncertainty
- `src/meta_ensemble.py` — CatBoost + LightGBM + Bayesian Ridge stacking

### Verification
- [x] All imports verified
- [x] Pipeline runs end-to-end
- [x] Submission.csv generated (41778 predictions)
- [x] Git committed and pushed

---

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
