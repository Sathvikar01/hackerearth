ST-DIFFUSION META-ENSEMBLE ARCHITECTURE (v6)
=============================================

COMPETITION: HackerEarth Traffic Demand Prediction
METRIC: score = max(0, 100 * R^2)
VALIDATION SCORE: 96.67 (Day 49 holdout — honest chronological validation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ST-Diffusion Meta-Ensemble (STD-ME):
A 4-phase architecture combining deep representation learning,
generative imputation, and meta-ensemble stacking.

Phase 1: Deep Representation Learning
  - Node2Vec graph embeddings (16D -> PCA 8D)
  - FFT spectral features (dominant frequencies, amplitudes, phases)
  - K-Means spatial clusters, rotated coordinates, Fourier harmonics

Phase 2: Generative Imputation (Diffusion Imputer)
  - Denoising MLP trained on rows with lag data
  - Generates N=10 samples for missing lags
  - Outputs: imputed_lag (mean), imputed_lag_var (uncertainty)
  - Inverse-variance sample weighting for downstream models

Phase 3: Meta-Ensemble Forecasting
  - Base Model 1: CatBoost (categorical interactions)
  - Base Model 2: LightGBM (fast gradient boosting)
  - Meta-Learner: Bayesian Ridge (stacked predictions)
  - Inverse-variance weighting prevents over-trusting imputed data

Phase 4: Lag Specialist + Blending
  - Model B: CatBoost trained on rows with real lag
  - Blending: W=1.0 for lag rows (100% Model B)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Meta-Ensemble (CB+LGB+BR):  81.48  (no-lag rows)
  Model B (Lag Specialist):    96.70  (lag rows only)
  Final Blended:               96.67  (full Day 49 holdout)

  Exact lag coverage: Val 81.6%, Test 88.9%
  Combined coverage:  Val 94.2%, Test 97.2%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPROVEMENT HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  v4 (toroidal + manual TE):     Blended = 93.74
  v5 (clusters + interactions):  Blended = 95.18
  v6 (STD-ME):                   Blended = 96.67

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW MODULES (v6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- src/graph_embeddings.py  — Node2Vec spatial embeddings
- src/temporal_fft.py      — FFT spectral features
- src/diffusion_imputer.py — Denoising MLP imputation + uncertainty
- src/meta_ensemble.py     — CatBoost + LightGBM + Bayesian Ridge

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOLS & LIBRARIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Python 3.11
- pandas, numpy, scikit-learn, scipy
- catboost, lightgbm
- torch (CPU), networkx, node2vec, gensim
- pygeohash

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- README.txt              (this file)
- src/config.py           (constants, model params)
- src/data_loader.py      (data loading, chronological split)
- src/features.py         (feature factory v6)
- src/graph_embeddings.py (Node2Vec spatial embeddings)
- src/temporal_fft.py     (FFT spectral features)
- src/diffusion_imputer.py(diffusion imputation + uncertainty)
- src/meta_ensemble.py    (CatBoost + LightGBM + Bayesian Ridge)
- src/models.py           (dual-model CatBoost)
- src/blending.py         (dynamic blending)
- src/utils.py            (scoring, submission)
- scripts/run_pipeline.py (full pipeline v6)
- tests/test_features.py  (unit tests)
- log.md                  (development log)
- lesson.md               (engineering lessons)
