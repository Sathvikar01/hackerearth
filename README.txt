DUAL-BRANCH SPATIO-TEMPORAL LAG ARCHITECTURE (v5)
===================================================

COMPETITION: HackerEarth Traffic Demand Prediction
METRIC: score = max(0, 100 * R^2)
VALIDATION SCORE: 95.18 (Day 49 holdout — honest chronological validation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dual-Branch Architecture:
  - Model A (Global Learner): Predicts demand using spatial + temporal
    features WITHOUT historical lag. Handles cold-start scenarios.
  - Model B (Lag Specialist): Predicts demand using exact/fuzzy/hour
    lag features from previous days. Only operates on rows with lag.
  - Blending: W=1.0 for lag rows (100% Model B), Model A for no-lag rows.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL A: GLOBAL LEARNER (v5 Upgrades)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. K-Means Spatial Clusters (K=10, K=50)
   - Groups geohashes into macro-regions and micro-neighborhoods
   - Data-driven boundaries (no grid artifacts)
   - Multiple K values capture different spatial scales

2. Rotated Coordinates (15, 30, 45 degrees)
   - Enables tree models to carve diagonal spatial boundaries
   - lat_rot = lat*cos(theta) - lon*sin(theta)
   - lon_rot = lat*sin(theta) + lon*cos(theta)

3. Higher-Order Fourier Harmonics
   - hour_sin_2, hour_cos_2 (12-hour cycle)
   - 15_min_slot_sin_2, 15_min_slot_cos_2 (8-hour cycle)
   - Captures complex traffic wave patterns

4. High-Order Interaction Keys (15+ features)
   - geo_hour: geohash + hour (localized hourly average)
   - geo_dow: geohash + day_of_week
   - cluster_hour: cluster + hour (neighborhood rush hour)
   - rt_hour: RoadType + hour (highway rush hour)
   - wx_hour: Weather + hour
   - And more...
   - CatBoost handles these via native ordered target encoding

5. Distance to Center
   - Euclidean distance from geographic center of dataset

6. Geohash Demand Statistics
   - Mean, std, median, count per geohash from training data

CatBoost Hyperparameters:
  iterations: 3000, learning_rate: 0.03, depth: 8, l2_leaf_reg: 3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL B: LAG SPECIALIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Lag Feature Cascade:
  1. Primary: exact (geohash, timestamp) match — 88.9% test coverage
  2. Fuzzy: +/- 30 minute rolling window average
  3. Secondary: (geohash, hour) average
  4. Combined coverage: 97.2% of test rows

Features: geohash, geohash_prefix_4, exact_lag_demand, Temperature,
          hour, minute, latitude, longitude, hour_sin, hour_cos

CatBoost Hyperparameters:
  iterations: 1000, learning_rate: 0.05, depth: 6, l2_leaf_reg: 5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model A (Global Learner):  72.51  (no lag, spatio-temporal features)
  Model B (Lag Specialist):  95.25  (lag rows only)
  Final Blended:             95.18  (full Day 49 holdout)

  Blending: W=1.0 (100% Model B for lag rows)
  Exact lag coverage: Val 81.6%, Test 88.9%
  Combined coverage:  Val 94.2%, Test 97.2%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPROVEMENT HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  v4 (toroidal + manual TE):  Model A=52.75, Blended=93.74
  v5 (clusters + interactions): Model A=72.51, Blended=95.18

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOLS & LIBRARIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Python 3.11
- pandas, numpy, scikit-learn
- catboost (CatBoostRegressor)
- pygeohash (geohash decoding)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILES INCLUDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- README.txt              (this file)
- pipeline.ipynb          (Jupyter notebook v1)
- pipeline_v2.ipynb       (Jupyter notebook v2)
- src/config.py           (constants, model params)
- src/data_loader.py      (data loading, chronological split)
- src/features.py         (feature factory v5)
- src/target_encoder.py   (manual Bayesian OOF encoder)
- src/models.py           (dual-model CatBoost)
- src/blending.py         (dynamic blending)
- src/utils.py            (scoring, submission)
- scripts/run_pipeline.py (full pipeline v5)
- tests/test_features.py  (unit tests)
- log.md                  (development log)
- lesson.md               (engineering lessons)
