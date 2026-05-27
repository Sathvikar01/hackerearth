DUAL-BRANCH SPATIO-TEMPORAL LAG ARCHITECTURE
==============================================

COMPETITION: HackerEarth Traffic Demand Prediction
METRIC: score = max(0, 100 * R²)
VALIDATION SCORE: 90.69 (Day 49 holdout — honest chronological validation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY INSIGHT & LESSONS LEARNED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Previous approach used KFold(shuffle=True) which created catastrophic
data leakage — the lookup feature saw the same (geohash, timestamp) in
both train and validation folds, inflating CV to 95.81 while actual
test score was only 72.95.

FIX: Chronological validation — Day 48 as train, Day 49 as validation.
This perfectly simulates the Day 48→Day 49 temporal shift of the test set.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE: 6-STAGE DUAL-BRANCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 1-2: Feature Factory
  - Temporal: hour, minute, 15_min_slot, sin/cos cyclical encodings
  - Spatial: geohash → lat/lon (pygeohash), prefix_3, prefix_4
  - Contextual: RoadType×hour, Weather×Temperature interactions
  - Golden Lag: exact_lag_demand from Day 48 via (geohash, timestamp) match

Stage 3: Leakage-Safe Target Encoding
  - Manual Bayesian Target Encoder (no external dependencies)
  - Formula: (count * cat_mean + m * global_mean) / (count + m)
  - Fit ONLY on train split, transform val/test
  - Encoded: geohash, geo+slot, geo_prefix4+hour

Stage 4: Dual-Model Training (CatBoost)
  - Model A (Global Learner): All features EXCEPT lag, trained on Day 48
  - Model B (Lag Specialist): Lag + stabilizers, trained on Day 49 lag rows
  - CatBoostRegressor with native categorical handling

Stage 5: Dynamic Blending
  - W * Model_B + (1-W) * Model_A for rows with lag
  - Model_A only for rows without lag
  - W optimized via np.linspace(0.5, 1.0, 51) on Day 49 validation
  - Optimal W = 1.0 (100% Model B for lag rows)

Stage 6: Final Prediction
  - Retrain on all train data (Day 48 + 49)
  - Predict test, clip negatives to 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model A (Global Learner):  52.41  (no lag, general patterns)
  Model B (Lag Specialist):  95.06  (lag rows only)
  Final Blended:             90.69  (full Day 49 holdout)

  Lag coverage: Val 81.6%, Test 88.9%

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
- pipeline_v2.ipynb       (Jupyter notebook)
- src/config.py           (constants)
- src/data_loader.py      (data loading, chronological split)
- src/features.py         (feature factory)
- src/target_encoder.py   (manual Bayesian OOF encoder)
- src/models.py           (dual-model CatBoost)
- src/blending.py         (dynamic blending)
- src/utils.py            (scoring, submission)
- scripts/run_pipeline.py (full pipeline)
