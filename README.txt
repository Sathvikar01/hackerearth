DUAL-BRANCH SPATIO-TEMPORAL LAG ARCHITECTURE (OPTIMIZED)
=========================================================

COMPETITION: HackerEarth Traffic Demand Prediction
METRIC: score = max(0, 100 * R²)
VALIDATION SCORE: 93.14 (Day 49 holdout — honest chronological validation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3 KEY OPTIMIZATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. JOIN GRANULARITY: Already at minute-level (H:M format)
   - Join keys: (geohash, timestamp) where timestamp = "H:M"
   - Coverage: 88.9% of test rows have exact match from Day 48
   - This is the finest granularity available in the data

2. HARDCODED W=1.0: Trust the lag 100% for rows with lag
   - For rows with lag feature: Final = Model_B (Lag Specialist)
   - For rows without lag: Final = Model_A (Global Learner)
   - This eliminated the blending dilution problem

3. SECONDARY FALLBACK LAG: geohash+hour average for missing exact lag
   - Primary: exact (geohash, timestamp) match — 88.9% test coverage
   - Fallback: (geohash, hour) average from Day 48
   - Combined coverage: 96.2% of test rows (up from 88.9%)
   - For validation: 92.8% (up from 81.6%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 1-2: Feature Factory
  - Temporal: hour, minute, 15_min_slot, sin/cos cyclical encodings
  - Spatial: geohash → lat/lon, prefix_3, prefix_4
  - Contextual: RoadType×hour, Weather×Temperature
  - Golden Lag: exact_lag_demand + hour_lag_demand + combined_lag

Stage 3: Leakage-Safe Target Encoding
  - Manual Bayesian Target Encoder (no external dependencies)
  - Fit ONLY on Day 48, transform val/test
  - Encoded: geohash, geo+slot, geo_prefix4+hour

Stage 4: Dual-Model Training (CatBoost)
  - Model A (Global Learner): All features except lag → 52.41
  - Model B (Lag Specialist): Lag + stabilizers → 95.13
  - Trained on Day 49 rows with lag (92.8% coverage)

Stage 5: Dynamic Blending (W=1.0)
  - W=1.0 for rows with lag: 100% Model B
  - Model A only for rows without lag
  - Blended score: 93.14

Stage 6: Final Prediction
  - Retrain on all train data (Day 48 + 49)
  - Rebuild lag with full train data
  - Test combined lag coverage: 96.3%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model A (Global Learner):  52.41  (no lag, general patterns)
  Model B (Lag Specialist):  95.13  (lag rows only)
  Final Blended:             93.14  (full Day 49 holdout)

  Exact lag coverage:  Val 81.6%, Test 88.9%
  Combined coverage:   Val 92.8%, Test 96.3%

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
