# Data Leakage Analyzer: Data Leakage Detection

## Focus Area
Data Leakage Detection

## Leak Score: 6/10

### Critical Issues
NONE - No critical data leakage detected

### High Priority Issues
1. **Lag Feature Temporal Leakage** (src/features.py)
   - The `exact_lag_demand` feature uses Day 48 demand for Day 49 prediction
   - This IS the target variable shifted by 1 day
   - HOWEVER: This is intentional for time-series prediction (lag features)
   - Severity: MEDIUM (acceptable for this problem type)
   - Recommendation: Current approach is correct

### Medium Priority Issues
1. **Validation Split Method**
   - Using Day 48 for training, Day 49 for validation
   - This is correct chronological split - no leakage
   - Severity: INFO

2. **Imputation on Validation Data**
   - Diffusion imputer trained on train_split, applied to val_split
   - This is correct - no leakage from imputation
   - Severity: INFO

### Low Priority Issues
1. **Graph Embeddings Fallback**
   - Using random embeddings when node2vec unavailable
   - Not a leakage issue, but less informative
   - Severity: LOW

### Summary
- Total issues: 2
- The lag feature leakage is intentional for time-series prediction
- No critical leakage detected
- Recommendation: Current approach is sound for this problem
