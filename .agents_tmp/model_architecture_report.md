# Model Architecture Reviewer: Model Architecture Evaluation

## Focus Area
Model Architecture Evaluation

## Architecture Score: 7/10

### Critical Issues
1. **Model A Meta-Ensemble Underperformance** (scripts/run_pipeline_v8.py)
   - Meta-ensemble achieves ~55% validation score
   - Most value comes from Model B (lag specialist) at 99%
   - Severity: HIGH (architecture could be simplified)
   - Recommendation: Consider focusing on Model B approach only

### High Priority Issues
1. **Dual CatBoost Training Duplication**
   - Training CatBoost twice (once for Model A, once for Model B)
   - Could share early stopping results
   - Severity: MEDIUM

2. **LightGBM Unused for Final Prediction**
   - LightGBM trained but only CatBoost used for test prediction
   - Potential for better ensemble
   - Severity: MEDIUM

### Medium Priority Issues
1. **Hard-coded Feature Selection**
   - MODEL_A_FEATURES and MODEL_B_FEATURES are hardcoded
   - Could benefit from automated feature selection
   - Severity: LOW

### Low Priority Issues
1. **No Model Averaging on Test**
   - Final prediction uses single CatBoost model
   - Could ensemble multiple seeds
   - Severity: LOW

### Summary
- Total issues: 5
- Main concern is Model A's complexity vs. value
- Consider simplifying to focus on lag-based approach
