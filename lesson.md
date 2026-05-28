# Lessons Learned

## 1. Graph Embeddings Capture Hidden Spatial Correlations
Node2Vec embeddings learned that geohashes far apart but with similar traffic patterns (e.g., two business districts) should be connected. This is impossible with K-Means clusters alone. The 8 PCA-reduced graph embedding features improved Model B from 95.25 to 96.70.

**Rule**: Always add graph-based spatial embeddings when working with spatial data. Distance-based clustering misses hidden correlations.

## 2. FFT Features Reveal Periodic Structure
Extracting dominant frequencies via FFT gave the model direct access to periodic demand patterns (rush hours, daily cycles). This is more informative than raw sin/cos harmonics because FFT finds the *actual* dominant frequencies in the data.

**Rule**: Use FFT to extract data-driven periodic features rather than assuming fixed harmonic frequencies.

## 3. Meta-Ensemble Stacking Beats Any Single Model
The Bayesian Ridge meta-learner (81.48) combined CatBoost (55.80) and LightGBM (61.36) predictions. Each base model captures different patterns — CatBoost excels at categorical interactions, LightGBM at numerical gradients. The meta-learner learns when to trust each.

**Rule**: Always stack multiple diverse models. The meta-learner will find the optimal combination.

## 4. Diffusion Imputation Provides Uncertainty for Free
The diffusion imputer generates multiple samples, giving us both a mean imputation and a variance. This uncertainty flows into the GBDT as a feature, letting the model learn to distrust imputed values when they're uncertain.

**Rule**: When imputing missing data, always generate multiple samples and pass the variance downstream.

## 5. Inverse-Variance Weighting Prevents Over-Trusting Imputed Data
Sample weights inversely proportional to imputation variance ensure the GBDT prioritizes rows with real lag data over rows with imputed lag.

**Rule**: Use inverse-variance sample weighting when mixing real and imputed data.

## 6. Feature Engineering > Model Architecture for Tabular Data
The single biggest improvement (Model A: 52.75 → 72.51) came from better features, not a better model. K-Means spatial clusters, rotated coordinates, and interaction keys gave CatBoost the information it needed.

**Rule**: Invest heavily in feature engineering before trying complex architectures.

## 7. Interaction Keys Are the "Silver Bullet" for Cold-Start Prediction
Without historical lag, the model needs to know "How busy is this location at this time?" Interaction keys like `geo_hour` and `cluster_dow` act as lookup keys into CatBoost's native ordered target encoding.

**Rule**: When lag is unavailable, create combined categorical keys (spatial x temporal).

## 8. CatBoost Native TE > Manual Target Encoding
The manual Bayesian Target Encoder was removed in favor of CatBoost's built-in ordered target encoding.

**Rule**: Don't reinvent the wheel — use CatBoost's `cat_features` parameter.

## 9. Spatial Clusters Beat Grid Mappings
K-Means clusters on lat/lon outperform toroidal grid traversals because clusters are data-driven and have no edge artifacts.

**Rule**: Use K-Means or DBSCAN for spatial grouping instead of arbitrary grid systems.

## 10. Rotated Coordinates Help Trees Split Diagonally
Rotating coordinates by 15°, 30°, 45° gives tree models the ability to create diagonal spatial boundaries.

**Rule**: Always add rotated coordinates when using tree models with spatial data.

## 11. CatBoost Pool Strictness
When calling `model.predict()`, always pass a `Pool` object with `cat_features` indices set.

**Rule**: Always wrap DataFrames in `Pool(X, cat_features=cat_indices)` before `predict()`.

## 12. DRY Principle
Code duplication between modules caused bugs to be fixed in one place but not the other.

**Rule**: Import shared logic from source modules.

## 13. Blending Strategy Matters
With W=1.0 for lag rows, the blended score is higher than any weighted blend because Model B is dramatically more accurate for rows with lag.

**Rule**: When one model is dramatically better for a subset, use hard switching (W=1.0).
