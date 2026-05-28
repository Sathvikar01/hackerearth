# Lessons Learned

## 1. Feature Engineering > Model Architecture for Tabular Data
The single biggest improvement (Model A: 52.75 → 72.51) came from better features, not a better model. K-Means spatial clusters, rotated coordinates, and interaction keys gave CatBoost the information it needed to learn spatio-temporal patterns without lag data.

**Rule**: Invest heavily in feature engineering before trying complex architectures.

## 2. Interaction Keys Are the "Silver Bullet" for Cold-Start Prediction
Without historical lag, the model needs to know "How busy is this location at this time?" Interaction keys like `geo_hour` and `cluster_dow` act as lookup keys into CatBoost's native ordered target encoding, which effectively gives the model a localized historical average without data leakage.

**Rule**: When lag is unavailable, create combined categorical keys (spatial x temporal) and let CatBoost's native TE handle the encoding.

## 3. CatBoost Native TE > Manual Target Encoding
The manual Bayesian Target Encoder was removed in favor of CatBoost's built-in ordered target encoding. CatBoost's approach is mathematically proven to prevent leakage and handles high-cardinality categoricals better.

**Rule**: Don't reinvent the wheel — use CatBoost's `cat_features` parameter for categorical encoding.

## 4. Spatial Clusters Beat Grid Mappings
K-Means clusters on lat/lon outperform toroidal grid traversals because:
- Clusters are data-driven (boundaries follow actual data density)
- No edge artifacts from grid wrapping
- Multiple K values capture different spatial scales

**Rule**: Use K-Means or DBSCAN for spatial grouping instead of arbitrary grid systems.

## 5. Rotated Coordinates Help Trees Split Diagonally
Tree-based models make orthogonal splits. For spatial data, this means they can only create rectangular decision boundaries. Rotating coordinates by 15°, 30°, 45° gives the model the ability to create diagonal boundaries, which better match real-world spatial patterns.

**Rule**: Always add rotated coordinates when using tree models with spatial data.

## 6. CatBoost Pool Strictness
When calling `model.predict()`, always pass a `Pool` object with `cat_features` indices set. Passing a raw DataFrame with string columns will crash or silently produce wrong results.

**Rule**: Always wrap DataFrames in `Pool(X, cat_features=cat_indices)` before `predict()`.

## 7. DRY Principle — Avoid Code Duplication
Code duplication between `src/models.py` and `scripts/run_pipeline.py` caused bugs to be fixed in one place but not the other.

**Rule**: Import shared logic from source modules. Never copy-paste core functions.

## 8. Safe Parameter Stripping
When retraining on full data (no eval set), strip `early_stopping_rounds` to avoid CatBoost errors.

**Rule**: Use a dedicated "full train params" config or explicitly strip eval-dependent params.

## 9. Blending Strategy Matters
With W=1.0 for lag rows, the blended score (95.18) is higher than any weighted blend because Model B (95.25) is significantly more accurate than Model A (72.51) for rows with lag. The optimal strategy is: trust the lag specialist 100% when lag is available.

**Rule**: When one model is dramatically better for a subset, use hard switching (W=1.0) rather than soft blending.
