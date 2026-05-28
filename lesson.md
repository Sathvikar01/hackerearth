# Lessons Learned

## 1. BallTree Replaces O(N²) Spatial Lookups
The nested `iterrows()` loop for distance-based graph construction was O(N²). BallTree with haversine metric does the same in O(N log N). For 1241 geohashes, this reduced edges from 278K to 12.7K while finding all pairs within 2km.

**Rule**: Always use BallTree or KDTree for spatial proximity queries.

## 2. Behavioral Edges Beat Physical Distance
Pearson-correlated demand patterns create edges between geohashes that *behave* the same (e.g., two business districts), even if they're far apart. This captures hidden spatial correlations that physical distance alone misses.

**Rule**: Add behavioral/correlation-based edges to spatial graphs.

## 3. FFT Must Be Leakage-Safe
Computing FFT on the entire dataset (including Day 49) leaks future data into training. The fix: compute FFT strictly on Day 48 and map forward.

**Rule**: Always apply strict chronological cutoffs for any feature computed from historical data.

## 4. Soft-Blending Prevents Discontinuities
The hard switch (W=1.0 for all lag rows) creates jagged prediction curves. Soft-blending using `W = 1/(1+var)` normalized to [0.5, 1.0] provides a smooth transition based on imputation confidence.

**Rule**: Replace hard switches with variance-weighted soft blends.

## 5. Diffusion Imputer Variance Is Naturally Small
When the denoising MLP converges well, the variance across N samples is tiny (~0.0001). This means the soft-blending degenerates to hard-blending, which is actually correct — high confidence means we should trust the lag.

**Rule**: If variance is naturally small, soft-blending ≈ hard-blending. That's OK.

## 6. Haversine Beats Euclidean for Coordinates
Flat Euclidean distance on lat/lon degrees is mathematically wrong (1° lat ≠ 1° lon in km). Haversine gives proper kilometer distances.

**Rule**: Always use haversine for geographic distance calculations.

## 7. Feature Pruning Removes Noise
Dropping the bottom 15% of features by CatBoost importance removes noisy/redundant features that confuse the meta-learner.

**Rule**: Always prune low-importance features before stacking.

## 8. FastKNN Is a Reliable Fallback
When the diffusion imputer fails or is too slow, FastKNN provides deterministic, fast imputation with reasonable accuracy.

**Rule**: Always have a deterministic fallback for stochastic imputers.

## 9. Graph Embeddings Capture Hidden Spatial Correlations
Node2Vec embeddings learned that geohashes far apart but with similar traffic patterns should be connected. This is impossible with K-Means clusters alone.

**Rule**: Always add graph-based spatial embeddings when working with spatial data.

## 10. FFT Features Reveal Periodic Structure
Extracting dominant frequencies via FFT gave the model direct access to periodic demand patterns.

**Rule**: Use FFT to extract data-driven periodic features rather than assuming fixed harmonic frequencies.

## 11. Meta-Ensemble Stacking Beats Any Single Model
The Bayesian Ridge meta-learner combined CatBoost and LightGBM predictions, each capturing different patterns.

**Rule**: Always stack multiple diverse models.

## 12. Inverse-Variance Weighting Prevents Over-Trusting Imputed Data
Sample weights inversely proportional to imputation variance ensure the GBDT prioritizes rows with real lag data.

**Rule**: Use inverse-variance sample weighting when mixing real and imputed data.

## 13. Interaction Keys Are the "Silver Bullet" for Cold-Start Prediction
Combined categorical keys (spatial × temporal) act as lookup keys into CatBoost's native ordered target encoding.

**Rule**: When lag is unavailable, create combined categorical keys.
