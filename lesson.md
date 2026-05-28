# Lessons Learned

## 1. CatBoost Pool Strictness
CatBoost is extremely strict about categorical features. When calling `model.predict()`, you **must** pass a `Pool` object with `cat_features` indices set — passing a raw Pandas DataFrame with string columns will crash. This is a silent trap because `model.fit()` with a Pool works fine, but the predict call on a DataFrame looks deceptively correct.

**Rule**: Always wrap DataFrames in `Pool(X, cat_features=cat_indices)` before calling `predict()` or `fit()`.

## 2. DRY Principle — Avoid Code Duplication
`scripts/run_pipeline.py` duplicated `train_model_a` from `src/models.py` with slightly different parameters. This led to:
- Bugs fixed in one place but not the other
- Configuration drift between modules
- Maintenance burden

**Rule**: Import shared logic from source modules. Never copy-paste core training functions.

## 3. Centralized Configuration
Constants like `TOROIDAL_N` were expected by multiple modules but never defined in the config file. This is a classic "interface contract" violation — the toroidal module declared its dependency but the config didn't fulfill it.

**Rule**: When a module imports from config, ensure all referenced constants exist. Consider using a config validation step at startup.

## 4. Test-Driven Validation
The test suite was completely blocked by the ImportError. Running tests early would have caught the missing config constants immediately.

**Rule**: Always run `pytest` after any config change to verify import chains are intact.

## 5. Safe Parameter Stripping
When retraining on full data (no eval set), the code strips `early_stopping_rounds` with dict comprehension. If other eval-dependent params are added later, this will silently break.

**Rule**: Use a dedicated "full train params" config or explicitly whitelist params to keep.

## 6. Impact of Proper Pool Usage
The CatBoost Pool fix didn't just prevent crashes — it also slightly improved model scores (93.14 → 93.74). This suggests CatBoost may have been silently mishandling categorical features in the raw DataFrame path, leading to degraded predictions even when it didn't crash outright.
