"""Stage 3: Manual Out-of-Fold (OOF) Target Encoder with Bayesian Smoothing.

No external dependencies. Total transparency. Zero leakage.
"""
import numpy as np
import pandas as pd


class BayesianTargetEncoder:
    """OOF Target Encoder with Bayesian smoothing.

    For each fold, the encoding is computed from the training fold only,
    using the formula:

        encoded_value = (count * category_mean + m * global_mean) / (count + m)

    where m is the smoothing weight (higher = more smoothing toward global mean).
    """

    def __init__(self, columns: list, target: str = "demand", m: int = 10):
        self.columns = columns
        self.target = target
        self.m = m
        self.encodings_ = {}  # column -> {category: encoded_value}
        self.global_mean_ = None

    def fit_transform_oof(self, df: pd.DataFrame, fold_indices: list) -> pd.DataFrame:
        """Fit and transform using OOF strategy.

        Args:
            df: Full training DataFrame
            fold_indices: List of (train_idx, val_idx) tuples

        Returns:
            DataFrame with new encoded columns added
        """
        self.global_mean_ = df[self.target].mean()
        result = df.copy()

        for col in self.columns:
            encoded_col = f"{col}_te"
            result[encoded_col] = np.nan

            for train_idx, val_idx in fold_indices:
                train_fold = df.iloc[train_idx]
                val_fold = df.iloc[val_idx]

                # Compute Bayesian smoothed means from train fold
                stats = train_fold.groupby(col)[self.target].agg(["count", "mean"])
                stats["encoded"] = (
                    (stats["count"] * stats["mean"] + self.m * self.global_mean_)
                    / (stats["count"] + self.m)
                )

                # Map to val fold
                mapping = stats["encoded"].to_dict()
                result.loc[df.index[val_idx], encoded_col] = val_fold[col].map(mapping)

            # Fill any remaining NaN with global mean
            result[encoded_col] = result[encoded_col].fillna(self.global_mean_)

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using full training encodings.

        Args:
            df: Validation or test DataFrame

        Returns:
            DataFrame with new encoded columns added
        """
        result = df.copy()

        for col in self.columns:
            encoded_col = f"{col}_te"
            if col in self.encodings_:
                mapping = self.encodings_[col]
                result[encoded_col] = result[col].map(mapping)
                result[encoded_col] = result[encoded_col].fillna(self.global_mean_)
            else:
                result[encoded_col] = self.global_mean_

        return result

    def fit(self, df: pd.DataFrame) -> "BayesianTargetEncoder":
        """Fit on full training data (for final model)."""
        self.global_mean_ = df[self.target].mean()

        for col in self.columns:
            stats = df.groupby(col)[self.target].agg(["count", "mean"])
            stats["encoded"] = (
                (stats["count"] * stats["mean"] + self.m * self.global_mean_)
                / (stats["count"] + self.m)
            )
            self.encodings_[col] = stats["encoded"].to_dict()

        return self
