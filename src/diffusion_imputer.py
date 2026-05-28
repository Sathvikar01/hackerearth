"""Diffusion Imputer with Fast Fallback for Missing Lag Features.

Two imputation strategies:
1. DiffusionImputer: Denoising MLP with uncertainty estimation (slow, stochastic)
2. FastKNNImputer: Deterministic KNN-based imputation (fast, deterministic)

The pipeline selects based on USE_FAST_IMPUTER flag in config.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor


class DenoisingMLP(nn.Module):
    """Small MLP that denoises lag features conditioned on context."""

    def __init__(self, context_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, noisy_lag, context):
        x = torch.cat([noisy_lag, context], dim=-1)
        return self.net(x)


class DiffusionImputer:
    """Denoising MLP imputer with uncertainty estimation."""

    def __init__(self, context_features: list, n_samples: int = 10,
                 noise_levels: list = None, epochs: int = 50,
                 lr: float = 1e-3, batch_size: int = 1024,
                 device: str = "cpu"):
        self.context_features = context_features
        self.n_samples = n_samples
        self.noise_levels = noise_levels or [0.1, 0.3, 0.5, 0.8, 1.0]
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.scaler = StandardScaler()
        self.lag_mean = 0.0
        self.lag_std = 1.0
        self.is_fitted = False

    def _prepare_context(self, df: pd.DataFrame) -> np.ndarray:
        available = [f for f in self.context_features if f in df.columns]
        return df[available].values.astype(np.float32)

    def fit(self, df: pd.DataFrame, lag_col: str = "exact_lag_demand"):
        mask = df[lag_col].notna()
        df_lag = df[mask].copy()

        if len(df_lag) < 100:
            print("    Diffusion Imputer: Not enough lag data, skipping")
            self.is_fitted = False
            return self

        context = self._prepare_context(df_lag)
        lag_values = df_lag[lag_col].values.astype(np.float32)

        self.lag_mean = lag_values.mean()
        self.lag_std = lag_values.std() + 1e-8
        lag_norm = (lag_values - self.lag_mean) / self.lag_std

        context_scaled = self.scaler.fit_transform(context)
        context_dim = context_scaled.shape[1]

        self.model = DenoisingMLP(context_dim, hidden_dim=128).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        context_t = torch.tensor(context_scaled, dtype=torch.float32).to(self.device)
        lag_t = torch.tensor(lag_norm, dtype=torch.float32).unsqueeze(1).to(self.device)

        n = len(context_scaled)
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                ctx_batch = context_t[idx]
                lag_batch = lag_t[idx]

                noise_level = np.random.choice(self.noise_levels)
                noise = torch.randn_like(lag_batch) * noise_level
                noisy_lag = lag_batch + noise

                pred = self.model(noisy_lag, ctx_batch)
                loss = loss_fn(pred, lag_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.is_fitted = True
        print(f"    Diffusion Imputer trained ({self.epochs} epochs, "
              f"final loss: {total_loss:.4f})")
        return self

    def impute(self, df: pd.DataFrame, lag_col: str = "exact_lag_demand") -> pd.DataFrame:
        df = df.copy()
        context = self._prepare_context(df)
        context_scaled = self.scaler.transform(context)

        df["imputed_lag"] = df[lag_col].copy()
        df["imputed_lag_var"] = 0.0
        df["is_lag_imputed"] = df[lag_col].isna().astype(int)

        if not self.is_fitted:
            if "combined_lag" in df.columns:
                df["imputed_lag"] = df["imputed_lag"].fillna(df["combined_lag"])
            df["imputed_lag"] = df["imputed_lag"].fillna(0.0)
            return df

        missing_mask = df[lag_col].isna()
        if missing_mask.sum() == 0:
            return df

        context_missing = context_scaled[missing_mask.values]
        context_t = torch.tensor(context_missing, dtype=torch.float32).to(self.device)

        self.model.eval()
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                noise = torch.randn(len(context_missing), 1).to(self.device)
                pred = self.model(noise, context_t)
                pred_np = pred.cpu().numpy().flatten() * self.lag_std + self.lag_mean
                samples.append(pred_np)

        samples = np.array(samples)
        imputed_mean = np.clip(np.mean(samples, axis=0), 0, None)
        imputed_var = np.var(samples, axis=0)

        df.loc[missing_mask, "imputed_lag"] = imputed_mean
        df.loc[missing_mask, "imputed_lag_var"] = imputed_var

        print(f"    Diffusion Imputer: imputed {missing_mask.sum()} rows, "
              f"mean_var={imputed_var.mean():.6f}")
        return df


class FastKNNImputer:
    """Deterministic KNN-based imputer for missing lag features.

    Fast fallback when diffusion is too slow or fails to converge.
    Uses KNN regressor on context features to predict missing lag values.
    Adds small Gaussian noise to the variance to enable uncertainty weighting.
    """

    def __init__(self, context_features: list, n_neighbors: int = 10):
        self.context_features = context_features
        self.n_neighbors = n_neighbors
        self.knn = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _prepare_context(self, df: pd.DataFrame) -> np.ndarray:
        available = [f for f in self.context_features if f in df.columns]
        return df[available].values.astype(np.float32)

    def fit(self, df: pd.DataFrame, lag_col: str = "exact_lag_demand"):
        mask = df[lag_col].notna()
        df_lag = df[mask].copy()

        if len(df_lag) < 10:
            print("    FastKNN Imputer: Not enough lag data, skipping")
            self.is_fitted = False
            return self

        context = self._prepare_context(df_lag)
        lag_values = df_lag[lag_col].values

        context_scaled = self.scaler.fit_transform(context)
        self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance")
        self.knn.fit(context_scaled, lag_values)

        self.is_fitted = True
        print(f"    FastKNN Imputer fitted on {len(df_lag)} rows")
        return self

    def impute(self, df: pd.DataFrame, lag_col: str = "exact_lag_demand") -> pd.DataFrame:
        df = df.copy()
        context = self._prepare_context(df)
        context_scaled = self.scaler.transform(context)

        df["imputed_lag"] = df[lag_col].copy()
        df["imputed_lag_var"] = 0.0
        df["is_lag_imputed"] = df[lag_col].isna().astype(int)

        if not self.is_fitted:
            if "combined_lag" in df.columns:
                df["imputed_lag"] = df["imputed_lag"].fillna(df["combined_lag"])
            df["imputed_lag"] = df["imputed_lag"].fillna(0.0)
            return df

        missing_mask = df[lag_col].isna()
        if missing_mask.sum() == 0:
            return df

        # KNN prediction
        imputed_values = self.knn.predict(context_scaled[missing_mask.values])
        imputed_values = np.clip(imputed_values, 0, None)

        # Small variance based on distance to nearest neighbors
        # This enables uncertainty weighting even for deterministic imputation
        distances, _ = self.knn.kneighbors(context_scaled[missing_mask.values])
        mean_dist = distances.mean(axis=1)
        imputed_var = mean_dist * 0.01  # Scale to reasonable variance

        df.loc[missing_mask, "imputed_lag"] = imputed_values
        df.loc[missing_mask, "imputed_lag_var"] = imputed_var

        print(f"    FastKNN Imputer: imputed {missing_mask.sum()} rows")
        return df


def add_diffusion_imputation(train_df: pd.DataFrame, val_or_test_df: pd.DataFrame,
                              context_features: list = None,
                              lag_col: str = "exact_lag_demand",
                              use_fast: bool = False) -> tuple:
    """Full imputation pipeline with fallback support.

    Args:
        train_df: Training DataFrame with lag data
        val_or_test_df: Validation or test DataFrame
        context_features: List of context feature column names
        lag_col: Name of the lag column
        use_fast: If True, use FastKNN instead of Diffusion

    Returns:
        (train_df, val_or_test_df) with imputation columns added
    """
    if context_features is None:
        context_features = [
            "hour", "minute", "minute_of_day", "15_min_slot", "day_of_week",
            "hour_sin", "hour_cos", "slot_sin", "slot_cos",
            "latitude", "longitude", "Temperature",
            "geo_demand_mean", "geo_demand_std",
            "dist_to_center",
        ]

    if use_fast:
        print("  Training FastKNN Imputer (deterministic fallback)...")
        imputer = FastKNNImputer(context_features=context_features)
    else:
        print("  Training Diffusion Imputer...")
        imputer = DiffusionImputer(context_features=context_features, epochs=30)

    imputer.fit(train_df, lag_col=lag_col)

    train_df = imputer.impute(train_df, lag_col=lag_col)
    val_or_test_df = imputer.impute(val_or_test_df, lag_col=lag_col)

    return train_df, val_or_test_df
