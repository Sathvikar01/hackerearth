"""Simplified Diffusion Imputer for Missing Lag Features.

Instead of a full conditional diffusion model (which requires weeks of training),
this implements a practical Tabular Diffusion approach:
1. Trains a small MLP denoiser to reconstruct lag from spatial + temporal context
2. Generates N imputation samples to compute Mean and Variance
3. Provides uncertainty-aware features for the downstream GBDT

This bridges the gap for rows where exact_lag_demand is NaN.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class DenoisingMLP(nn.Module):
    """Small MLP that denoises lag features conditioned on context."""

    def __init__(self, context_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + 1, hidden_dim),  # +1 for noisy lag
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),  # Predict clean lag
        )

    def forward(self, noisy_lag, context):
        x = torch.cat([noisy_lag, context], dim=-1)
        return self.net(x)


class DiffusionImputer:
    """Simplified diffusion-based imputer for missing lag features.

    Instead of full diffusion (which needs extensive training), this uses
    a denoising MLP trained with noise augmentation to:
    1. Impute missing lag values from context features
    2. Generate multiple samples for uncertainty estimation
    """

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
        """Extract and scale context features."""
        available = [f for f in self.context_features if f in df.columns]
        return df[available].values.astype(np.float32)

    def fit(self, df: pd.DataFrame, lag_col: str = "exact_lag_demand",
            target_col: str = "demand"):
        """Train the denoising model on rows WITH lag data.

        The model learns: given noisy_lag + context features -> clean lag.
        This way, at inference time, we can impute missing lags from context alone.
        """
        # Filter to rows with lag
        mask = df[lag_col].notna()
        df_lag = df[mask].copy()

        if len(df_lag) < 100:
            print("    Diffusion Imputer: Not enough lag data, skipping")
            self.is_fitted = False
            return self

        context = self._prepare_context(df_lag)
        lag_values = df_lag[lag_col].values.astype(np.float32)

        # Normalize
        self.lag_mean = lag_values.mean()
        self.lag_std = lag_values.std() + 1e-8
        lag_norm = (lag_values - self.lag_mean) / self.lag_std

        context_scaled = self.scaler.fit_transform(context)
        context_dim = context_scaled.shape[1]

        # Build model
        self.model = DenoisingMLP(context_dim, hidden_dim=128).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # Training loop with noise augmentation
        context_t = torch.tensor(context_scaled, dtype=torch.float32).to(self.device)
        lag_t = torch.tensor(lag_norm, dtype=torch.float32).unsqueeze(1).to(self.device)

        n = len(context_scaled)
        for epoch in range(self.epochs):
            # Shuffle
            perm = torch.randperm(n)
            total_loss = 0.0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                ctx_batch = context_t[idx]
                lag_batch = lag_t[idx]

                # Add noise at random level
                noise_level = np.random.choice(self.noise_levels)
                noise = torch.randn_like(lag_batch) * noise_level
                noisy_lag = lag_batch + noise

                # Predict clean lag from noisy lag + context
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
        """Impute missing lag values with uncertainty estimates.

        For rows WITH lag: imputed_mean = actual lag, imputed_var = 0
        For rows WITHOUT lag: generates N samples, computes mean and variance

        Adds columns:
        - 'imputed_lag': Best estimate (actual or imputed mean)
        - 'imputed_lag_var': Uncertainty (0 for real, variance for imputed)
        - 'is_lag_imputed': Binary flag (0=real, 1=imputed)
        """
        df = df.copy()
        context = self._prepare_context(df)
        context_scaled = self.scaler.transform(context)

        # Initialize columns
        df["imputed_lag"] = df[lag_col].copy()
        df["imputed_lag_var"] = 0.0
        df["is_lag_imputed"] = df[lag_col].isna().astype(int)

        if not self.is_fitted:
            # Fallback: use combined_lag or 0
            if "combined_lag" in df.columns:
                df["imputed_lag"] = df["imputed_lag"].fillna(df["combined_lag"])
            df["imputed_lag"] = df["imputed_lag"].fillna(0.0)
            return df

        # Find rows needing imputation
        missing_mask = df[lag_col].isna()
        if missing_mask.sum() == 0:
            return df

        # Generate N samples for missing rows
        context_missing = context_scaled[missing_mask.values]
        context_t = torch.tensor(context_missing, dtype=torch.float32).to(self.device)

        self.model.eval()
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Start from noise (simplified diffusion: single-step denoise)
                noise = torch.randn(len(context_missing), 1).to(self.device)
                pred = self.model(noise, context_t)
                # Denormalize
                pred_np = pred.cpu().numpy().flatten() * self.lag_std + self.lag_mean
                samples.append(pred_np)

        samples = np.array(samples)  # (n_samples, n_missing)
        imputed_mean = np.mean(samples, axis=0)
        imputed_var = np.var(samples, axis=0)

        # Clip to non-negative
        imputed_mean = np.clip(imputed_mean, 0, None)

        # Fill in
        df.loc[missing_mask, "imputed_lag"] = imputed_mean
        df.loc[missing_mask, "imputed_lag_var"] = imputed_var

        print(f"    Diffusion Imputer: imputed {missing_mask.sum()} rows, "
              f"mean_var={imputed_var.mean():.4f}")

        return df


def add_diffusion_imputation(train_df: pd.DataFrame, val_or_test_df: pd.DataFrame,
                              context_features: list = None,
                              lag_col: str = "exact_lag_demand") -> tuple:
    """Full diffusion imputation pipeline.

    Fits on train_df (rows with lag), imputes missing lags in val_or_test_df.

    Args:
        train_df: Training DataFrame with lag data
        val_or_test_df: Validation or test DataFrame
        context_features: List of context feature column names
        lag_col: Name of the lag column

    Returns:
        (train_df, val_or_test_df) with imputation columns added
    """
    if context_features is None:
        # Default context features (numerical only)
        context_features = [
            "hour", "minute", "minute_of_day", "15_min_slot", "day_of_week",
            "hour_sin", "hour_cos", "slot_sin", "slot_cos",
            "latitude", "longitude", "Temperature",
            "geo_demand_mean", "geo_demand_std",
            "dist_to_center",
        ]

    print("  Training Diffusion Imputer...")
    imputer = DiffusionImputer(context_features=context_features, epochs=30)
    imputer.fit(train_df, lag_col=lag_col)

    # Impute train (for consistency)
    train_df = imputer.impute(train_df, lag_col=lag_col)
    # Impute val/test
    val_or_test_df = imputer.impute(val_or_test_df, lag_col=lag_col)

    return train_df, val_or_test_df
