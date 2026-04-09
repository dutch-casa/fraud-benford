"""Model wrappers. All expose fit(X, y) and predict_proba(X) -> (n, 2) array."""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
import pandas as pd

RANDOM_SEED = 42


class FraudModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def logistic_regression(class_weight: str | None = None) -> FraudModel:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def xgboost_classifier(scale_pos_weight: float = 1.0, **kwargs) -> FraudModel:
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    params.update(kwargs)
    return XGBClassifier(**params)


def lightgbm_classifier(class_weight: str | None = "balanced", **kwargs) -> FraudModel:
    from lightgbm import LGBMClassifier

    params = dict(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kwargs)
    return LGBMClassifier(**params)


def mlp_classifier(hidden: tuple[int, ...] = (64, 32)) -> FraudModel:
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=40,
        batch_size=512,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_SEED,
    )


class AutoencoderAnomaly:
    """Train a small autoencoder on legit-only rows. Score = reconstruction error."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        epochs: int = 20,
        batch_size: int = 512,
        lr: float = 1e-3,
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model = None
        self._scaler = None
        self._max_error: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        import torch
        from sklearn.preprocessing import StandardScaler
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(RANDOM_SEED)

        X_legit = X.loc[y == 0].to_numpy(dtype=np.float32)
        self._scaler = StandardScaler().fit(X_legit)
        X_scaled = self._scaler.transform(X_legit).astype(np.float32)

        n_features = X_scaled.shape[1]
        self._model = nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_features),
        )

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_scaled)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                loss = loss_fn(self._model(batch), batch)
                loss.backward()
                optimizer.step()

        self._model.eval()
        with torch.no_grad():
            recon = self._model(torch.from_numpy(X_scaled))
            errors = ((recon - torch.from_numpy(X_scaled)) ** 2).mean(dim=1).numpy()
        self._max_error = float(errors.max()) or 1.0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self._model is None or self._scaler is None or self._max_error is None:
            raise RuntimeError("AutoencoderAnomaly must be fit before predict_proba")

        X_scaled = self._scaler.transform(X.to_numpy(dtype=np.float32)).astype(np.float32)
        with torch.no_grad():
            recon = self._model(torch.from_numpy(X_scaled))
            errors = ((recon - torch.from_numpy(X_scaled)) ** 2).mean(dim=1).numpy()

        scores = np.clip(errors / self._max_error, 0.0, 1.0)
        return np.stack([1.0 - scores, scores], axis=1)


def autoencoder_anomaly(latent_dim: int = 8) -> AutoencoderAnomaly:
    return AutoencoderAnomaly(latent_dim=latent_dim)
