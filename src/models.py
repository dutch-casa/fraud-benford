"""Model wrappers. All expose fit(X, y) and predict_proba(X) -> P(fraud)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class FraudModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudModel": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def logistic_regression(class_weight: str | None = None) -> FraudModel:
    raise NotImplementedError


def xgboost_classifier(**kwargs) -> FraudModel:
    raise NotImplementedError


def lightgbm_classifier(**kwargs) -> FraudModel:
    raise NotImplementedError


def mlp_classifier(hidden: tuple[int, ...] = (64, 32)) -> FraudModel:
    raise NotImplementedError


def autoencoder_anomaly(latent_dim: int = 8) -> FraudModel:
    """Train on legit-only, score test rows by reconstruction error."""
    raise NotImplementedError
