"""Evaluation metrics and plots for imbalanced binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalReport:
    auprc: float
    roc_auc: float
    precision_at_recall_80: float
    confusion_at_best_f1: np.ndarray
    best_f1_threshold: float


def evaluate(y_true: pd.Series, y_score: np.ndarray) -> EvalReport:
    """Compute AUPRC, ROC-AUC, and threshold-dependent metrics at best F1."""
    raise NotImplementedError


def plot_pr_curve(y_true: pd.Series, y_score: np.ndarray, label: str) -> None:
    raise NotImplementedError


def plot_roc_curve(y_true: pd.Series, y_score: np.ndarray, label: str) -> None:
    raise NotImplementedError


def plot_benford_histogram(
    digits: pd.Series, title: str = "Leading digit distribution"
) -> None:
    """Empirical histogram overlaid with Benford's predicted curve."""
    raise NotImplementedError


def model_comparison_table(reports: dict[str, EvalReport]) -> pd.DataFrame:
    """One row per model, columns are the metrics from EvalReport."""
    raise NotImplementedError
